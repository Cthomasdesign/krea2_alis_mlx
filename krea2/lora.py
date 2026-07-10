"""Runtime LoRA adapters for the Krea-2 MLX transformer.

Loads Krea-2 LoRAs (https://huggingface.co/collections/krea/krea-2-loras) and
applies each as a low-rank side branch on top of the base layer:

    y = base(x) + Σᵢ scaleᵢ · (x · Aᵢᵀ) · Bᵢᵀ

Because the branch only *calls* the base layer, it works unchanged on both
`nn.Linear` and `nn.QuantizedLinear` — the 8-bit and mixed-4/8 builds take
LoRAs with no dequantize/merge/requantize round-trip. Adapters stack (summed
per layer) and are fully reversible: `clear_loras` restores the exact base
modules, so scale 0 and a cleared model are byte-identical to no LoRA at all.

Key mapping is ~1:1 because this port kept the reference tensor names: a
trained key like `base_model.model.blocks.3.attn.wq.lora_A.weight` strips to
the module path `blocks.3.attn.wq`. Unmapped keys raise instead of silently
mis-applying.
"""

from __future__ import annotations

import json
import os

import mlx.core as mx
from mlx import nn

# training-framework prefixes stripped (repeatedly) from checkpoint keys
_PREFIXES = ("base_model.model.", "diffusion_model.", "transformer.", "unet.")
# suffix -> role; longest match wins (`.lora_A.default.weight` before `.lora_A.weight`)
_SUFFIXES = {
    ".lora_A.default.weight": "A",   # PEFT with a named adapter
    ".lora_B.default.weight": "B",
    ".lora_A.weight": "A",           # PEFT
    ".lora_B.weight": "B",
    ".lora.down.weight": "A",        # diffusers attn-processor style
    ".lora.up.weight": "B",
    ".lora_down.weight": "A",
    ".lora_up.weight": "B",
    ".alpha": "alpha",               # per-module alpha (kohya-style)
}
_SUFFIX_ORDER = sorted(_SUFFIXES, key=len, reverse=True)


def _parse_key(key: str):
    """checkpoint key -> (module_path, role) or None if not a LoRA key we understand."""
    for suf in _SUFFIX_ORDER:
        if key.endswith(suf):
            path = key[: -len(suf)]
            stripped = True
            while stripped:  # some exporters stack prefixes (e.g. base_model.model.transformer.)
                stripped = False
                for pre in _PREFIXES:
                    if path.startswith(pre):
                        path = path[len(pre):]
                        stripped = True
            return path, _SUFFIXES[suf]
    return None


def load_lora(path: str) -> dict[str, tuple[mx.array, mx.array, float]]:
    """Load a LoRA .safetensors file -> {module_path: (A, B, alpha_scale)}.

    A is (rank, in_features), B is (out_features, rank); alpha_scale is the
    trained alpha/rank factor (1.0 when the file carries no alpha). Raises
    ValueError on keys that don't parse — loud beats silently mis-applied.
    """
    weights = mx.load(path)
    per_module: dict[str, dict[str, mx.array]] = {}
    bad = []
    for key, value in weights.items():
        parsed = _parse_key(key)
        if parsed is None:
            bad.append(key)
            continue
        mpath, role = parsed
        per_module.setdefault(mpath, {})[role] = value
    if bad:
        raise ValueError(
            f"{path}: {len(bad)} key(s) don't look like a Krea-2 LoRA "
            f"(expected PEFT/diffusers `…lora_A/lora_B` keys); first few: {bad[:5]}")
    if not per_module:
        raise ValueError(f"{path}: no LoRA keys found.")

    # PEFT stores alpha in a sidecar adapter_config.json rather than in the tensors
    default_alpha = None
    cfg = os.path.join(os.path.dirname(os.path.abspath(path)), "adapter_config.json")
    if os.path.exists(cfg):
        try:
            default_alpha = json.load(open(cfg)).get("lora_alpha")
        except (OSError, ValueError):
            default_alpha = None

    out = {}
    for mpath, parts in per_module.items():
        if "A" not in parts or "B" not in parts:
            raise ValueError(f"{path}: module '{mpath}' is missing its lora_A or lora_B tensor.")
        a, b = parts["A"], parts["B"]
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]:
            raise ValueError(
                f"{path}: module '{mpath}' has inconsistent shapes A={list(a.shape)} B={list(b.shape)} "
                "(want A=(rank, in), B=(out, rank)).")
        rank = a.shape[0]
        alpha = parts["alpha"].item() if "alpha" in parts else default_alpha
        out[mpath] = (a, b, float(alpha) / rank if alpha else 1.0)
    return out


class LoRALinear(nn.Module):
    """Wraps a Linear/QuantizedLinear; adds the summed low-rank branches to its output.

    Adapters live in the underscore-prefixed `_adapters`, so they are invisible
    to `parameters()` / `load_weights` — the wrapped model's parameter tree
    stays exactly the base tree plus a `.base` hop.
    """

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base
        self._adapters: list[tuple[mx.array, mx.array, float]] = []

    def __call__(self, x: mx.array) -> mx.array:
        y = self.base(x)
        for a, b, scale in self._adapters:
            y = y + (x @ a.T.astype(x.dtype) @ b.T.astype(x.dtype)) * scale
        return y


def _linear_dims(module) -> tuple[int, int] | None:
    """(in_features, out_features) of a Linear or QuantizedLinear, else None."""
    if isinstance(module, nn.QuantizedLinear):
        return module.scales.shape[1] * module.group_size, module.scales.shape[0]
    if isinstance(module, nn.Linear):
        return module.weight.shape[1], module.weight.shape[0]
    return None


def _resolve(model, mpath: str):
    """module path like 'blocks.3.attn.wq' -> (parent_container, last_key, module)."""
    parts = mpath.split(".")
    obj = model
    for i, p in enumerate(parts):
        parent = obj
        if isinstance(obj, (list, tuple)):
            if not p.isdigit() or int(p) >= len(obj):
                raise KeyError(mpath)
            obj = obj[int(p)]
        elif isinstance(obj, nn.Module) and p in obj:
            obj = obj[p]
        elif isinstance(obj, dict) and p in obj:
            obj = obj[p]
        else:
            raise KeyError(mpath)
        if i == len(parts) - 1:
            return parent, (int(p) if isinstance(parent, list) else p), obj
    raise KeyError(mpath)  # unreachable (mpath is never empty)


def apply_lora(model: nn.Module, lora, scale: float = 1.0) -> int:
    """Apply one LoRA (path or `load_lora` dict) on top of `model` at `scale`.

    Stacks with already-applied adapters. All-or-nothing: every target is
    resolved and shape-checked before the first module is touched, and any
    unmapped/mismatched target raises ValueError. Returns the target count.
    """
    if isinstance(lora, str):
        lora = load_lora(lora)
    plan, bad = [], []
    for mpath, (a, b, alpha_scale) in lora.items():
        try:
            parent, key, module = _resolve(model, mpath)
        except KeyError:
            bad.append(f"{mpath} (no such module)")
            continue
        base = module.base if isinstance(module, LoRALinear) else module
        dims = _linear_dims(base)
        if dims is None:
            bad.append(f"{mpath} (not a Linear)")
        elif dims != (a.shape[1], b.shape[0]):
            bad.append(f"{mpath} (LoRA is {a.shape[1]}->{b.shape[0]}, layer is {dims[0]}->{dims[1]})")
        else:
            plan.append((parent, key, module, a, b, scale * alpha_scale))
    if bad:
        raise ValueError(f"LoRA does not match this model — {len(bad)} unmapped target(s): "
                         + "; ".join(bad[:8]) + ("; …" if len(bad) > 8 else ""))
    for parent, key, module, a, b, s in plan:
        if not isinstance(module, LoRALinear):
            wrapper = LoRALinear(module)
            if isinstance(parent, list):
                parent[key] = wrapper
            elif isinstance(parent, nn.Module):
                setattr(parent, key, wrapper)
            else:  # plain dict container
                parent[key] = wrapper
            module = wrapper
        module._adapters.append((a, b, s))
        mx.eval(a, b)
    return len(plan)


def _walk_wrapped(container):
    """Yield (parent_container, key, LoRALinear) for every wrapper in the tree."""
    if isinstance(container, LoRALinear):
        return  # never wrapped twice; nothing below to restore
    if isinstance(container, nn.Module) or isinstance(container, dict):
        items = [(k, v) for k, v in container.items() if not str(k).startswith("_")]
    elif isinstance(container, list):
        items = list(enumerate(container))
    else:
        return
    for key, value in items:
        if isinstance(value, LoRALinear):
            yield container, key, value
        elif isinstance(value, (nn.Module, dict, list)):
            yield from _walk_wrapped(value)


def clear_loras(model: nn.Module) -> int:
    """Remove every applied LoRA, restoring the exact original modules."""
    wrapped = list(_walk_wrapped(model))
    for parent, key, wrapper in wrapped:
        if isinstance(parent, nn.Module):
            setattr(parent, key, wrapper.base)
        else:
            parent[key] = wrapper.base
    return len(wrapped)


def set_loras(model: nn.Module, loras) -> list[tuple[str, float]]:
    """Replace the active LoRA set. `loras`: iterable of path or (path, scale);
    empty/None clears. Returns the applied [(path, scale), …]. If any LoRA in
    the set fails to apply, the model is restored to base before re-raising —
    it is never left with a partial set."""
    clear_loras(model)
    applied = []
    try:
        for item in loras or []:
            path, scale = item if isinstance(item, (tuple, list)) else (item, 1.0)
            scale = float(scale)
            n = apply_lora(model, path, scale=scale)
            applied.append((path, scale))
            print(f"LoRA {os.path.basename(str(path))}: {n} layers @ scale {scale:g}")
    except Exception:
        clear_loras(model)
        raise
    return applied
