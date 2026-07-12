"""LoRA parity — a LoRA at scale 0 must be a no-op, and every LoRA key must map.

Self-consistency checks for the runtime LoRA branch (krea2/lora.py); MLX-only, no PT reference:
  1. every LoRA target resolves to a linear in the transformer          -> 0 unmatched keys
  2. the official diffusers naming (darkbrush key fixture) maps 1:1     -> translate table covered
  3. at scale 0, a wrapped layer's output is bit-identical to the base  -> clean disable
  4. at scale 1, the output changes                                     -> adapter is live
  5. unload restores the original layer                                 -> reversible
  6. a spec that fails mid-validation leaves the model untouched        -> two-phase apply
  7. set_loras: apply -> replace -> fail -> retry all behave            -> exception-safe state

No weights are committed to the repo: the quantized base build is fetched with resolve_weights
and the LoRA is downloaded from Hugging Face (gokaygokay/Krea-2-Realism-LoRA), both cached. The
diffusers-convention check needs no download at all — it dry-resolves the committed key list of
krea/Krea-2-LoRA-darkbrush (fixtures/darkbrush_keys.txt) against the module tree. Point the
runtime checks at local files instead via the env vars:

  KREA2_PRECISION=8bit \\           # or mixed-4-8   (default: auto -> 8bit)
  KREA2_TRANSFORMER=transformer_8bit.safetensors \\   # optional: a local base build
  KREA2_LORA=loras/my_lora.safetensors \\             # optional: a local LoRA
  python3 validation/validate_lora.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

import mlx.core as mx
from mlx import nn

from krea2.lora import LoRALinear, _get_submodule, _to_mlx_path, apply_loras, unload_loras
from krea2.pipeline import Krea2Pipeline, resolve_weights
from krea2.quant_recipes import mixed_4_8, quantize_bulk
from krea2.transformer import Krea2Config, SingleStreamDiT

# The LoRA validated against: a Krea-2 Realism LoRA on the Hub. Downloaded (and cached) on demand —
# no weight files live in this repo. Set KREA2_LORA to use a local .safetensors instead.
LORA_REPO = "gokaygokay/Krea-2-Realism-LoRA"
LORA_FILE = "krea2_realism_lora.safetensors"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE = os.path.join(REPO_ROOT, "validation", "fixtures", "darkbrush_keys.txt")

_SUFFIXES = (".lora_A.weight", ".lora_B.weight", ".lora_down.weight", ".lora_up.weight")


def _resolve_lora() -> str:
    local = os.environ.get("KREA2_LORA")
    if local:
        return local
    from huggingface_hub import hf_hub_download
    return hf_hub_download(LORA_REPO, LORA_FILE)


def _probe(m, x):
    y = m.blocks[0].attn.wq(x)
    mx.eval(y)
    return y


def _diffusers_fixture_check(m) -> tuple[int, int]:
    """Dry-resolve the official darkbrush key list (diffusers naming) through the mapper
    against the real module tree. Returns (targets, unmatched)."""
    targets = set()
    with open(FIXTURE) as f:
        for line in f:
            k = line.strip()
            if not k or k.startswith("#"):
                continue
            for suf in _SUFFIXES:
                if k.endswith(suf):
                    targets.add(_to_mlx_path(k[: -len(suf)]))
                    break
    unmatched = 0
    for tgt in sorted(targets):
        try:
            sub = _get_submodule(m, tgt)
            if not isinstance(sub, (nn.Linear, nn.QuantizedLinear, LoRALinear)):
                unmatched += 1
        except (AttributeError, IndexError, KeyError, ValueError):
            unmatched += 1
    return len(targets), unmatched


def main() -> int:
    precision = os.environ.get("KREA2_PRECISION")
    tpath = os.environ.get("KREA2_TRANSFORMER")
    if tpath:                                # explicit local base -> trust KREA2_PRECISION (default 8bit)
        precision = precision or "8bit"
    else:                                    # otherwise resolve/download the build (cached)
        precision, tpath = resolve_weights(REPO_ROOT, precision=precision, download=True)
    lora = _resolve_lora()
    print(f"base   : {precision}  {tpath}")
    print(f"lora   : {lora}")

    cfg = Krea2Config()
    m = SingleStreamDiT(cfg)
    if precision == "mixed-4-8":             # mirror Krea2Pipeline's per-precision quantization
        nn.quantize(m, group_size=64, bits=4, class_predicate=mixed_4_8)
    else:                                    # 8bit
        nn.quantize(m, group_size=64, bits=8, class_predicate=quantize_bulk)
    m.load_weights(tpath, strict=True)
    mx.eval(m.parameters())

    # 2. official diffusers naming maps 1:1 onto the module tree (no weights needed)
    fx_targets, fx_unmatched = _diffusers_fixture_check(m)

    x = mx.random.normal((1, 4, cfg.features)).astype(mx.bfloat16)
    mx.eval(x)
    base = _probe(m, x)

    paths = apply_loras(m, [(lora, 0.0)])           # scale 0 -> must be a no-op
    d0 = float(mx.abs(_probe(m, x) - base).max())
    n = len(paths)

    unload_loras(m, paths)
    paths = apply_loras(m, [(lora, 1.0)])           # scale 1 -> must change output
    d1 = float(mx.abs(_probe(m, x) - base).max())

    unload_loras(m, paths)                          # must restore the base exactly
    restored = type(m.blocks[0].attn.wq).__name__
    db = float(mx.abs(_probe(m, x) - base).max())

    # 6. two-phase apply: 1 valid + 1 unmapped target -> raises, model untouched (no leaked wrapper)
    tmp = tempfile.mkdtemp()
    rank, feats = 4, cfg.features
    bad_mixed = os.path.join(tmp, "mixed.safetensors")
    mx.save_safetensors(bad_mixed, {
        "blocks.0.attn.wq.lora_A.weight": mx.random.normal((rank, feats)) * 0.01,
        "blocks.0.attn.wq.lora_B.weight": mx.random.normal((feats, rank)) * 0.01,
        "no_such_layer.lora_A.weight": mx.random.normal((rank, feats)),
        "no_such_layer.lora_B.weight": mx.random.normal((feats, rank)),
    })
    try:
        apply_loras(m, [(bad_mixed, 1.0)])
        two_phase = False                            # should have raised
    except ValueError:
        two_phase = (not isinstance(m.blocks[0].attn.wq, LoRALinear)
                     and float(mx.abs(_probe(m, x) - base).max()) == 0.0)

    # 7. set_loras lifecycle on a pipeline shell (transformer only — no VAE / text encoder)
    pipe = Krea2Pipeline.__new__(Krea2Pipeline)
    pipe.transformer, pipe._lora_sig, pipe._lora_paths = m, (), []
    pipe.set_loras([(lora, 0.9)])
    lc_apply = float(mx.abs(_probe(m, x) - base).max()) > 0
    pipe.set_loras([(lora, 0.9)])                    # unchanged set -> cheap no-op, still active
    lc_noop = float(mx.abs(_probe(m, x) - base).max()) > 0
    try:
        pipe.set_loras([(bad_mixed, 1.0)])           # replace with a failing set
        lc_fail = False                              # should have raised
    except ValueError:
        lc_fail = float(mx.abs(_probe(m, x) - base).max()) == 0.0   # -> exact base state
    pipe.set_loras([(lora, 0.9)])                    # retry the working set -> must re-apply
    lc_retry = float(mx.abs(_probe(m, x) - base).max()) > 0
    pipe.set_loras([])                               # clear -> exact base
    lc_clear = float(mx.abs(_probe(m, x) - base).max()) == 0.0
    lifecycle = lc_apply and lc_noop and lc_fail and lc_retry and lc_clear

    print(f"targets mapped   : {n}  (0 unmatched keys)")
    print(f"diffusers fixture: {fx_targets} targets, {fx_unmatched} unmatched  (expect 264, 0)")
    print(f"scale 0 vs base  : {d0:.6f}  (expect 0.000000)")
    print(f"scale 1 vs base  : {d1:.6f}  (expect > 0)")
    print(f"after unload     : {restored}, max|delta|={db:.6f}  (expect base layer, 0.000000)")
    print(f"two-phase apply  : {'model untouched after failed spec' if two_phase else 'LEAKED'}")
    print(f"set_loras        : apply={lc_apply} noop={lc_noop} fail->base={lc_fail} "
          f"retry={lc_retry} clear={lc_clear}")

    ok = (n > 0 and fx_targets == 264 and fx_unmatched == 0
          and d0 == 0.0 and d1 > 0.0 and db == 0.0
          and restored in ("QuantizedLinear", "Linear")
          and two_phase and lifecycle)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
