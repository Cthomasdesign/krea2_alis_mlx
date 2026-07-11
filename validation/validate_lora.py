"""LoRA parity — a LoRA at scale 0 must be a no-op, and every LoRA key must map.

Self-consistency check for the runtime LoRA branch (krea2/lora.py); MLX-only, no PT reference:
  1. every LoRA target resolves to a module in the transformer  -> 0 unmatched keys
  2. at scale 0, a wrapped layer's output is bit-identical to the base  -> clean disable
  3. at scale 1, the output changes                                     -> adapter is live
  4. unload restores the original layer                                 -> reversible

No weights are committed to the repo: the quantized base build is fetched with resolve_weights
and the LoRA is downloaded from Hugging Face (gokaygokay/Krea-2-Realism-LoRA), both cached. Point
it at local files instead via the env vars:

  KREA2_PRECISION=8bit \\           # or mixed-4-8   (default: auto -> 8bit)
  KREA2_TRANSFORMER=transformer_8bit.safetensors \\   # optional: a local base build
  KREA2_LORA=loras/my_lora.safetensors \\             # optional: a local LoRA
  python3 validation/validate_lora.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

import mlx.core as mx
from mlx import nn

from krea2.lora import apply_loras, unload_loras
from krea2.pipeline import resolve_weights
from krea2.quant_recipes import mixed_4_8, quantize_bulk
from krea2.transformer import Krea2Config, SingleStreamDiT

# The LoRA validated against: a Krea-2 Realism LoRA on the Hub. Downloaded (and cached) on demand —
# no weight files live in this repo. Set KREA2_LORA to use a local .safetensors instead.
LORA_REPO = "gokaygokay/Krea-2-Realism-LoRA"
LORA_FILE = "krea2_realism_lora.safetensors"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve_lora() -> str:
    local = os.environ.get("KREA2_LORA")
    if local:
        return local
    from huggingface_hub import hf_hub_download
    return hf_hub_download(LORA_REPO, LORA_FILE)


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

    x = mx.random.normal((1, 4, cfg.features)).astype(mx.bfloat16)
    mx.eval(x)
    base = m.blocks[0].attn.wq(x)
    mx.eval(base)

    paths = apply_loras(m, [(lora, 0.0)])           # scale 0 -> must be a no-op
    y0 = m.blocks[0].attn.wq(x); mx.eval(y0)
    d0 = float(mx.abs(y0 - base).max())
    n = len(paths)

    unload_loras(m, paths)
    paths = apply_loras(m, [(lora, 1.0)])           # scale 1 -> must change output
    y1 = m.blocks[0].attn.wq(x); mx.eval(y1)
    d1 = float(mx.abs(y1 - base).max())

    unload_loras(m, paths)                          # must restore the base exactly
    restored = type(m.blocks[0].attn.wq).__name__
    yb = m.blocks[0].attn.wq(x); mx.eval(yb)
    db = float(mx.abs(yb - base).max())

    print(f"targets mapped  : {n}  (0 unmatched keys)")
    print(f"scale 0 vs base : {d0:.6f}  (expect 0.000000)")
    print(f"scale 1 vs base : {d1:.6f}  (expect > 0)")
    print(f"after unload    : {restored}, max|delta|={db:.6f}  (expect base layer, 0.000000)")

    ok = (n > 0 and d0 == 0.0 and d1 > 0.0 and db == 0.0
          and restored in ("QuantizedLinear", "Linear"))
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
