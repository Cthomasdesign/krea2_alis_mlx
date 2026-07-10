"""LoRA parity test — no weights, no download (runs on the CI smoke job).

On a small random-init SingleStreamDiT (bf16 and quantized mixed-4/8):
  1. a synthetic PEFT-format LoRA maps with 0 unmatched keys,
  2. scale 0 is byte-identical to the base model (clean disable),
  3. scale 1 changes the output; stacking two adapters changes it again,
  4. clear_loras restores byte-identical base outputs,
  5. unknown module paths / bad key formats / shape mismatches raise ValueError.
"""

import os
import sys
import tempfile

import mlx.core as mx
import numpy as np
from mlx import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from krea2.lora import LoRALinear, apply_lora, clear_loras, load_lora, set_loras  # noqa: E402
from krea2.quant_recipes import mixed_4_8  # noqa: E402
from krea2.transformer import Krea2Config, SingleStreamDiT  # noqa: E402

CFG = Krea2Config(features=384, tdim=32, txtdim=128, heads=6, kvheads=3,
                  multiplier=2, layers=2, patch=2, channels=4,
                  txtheads=4, txtkvheads=4, txtlayers=2)
RANK = 4
# targets mirror what Krea-2 LoRAs train: block attn/mlp, txtfusion, endpoints
TARGETS = [
    ("blocks.0.attn.wq", CFG.features, CFG.features),
    ("blocks.1.mlp.down", 512, CFG.features),  # SwiGLU mlpdim for these dims
    ("txtfusion.refiner_blocks.0.attn.wo", CFG.txtdim, CFG.txtdim),
    ("first", CFG.channels * CFG.patch**2, CFG.features),
    ("last.linear", CFG.features, CFG.patch**2 * CFG.channels),
]


def make_lora(tmp, name, seed, targets=TARGETS, rank=RANK):
    mx.random.seed(seed)
    tensors = {}
    for path, fin, fout in targets:
        tensors[f"base_model.model.{path}.lora_A.weight"] = \
            (mx.random.normal((rank, fin)) * 0.1).astype(mx.float32)
        tensors[f"base_model.model.{path}.lora_B.weight"] = \
            (mx.random.normal((fout, rank)) * 0.1).astype(mx.float32)
    p = os.path.join(tmp, f"{name}.safetensors")
    mx.save_safetensors(p, tensors)
    return p


def forward(m):
    B, seq, h_, w_ = 1, 4, 4, 4
    Limg = h_ * w_
    mx.random.seed(7)
    img = mx.random.normal((B, Limg, CFG.channels * CFG.patch**2)).astype(mx.bfloat16)
    context = mx.random.normal((B, seq, CFG.txtlayers, CFG.txtdim)).astype(mx.bfloat16)
    t = mx.array([1.0])
    txtpos = np.zeros((seq, 3), np.float32)
    imgids = np.zeros((h_, w_, 3), np.float32)
    imgids[..., 1] = np.arange(h_)[:, None]
    imgids[..., 2] = np.arange(w_)[None, :]
    pos = mx.array(np.concatenate([txtpos, imgids.reshape(-1, 3)], 0))
    mask = mx.ones((B, seq + Limg))
    out = m(img, context, t, pos, mask)
    mx.eval(out)
    return np.array(out.astype(mx.float32))


def build(quantized):
    mx.random.seed(0)
    m = SingleStreamDiT(CFG)
    if quantized:
        mx.eval(m.parameters())
        nn.quantize(m, group_size=64, bits=4, class_predicate=mixed_4_8)
    mx.eval(m.parameters())
    return m


def run_suite(quantized, tmp):
    tag = "quant" if quantized else "bf16 "
    m = build(quantized)
    base = forward(m)

    lora1 = make_lora(tmp, "l1", seed=1)
    lora2 = make_lora(tmp, "l2", seed=2)

    n = apply_lora(m, lora1, scale=0.0)
    assert n == len(TARGETS), f"mapped {n}, want {len(TARGETS)}"
    out0 = forward(m)
    assert np.array_equal(base, out0), "scale 0 must be byte-identical to base"
    print(f"[{tag}] scale-0 parity OK ({n}/{len(TARGETS)} targets mapped)")

    clear_loras(m)
    apply_lora(m, lora1, scale=1.0)
    out1 = forward(m)
    assert not np.array_equal(base, out1), "scale 1 must change the output"
    apply_lora(m, lora2, scale=0.7)  # stack on top
    out12 = forward(m)
    assert not np.array_equal(out1, out12), "stacked adapter must change the output"
    print(f"[{tag}] scale-1 shifts output; stacking shifts it again OK")

    removed = clear_loras(m)
    assert removed == len(TARGETS)
    assert np.array_equal(base, forward(m)), "clear_loras must restore the base exactly"
    print(f"[{tag}] clear restores base byte-identical OK")

    # set_loras replaces (not stacks) the active set
    set_loras(m, [(lora1, 1.0)])
    set_loras(m, [(lora1, 1.0)])
    assert np.array_equal(out1, forward(m)), "set_loras must replace, not stack"
    set_loras(m, [])
    assert np.array_equal(base, forward(m))
    print(f"[{tag}] set_loras replace/clear OK")


def run_errors(tmp):
    m = build(quantized=False)

    p = make_lora(tmp, "bad_target", seed=3, targets=[("blocks.9.attn.nope", 384, 384)])
    try:
        apply_lora(m, p)
        raise AssertionError("unknown module path must raise")
    except ValueError as e:
        assert "unmapped" in str(e)

    p = make_lora(tmp, "bad_shape", seed=4, targets=[("blocks.0.attn.wq", 384, 999)])
    try:
        apply_lora(m, p)
        raise AssertionError("shape mismatch must raise")
    except ValueError as e:
        assert "unmapped" in str(e)

    junk = os.path.join(tmp, "junk.safetensors")
    mx.save_safetensors(junk, {"some.random.tensor": mx.zeros((2, 2))})
    try:
        load_lora(junk)
        raise AssertionError("non-LoRA keys must raise")
    except ValueError:
        pass
    assert not any(isinstance(x, LoRALinear) for x in m.modules()), \
        "failed applies must not leave wrappers behind"

    # a failing set_loras must restore the base model, not leave a partial set
    base = forward(m)
    good = make_lora(tmp, "good", seed=6)
    try:
        set_loras(m, [(good, 1.0), (p, 1.0)])
        raise AssertionError("bad second LoRA must raise")
    except ValueError:
        pass
    assert not any(isinstance(x, LoRALinear) for x in m.modules())
    assert np.array_equal(base, forward(m)), "failed set_loras must restore base"
    print("[err ] unmapped path / shape mismatch / junk keys / partial-set rollback OK")


def run_alpha(tmp):
    # kohya-style per-module alpha tensor scales by alpha/rank
    m = build(quantized=False)
    base = forward(m)
    mx.random.seed(5)
    a = (mx.random.normal((RANK, CFG.features)) * 0.1).astype(mx.float32)
    b = (mx.random.normal((CFG.features, RANK)) * 0.1).astype(mx.float32)
    p_noalpha = os.path.join(tmp, "noalpha.safetensors")
    mx.save_safetensors(p_noalpha, {
        "base_model.model.blocks.0.attn.wq.lora_A.weight": a,
        "base_model.model.blocks.0.attn.wq.lora_B.weight": b,
    })
    p_alpha = os.path.join(tmp, "alpha.safetensors")
    mx.save_safetensors(p_alpha, {
        "base_model.model.blocks.0.attn.wq.lora_A.weight": a,
        "base_model.model.blocks.0.attn.wq.lora_B.weight": b,
        "base_model.model.blocks.0.attn.wq.alpha": mx.array(float(2 * RANK)),
    })
    apply_lora(m, p_noalpha, scale=2.0)  # alpha/rank == 2 folded into scale…
    out_scaled = forward(m)
    clear_loras(m)
    apply_lora(m, p_alpha, scale=1.0)  # …must equal alpha-carrying file at scale 1
    assert np.array_equal(out_scaled, forward(m)), "alpha/rank scaling must fold into scale"
    clear_loras(m)
    assert np.array_equal(base, forward(m))
    print("[alph] alpha/rank scaling OK")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        run_suite(quantized=False, tmp=tmp)
        run_suite(quantized=True, tmp=tmp)
        run_errors(tmp)
        run_alpha(tmp)
    print("ALL LORA TESTS PASSED")
