#!/usr/bin/env python
"""Generate images with Krea-2-Turbo on Apple MLX.

    python3 -m pip install -r requirements.txt
    python3 generate.py "a fox in the snow"

8-bit (default) uses this repo's transformer_8bit.safetensors. Use --precision bf16
to run the full-precision transformer from krea/Krea-2-Turbo instead.
"""

import argparse
import os

from krea2.pipeline import Krea2Pipeline, resolve_weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--precision", choices=["8bit", "mixed-4-8", "bf16"], default=None,
                    help="default: auto-detect the weights shipped in this folder")
    ap.add_argument("--transformer", default=None,
                    help="path to transformer weights (default: the matching file in this repo)")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-images", type=int, default=1)
    ap.add_argument("--init-image", default=None,
                    help="img2img: path to an input image to transform (scaled to --width/--height)")
    ap.add_argument("--strength", type=float, default=0.6,
                    help="img2img: how much to change the input, (0, 1] — higher = more change (default 0.6)")
    ap.add_argument("--lora-paths", nargs="*", default=[],
                    help="one or more LoRA .safetensors to apply (stackable)")
    ap.add_argument("--lora-scales", nargs="*", type=float, default=None,
                    help="strength per LoRA (default 1.0; a single value applies to all)")
    ap.add_argument("--out", default="out.png")
    ap.add_argument("--no-safety", action="store_true",
                    help="disable the NSFW content filter (on by default; see the license)")
    args = ap.parse_args()

    if args.init_image:  # fail on a bad path/file BEFORE loading ~14 GB of weights
        from PIL import Image, UnidentifiedImageError
        try:
            with Image.open(args.init_image) as im:
                im.verify()
        except (OSError, UnidentifiedImageError, ValueError) as e:
            ap.error(f"--init-image: cannot read {args.init_image!r}: {e}")

    lora_specs = []
    if args.lora_paths:  # likewise, catch bad LoRA args before the big load
        scales = args.lora_scales if args.lora_scales is not None else [1.0]
        if len(scales) == 1:
            scales = scales * len(args.lora_paths)
        if len(scales) != len(args.lora_paths):
            ap.error("--lora-scales must give one value, or one per --lora-paths")
        for p in args.lora_paths:
            if not os.path.isfile(p):
                ap.error(f"--lora-paths: no such file: {p!r}")
        lora_specs = list(zip(args.lora_paths, scales))

    here = os.path.dirname(os.path.abspath(__file__))
    precision, tpath = args.precision, args.transformer
    if precision != "bf16" and tpath is None:
        # local file if present, else download the chosen build (8bit / mixed-4-8; default 8bit)
        precision, tpath = resolve_weights(here, precision=precision, download=True)

    pipe = Krea2Pipeline(transformer_path=tpath, precision=precision)
    if lora_specs:
        try:
            pipe.set_loras(lora_specs)
        except (ValueError, FileNotFoundError) as e:
            ap.error(str(e))  # e.g. an unmapped key or a non-LoRA safetensors
    try:
        images = pipe.generate(args.prompt, width=args.width, height=args.height,
                               steps=args.steps, seed=args.seed, num_images=args.num_images,
                               init_image=args.init_image, strength=args.strength)
    except ValueError as e:
        ap.error(str(e))  # clean "generate.py: error: ..." instead of a traceback
    from krea2 import safety
    images, flagged = safety.apply(images, enabled=not args.no_safety)
    if flagged:
        print(f"⚠  {flagged} image(s) redacted by the NSFW safety filter (disable with --no-safety)")
    base, ext = os.path.splitext(args.out)
    ext = ext or ".png"
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    for i, im in enumerate(images):
        path = args.out if len(images) == 1 else f"{base}_{i + 1}{ext}"  # foo_1.png, foo_2.png …
        im.save(path)
        print(f"saved {path}")


if __name__ == "__main__":
    main()
