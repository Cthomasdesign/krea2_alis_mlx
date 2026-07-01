"""End-to-end Krea-2-Turbo MLX pipeline.

Ships only the (quantized or bf16) transformer; the Qwen-Image VAE, Qwen3-VL-4B text
encoder, and tokenizer are pulled at runtime from `krea/Krea-2-Turbo` (you accept
Krea's license there). The VAE reuses mflux's `QwenVAE` — install with `pip install mflux`.
"""

from __future__ import annotations

import os

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_map

from .quant_recipes import mixed_4_8, quantize_bulk
from .sampling import sample, to_pil
from .text_encoder import Qwen3VLConditioner
from .transformer import Krea2Config, SingleStreamDiT

BASE_REPO = "krea/Krea-2-Turbo"
# quantized builds the apps can use: precision -> (HF repo, transformer filename)
BUILDS = {
    "8bit": ("avlp12/Krea-2-Turbo-Alis-MLX-8bit", "transformer_8bit.safetensors"),
    "mixed-4-8": ("avlp12/Krea-2-Turbo-Alis-MLX-mixed-4-8", "transformer_mixed_4_8.safetensors"),
}
_CACHE = os.path.expanduser("~/.cache/krea2_alis_mlx")


def _http_download(repo: str, filename: str, dest_root: str) -> str:
    """Download {repo}/resolve/main/{filename} over plain HTTP (the HF CDN / Xet bridge).

    We bypass huggingface_hub's downloader on purpose: its Xet path hangs with no fallback
    when cas-server.xethub.hf.co is unreachable (corporate firewalls, some ISPs), while the
    public resolve URL always works via the CDN bridge. Cached under ~/.cache/krea2_alis_mlx;
    skips the download if the local file already matches the remote size.
    """
    import requests

    url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
    dest = os.path.join(dest_root, filename)
    try:
        total = int(requests.head(url, allow_redirects=True, timeout=30).headers.get("content-length") or 0)
    except Exception:
        total = 0
    if os.path.exists(dest) and total and os.path.getsize(dest) == total:
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    pos = os.path.getsize(tmp) if os.path.exists(tmp) else 0  # resume a partial download
    if total and pos == total:  # .part already holds the whole file (died before the rename) -> finalize
        os.replace(tmp, dest)    # (avoids re-requesting Range: bytes={total}- which the CDN answers 416)
        return dest
    if total and pos > total:   # stale/corrupt leftover .part (wrong or changed remote) -> restart clean
        os.remove(tmp)
        pos = 0
    headers = {"Range": f"bytes={pos}-"} if pos else {}
    with requests.get(url, headers=headers, stream=True, timeout=(30, 120), allow_redirects=True) as r:
        r.raise_for_status()  # 4xx/5xx (gated repo, missing file) surface here, not as a resume hint
        resume = bool(pos) and r.status_code == 206  # 206 => server honored the range
        pos = pos if resume else 0
        total = total or (pos + int(r.headers.get("content-length") or 0))  # prefer GET's length
        done = pos
        try:
            with open(tmp, "ab" if resume else "wb") as f:
                for chunk in r.iter_content(4 << 20):
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        print(f"\r  ↓ {filename}  {done // 1048576} / {total // 1048576} MB", end="", flush=True)
        except requests.exceptions.RequestException as e:
            # a mid-stream drop (incl. a truncated chunked transfer) lands here; .part is kept for resume
            raise OSError(f"Download of {filename} interrupted; re-run to resume "
                          "(the partial file is kept).") from e
    if total:
        print()
    # Commit only a verified-complete file. When the length is known we size-check; for a length-less
    # (chunked) transfer requests raises above on a short read, so a clean loop here means complete —
    # but never commit an empty result (an immediate disconnect that produced no bytes).
    if (total and done != total) or done == 0:
        raise OSError(f"Incomplete download of {filename}: got {done} of {total or '?'} bytes. "
                      "Re-run to resume (the partial file is kept).")
    os.replace(tmp, dest)
    return dest


def resolve_weights(folder: str = ".", precision: str | None = None, download: bool = True):
    """Resolve the transformer weights to use. Returns (precision, path).

    - precision given ('8bit'|'mixed-4-8'): use the local file if present, else (download)
      fetch that build from its HF repo (cached under ~/.cache/krea2_alis_mlx).
    - precision None (auto): use whichever build's file is already in `folder`; if none,
      default to 8-bit (downloaded on load).
    """
    def _resolve(prec):
        repo, fname = BUILDS[prec]
        local = os.path.join(folder, fname)
        if os.path.exists(local):
            return prec, local
        if not download:
            return prec, None
        return prec, _http_download(repo, fname, os.path.join(_CACHE, repo.replace("/", "__")))

    if precision in BUILDS:
        return _resolve(precision)
    for prec, (_, fname) in BUILDS.items():  # auto: prefer a locally-present build
        if os.path.exists(os.path.join(folder, fname)):
            return prec, os.path.join(folder, fname)
    return _resolve("8bit")


def _base_dir() -> str:
    """Fetch the VAE / Qwen3-VL-4B encoder / tokenizer from krea/Krea-2-Turbo over HTTP."""
    from huggingface_hub import HfApi

    dest = os.path.join(_CACHE, BASE_REPO.replace("/", "__"))
    exts = (".safetensors", ".json", ".jinja", ".txt", ".model")
    want = [
        s.rfilename for s in HfApi().model_info(BASE_REPO).siblings
        if (s.rfilename.startswith(("vae/", "text_encoder/", "tokenizer/")) or s.rfilename == "model_index.json")
        and s.rfilename.endswith(exts)
    ]
    for f in want:
        _http_download(BASE_REPO, f, dest)
    return dest


def _load_vae(base_dir: str):
    from mflux.models.common.weights.loading.weight_definition import ComponentDefinition
    from mflux.models.common.weights.loading.weight_loader import WeightLoader
    from mflux.models.qwen.model.qwen_vae.qwen_vae import QwenVAE
    from mflux.models.qwen.weights.qwen_weight_mapping import QwenWeightMapping

    class _VaeDef:
        @staticmethod
        def get_components():
            return [ComponentDefinition(name="vae", hf_subdir="vae", loading_mode="single",
                                        mapping_getter=QwenWeightMapping.get_vae_mapping)]

        @staticmethod
        def get_download_patterns():
            return ["vae/*.safetensors", "vae/*.json"]

    vae = QwenVAE()
    vae.update(WeightLoader.load(weight_definition=_VaeDef, model_path=base_dir).components["vae"])
    mx.eval(vae.parameters())
    return vae


class Krea2Pipeline:
    """precision:
    - '8bit'      : transformer_8bit.safetensors (28-block attn+mlp @ 8-bit)
    - 'mixed-4-8' : transformer_mixed_4_8.safetensors (down_proj+endpoints @8, rest @4)
    - 'bf16'      : krea/Krea-2-Turbo/turbo.safetensors (auto-downloaded)"""

    def __init__(self, transformer_path: str | None = None, precision: str = "8bit", base_dir: str | None = None):
        base = base_dir or _base_dir()
        m = SingleStreamDiT(Krea2Config())
        if precision == "8bit":
            nn.quantize(m, group_size=64, bits=8, class_predicate=quantize_bulk)
            m.load_weights(transformer_path, strict=True)
        elif precision == "mixed-4-8":
            nn.quantize(m, group_size=64, bits=4, class_predicate=mixed_4_8)
            m.load_weights(transformer_path, strict=True)
        elif precision == "bf16":
            if transformer_path is None:
                from huggingface_hub import hf_hub_download
                transformer_path = hf_hub_download(BASE_REPO, "turbo.safetensors")
            m.load_weights(transformer_path, strict=True)
            m.update(tree_map(lambda a: a.astype(mx.bfloat16), m.parameters()))
        else:
            raise ValueError(f"precision must be '8bit', 'mixed-4-8' or 'bf16', got {precision}")
        mx.eval(m.parameters())
        self.transformer = m
        self.vae = _load_vae(base)
        self.encoder = Qwen3VLConditioner(base, dtype=mx.bfloat16)
        self._lora_sig: tuple = ()   # currently-applied (path, scale) set, to skip redundant rebuilds
        self._lora_paths: list = []  # wrapped target paths, for clean unload

    def set_loras(self, specs) -> None:
        """Apply a set of LoRAs (list of (path, scale)) to the transformer, replacing any
        previously-applied set. Empty/None clears all LoRAs. Cheap no-op if unchanged."""
        specs = [(str(p), float(s)) for p, s in (specs or [])]
        sig = tuple(specs)
        if sig == self._lora_sig:
            return
        from .lora import apply_loras, unload_loras
        unload_loras(self.transformer, self._lora_paths)
        self._lora_paths = apply_loras(self.transformer, specs) if specs else []
        self._lora_sig = sig

    def generate(self, prompt, *, width=1024, height=1024, steps=8, seed=0, num_images=1,
                 init_image=None, strength=0.6, step_callback=None):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string.")
        try:  # uniform ValueError for None / non-numeric / inf / nan (not TypeError / OverflowError)
            width, height, steps, num_images, seed = (
                int(width), int(height), int(steps), int(num_images), int(seed))
        except (TypeError, ValueError, OverflowError):
            raise ValueError("width, height, steps, num_images, seed must be integers.") from None
        # the VAE downsamples ×8 and the DiT patchifies ×2 → dims must be multiples of 16
        for name, v in (("width", width), ("height", height)):
            if v < 256 or v > 2048 or v % 16:
                raise ValueError(f"{name} must be a multiple of 16 in [256, 2048], got {v}.")
        if not 1 <= steps <= 50:
            raise ValueError(f"steps must be in [1, 50], got {steps}.")
        if not 1 <= num_images <= 8:
            raise ValueError(f"num_images must be in [1, 8], got {num_images}.")
        if not 0 <= seed < 2**64:  # mx.random.seed wants a non-negative uint64 (else a bare TypeError)
            raise ValueError(f"seed must be in [0, 2^64), got {seed}.")
        init_latent = None
        if init_image is not None:
            try:
                strength = float(strength)
            except (TypeError, ValueError):
                raise ValueError("strength must be a number in (0, 1].") from None
            if not 0.0 < strength <= 1.0:
                raise ValueError(f"strength must be in (0, 1], got {strength}.")
            if strength < 1.0:  # at 1.0 the image is ignored by design — don't pay for the encode
                # img2img: encode the init image (a file path or PIL image) with the same Qwen VAE
                # the sampler decodes with — the returned latents are mean/std-normalized, i.e.
                # already in the sampler's latent space. Scaled to (width, height) before encoding.
                from mflux.models.common.latent_creator.latent_creator import LatentCreator
                tiling = None
                if width * height >= 1536 * 1536:  # bound the encoder's full-res activations on big inputs
                    from mflux.models.common.vae.tiling_config import TilingConfig
                    tiling = TilingConfig()
                init_latent = LatentCreator.encode_image(
                    vae=self.vae, image_path=init_image, height=height, width=width,
                    tiling_config=tiling)
                if init_latent.ndim == 5:  # the tiled path keeps the temporal dim — the sampler is 4D
                    init_latent = init_latent[:, :, 0]
                mx.eval(init_latent)
        dec = sample(self.transformer, self.vae, self.encoder, [prompt] * num_images,
                     width=width, height=height, steps=steps, guidance=0.0, seed=seed,
                     init_latent=init_latent, strength=strength if init_latent is not None else 1.0,
                     step_callback=step_callback)
        return to_pil(dec)
