#!/usr/bin/env python3
"""
Cosmetic Contact Lens Design Reconstruction
============================================

Reconstructs a cosmetic contact lens (美瞳) pattern in two modes:

1. **Diffusion model** (default) – uses Stable Diffusion via the
   HuggingFace ``diffusers`` library to generate the pattern.
   Install the optional dependency with::

       pip install diffusers accelerate transformers

2. **Procedural fallback** – when ``diffusers`` is not installed or
   the ``--procedural`` flag is used, a high-quality fBm (fractional
   Brownian Motion) noise renderer produces a graphic that closely
   matches the reference design characteristics:

   * Deep grape-purple outer lock ring (dense dot-matrix)
   * Cold purple-grey radial gradient (outer-dark → inner-light)
   * Fine diamond/mesh dot-matrix base texture
   * Radial fibre streaks from inner edge to outer ring
   * Heavier brush strokes on the lower half
   * Large transparent centre (pupil area)

Usage
-----
::

    # Diffusion model (requires diffusers):
    python generate_lens_design.py

    # Force procedural renderer:
    python generate_lens_design.py --procedural

    # Custom output path and resolution:
    python generate_lens_design.py --output outputs/my_lens.png --size 1024

Output
------
``lens_design_output.png`` (or the path given by ``--output``) in the
current working directory.
"""

import argparse
import math
import os

import numpy as np
from PIL import Image, ImageFilter

# ---------------------------------------------------------------------------
# Global geometry defaults (overridable via CLI)
# ---------------------------------------------------------------------------
_SIZE = 1024
_OUTER_FRAC = 0.46   # outer radius as fraction of image half-width
_INNER_FRAC = 0.145  # inner (pupil) radius fraction

# ---------------------------------------------------------------------------
# Reference colour palette (cold purple-grey, inferred from source image)
# ---------------------------------------------------------------------------
_C_DEEP_OUTER = np.array([38, 26, 40], dtype=np.float32)    # #261A28
_C_DARK_PURPLE = np.array([74, 54, 81], dtype=np.float32)   # #4A3651
_C_MID_GREY = np.array([119, 101, 125], dtype=np.float32)   # #77657D
_C_SOFT_LILAC = np.array([183, 166, 200], dtype=np.float32) # #B7A6C8
_C_HIGH_LAVEND = np.array([217, 204, 241], dtype=np.float32) # #D9CCF1

# ---------------------------------------------------------------------------
# Diffusion-model prompt
# ---------------------------------------------------------------------------
_LENS_PROMPT = (
    "flat lay product design sheet of a cosmetic contact lens pattern, "
    "deep grape-purple outer ring, cold purple-grey radial gradient mid zone, "
    "fine diamond dot matrix texture, radial fibre streaks, transparent pupil hole, "
    "lavender highlights, dark brush strokes on lower half, "
    "white background, top-down orthographic view, high quality graphic design"
)
_NEGATIVE_PROMPT = (
    "eye, face, person, blurry, low quality, photorealistic photo, "
    "text, watermark, cropped, deformed"
)


# ===========================================================================
# Helpers
# ===========================================================================

def _make_coord_grids(size: int):
    cy = cx = size / 2.0
    y, x = np.mgrid[0:size, 0:size]
    dx = x - cx
    dy = y - cy
    r = np.sqrt(dx ** 2 + dy ** 2)
    theta = np.arctan2(dy, dx)
    return r, theta, dx, dy


def _smoothstep(edge0, edge1, x):
    t = np.clip((x - edge0) / (edge1 - edge0 + 1e-9), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# ---------------------------------------------------------------------------
# Lightweight 2-D value noise (no external dependency)
# ---------------------------------------------------------------------------

class _ValueNoise2D:
    """Bilinear interpolation of a random lattice – tileable 2-D value noise."""

    def __init__(self, freq: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        # +1 so we can wrap: grid[f] == grid[0]
        self._grid = rng.random((freq + 1, freq + 1)).astype(np.float32)
        self._freq = freq

    def __call__(self, nx: np.ndarray, ny: np.ndarray) -> np.ndarray:
        f = self._freq
        gx = nx * f
        gy = ny * f
        x0 = np.floor(gx).astype(int) % f
        y0 = np.floor(gy).astype(int) % f
        x1 = (x0 + 1) % (f + 1)
        y1 = (y0 + 1) % (f + 1)
        fx = (gx - np.floor(gx)).astype(np.float32)
        fy = (gy - np.floor(gy)).astype(np.float32)
        v00 = self._grid[y0, x0]
        v10 = self._grid[y0, x1]
        v01 = self._grid[y1, x0]
        v11 = self._grid[y1, x1]
        return (v00 * (1 - fx) * (1 - fy)
                + v10 * fx * (1 - fy)
                + v01 * (1 - fx) * fy
                + v11 * fx * fy)


def _fbm(nx: np.ndarray, ny: np.ndarray, octaves: int = 6, seed_base: int = 0) -> np.ndarray:
    """Fractional Brownian Motion via summed octaves of value noise."""
    value = np.zeros_like(nx, dtype=np.float32)
    amp = 0.5
    freq = 4
    for i in range(octaves):
        noise = _ValueNoise2D(freq, seed=seed_base + i)
        value += amp * noise(nx, ny)
        amp *= 0.5
        freq *= 2
    return value


# ---------------------------------------------------------------------------
# Radial gradient
# ---------------------------------------------------------------------------

def _radial_gradient(t_r: np.ndarray, in_annulus: np.ndarray) -> np.ndarray:
    """Evaluate the multi-stop radial colour gradient."""
    stops = [
        (0.00, _C_HIGH_LAVEND),
        (0.18, _C_SOFT_LILAC),
        (0.50, _C_MID_GREY),
        (0.80, _C_DARK_PURPLE),
        (1.00, _C_DEEP_OUTER),
    ]
    H, W = t_r.shape
    result = np.zeros((H, W, 3), dtype=np.float32)
    for i in range(len(stops) - 1):
        t0, c0 = stops[i]
        t1, c1 = stops[i + 1]
        seg_mask = (t_r >= t0) & (t_r <= t1)
        local_t = np.clip((t_r - t0) / (t1 - t0 + 1e-9), 0.0, 1.0)
        # Smoothstep for soft stop transitions
        local_t = local_t * local_t * (3.0 - 2.0 * local_t)
        color = ((1 - local_t[..., None]) * c0[None, None, :]
                 + local_t[..., None] * c1[None, None, :])
        result = np.where(seg_mask[..., None], color, result)
    return result * in_annulus[..., None]


# ===========================================================================
# Procedural renderer
# ===========================================================================

def procedural_lens(size: int = _SIZE) -> Image.Image:
    """
    Render the contact-lens pattern procedurally.

    All layer deltas are kept in the correct [0, 255] numeric range before
    compositing, so the final ``np.clip`` is only a safety guard.

    Returns a PIL RGBA image with a transparent pupil hole and soft outer edge.
    """
    outer_r = int(size * _OUTER_FRAC)
    inner_r = int(size * _INNER_FRAC)

    r, theta, dx, dy = _make_coord_grids(size)
    t_r = np.clip((r - inner_r) / (outer_r - inner_r + 1e-9), 0.0, 1.0)
    in_annulus = ((r >= inner_r) & (r <= outer_r)).astype(np.float32)

    # ------------------------------------------------------------------
    # 1. Base radial gradient  (values in [0..255])
    # ------------------------------------------------------------------
    palette = _radial_gradient(t_r, in_annulus)

    # Normalised coordinates for noise sampling
    nx = (dx / (outer_r * 1.1)) * 0.5 + 0.5
    ny = (dy / (outer_r * 1.1)) * 0.5 + 0.5

    # ------------------------------------------------------------------
    # 2. fBm noise – iris-fibre texture (delta ±20 per channel)
    # ------------------------------------------------------------------
    noise = _fbm(nx, ny, octaves=6, seed_base=0)
    noise2 = _fbm(nx * 1.4, ny * 1.4, octaves=5, seed_base=99)
    noise = (noise * 0.65 + noise2 * 0.35).astype(np.float32)
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-9)   # [0, 1]

    # Strongest in the mid-annulus, zero outside
    noise_weight = np.clip(1.0 - np.abs(t_r - 0.5) * 2.2, 0.0, 1.0) * in_annulus
    # Delta: (noise−0.5)∈[−0.5, 0.5], weight∈[0,1], scale 40 → ±20 per channel
    noise_delta = (noise - 0.5) * noise_weight * 40.0
    # Cool/purple tint (multiply component-wise, all ≈1)
    noise_tint = np.array([0.84, 0.88, 1.06], dtype=np.float32)
    noise_layer = noise_delta[..., None] * noise_tint[None, None, :]     # ±~17–21 per ch

    # ------------------------------------------------------------------
    # 3. Radial fibre streaks (delta ±12 per channel, cool-purple tinted)
    # ------------------------------------------------------------------
    ang_noise = _ValueNoise2D(64, seed=13)
    perturb = (_fbm(nx, ny * 1.5, octaves=4, seed_base=42) - 0.5) * 0.30
    nx_s = np.cos(theta + perturb) * 0.5 + 0.5
    ny_s = np.sin(theta + perturb) * 0.5 + 0.5
    streaks = ang_noise(nx_s, ny_s).astype(np.float32)
    streaks = (streaks - streaks.min()) / (streaks.max() - streaks.min() + 1e-9)  # [0,1]

    streak_weight = (1.0 - t_r * 0.45) * in_annulus * 0.7     # [0, 0.7]
    # Delta: (streaks−0.45)∈[−0.45, 0.55], weight≤0.7, scale 35 → up to ±14 per channel
    streak_delta = (streaks - 0.45) * streak_weight * 35.0
    streak_tint = np.array([0.80, 0.75, 1.05], dtype=np.float32)        # cool tint
    streak_layer = streak_delta[..., None] * streak_tint[None, None, :]  # ±~11–15 per ch

    # ------------------------------------------------------------------
    # 4. Diamond dot-matrix: alpha-blend palette toward dark outer colour
    # ------------------------------------------------------------------
    dot_pitch = max(8, size // 100)
    rng_d = np.random.default_rng(7)
    yi, xi = np.mgrid[0:size, 0:size]
    ui = (xi + yi) // dot_pitch
    vi = (xi - yi) // dot_pitch
    dot_pattern = ((ui + vi) % 2 == 0).astype(np.float32)
    jitter = rng_d.random((size, size)).astype(np.float32)
    density = 0.25 + 0.55 * _smoothstep(0.1, 0.9, t_r)
    dot_mask = (dot_pattern * (jitter < density) * in_annulus).astype(np.float32)
    # Blend factor: 0→no change, 1→fully dark outer colour
    dot_blend = np.clip((0.15 + 0.30 * t_r) * dot_mask, 0.0, 0.55)
    # dot_layer = blend × (target − current); negative ⇒ darkening
    dot_layer = dot_blend[..., None] * (_C_DEEP_OUTER[None, None, :] - palette)

    # ------------------------------------------------------------------
    # 5. Lower-half brush strokes: blend palette toward dark brush colour
    # ------------------------------------------------------------------
    lower_mask = ((dy / outer_r) > 0.03).astype(np.float32) * in_annulus
    brush_color = (_C_DEEP_OUTER * 0.70 + _C_DARK_PURPLE * 0.30).astype(np.float32)

    rng_b = np.random.default_rng(77)
    center = size / 2.0
    brush_accum = np.zeros((size, size), dtype=np.float32)

    for _ in range(7):
        y_c = center + rng_b.uniform(0.12, 0.38) * outer_r
        band_w = rng_b.uniform(0.030, 0.065) * outer_r
        x_lo = center - outer_r * rng_b.uniform(0.50, 0.72)
        x_hi = center + outer_r * rng_b.uniform(0.05, 0.38)
        intensity = rng_b.uniform(0.30, 0.65)

        row_w = np.exp(-0.5 * ((np.arange(size) - y_c) / (band_w + 1e-9)) ** 2).astype(np.float32)
        col_w = np.zeros(size, dtype=np.float32)
        xl, xh = int(max(0, x_lo)), int(min(size, x_hi))
        if xl < xh:
            col_w[xl:xh] = 1.0
            feather = int(band_w * 2.5)
            for fi in range(feather):
                frac = fi / (feather + 1e-9)
                if xl + fi < size:
                    col_w[xl + fi] = frac
                xe = xh + fi
                if xe < size:
                    col_w[xe] = 1.0 - frac

        brush_2d = row_w[:, None] * col_w[None, :] * lower_mask * intensity
        # Use max so overlapping strokes don't double-darken
        brush_accum = np.maximum(brush_accum, brush_2d)

    brush_blend = np.clip(brush_accum * 0.65, 0.0, 0.65)
    brush_layer = brush_blend[..., None] * (brush_color[None, None, :] - palette)

    # ------------------------------------------------------------------
    # 6. Compose all layers
    # ------------------------------------------------------------------
    image_rgb = palette + noise_layer + streak_layer + dot_layer + brush_layer
    image_rgb = np.clip(image_rgb, 0.0, 255.0).astype(np.uint8)

    # ------------------------------------------------------------------
    # 7. Alpha channel (transparent pupil + soft outer edge)
    # ------------------------------------------------------------------
    alpha = np.ones((size, size), dtype=np.float32)
    alpha = np.where(r < inner_r, 0.0, alpha)
    inner_feather = _smoothstep(inner_r - 2, inner_r + 22, r)
    alpha = np.where((r >= inner_r - 2) & (r <= inner_r + 22), inner_feather, alpha)
    outer_feather = 1.0 - _smoothstep(outer_r - 14, outer_r + 2, r)
    alpha = np.where(r > outer_r - 14, outer_feather, alpha)
    alpha = np.clip(alpha, 0.0, 1.0)

    # ------------------------------------------------------------------
    # 8. Assemble RGBA and apply subtle smoothing
    # ------------------------------------------------------------------
    rgba = np.zeros((size, size, 4), dtype=np.uint8)
    rgba[:, :, :3] = image_rgb
    rgba[:, :, 3] = (alpha * 255).astype(np.uint8)
    pil_img = Image.fromarray(rgba, "RGBA")
    pil_img = pil_img.filter(ImageFilter.GaussianBlur(0.7))
    return pil_img


# ===========================================================================
# Diffusion-model path (Stable Diffusion via HuggingFace diffusers)
# ===========================================================================

def diffusion_lens(size: int = 512) -> Image.Image:
    """
    Generate the lens design using Stable Diffusion.

    Tries ``stabilityai/sd-turbo`` first (fast, 1-step capable), then
    falls back to ``runwayml/stable-diffusion-v1-5``.

    Requires::

        pip install diffusers accelerate transformers

    Uses CUDA when available; otherwise runs on CPU (slow but functional).
    """
    import torch  # noqa: PLC0415

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    try:
        from diffusers import AutoPipelineForText2Image  # noqa: PLC0415

        pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sd-turbo",
            torch_dtype=dtype,
            variant="fp16" if dtype == torch.float16 else None,
        ).to(device)
        result = pipe(
            prompt=_LENS_PROMPT,
            negative_prompt=_NEGATIVE_PROMPT,
            num_inference_steps=4,
            guidance_scale=0.0,
            height=size,
            width=size,
        )
    except Exception:
        from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline  # noqa: PLC0415

        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=dtype,
            safety_checker=None,
        ).to(device)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        result = pipe(
            prompt=_LENS_PROMPT,
            negative_prompt=_NEGATIVE_PROMPT,
            num_inference_steps=25,
            guidance_scale=7.5,
            height=size,
            width=size,
        )

    return result.images[0]


# ===========================================================================
# CLI entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate a cosmetic contact lens design PNG.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--procedural",
        action="store_true",
        help="Skip diffusion model; use the built-in fBm procedural renderer.",
    )
    parser.add_argument(
        "--output",
        default="lens_design_output.png",
        help="Output PNG file path (default: lens_design_output.png).",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=_SIZE,
        help=f"Canvas size in pixels, square (default: {_SIZE}).",
    )
    args = parser.parse_args()

    img = None

    if not args.procedural:
        try:
            print("[INFO] Attempting Stable Diffusion generation …")
            img = diffusion_lens(size=min(args.size, 1024))
            print("[INFO] Diffusion generation succeeded.")
        except ImportError:
            print("[WARN] `diffusers` not installed.")
            print("       Install with:  pip install diffusers accelerate transformers")
            print("[INFO] Falling back to procedural renderer.")
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Diffusion generation failed: {exc}")
            print("[INFO] Falling back to procedural renderer.")

    if img is None:
        print("[INFO] Running procedural fBm lens renderer …")
        img = procedural_lens(size=args.size)
        print("[INFO] Procedural rendering complete.")

    out_path = args.output
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    img.save(out_path)
    print(f"[INFO] Saved → {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
