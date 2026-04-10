"""
generate_lens_pattern.py
========================
Reconstructs a cosmetic colored contact lens pattern based on visual analysis
of a real lens photograph.

Design features reproduced:
  - Cool purple-gray base color palette
  - Dark outer limbal ring (dense dot-matrix)
  - Radial feather/streak texture throughout
  - Darker brush-like accents in the lower half
  - Soft transparent pupil opening with feathered inner edge

Usage
-----
    python generate_lens_pattern.py

Outputs
-------
    outputs/lens_pattern.png   – raster image (3000 × 3000 px)
    outputs/lens_pattern.svg   – vector image

Dependencies
------------
    numpy, matplotlib, Pillow  (all installable via pip)
"""

import math
import os

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, FancyArrowPatch
from PIL import Image, ImageDraw, ImageFilter

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SIZE = 3000          # canvas size in pixels
CX = CY = SIZE // 2  # centre

# Radii (in pixels, relative to SIZE=3000)
R_OUTER = 1380       # outer edge of the lens pattern
R_INNER_FADE = 290   # inner transparent pupil — hard boundary
R_INNER_SOFT = 420   # start of feathered inner transition

# Colour palette (RGB 0-255)
C_DEEP_OUTER = (38, 26, 40)        # #261A28
C_DARK_PURPLE = (74, 54, 81)       # #4A3651
C_MID_PURPLE = (119, 101, 125)     # #77657D
C_LIGHT_FOG = (183, 166, 200)      # #B7A6C8
C_HIGHLIGHT = (217, 204, 241)      # #D9CCF1
C_COOL_GRAY = (142, 136, 149)      # #8E8895
C_BRUSH_DARK = (45, 30, 48)        # #2D1E30

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def hex_to_rgb(h: str) -> tuple:
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def lerp_color(c1, c2, t):
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def radial_gradient_image(size, cx, cy, stops):
    """
    Build a numpy RGBA image with a radial gradient.
    stops: list of (radius_fraction, (R,G,B), alpha_float)
    radius_fraction is relative to size/2
    """
    img = np.zeros((size, size, 4), dtype=np.float32)
    ys, xs = np.mgrid[0:size, 0:size]
    dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    max_r = size / 2.0

    # Sort stops
    stops = sorted(stops, key=lambda s: s[0])

    for i in range(len(stops) - 1):
        r0, col0, a0 = stops[i]
        r1, col1, a1 = stops[i + 1]
        r0_px = r0 * max_r
        r1_px = r1 * max_r
        mask = (dist >= r0_px) & (dist < r1_px)
        t = np.clip((dist[mask] - r0_px) / max(r1_px - r0_px, 1e-6), 0, 1)
        for ch, (v0, v1) in enumerate(zip(col0, col1)):
            img[mask, ch] = (v0 + (v1 - v0) * t) / 255.0
        img[mask, 3] = a0 + (a1 - a0) * t

    # Beyond last stop
    last_r, last_col, last_a = stops[-1]
    mask_last = dist >= last_r * max_r
    for ch, v in enumerate(last_col):
        img[mask_last, ch] = v / 255.0
    img[mask_last, 3] = last_a

    return img


# ---------------------------------------------------------------------------
# Layer builders
# ---------------------------------------------------------------------------

def build_base_layer(size):
    """Radial gradient base: transparent pupil → light fog → mid purple → dark outer ring."""
    stops = [
        (0.00, (0, 0, 0), 0.0),
        (0.14, (0, 0, 0), 0.0),
        (0.20, C_COOL_GRAY, 0.08),
        (0.32, C_LIGHT_FOG, 0.50),
        (0.50, C_HIGHLIGHT, 0.72),
        (0.68, C_MID_PURPLE, 0.82),
        (0.78, C_DARK_PURPLE, 0.94),
        (0.86, C_DEEP_OUTER, 1.00),
        (1.00, C_DEEP_OUTER, 1.00),
    ]
    return radial_gradient_image(size, CX, CY, stops)


def build_dot_matrix(size, dot_spacing=18, dot_radius=2.8):
    """
    Diamond-oriented dot matrix across the full canvas.
    Outer zone dots are darker, inner zone dots are lighter purple.
    """
    img = np.zeros((size, size, 4), dtype=np.float32)
    ys, xs = np.mgrid[0:size, 0:size]
    dist = np.sqrt((xs - CX) ** 2 + (ys - CY) ** 2)

    # Rotate grid 45° → diamond pattern
    angle = math.radians(45)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    xu = (xs - CX) * cos_a + (ys - CY) * sin_a
    yu = -(xs - CX) * sin_a + (ys - CY) * cos_a
    # Nearest grid point in rotated space
    gx = np.round(xu / dot_spacing) * dot_spacing
    gy = np.round(yu / dot_spacing) * dot_spacing
    dot_dist = np.sqrt((xu - gx) ** 2 + (yu - gy) ** 2)
    on_dot = dot_dist < dot_radius

    # Outer-ring: dark dots
    mask_outer = on_dot & (dist >= R_INNER_SOFT) & (dist < R_OUTER)
    outer_alpha = np.clip((dist[mask_outer] - R_INNER_SOFT) / (R_OUTER - R_INNER_SOFT), 0, 1)
    img[mask_outer, 0] = C_BRUSH_DARK[0] / 255.0
    img[mask_outer, 1] = C_BRUSH_DARK[1] / 255.0
    img[mask_outer, 2] = C_BRUSH_DARK[2] / 255.0
    img[mask_outer, 3] = 0.45 + 0.35 * outer_alpha

    # Inner-ring: soft purple dots
    mask_inner = on_dot & (dist >= R_INNER_FADE) & (dist < R_INNER_SOFT)
    img[mask_inner, 0] = C_HIGHLIGHT[0] / 255.0
    img[mask_inner, 1] = C_HIGHLIGHT[1] / 255.0
    img[mask_inner, 2] = C_HIGHLIGHT[2] / 255.0
    img[mask_inner, 3] = 0.28

    return img


def build_radial_streaks(size, n_streaks=320, rng_seed=42):
    """Radial feather-like streaks from inner edge to outer ring."""
    img = np.zeros((size, size, 4), dtype=np.float32)
    rng = np.random.default_rng(rng_seed)

    ys, xs = np.mgrid[0:size, 0:size]
    dist = np.sqrt((xs - CX) ** 2 + (ys - CY) ** 2)
    theta = np.arctan2(ys - CY, xs - CX)  # −π … π

    for _ in range(n_streaks):
        angle = rng.uniform(-math.pi, math.pi)
        width = rng.uniform(0.004, 0.018)   # angular half-width in radians
        r_start = rng.uniform(R_INNER_SOFT * 0.85, R_INNER_SOFT * 1.2)
        r_end = rng.uniform(R_OUTER * 0.55, R_OUTER * 0.97)

        # Depth: lower half has darker, heavier streaks
        is_lower_half = math.sin(angle) > 0
        alpha_base = rng.uniform(0.10, 0.35) if not is_lower_half else rng.uniform(0.12, 0.40)

        # Angular distance (wrapping)
        dtheta = np.abs(((theta - angle) + math.pi) % (2 * math.pi) - math.pi)
        in_streak = (dtheta < width) & (dist >= r_start) & (dist < r_end)

        # Fade along streak length
        t_len = np.clip((dist[in_streak] - r_start) / max(r_end - r_start, 1), 0, 1)
        # Fade on angular edges
        t_ang = 1.0 - (dtheta[in_streak] / width)
        alpha = alpha_base * t_ang * (1.0 - t_len ** 2)

        # Streak colour: mix between light fog and highlight
        mix = rng.uniform(0.2, 0.9)
        col = lerp_color(C_LIGHT_FOG, C_HIGHLIGHT, mix)

        img[in_streak, 0] = col[0] / 255.0
        img[in_streak, 1] = col[1] / 255.0
        img[in_streak, 2] = col[2] / 255.0
        img[in_streak, 3] = np.maximum(img[in_streak, 3], alpha)

    return img


def build_lower_brush_accents(size, rng_seed=7):
    """
    A few wide, sweeping dark brush strokes in the lower half,
    giving the lens its characteristic deeper lower-half appearance.
    """
    img = np.zeros((size, size, 4), dtype=np.float32)
    rng = np.random.default_rng(rng_seed)

    ys, xs = np.mgrid[0:size, 0:size]
    dist = np.sqrt((xs - CX) ** 2 + (ys - CY) ** 2)

    # Only in lower half (positive y in image coords)
    lower_mask = (ys > CY) & (dist >= R_INNER_SOFT * 0.9) & (dist < R_OUTER * 0.97)

    # Broad sweep bands at several polar angles (π/4 … 3π/4)
    theta = np.arctan2(ys - CY, xs - CX)

    brush_centers = [math.pi * 0.35, math.pi * 0.52, math.pi * 0.65, math.pi * 0.78, math.pi * 0.90]
    brush_widths  = [0.18, 0.22, 0.20, 0.16, 0.14]
    brush_alphas  = [0.42, 0.55, 0.48, 0.38, 0.32]

    for angle, bw, ba in zip(brush_centers, brush_widths, brush_alphas):
        dtheta = np.abs(((theta - angle) + math.pi) % (2 * math.pi) - math.pi)
        in_brush = lower_mask & (dtheta < bw)

        # Radial alpha: strongest in mid-ring
        r_frac = (dist[in_brush] - R_INNER_SOFT * 0.9) / (R_OUTER * 0.97 - R_INNER_SOFT * 0.9)
        r_frac = np.clip(r_frac, 0, 1)
        env = 4 * r_frac * (1 - r_frac)  # bell-shaped in r

        ang_env = 1.0 - dtheta[in_brush] / bw
        alpha = ba * env * ang_env

        img[in_brush, 0] = C_BRUSH_DARK[0] / 255.0
        img[in_brush, 1] = C_BRUSH_DARK[1] / 255.0
        img[in_brush, 2] = C_BRUSH_DARK[2] / 255.0
        img[in_brush, 3] = np.maximum(img[in_brush, 3], alpha)

    return img


def build_inner_feather(size):
    """Soft feathered edge around the transparent pupil opening."""
    img = np.zeros((size, size, 4), dtype=np.float32)
    ys, xs = np.mgrid[0:size, 0:size]
    dist = np.sqrt((xs - CX) ** 2 + (ys - CY) ** 2)
    zone = (dist >= R_INNER_FADE) & (dist < R_INNER_SOFT)
    t = (dist[zone] - R_INNER_FADE) / (R_INNER_SOFT - R_INNER_FADE)
    img[zone, 0] = C_LIGHT_FOG[0] / 255.0
    img[zone, 1] = C_LIGHT_FOG[1] / 255.0
    img[zone, 2] = C_LIGHT_FOG[2] / 255.0
    img[zone, 3] = 0.30 * t * (1 - t) * 4  # bell fade
    return img


# ---------------------------------------------------------------------------
# Compositing
# ---------------------------------------------------------------------------

def alpha_over(dst: np.ndarray, src: np.ndarray) -> np.ndarray:
    """Porter-Duff 'over' compositing."""
    src_a = src[..., 3:4]
    dst_a = dst[..., 3:4]
    out_a = src_a + dst_a * (1.0 - src_a)
    out_a_safe = np.where(out_a > 0, out_a, 1.0)
    out_rgb = (src[..., :3] * src_a + dst[..., :3] * dst_a * (1.0 - src_a)) / out_a_safe
    return np.concatenate([out_rgb, out_a], axis=-1)


def apply_lens_mask(img: np.ndarray) -> np.ndarray:
    """Zero out alpha outside the lens ring and inside the pupil hole."""
    ys, xs = np.mgrid[0 : img.shape[0], 0 : img.shape[1]]
    dist = np.sqrt((xs - CX) ** 2 + (ys - CY) ** 2)
    outside = (dist > R_OUTER) | (dist < R_INNER_FADE)
    img[outside, 3] = 0.0
    return img


# ---------------------------------------------------------------------------
# PNG generation
# ---------------------------------------------------------------------------

def generate_png(output_path: str):
    print("  Building base gradient layer …")
    base = build_base_layer(SIZE)

    print("  Building dot matrix layer …")
    dots = build_dot_matrix(SIZE)

    print("  Building radial streak layer …")
    streaks = build_radial_streaks(SIZE)

    print("  Building lower brush accent layer …")
    brushes = build_lower_brush_accents(SIZE)

    print("  Building inner feather layer …")
    feather = build_inner_feather(SIZE)

    print("  Compositing layers …")
    composite = base.copy()
    composite = alpha_over(composite, dots)
    composite = alpha_over(composite, streaks)
    composite = alpha_over(composite, brushes)
    composite = alpha_over(composite, feather)

    # Apply lens ring mask
    composite = apply_lens_mask(composite)

    # Convert to uint8
    out = np.clip(composite * 255, 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(out, mode="RGBA")

    # Slight soft-focus blur to blend dot artifacts
    pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=0.8))

    pil_img.save(output_path, format="PNG")
    print(f"  PNG saved → {output_path}")
    return pil_img


# ---------------------------------------------------------------------------
# SVG generation
# ---------------------------------------------------------------------------

SVG_TEMPLATE = """\
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 1200" width="1200" height="1200">
  <defs>
    <radialGradient id="baseGrad" cx="50%" cy="50%" r="50%">
      <stop offset="0%"   stop-color="#000000" stop-opacity="0"/>
      <stop offset="14%"  stop-color="#000000" stop-opacity="0"/>
      <stop offset="22%"  stop-color="#8E8895" stop-opacity="0.10"/>
      <stop offset="38%"  stop-color="#B7A6C8" stop-opacity="0.50"/>
      <stop offset="55%"  stop-color="#D9CCF1" stop-opacity="0.72"/>
      <stop offset="72%"  stop-color="#77657D" stop-opacity="0.82"/>
      <stop offset="84%"  stop-color="#4A3651" stop-opacity="0.94"/>
      <stop offset="92%"  stop-color="#261A28" stop-opacity="1"/>
      <stop offset="100%" stop-color="#261A28" stop-opacity="1"/>
    </radialGradient>

    <radialGradient id="outerRing" cx="50%" cy="50%" r="50%">
      <stop offset="70%"  stop-color="#000000"  stop-opacity="0"/>
      <stop offset="82%"  stop-color="#3A2740"  stop-opacity="0.35"/>
      <stop offset="92%"  stop-color="#2A1B2B"  stop-opacity="0.80"/>
      <stop offset="100%" stop-color="#241622"  stop-opacity="1"/>
    </radialGradient>

    <radialGradient id="innerFade" cx="50%" cy="50%" r="50%">
      <stop offset="0%"   stop-color="#000000" stop-opacity="0"/>
      <stop offset="66%"  stop-color="#000000" stop-opacity="0"/>
      <stop offset="82%"  stop-color="#CFC8D8" stop-opacity="0.10"/>
      <stop offset="100%" stop-color="#FFFFFF"  stop-opacity="0.18"/>
    </radialGradient>

    <pattern id="dots" x="0" y="0" width="14" height="14"
             patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
      <circle cx="3"  cy="3"  r="1.8" fill="#2D1E30" fill-opacity="0.72"/>
      <circle cx="10" cy="10" r="1.8" fill="#2D1E30" fill-opacity="0.72"/>
    </pattern>

    <pattern id="softDots" x="0" y="0" width="16" height="16"
             patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
      <circle cx="4"  cy="4"  r="1.4" fill="#CBB7E8" fill-opacity="0.36"/>
      <circle cx="12" cy="12" r="1.4" fill="#CBB7E8" fill-opacity="0.36"/>
    </pattern>

    <mask id="lensMask">
      <rect width="1200" height="1200" fill="black"/>
      <circle cx="600" cy="600" r="500" fill="white"/>
      <circle cx="600" cy="600" r="173" fill="black"/>
    </mask>

    <filter id="blur2"><feGaussianBlur stdDeviation="2.2"/></filter>
    <filter id="blur5"><feGaussianBlur stdDeviation="5"/></filter>
  </defs>

  <rect width="1200" height="1200" fill="transparent"/>

  <g mask="url(#lensMask)">
    <!-- Base radial gradient -->
    <circle cx="600" cy="600" r="500" fill="url(#baseGrad)"/>

    <!-- Dot matrix -->
    <circle cx="600" cy="600" r="490" fill="url(#dots)"     opacity="0.82"/>
    <circle cx="600" cy="600" r="420" fill="url(#softDots)" opacity="0.60"/>

    <!-- Radial streaks (upper half, lighter) -->
    <g stroke-linecap="round" opacity="0.34">
{streak_lines_upper}
    </g>

    <!-- Radial streaks (all-round, fine) -->
    <g opacity="0.28" filter="url(#blur2)" stroke-linecap="round">
{streak_lines_all}
    </g>

    <!-- Lower-half brush accents -->
    <g opacity="0.52" filter="url(#blur5)" stroke-linecap="round">
      <path d="M180 710 C320 678, 380 660, 460 638" stroke="#352238" stroke-width="20" fill="none"/>
      <path d="M160 768 C295 730, 415 698, 512 672" stroke="#2B1C2D" stroke-width="25" fill="none"/>
      <path d="M185 828 C330 786, 462 748, 568 718" stroke="#3E2942" stroke-width="18" fill="none"/>
      <path d="M688 718 C808 686, 912 672, 1020 688"
            stroke="#2A1B2B" stroke-width="16" fill="none" opacity="0.32"/>
    </g>

    <!-- Outer limbal ring overlay -->
    <circle cx="600" cy="600" r="500" fill="url(#outerRing)"/>

    <!-- Inner feathered edge -->
    <circle cx="600" cy="600" r="214" fill="url(#innerFade)" opacity="0.72"/>
  </g>

  <!-- Pupil area hint -->
  <circle cx="600" cy="600" r="173" fill="#808080" fill-opacity="0.08"/>
</svg>
"""


def _make_svg_streak_lines():
    """Generate SVG <line> elements for radial streaks."""
    rng = np.random.default_rng(42)
    upper_lines = []
    all_lines = []

    # All-round fine streaks
    n_all = 40
    for i in range(n_all):
        angle = -math.pi + i * 2 * math.pi / n_all + rng.uniform(-0.05, 0.05)
        r0 = rng.uniform(200, 250)
        r1 = rng.uniform(430, 530)
        x1 = 600 + r0 * math.cos(angle)
        y1 = 600 + r0 * math.sin(angle)
        x2 = 600 + r1 * math.cos(angle)
        y2 = 600 + r1 * math.sin(angle)
        col = "#E8DDFF" if rng.random() > 0.5 else "#D8C8F2"
        w = rng.uniform(1.5, 3.0)
        all_lines.append(
            f'      <line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}"'
            f' stroke="{col}" stroke-width="{w:.1f}"/>'
        )

    # Upper-half emphasis streaks
    angles_upper = np.linspace(-math.pi, 0, 14)
    for angle in angles_upper:
        r0 = rng.uniform(215, 260)
        r1 = rng.uniform(390, 510)
        x1 = 600 + r0 * math.cos(angle)
        y1 = 600 + r0 * math.sin(angle)
        x2 = 600 + r1 * math.cos(angle)
        y2 = 600 + r1 * math.sin(angle)
        col = "#E6DAFF" if rng.random() > 0.5 else "#CDBCE8"
        w = rng.uniform(2.5, 4.5)
        upper_lines.append(
            f'      <line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}"'
            f' stroke="{col}" stroke-width="{w:.1f}"/>'
        )

    return "\n".join(upper_lines), "\n".join(all_lines)


def generate_svg(output_path: str):
    upper, all_round = _make_svg_streak_lines()
    svg_content = SVG_TEMPLATE.format(
        streak_lines_upper=upper,
        streak_lines_all=all_round,
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg_content)
    print(f"  SVG saved → {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("Generating reconstructed cosmetic lens pattern …")
    png_path = os.path.join(OUTPUT_DIR, "lens_pattern.png")
    svg_path = os.path.join(OUTPUT_DIR, "lens_pattern.svg")

    generate_png(png_path)
    generate_svg(svg_path)

    print("\nDone. Output files:")
    print(f"  {png_path}")
    print(f"  {svg_path}")


if __name__ == "__main__":
    main()
