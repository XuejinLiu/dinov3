from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from PIL.ImageOps import colorize
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import normalize, resize, to_tensor

from dinov3.data.transforms import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from dinov3.hub.backbones import dinov3_vitl16


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_CONFIG = {
    "reference_dir": None,
    "query": None,
    "memory_bank_image_count": 50,
    "backbone_weights": None,
    "device": None,
    "resize_long_side": 0,
    "min_long_side": 384,
    "max_long_side": 896,
    "patch_size": 16,
    "coreset_ratio": 0.1,
    "max_memory": 10000,
    "proj_dim": 128,
    "nn_batch_size": 2048,
    "image_score_topk_ratio": 0.05,
    "image_score_mode": "topk_mean",
    "save_map_dir": "outputs/patchcore_maps",
    "save_json": None,
    "overlay_alpha": 0.45,
    "mask_mode": "none",
    "mask_image": None,
    "circle_center_x_ratio": 0.5,
    "circle_center_y_ratio": 0.5,
    "circle_radius_ratio": 0.45,
    "memory_bank_path": None,
    "rebuild_memory_bank": False,
    "calib_score_threshold": 3.0,
}


def mould_edge_aligning_ty(img_gray):
    if img_gray is None:
        return None

    if img_gray.ndim == 3:
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)
    if img_gray.dtype != np.uint8:
        img_gray = np.clip(img_gray, 0, 255).astype(np.uint8)

    h, w = img_gray.shape[:2]
    if h < 8 or w < 8:
        return None

    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

    # OpenCV 4.13下自适应阈值对block size更敏感，按图像尺寸自适应设置为奇数。
    blk = max(17, (min(h, w) // 16) | 1)
    img_binary = cv2.adaptiveThreshold(
        img_blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blk,
        3,
    )

    k = max(3, (min(h, w) // 128) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    img_binary = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    img_binary = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, kernel, iterations=1)

    contours_info = cv2.findContours(
        img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]
    if not contours:
        return None

    image_area = float(h * w)
    cx0, cy0 = (w - 1) * 0.5, (h - 1) * 0.5
    best: tuple[float, tuple[int, int], int] | None = None

    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        area_ratio = area / image_area
        if area_ratio < 0.10 or area_ratio > 0.95:
            continue

        peri = float(cv2.arcLength(cnt, True))
        if peri <= 1e-6:
            continue
        circularity = 4.0 * math.pi * area / (peri * peri)
        if circularity < 0.45:
            continue

        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        if radius < 4:
            continue

        center_dist = math.hypot(cx - cx0, cy - cy0)
        center_score = center_dist / max(1.0, min(h, w) * 0.5)
        shape_score = abs(1.0 - circularity)

        # 轻微偏向“面积较大且近中心”的轮廓，减少边缘噪声误检。
        score = center_score * 0.7 + shape_score * 0.2 + (1.0 - area_ratio) * 0.1
        candidate = (score, (int(round(cx)), int(round(cy))), int(round(radius)))
        if best is None or candidate[0] < best[0]:
            best = candidate

    if best is not None:
        _, center, radius = best
        return (center, radius)

    # 回退：允许椭圆轮廓，按中心距离优先。
    fallback: tuple[float, tuple[int, int], int] | None = None
    for cnt in contours:
        if len(cnt) < 5:
            continue
        ellipse = cv2.fitEllipse(cnt)
        (cx, cy), (ma, mi), _ = ellipse
        if ma <= 0 or mi <= 0:
            continue
        area = math.pi * ma * mi * 0.25
        area_ratio = area / image_area
        if area_ratio < 0.10 or area_ratio > 0.98:
            continue

        center_dist = math.hypot(cx - cx0, cy - cy0)
        score = center_dist / max(1.0, min(h, w) * 0.5)
        radius = int(round((ma + mi) * 0.25))
        candidate = (score, (int(round(cx)), int(round(cy))), radius)
        if fallback is None or candidate[0] < fallback[0]:
            fallback = candidate

    if fallback is None:
        return None

    _, center, radius = fallback
    return (center, radius)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Few-shot anomaly detection (PatchCore-style) with frozen DINOv3 ViT-L dense patch features."
        )
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=None,
        help="Directory containing normal images.",
    )
    parser.add_argument(
        "--query",
        type=Path,
        default=None,
        help="A query image path or a directory of query images.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("patchcore_config.json"),
        help="Path to PatchCore runtime config JSON.",
    )
    parser.add_argument(
        "--memory-bank-image-count",
        type=int,
        default=None,
        help="Number of first images in --reference-dir used to build memory bank (overrides config).",
    )
    parser.add_argument(
        "--backbone-weights",
        type=Path,
        default=None,
        help="Optional local path to dinov3_vitl16 backbone weights (.pth).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Inference device, e.g. cuda or cpu.",
    )
    parser.add_argument(
        "--resize-long-side",
        type=int,
        default=None,
        help="Target long side. Use 0 for automatic sizing.",
    )
    parser.add_argument(
        "--min-long-side",
        type=int,
        default=None,
        help="Minimum long side used by automatic sizing.",
    )
    parser.add_argument(
        "--max-long-side",
        type=int,
        default=None,
        help="Maximum long side used by automatic sizing.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=None,
        help="Backbone patch size used to infer patch grid.",
    )
    parser.add_argument(
        "--coreset-ratio",
        type=float,
        default=None,
        help="Ratio of normal patch features kept after coreset compression.",
    )
    parser.add_argument(
        "--max-memory",
        type=int,
        default=None,
        help="Maximum number of patch vectors in compressed memory bank.",
    )
    parser.add_argument(
        "--proj-dim",
        type=int,
        default=None,
        help="Projection dimension for k-center coreset selection.",
    )
    parser.add_argument(
        "--nn-batch-size",
        type=int,
        default=None,
        help="Chunk size for nearest-neighbor distance calculation.",
    )
    parser.add_argument(
        "--image-score-topk-ratio",
        type=float,
        default=None,
        help="Top-k patch ratio used to aggregate image-level anomaly score.",
    )
    parser.add_argument(
        "--image-score-mode",
        type=str,
        default=None,
        choices=("topk_mean", "p95", "p99"),
        help=(
            "Image score aggregation mode: topk_mean (existing), p95, or p99. "
            "P95/P99 are more sensitive to local anomalies."
        ),
    )
    parser.add_argument(
        "--save-json",
        type=Path,
        default=None,
        help="Optional output path for scoring results as JSON.",
    )
    parser.add_argument(
        "--save-map-dir",
        type=Path,
        default=None,
        help="Directory to save patch-level anomaly maps and overlays.",
    )
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=None,
        help="Alpha for heatmap overlay on original image, between 0 and 1.",
    )
    parser.add_argument(
        "--mask-mode",
        type=str,
        default=None,
        choices=("none", "circle", "mask-image", "mould-circle"),
        help="Valid anomaly region mode.",
    )
    parser.add_argument(
        "--mask-image",
        type=Path,
        default=None,
        help="Binary mask image used when --mask-mode=mask-image.",
    )
    parser.add_argument(
        "--circle-center-x-ratio",
        type=float,
        default=None,
        help="Circle center x ratio in [0,1] for --mask-mode=circle.",
    )
    parser.add_argument(
        "--circle-center-y-ratio",
        type=float,
        default=None,
        help="Circle center y ratio in [0,1] for --mask-mode=circle.",
    )
    parser.add_argument(
        "--circle-radius-ratio",
        type=float,
        default=None,
        help="Circle radius ratio against min(H, W) for --mask-mode=circle.",
    )
    parser.add_argument(
        "--memory-bank-path",
        type=Path,
        default=None,
        help="Optional path to save/load compressed PatchCore memory bank (.pt).",
    )
    parser.add_argument(
        "--rebuild-memory-bank",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Whether to force rebuilding memory bank even if --memory-bank-path exists. "
            "If omitted, use config value."
        ),
    )
    parser.add_argument(
        "--calib-score-threshold",
        type=float,
        default=None,
        help=(
            "Anomaly threshold in calibrated score units (z-score relative to good validation set). "
            "If omitted, use config value (default 3.0)."
        ),
    )
    return parser.parse_args()


def iter_images(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() in IMAGE_EXTS:
            return [path]
        raise ValueError(f"Unsupported image extension: {path}")
    if not path.is_dir():
        raise FileNotFoundError(f"Path not found: {path}")

    images = sorted(
        p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if not images:
        raise ValueError(f"No images found under: {path}")
    return images


def load_runtime_config(config_path: Path) -> dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    if config_path.exists():
        loaded = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"Config must be a JSON object: {config_path}")
        config.update(loaded)
    return config


def _round_to_multiple(value: int, base: int) -> int:
    return max(base, int(round(value / base) * base))


def _compute_resize_shape(
    height: int,
    width: int,
    patch_size: int,
    resize_long_side: int,
    min_long_side: int,
    max_long_side: int,
) -> tuple[int, int]:
    long_side = max(height, width)
    if resize_long_side > 0:
        target_long = resize_long_side
    else:
        # In auto mode, always use the allowed maximum resolution.
        target_long = max_long_side

    scale = target_long / long_side
    new_h = max(1, round(height * scale))
    new_w = max(1, round(width * scale))

    # Align to patch grid for stable dense feature extraction.
    new_h = _round_to_multiple(new_h, patch_size)
    new_w = _round_to_multiple(new_w, patch_size)
    return new_h, new_w


def load_image_tensor(
    image_path: Path,
    resize_long_side: int,
    patch_size: int,
    min_long_side: int,
    max_long_side: int,
) -> tuple[Image.Image, torch.Tensor]:
    image = Image.open(image_path).convert("RGB")
    tensor = to_tensor(image)
    h, w = tensor.shape[-2:]
    new_h, new_w = _compute_resize_shape(
        h,
        w,
        patch_size,
        resize_long_side,
        min_long_side,
        max_long_side,
    )
    if (new_h, new_w) != (h, w):
        tensor = resize(
            tensor,
            (new_h, new_w),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
    normalized = normalize(tensor, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    return image, normalized


@torch.inference_mode()
def extract_patch_tokens(
    model: torch.nn.Module, image_tensor: torch.Tensor
) -> torch.Tensor:
    outputs = model.forward_features(image_tensor.unsqueeze(0))
    patch_tokens = outputs["x_norm_patchtokens"][0].float().cpu()
    return F.normalize(patch_tokens, dim=1)


def estimate_patch_grid(
    n_patches: int, image_tensor: torch.Tensor, patch_size: int
) -> tuple[int, int]:
    h, w = image_tensor.shape[-2:]
    gh = max(1, h // patch_size)
    gw = max(1, w // patch_size)
    if gh * gw == n_patches:
        return gh, gw

    side = int(round(n_patches**0.5))
    if side * side == n_patches:
        return side, side

    raise ValueError(
        f"Cannot infer patch grid for {n_patches} patches with image size {(h, w)}"
    )


def pairwise_min_dist(
    query: torch.Tensor, memory: torch.Tensor, batch_size: int
) -> torch.Tensor:
    mins: list[torch.Tensor] = []
    for i in range(0, query.shape[0], batch_size):
        q = query[i : i + batch_size]
        d = torch.cdist(q, memory)
        mins.append(d.min(dim=1).values)
    return torch.cat(mins, dim=0)


def build_valid_pixel_mask(
    image_tensor: torch.Tensor,
    mask_mode: str,
    mask_image: Path | None,
    circle_center_x_ratio: float,
    circle_center_y_ratio: float,
    circle_radius_ratio: float,
) -> torch.Tensor:
    h, w = image_tensor.shape[-2:]

    if mask_mode == "none":
        return torch.ones((h, w), dtype=torch.bool)

    if mask_mode == "circle":
        cx = circle_center_x_ratio * (w - 1)
        cy = circle_center_y_ratio * (h - 1)
        radius = max(1.0, circle_radius_ratio * min(h, w))
        yy, xx = torch.meshgrid(
            torch.arange(h, dtype=torch.float32),
            torch.arange(w, dtype=torch.float32),
            indexing="ij",
        )
        dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
        return dist2 <= radius**2

    if mask_mode == "mould-circle":
        # Convert normalized tensor back to uint8 grayscale for contour-based circle fitting.
        mean = torch.tensor(
            IMAGENET_DEFAULT_MEAN, dtype=image_tensor.dtype, device=image_tensor.device
        ).view(3, 1, 1)
        std = torch.tensor(
            IMAGENET_DEFAULT_STD, dtype=image_tensor.dtype, device=image_tensor.device
        ).view(3, 1, 1)
        rgb = torch.clamp(image_tensor * std + mean, 0.0, 1.0)
        rgb_u8 = (rgb.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
        gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)

        loc_mould = mould_edge_aligning_ty(gray)
        if loc_mould is not None:
            (cx, cy), radius = loc_mould
        else:
            # Fallback to ratio-based circle if edge fitting fails.
            cx = round(circle_center_x_ratio * (w - 1))
            cy = round(circle_center_y_ratio * (h - 1))
            radius = max(1, round(circle_radius_ratio * min(h, w)))

        yy, xx = torch.meshgrid(
            torch.arange(h, dtype=torch.float32),
            torch.arange(w, dtype=torch.float32),
            indexing="ij",
        )
        dist2 = (xx - float(cx)) ** 2 + (yy - float(cy)) ** 2
        return dist2 <= float(radius) ** 2

    if mask_mode == "mask-image":
        if mask_image is None:
            raise ValueError("--mask-image is required when --mask-mode=mask-image")
        if not mask_image.exists():
            raise FileNotFoundError(f"Mask image not found: {mask_image}")

        mask_img = Image.open(mask_image).convert("L")
        mask_tensor = to_tensor(mask_img)[0]
        if tuple(mask_tensor.shape) != (h, w):
            mask_tensor = resize(
                mask_tensor.unsqueeze(0),
                (h, w),
                interpolation=InterpolationMode.NEAREST,
                antialias=False,
            )[0]
        return mask_tensor > 0.5

    raise ValueError(f"Unsupported mask mode: {mask_mode}")


def build_valid_patch_mask(
    valid_pixel_mask: torch.Tensor,
    gh: int,
    gw: int,
) -> torch.Tensor:
    # Use patch coverage ratio instead of nearest-neighbor sampling to reduce
    # boundary misclassification for circular/irregular masks.
    mask = valid_pixel_mask.float().unsqueeze(0).unsqueeze(0)
    patch_coverage = F.adaptive_avg_pool2d(mask, output_size=(gh, gw))[0, 0]
    patch_mask = patch_coverage >= 0.5
    return patch_mask.reshape(-1)


def coreset_kcenter(
    features: torch.Tensor, ratio: float, max_memory: int, proj_dim: int
) -> torch.Tensor:
    n, dim = features.shape
    target = min(max(1, int(n * ratio)), max_memory, n)
    if target >= n:
        return features

    proj_dim = min(proj_dim, dim)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(42)
    proj = torch.randn(dim, proj_dim, generator=generator, dtype=features.dtype)
    proj = F.normalize(proj, dim=0)
    reduced = features @ proj

    selected = torch.empty(target, dtype=torch.long)
    selected[0] = 0
    min_d = torch.cdist(reduced[0:1], reduced).squeeze(0)

    for i in range(1, target):
        idx = torch.argmax(min_d)
        selected[i] = idx
        d_new = torch.cdist(reduced[idx : idx + 1], reduced).squeeze(0)
        min_d = torch.minimum(min_d, d_new)

    return features[selected]


def heatmap_from_score_map(
    score_map: torch.Tensor, target_size: tuple[int, int]
) -> Image.Image:
    grayscale = (score_map.numpy() * 255.0).clip(0, 255).astype("uint8")
    heatmap = colorize(
        Image.fromarray(grayscale, mode="L"), black="#0a1f44", white="#ff4d4d"
    )
    return heatmap.resize(target_size, resample=Image.Resampling.BICUBIC)


def save_patch_map_images(
    original_image: Image.Image,
    score_map: torch.Tensor,
    image_score: float,
    save_map_dir: Path,
    stem: str,
    overlay_alpha: float,
) -> tuple[Path, Path]:
    save_map_dir.mkdir(parents=True, exist_ok=True)
    alpha = max(0.0, min(1.0, overlay_alpha))

    original_rgb = original_image.convert("RGB")
    heatmap = heatmap_from_score_map(score_map, target_size=original_rgb.size)

    # PIL blend 要求两张图的 mode 和 size 完全一致，做一次防御性对齐。
    if heatmap.mode != original_rgb.mode:
        heatmap = heatmap.convert(original_rgb.mode)
    if heatmap.size != original_rgb.size:
        heatmap = heatmap.resize(original_rgb.size, resample=Image.Resampling.BICUBIC)

    overlay = Image.blend(original_rgb, heatmap, alpha=alpha)

    draw = ImageDraw.Draw(overlay)
    score_text = f"PatchCore score: {image_score:.4f}"

    # Use a larger dynamic font size for better readability on overlays.
    font_size = max(18, int(min(overlay.size) * 0.035))
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except OSError:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        except OSError:
            font = ImageFont.load_default()

    text_bbox = draw.textbbox((0, 0), score_text, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]
    pad = 8
    x0, y0 = 8, 8
    x1 = x0 + text_w + pad * 2
    y1 = y0 + text_h + pad * 2
    draw.rectangle((x0, y0, x1, y1), fill=(0, 0, 0, 160))
    draw.text((x0 + pad, y0 + pad), score_text, fill=(255, 255, 255), font=font)

    heatmap_path = save_map_dir / f"{stem}_patchcore_heatmap.png"
    overlay_path = save_map_dir / f"{stem}_patchcore_overlay.png"
    heatmap.save(heatmap_path)
    overlay.save(overlay_path)
    return heatmap_path, overlay_path


def make_anomaly_maps(
    patch_tokens: torch.Tensor,
    memory_bank: torch.Tensor,
    image_tensor: torch.Tensor,
    patch_size: int,
    nn_batch_size: int,
    valid_pixel_mask: torch.Tensor,
    global_min: float = None,
    global_max: float = None,
    # patch_score_clip 已移除
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    支持全局归一化的 anomaly map 生成。
    global_min/global_max: 若提供，则用于全局归一化（如 good 验证集统计），否则为当前图像自适应归一化。
    """
    patch_dists = pairwise_min_dist(patch_tokens, memory_bank, batch_size=nn_batch_size)
    gh, gw = estimate_patch_grid(patch_dists.numel(), image_tensor, patch_size)
    valid_patch_mask = build_valid_patch_mask(valid_pixel_mask, gh, gw)

    patch_map = patch_dists.reshape(gh, gw)
    full_map = F.interpolate(
        patch_map.unsqueeze(0).unsqueeze(0),
        size=image_tensor.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )[0, 0]

    valid_pixel_mask_f = valid_pixel_mask.float()
    full_map = full_map * valid_pixel_mask_f

    valid_values = full_map[valid_pixel_mask]
    if valid_values.numel() > 0:
        if (
            global_min is not None
            and global_max is not None
            and global_max > global_min
        ):
            # 全局归一化（推荐：good 验证集 patch 分数统计）
            min_v, max_v = float(global_min), float(global_max)
            denom = max(max_v - min_v, 1e-6)
            norm_map = (full_map - min_v) / denom
        else:
            min_v = valid_values.min()
            max_v = valid_values.max()
            denom = (max_v - min_v).clamp(min=1e-6)
            norm_map = (full_map - min_v) / denom
        full_map = torch.where(
            valid_pixel_mask,
            norm_map,
            torch.zeros_like(full_map),
        )
    else:
        full_map = torch.zeros_like(full_map)

    return patch_dists, valid_patch_mask, full_map.cpu()


def compute_image_score(
    patch_dists: torch.Tensor,
    topk_ratio: float,
    valid_patch_mask: torch.Tensor,
    mode: str = "topk_mean",
) -> float:
    effective_patch_dists = patch_dists[valid_patch_mask]
    if effective_patch_dists.numel() == 0:
        return 0.0

    if mode == "p95":
        return float(torch.quantile(effective_patch_dists, 0.95).item())
    if mode == "p99":
        return float(torch.quantile(effective_patch_dists, 0.99).item())

    n = effective_patch_dists.numel()
    k = max(1, int(n * topk_ratio))
    topk = torch.topk(effective_patch_dists, k=k, largest=True).values
    return float(topk.mean().item())


def compute_calibration_stats(scores: list[float]) -> tuple[float, float]:
    """Robust calibration using median and MAD (resistant to outliers in good val set).

    Returns (median, robust_std) where robust_std = 1.4826 * MAD.
    A calibrated_score = (raw - median) / robust_std, so:
      ~0   => typical good image
      >3   => likely anomalous (configurable threshold)
    """
    if not scores:
        return 0.0, 1.0
    arr = sorted(scores)
    n = len(arr)
    median = arr[n // 2] if n % 2 == 1 else (arr[n // 2 - 1] + arr[n // 2]) * 0.5
    deviations = sorted(abs(x - median) for x in arr)
    mad = (
        deviations[n // 2]
        if n % 2 == 1
        else (deviations[n // 2 - 1] + deviations[n // 2]) * 0.5
    )
    robust_std = max(mad * 1.4826, 1e-6)
    return median, robust_std


def calibrate_score(raw: float, calib_median: float, calib_std: float) -> float:
    """Return z-score-like calibrated anomaly score relative to good validation set."""
    return (raw - calib_median) / calib_std


def save_memory_bank(
    memory_bank: torch.Tensor,
    args: argparse.Namespace,
    path: Path,
    memory_bank_image_count: int,
    calib_stats: tuple[float, float] | None = None,
) -> None:
    payload = {
        "memory_bank": memory_bank.cpu(),
        "meta": {
            "model": "dinov3_vitl16",
            "patch_size": int(args.patch_size),
            "resize_long_side": int(args.resize_long_side),
            "min_long_side": int(args.min_long_side),
            "max_long_side": int(args.max_long_side),
            "coreset_ratio": float(args.coreset_ratio),
            "max_memory": int(args.max_memory),
            "proj_dim": int(args.proj_dim),
            "reference_dir": str(args.reference_dir.resolve()),
            "memory_bank_image_count": int(memory_bank_image_count),
            "mask_mode": str(args.mask_mode),
            "mask_image": (
                str(args.mask_image.resolve()) if args.mask_image is not None else None
            ),
            "circle_center_x_ratio": float(args.circle_center_x_ratio),
            "circle_center_y_ratio": float(args.circle_center_y_ratio),
            "circle_radius_ratio": float(args.circle_radius_ratio),
        },
    }
    if calib_stats is not None:
        payload["calib_stats"] = {
            "median": float(calib_stats[0]),
            "std": float(calib_stats[1]),
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_memory_bank(
    path: Path, args: argparse.Namespace
) -> tuple[torch.Tensor, tuple[float, float] | None]:
    payload = torch.load(path, map_location="cpu")
    memory_bank = payload["memory_bank"]
    meta = payload.get("meta", {})

    if meta:
        checks = {
            "patch_size": int(args.patch_size),
            "resize_long_side": int(args.resize_long_side),
            "min_long_side": int(args.min_long_side),
            "max_long_side": int(args.max_long_side),
            "mask_mode": str(args.mask_mode),
            "mask_image": (
                str(args.mask_image.resolve()) if args.mask_image is not None else None
            ),
            "circle_center_x_ratio": float(args.circle_center_x_ratio),
            "circle_center_y_ratio": float(args.circle_center_y_ratio),
            "circle_radius_ratio": float(args.circle_radius_ratio),
        }
        for key, current in checks.items():
            stored = meta.get(key)
            if stored is None:
                continue
            if isinstance(current, float):
                if abs(float(stored) - current) > 1e-8:
                    raise ValueError(
                        f"Memory bank config mismatch for {key}: stored={stored}, current={current}. "
                        f"Use --rebuild-memory-bank or align arguments."
                    )
            else:
                if str(stored) != str(current):
                    raise ValueError(
                        f"Memory bank config mismatch for {key}: stored={stored}, current={current}. "
                        f"Use --rebuild-memory-bank or align arguments."
                    )
    raw_calib = payload.get("calib_stats")
    calib_stats: tuple[float, float] | None = None
    if raw_calib and "median" in raw_calib and "std" in raw_calib:
        calib_stats = (float(raw_calib["median"]), float(raw_calib["std"]))
    return memory_bank, calib_stats


def main() -> None:
    args = parse_args()
    runtime_config = load_runtime_config(args.config)

    def _resolve(name: str):
        cli_val = getattr(args, name)
        if cli_val is not None:
            return cli_val
        return runtime_config.get(name)

    reference_dir_val = _resolve("reference_dir")
    if not reference_dir_val:
        raise ValueError(
            "reference_dir is required (set in config or pass --reference-dir)"
        )
    args.reference_dir = Path(str(reference_dir_val))

    query_val = _resolve("query")
    args.query = Path(str(query_val)) if query_val else None

    backbone_weights_val = _resolve("backbone_weights")
    args.backbone_weights = (
        Path(str(backbone_weights_val)) if backbone_weights_val else None
    )

    args.save_map_dir = Path(str(_resolve("save_map_dir")))
    save_json_val = _resolve("save_json")
    args.save_json = Path(str(save_json_val)) if save_json_val else None

    memory_bank_path_val = _resolve("memory_bank_path")
    args.memory_bank_path = (
        Path(str(memory_bank_path_val)) if memory_bank_path_val else None
    )

    device_name = _resolve("device")
    if not device_name:
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = str(device_name)
    device = torch.device(args.device)

    args.resize_long_side = int(_resolve("resize_long_side"))
    args.min_long_side = int(_resolve("min_long_side"))
    args.max_long_side = int(_resolve("max_long_side"))
    args.patch_size = int(_resolve("patch_size"))
    args.coreset_ratio = float(_resolve("coreset_ratio"))
    args.max_memory = int(_resolve("max_memory"))
    args.proj_dim = int(_resolve("proj_dim"))
    args.nn_batch_size = int(_resolve("nn_batch_size"))
    args.image_score_topk_ratio = float(_resolve("image_score_topk_ratio"))
    args.image_score_mode = str(_resolve("image_score_mode"))
    args.overlay_alpha = float(_resolve("overlay_alpha"))
    args.mask_mode = str(_resolve("mask_mode"))
    mask_image_val = _resolve("mask_image")
    args.mask_image = Path(str(mask_image_val)) if mask_image_val else None
    args.circle_center_x_ratio = float(_resolve("circle_center_x_ratio"))
    args.circle_center_y_ratio = float(_resolve("circle_center_y_ratio"))
    args.circle_radius_ratio = float(_resolve("circle_radius_ratio"))
    rebuild_memory_bank_val = _resolve("rebuild_memory_bank")
    args.rebuild_memory_bank = bool(rebuild_memory_bank_val)
    calib_threshold = float(_resolve("calib_score_threshold"))

    memory_bank_image_count = int(_resolve("memory_bank_image_count"))
    if memory_bank_image_count <= 0:
        raise ValueError("memory_bank_image_count must be > 0")

    reference_paths = iter_images(args.reference_dir)
    if memory_bank_image_count >= len(reference_paths):
        raise ValueError(
            "memory_bank_image_count must be smaller than total images in reference-dir "
            f"({len(reference_paths)})."
        )

    memory_bank_paths = reference_paths[:memory_bank_image_count]
    # Good validation set: always the remainder of reference_dir after memory bank images.
    good_val_paths = reference_paths[memory_bank_image_count:]
    if args.query is None:
        query_paths = good_val_paths
        if not query_paths:
            raise ValueError(
                "No query images available from reference-dir remainder. "
                "Reduce memory_bank_image_count or provide --query."
            )
    else:
        query_paths = iter_images(args.query)

    model_kwargs: dict[str, object] = {"pretrained": True}
    if args.backbone_weights is not None:
        model_kwargs["weights"] = str(args.backbone_weights)

    model = dinov3_vitl16(**model_kwargs).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    all_ref_patch_count: int | None = None
    use_cached_bank = (
        args.memory_bank_path is not None
        and args.memory_bank_path.exists()
        and not args.rebuild_memory_bank
    )

    if use_cached_bank:
        memory_bank, stored_calib = load_memory_bank(args.memory_bank_path, args)
        print(f"Loaded memory bank: {args.memory_bank_path}")
    else:
        stored_calib = None
        ref_patch_features: list[torch.Tensor] = []
        for path in memory_bank_paths:
            _, image_tensor = load_image_tensor(
                path,
                args.resize_long_side,
                args.patch_size,
                args.min_long_side,
                args.max_long_side,
            )
            image_tensor = image_tensor.to(device)
            patch_tokens = extract_patch_tokens(model, image_tensor)
            valid_pixel_mask = build_valid_pixel_mask(
                image_tensor,
                args.mask_mode,
                args.mask_image,
                args.circle_center_x_ratio,
                args.circle_center_y_ratio,
                args.circle_radius_ratio,
            ).cpu()
            gh, gw = estimate_patch_grid(
                patch_tokens.shape[0], image_tensor, args.patch_size
            )
            valid_patch_mask = build_valid_patch_mask(valid_pixel_mask, gh, gw)
            masked_patch_tokens = patch_tokens[valid_patch_mask]
            if masked_patch_tokens.numel() == 0:
                continue
            ref_patch_features.append(masked_patch_tokens)

        if not ref_patch_features:
            raise ValueError(
                "No valid normal patches found for memory bank after applying mask. "
                "Please adjust mask parameters."
            )

        all_ref_features = torch.cat(ref_patch_features, dim=0)
        all_ref_patch_count = int(all_ref_features.shape[0])
        memory_bank = coreset_kcenter(
            all_ref_features,
            ratio=args.coreset_ratio,
            max_memory=args.max_memory,
            proj_dim=args.proj_dim,
        )
        if args.memory_bank_path is not None:
            save_memory_bank(
                memory_bank,
                args,
                args.memory_bank_path,
                memory_bank_image_count,
            )
            print(f"Saved memory bank: {args.memory_bank_path}")

    # ---- Calibration on good validation set --------------------------------

    # --------- 全局 patch 分数 min/max 统计（good 验证集） ---------
    global_patch_min, global_patch_max = None, None

    if stored_calib is not None:
        calib_median, calib_std = stored_calib
        print(
            f"Loaded calibration stats from bank: "
            f"median={calib_median:.6f} robust_std={calib_std:.6f}"
        )
    else:
        if not good_val_paths:
            print("Warning: no good validation images available; calibration disabled.")
            calib_median, calib_std = 0.0, 1.0
        else:
            print(
                f"Computing calibration from {len(good_val_paths)} good validation images..."
            )
            val_scores: list[float] = []
            patch_min_list: list[float] = []
            patch_max_list: list[float] = []
            patch_all_list: list[float] = []
            for val_path in good_val_paths:
                _, val_tensor = load_image_tensor(
                    val_path,
                    args.resize_long_side,
                    args.patch_size,
                    args.min_long_side,
                    args.max_long_side,
                )
                val_tensor = val_tensor.to(device)
                val_pixel_mask = build_valid_pixel_mask(
                    val_tensor,
                    args.mask_mode,
                    args.mask_image,
                    args.circle_center_x_ratio,
                    args.circle_center_y_ratio,
                    args.circle_radius_ratio,
                ).cpu()
                val_patch_tokens = extract_patch_tokens(model, val_tensor)
                val_dists, val_patch_mask, _ = make_anomaly_maps(
                    val_patch_tokens,
                    memory_bank,
                    val_tensor,
                    patch_size=args.patch_size,
                    nn_batch_size=args.nn_batch_size,
                    valid_pixel_mask=val_pixel_mask,
                )
                # 统计 patch 分数 min/max/全量
                valid_patch_scores = val_dists[val_patch_mask]
                if valid_patch_scores.numel() > 0:
                    patch_min_list.append(float(valid_patch_scores.min().item()))
                    patch_max_list.append(float(valid_patch_scores.max().item()))
                    patch_all_list.extend(valid_patch_scores.cpu().numpy().tolist())
                val_scores.append(
                    compute_image_score(
                        val_dists,
                        topk_ratio=args.image_score_topk_ratio,
                        valid_patch_mask=val_patch_mask,
                        mode=args.image_score_mode,
                    )
                )
            calib_median, calib_std = compute_calibration_stats(val_scores)
            # 全局 patch 分数 min/max
            if patch_min_list and patch_max_list and patch_all_list:
                global_patch_min = float(np.min(patch_min_list))
                global_patch_max = float(np.max(patch_max_list))
                print(
                    f"Global patch score min={global_patch_min:.6f} max={global_patch_max:.6f}"
                )
            # patch 分数 clip 阈值
            print(
                f"Calibration: median={calib_median:.6f} robust_std={calib_std:.6f} "
                f"threshold@{calib_threshold:.1f}sigma "
                f"=> raw>={calib_median + calib_threshold * calib_std:.6f}"
            )
            # Persist calib stats into bank file for future runs.
            if args.memory_bank_path is not None and args.memory_bank_path.exists():
                payload = torch.load(args.memory_bank_path, map_location="cpu")
                payload["calib_stats"] = {"median": calib_median, "std": calib_std}
                torch.save(payload, args.memory_bank_path)
                print(f"Updated bank with calibration stats: {args.memory_bank_path}")

    print(f"Reference images (total in good): {len(reference_paths)}")
    print(f"Memory bank images (first N): {memory_bank_image_count}")
    if all_ref_patch_count is not None:
        print(f"Reference patches: {all_ref_patch_count}")
    print(f"Memory bank patches after coreset: {memory_bank.shape[0]}")
    print(f"Query images: {len(query_paths)}")

    results: list[dict[str, object]] = []

    for query_path in query_paths:
        original_image, image_tensor = load_image_tensor(
            query_path,
            args.resize_long_side,
            args.patch_size,
            args.min_long_side,
            args.max_long_side,
        )
        image_tensor = image_tensor.to(device)
        valid_pixel_mask = build_valid_pixel_mask(
            image_tensor,
            args.mask_mode,
            args.mask_image,
            args.circle_center_x_ratio,
            args.circle_center_y_ratio,
            args.circle_radius_ratio,
        ).cpu()

        query_patch_tokens = extract_patch_tokens(model, image_tensor)
        patch_dists, valid_patch_mask, score_map = make_anomaly_maps(
            query_patch_tokens,
            memory_bank,
            image_tensor,
            patch_size=args.patch_size,
            nn_batch_size=args.nn_batch_size,
            valid_pixel_mask=valid_pixel_mask,
            global_min=global_patch_min,
            global_max=global_patch_max,
        )

        image_score = compute_image_score(
            patch_dists,
            topk_ratio=args.image_score_topk_ratio,
            valid_patch_mask=valid_patch_mask,
            mode=args.image_score_mode,
        )
        heatmap_path, overlay_path = save_patch_map_images(
            original_image,
            score_map,
            image_score,
            args.save_map_dir,
            query_path.stem,
            args.overlay_alpha,
        )

        cal_score = calibrate_score(image_score, calib_median, calib_std)
        is_anomaly = cal_score >= calib_threshold

        result = {
            "image": str(query_path),
            "image_anomaly_score": image_score,
            "calibrated_score": round(cal_score, 6),
            "is_anomaly": is_anomaly,
            "calib_threshold": calib_threshold,
            "patch_score_stats": {
                "max": float(patch_dists.max().item()),
                "mean": float(patch_dists.mean().item()),
                "topk_mean": image_score,
                "image_score_mode": args.image_score_mode,
            },
            "patch_map": {
                "heatmap": str(heatmap_path),
                "overlay": str(overlay_path),
            },
        }
        results.append(result)

        verdict = "ANOMALY" if is_anomaly else "ok"
        print(
            f"{query_path.name} | "
            f"score={image_score:.6f} cal={cal_score:+.3f} [{verdict}] "
            f"overlay={overlay_path}"
        )

    if args.save_json is not None:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Saved JSON: {args.save_json}")


if __name__ == "__main__":
    main()
