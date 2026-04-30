from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from PIL.ImageOps import colorize
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import normalize, resize, to_tensor

from dinov3.data.transforms import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from dinov3.hub.backbones import dinov3_vitl16


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Zero-shot anomaly detection with frozen DINOv3 ViT-L features."
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        required=True,
        help="Directory of normal/reference images.",
    )
    parser.add_argument(
        "--query",
        type=Path,
        required=True,
        help="A query image path or a directory of query images.",
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
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device, e.g. cuda or cpu.",
    )
    parser.add_argument(
        "--resize-long-side",
        type=int,
        default=518,
        help="Resize each image so its long side equals this value.",
    )
    parser.add_argument(
        "--knn-k",
        type=int,
        default=5,
        help="k for kNN anomaly score.",
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
        default=Path("outputs") / "patch_maps",
        help="Directory to save patch-level anomaly maps and overlays.",
    )
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.45,
        help="Alpha for heatmap overlay on original image, between 0 and 1.",
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


def load_image_tensor(
    image_path: Path, resize_long_side: int
) -> tuple[Image.Image, torch.Tensor]:
    image = Image.open(image_path).convert("RGB")
    tensor = to_tensor(image)
    h, w = tensor.shape[-2:]
    long_side = max(h, w)
    if long_side != resize_long_side:
        scale = resize_long_side / long_side
        new_size = (max(1, round(h * scale)), max(1, round(w * scale)))
        tensor = resize(
            tensor, new_size, interpolation=InterpolationMode.BICUBIC, antialias=True
        )
    normalized = normalize(tensor, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    return image, normalized


@torch.inference_mode()
def extract_features(
    model: torch.nn.Module, image_tensor: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    outputs = model.forward_features(image_tensor.unsqueeze(0))
    patch_tokens = outputs["x_norm_patchtokens"][0].float().cpu()
    cls_token = outputs["x_norm_clstoken"][0].float().cpu()
    return patch_tokens, cls_token


def make_embedding(patch_tokens: torch.Tensor, cls_token: torch.Tensor) -> torch.Tensor:
    patch_mean = patch_tokens.mean(dim=0)
    emb = torch.cat(
        [F.normalize(cls_token, dim=0), F.normalize(patch_mean, dim=0)], dim=0
    )
    return F.normalize(emb, dim=0)


def build_reference_bank(
    model: torch.nn.Module,
    reference_paths: list[Path],
    resize_long_side: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    embeddings: list[torch.Tensor] = []
    all_patch_tokens: list[torch.Tensor] = []
    for path in reference_paths:
        _, image_tensor = load_image_tensor(path, resize_long_side)
        image_tensor = image_tensor.to(device)
        patch_tokens, cls_token = extract_features(model, image_tensor)
        embeddings.append(make_embedding(patch_tokens, cls_token))
        all_patch_tokens.append(F.normalize(patch_tokens, dim=1))
    reference_bank = torch.stack(embeddings, dim=0)
    patch_bank = torch.cat(all_patch_tokens, dim=0)
    return reference_bank, patch_bank


def estimate_patch_grid(
    n_patches: int, image_tensor: torch.Tensor, patch_size: int = 16
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


def make_patch_anomaly_map(
    patch_tokens: torch.Tensor,
    patch_prototype: torch.Tensor,
    image_tensor: torch.Tensor,
    patch_size: int = 16,
) -> torch.Tensor:
    normalized_tokens = F.normalize(patch_tokens, dim=1)
    cosine_sim = normalized_tokens @ patch_prototype
    patch_scores = 1.0 - cosine_sim
    gh, gw = estimate_patch_grid(
        patch_scores.numel(), image_tensor, patch_size=patch_size
    )
    patch_map = patch_scores.reshape(gh, gw)
    full_map = F.interpolate(
        patch_map.unsqueeze(0).unsqueeze(0),
        size=image_tensor.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    full_map = full_map - full_map.min()
    full_map = full_map / full_map.max().clamp(min=1e-6)
    return full_map.cpu()


def heatmap_from_score_map(
    score_map: torch.Tensor, target_size: tuple[int, int]
) -> Image.Image:
    grayscale = (score_map.numpy() * 255.0).clip(0, 255).astype("uint8")
    heatmap = colorize(
        Image.fromarray(grayscale, mode="L"), black="#0a1f44", white="#ff4d4d"
    )
    return heatmap.resize(target_size, resample=Image.Resampling.BICUBIC)


def save_patch_level_maps(
    original_image: Image.Image,
    score_map: torch.Tensor,
    save_map_dir: Path,
    stem: str,
    overlay_alpha: float,
) -> tuple[Path, Path]:
    save_map_dir.mkdir(parents=True, exist_ok=True)
    alpha = max(0.0, min(1.0, overlay_alpha))

    heatmap = heatmap_from_score_map(score_map, target_size=original_image.size)
    overlay = Image.blend(original_image, heatmap, alpha=alpha)

    heatmap_path = save_map_dir / f"{stem}_patch_anomaly_heatmap.png"
    overlay_path = save_map_dir / f"{stem}_patch_anomaly_overlay.png"
    heatmap.save(heatmap_path)
    overlay.save(overlay_path)
    return heatmap_path, overlay_path


def fit_mahalanobis(reference_bank: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = reference_bank.mean(dim=0)
    centered = reference_bank - mean
    n = reference_bank.shape[0]
    cov = (centered.T @ centered) / max(1, n - 1)
    eye = torch.eye(cov.shape[0], dtype=cov.dtype)
    cov = cov + 1e-4 * eye
    inv_cov = torch.linalg.pinv(cov)
    return mean, inv_cov


def score_mahalanobis(
    embedding: torch.Tensor, mean: torch.Tensor, inv_cov: torch.Tensor
) -> float:
    delta = embedding - mean
    value = torch.sqrt(torch.clamp(delta @ inv_cov @ delta, min=0.0))
    return float(value.item())


def score_cosine(embedding: torch.Tensor, reference_bank: torch.Tensor) -> float:
    sims = reference_bank @ embedding
    return float((1.0 - sims.max()).item())


def score_knn(embedding: torch.Tensor, reference_bank: torch.Tensor, k: int) -> float:
    dists = torch.norm(reference_bank - embedding.unsqueeze(0), dim=1)
    k = min(k, dists.numel())
    return float(torch.topk(dists, k=k, largest=False).values.mean().item())


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    reference_paths = iter_images(args.reference_dir)
    query_paths = iter_images(args.query)

    model_kwargs: dict[str, object] = {"pretrained": True}
    if args.backbone_weights is not None:
        model_kwargs["weights"] = str(args.backbone_weights)
    model = dinov3_vitl16(**model_kwargs).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    reference_bank, patch_bank = build_reference_bank(
        model, reference_paths, args.resize_long_side, device
    )
    mean, inv_cov = fit_mahalanobis(reference_bank)
    patch_prototype = F.normalize(patch_bank.mean(dim=0), dim=0)

    results: list[dict[str, object]] = []
    print(f"Reference images: {len(reference_paths)}")
    print(f"Query images: {len(query_paths)}")

    for query_path in query_paths:
        original_image, image_tensor = load_image_tensor(
            query_path, args.resize_long_side
        )
        image_tensor = image_tensor.to(device)
        patch_tokens, cls_token = extract_features(model, image_tensor)
        embedding = make_embedding(patch_tokens, cls_token)
        patch_map = make_patch_anomaly_map(
            patch_tokens,
            patch_prototype,
            image_tensor,
            patch_size=16,
        )
        heatmap_path, overlay_path = save_patch_level_maps(
            original_image,
            patch_map,
            args.save_map_dir,
            query_path.stem,
            args.overlay_alpha,
        )

        maha = score_mahalanobis(embedding, mean, inv_cov)
        cosine = score_cosine(embedding, reference_bank)
        knn = score_knn(embedding, reference_bank, args.knn_k)

        result = {
            "image": str(query_path),
            "scores": {
                "mahalanobis": maha,
                "cosine": cosine,
                "knn": knn,
            },
            "anomaly_score": {
                "mahalanobis": maha,
                "cosine": cosine,
                "knn": knn,
            },
            "patch_map": {
                "heatmap": str(heatmap_path),
                "overlay": str(overlay_path),
            },
        }
        results.append(result)

        print(
            f"{query_path.name} | "
            f"mahalanobis={maha:.6f} "
            f"cosine={cosine:.6f} "
            f"knn={knn:.6f} "
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
