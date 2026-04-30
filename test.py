from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from PIL.ImageOps import colorize
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import normalize, resize, to_tensor


WORKSPACE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = WORKSPACE_ROOT / "dinov3"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LOCAL_MODEL_DIRS = (
    WORKSPACE_ROOT / "model",
    WORKSPACE_ROOT / "models",
    REPO_ROOT / "model",
    REPO_ROOT / "models",
)

from dinov3.data.transforms import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from dinov3.hub.backbones import (
    dinov3_vit7b16,
    dinov3_vitb16,
    dinov3_vitl16,
    dinov3_vitl16plus,
    dinov3_vits16,
    dinov3_vits16plus,
)
from dinov3.hub.detectors import dinov3_vit7b16_de, dinov3_vitl16plus_de

# from dinov3.hub.depthers import dinov3_vitl16_chmv2


COCO_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "N/A",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "N/A",
    "backpack",
    "umbrella",
    "N/A",
    "N/A",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "N/A",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "N/A",
    "dining table",
    "N/A",
    "N/A",
    "toilet",
    "N/A",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "N/A",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def parse_args() -> argparse.Namespace:
    model_choices = (
        "dinov3_vitl16plus_de",
        "dinov3_vit7b16_de",
        "dinov3_vits16",
        "dinov3_vits16plus",
        "dinov3_vitb16",
        "dinov3_vitl16",
        "dinov3_vitl16plus",
        "dinov3_vit7b16",
        "dinov3_vitl16_chmv2",
    )
    parser = argparse.ArgumentParser(
        description="Run a DINOv3 detector on one image and save detections plus feature map outputs."
    )
    parser.add_argument("image", type=Path, help="Path to the input image.")
    parser.add_argument(
        "--model-name",
        type=str,
        default="dinov3_vitl16plus_de",
        choices=model_choices,
        help="Model name.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WORKSPACE_ROOT / "outputs",
        help="Directory used to store the annotated image, feature heatmap and raw tensor.",
    )
    parser.add_argument(
        "--score-thresh",
        type=float,
        default=0.4,
        help="Keep detections whose score is >= this threshold.",
    )
    parser.add_argument(
        "--resize-long-side",
        type=int,
        default=None,
        help="Optionally resize the image before inference so its long side equals this value.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device, for example cuda or cpu.",
    )
    parser.add_argument(
        "--detector-weights",
        type=Path,
        default=None,
        help="Optional local path for DETR head weights (.pth).",
    )
    parser.add_argument(
        "--backbone-weights",
        type=Path,
        default=None,
        help="Optional local path for DINOv3 backbone weights (.pth).",
    )
    return parser.parse_args()


DETECTOR_MODELS = {
    "dinov3_vitl16plus_de": dinov3_vitl16plus_de,
    "dinov3_vit7b16_de": dinov3_vit7b16_de,
}


DEPTHER_MODELS = {
    "dinov3_vitl16_chmv2": dinov3_vitl16_chmv2,
}


BACKBONE_MODELS = {
    "dinov3_vits16": dinov3_vits16,
    "dinov3_vits16plus": dinov3_vits16plus,
    "dinov3_vitb16": dinov3_vitb16,
    "dinov3_vitl16": dinov3_vitl16,
    "dinov3_vitl16plus": dinov3_vitl16plus,
    "dinov3_vit7b16": dinov3_vit7b16,
}


MODEL_WEIGHT_HINTS = {
    "dinov3_vitl16plus_de": ("vitl16plus", "detr_head"),
    "dinov3_vit7b16_de": ("vit7b16", "detr_head"),
    "dinov3_vits16": ("vits16", "pretrain"),
    "dinov3_vits16plus": ("vits16plus", "pretrain"),
    "dinov3_vitb16": ("vitb16", "pretrain"),
    "dinov3_vitl16": ("vitl16", "pretrain"),
    "dinov3_vitl16plus": ("vitl16plus", "pretrain"),
    "dinov3_vit7b16": ("vit7b16", "pretrain"),
    "dinov3_vitl16_chmv2": ("vitl16", "dpt_head"),
}


def _find_local_weight(model_name: str, *, kind: str) -> Path | None:
    expected_tokens = MODEL_WEIGHT_HINTS[model_name]
    if kind == "backbone":
        required = (expected_tokens[0], "pretrain")
    else:
        required = (expected_tokens[0], expected_tokens[1])

    for directory in LOCAL_MODEL_DIRS:
        if not directory.exists():
            continue
        for candidate in sorted(directory.glob("*.pth")):
            candidate_name = candidate.name.lower()
            if all(token in candidate_name for token in required):
                return candidate
    return None


def _validate_weight_path(model_name: str, weight_path: Path, *, kind: str) -> None:
    expected_tokens = MODEL_WEIGHT_HINTS[model_name]
    weight_name = weight_path.name.lower()
    if kind == "backbone":
        required = (expected_tokens[0], "pretrain")
    else:
        required = (expected_tokens[0], expected_tokens[1])

    if all(token in weight_name for token in required):
        return

    joined = ", ".join(required)
    raise ValueError(
        f"{kind.capitalize()} weights {weight_path.name!r} do not look compatible with model "
        f"{model_name!r}. Expected filename tokens: {joined}."
    )


def build_model(
    model_name: str,
    detector_weights: Path | None,
    backbone_weights: Path | None,
    device: torch.device,
) -> tuple[torch.nn.Module, str]:
    model_kwargs: dict[str, object] = {"pretrained": True}

    if model_name in DETECTOR_MODELS:
        detector_weights = detector_weights or _find_local_weight(
            model_name, kind="detector"
        )
        backbone_weights = backbone_weights or _find_local_weight(
            model_name, kind="backbone"
        )
        if detector_weights is not None:
            _validate_weight_path(model_name, detector_weights, kind="detector")
            model_kwargs["weights"] = str(detector_weights)
        if backbone_weights is not None:
            _validate_weight_path(model_name, backbone_weights, kind="backbone")
            model_kwargs["backbone_weights"] = str(backbone_weights)
        model = DETECTOR_MODELS[model_name](**model_kwargs).to(device)
        return model, "detector"

    if model_name in DEPTHER_MODELS:
        detector_weights = detector_weights or _find_local_weight(
            model_name, kind="head"
        )
        backbone_weights = backbone_weights or _find_local_weight(
            model_name, kind="backbone"
        )
        if detector_weights is not None:
            _validate_weight_path(model_name, detector_weights, kind="head")
            model_kwargs["weights"] = str(detector_weights)
        if backbone_weights is not None:
            _validate_weight_path(model_name, backbone_weights, kind="backbone")
            model_kwargs["backbone_weights"] = str(backbone_weights)
        model = DEPTHER_MODELS[model_name](**model_kwargs).to(device)
        return model, "depther"

    if detector_weights is not None:
        raise ValueError(
            f"Model {model_name!r} is a backbone-only model and does not accept --detector-weights."
        )
    backbone_weights = backbone_weights or _find_local_weight(
        model_name, kind="backbone"
    )
    if backbone_weights is not None:
        _validate_weight_path(model_name, backbone_weights, kind="backbone")
        model_kwargs["weights"] = str(backbone_weights)
    model = BACKBONE_MODELS[model_name](**model_kwargs).to(device)
    return model, "backbone"


def load_image(
    image_path: Path, resize_long_side: int | None
) -> tuple[Image.Image, torch.Tensor]:
    image = Image.open(image_path).convert("RGB")
    tensor = to_tensor(image)
    if resize_long_side is not None:
        height, width = tensor.shape[-2:]
        long_side = max(height, width)
        scale = resize_long_side / long_side
        new_size = (max(1, round(height * scale)), max(1, round(width * scale)))
        tensor = resize(
            tensor, new_size, interpolation=InterpolationMode.BICUBIC, antialias=True
        )
        image = Image.fromarray(
            (tensor.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
        )

    normalized = normalize(tensor, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    return image, normalized


def unwrap_backbone(module: torch.nn.Module) -> torch.nn.Module:
    current = module
    seen: set[int] = set()
    while not hasattr(current, "get_intermediate_layers"):
        identifier = id(current)
        if identifier in seen:
            raise RuntimeError(
                "Failed to unwrap the detector backbone to a feature extractor."
            )
        seen.add(identifier)

        if hasattr(current, "_backbone"):
            current = current._backbone
            continue
        if hasattr(current, "backbone"):
            current = current.backbone
            continue
        if isinstance(current, torch.nn.Sequential) and len(current) > 0:
            current = current[0]
            continue
        raise RuntimeError(f"Unsupported backbone wrapper type: {type(current)!r}")
    return current


def label_name(label_id: int) -> str:
    if 0 <= label_id < len(COCO_CLASSES):
        return COCO_CLASSES[label_id]
    return f"class_{label_id}"


def filter_detections(
    prediction: dict[str, torch.Tensor], score_thresh: float
) -> list[dict[str, object]]:
    scores = prediction["scores"].detach().cpu()
    labels = prediction["labels"].detach().cpu()
    boxes = prediction["boxes"].detach().cpu()

    detections: list[dict[str, object]] = []
    for score, label, box in zip(scores, labels, boxes):
        score_value = float(score.item())
        label_id = int(label.item())
        name = label_name(label_id)
        if score_value < score_thresh or name == "N/A":
            continue
        detections.append(
            {
                "label_id": label_id,
                "label": name,
                "score": score_value,
                "box_xyxy": [round(float(v), 2) for v in box.tolist()],
            }
        )
    return detections


def draw_detections(
    image: Image.Image, detections: list[dict[str, object]]
) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    for det in detections:
        x1, y1, x2, y2 = det["box_xyxy"]
        caption = f"{det['label']} {det['score']:.3f}"
        draw.rectangle((x1, y1, x2, y2), outline=(255, 64, 64), width=3)
        draw.text((x1 + 4, max(0, y1 - 14)), caption, fill=(255, 64, 64))
    return annotated


def feature_map_to_heatmap(feature_map: torch.Tensor) -> Image.Image:
    if feature_map.ndim == 3:
        activation = feature_map.norm(dim=0)
    elif feature_map.ndim == 2:
        activation = feature_map
    else:
        raise ValueError(f"Unsupported feature map shape: {tuple(feature_map.shape)}")
    activation = activation - activation.min()
    activation = activation / activation.max().clamp(min=1e-6)
    grayscale = (activation.numpy() * 255.0).astype(np.uint8)
    return colorize(
        Image.fromarray(grayscale, mode="L"), black="#0a1f44", white="#ffd166"
    )


def save_outputs(
    original_image: Image.Image,
    detections: list[dict[str, object]],
    feature_map: torch.Tensor,
    output_dir: Path,
    image_stem: str,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    detections_path = output_dir / f"{image_stem}_detections.json"
    annotated_path = output_dir / f"{image_stem}_detections.png"
    feature_tensor_path = output_dir / f"{image_stem}_feature_map.pt"
    feature_heatmap_path = output_dir / f"{image_stem}_feature_heatmap.png"

    detections_path.write_text(
        json.dumps(detections, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    draw_detections(original_image, detections).save(annotated_path)
    torch.save(feature_map, feature_tensor_path)

    heatmap = feature_map_to_heatmap(feature_map)
    heatmap = heatmap.resize(original_image.size, resample=Image.Resampling.BICUBIC)
    heatmap.save(feature_heatmap_path)

    return {
        "detections_json": detections_path,
        "annotated_image": annotated_path,
        "feature_tensor": feature_tensor_path,
        "feature_heatmap": feature_heatmap_path,
    }


def main() -> None:
    args = parse_args()
    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    device = torch.device(args.device)
    image, normalized_tensor = load_image(args.image, args.resize_long_side)
    model, model_kind = build_model(
        args.model_name,
        args.detector_weights,
        args.backbone_weights,
        device,
    )
    model.eval()

    batched_tensor = normalized_tensor.unsqueeze(0).to(device)

    with torch.inference_mode():
        if model_kind == "detector":
            detector_backbone = unwrap_backbone(model.detector.backbone[0])
            prediction = model([batched_tensor[0]])[0]
            feature_map = detector_backbone.get_intermediate_layers(
                batched_tensor, n=1, reshape=True
            )[0][0].cpu()
            detections = filter_detections(prediction, args.score_thresh)
        elif model_kind == "backbone":
            feature_map = model.get_intermediate_layers(
                batched_tensor, n=1, reshape=True
            )[0][0].cpu()
            detections = []
        else:
            feature_map = model(batched_tensor)[0, 0].cpu()
            detections = []

    image_output_dir = args.output_dir / args.image.stem
    saved_paths = save_outputs(
        image, detections, feature_map, image_output_dir, args.image.stem
    )

    print(f"Device: {device}")
    print(f"Input image: {args.image}")
    print(f"Detected objects: {len(detections)}")
    for index, det in enumerate(detections, start=1):
        print(
            f"[{index}] label={det['label']} score={det['score']:.4f} box_xyxy={det['box_xyxy']}"
        )

    print("Saved files:")
    for name, path in saved_paths.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
