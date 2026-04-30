from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find top-N highest image_anomaly_score across *_patchcore_scores.json files."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Root directory to search.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="ksd_20250520_patchcore_scores.json",
        help="Filename glob pattern.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=100,
        help="How many top records to print.",
    )
    return parser.parse_args()


def load_json_items(json_path: Path) -> list[dict[str, Any]]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return []

    valid_items: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        score = item.get("image_anomaly_score")
        if isinstance(score, (int, float)):
            valid_items.append(item)
    return valid_items


def main() -> None:
    args = parse_args()
    if args.topk <= 0:
        raise ValueError("--topk must be greater than 0")

    root = args.root.resolve()
    score_files = sorted(root.rglob(args.pattern))
    if not score_files:
        print(f"No score files found under {root} with pattern: {args.pattern}")
        return

    records: list[dict[str, Any]] = []
    for score_file in score_files:
        try:
            items = load_json_items(score_file)
        except Exception as exc:
            print(f"Skip invalid JSON: {score_file} ({exc})")
            continue

        for item in items:
            records.append(
                {
                    "score_file": str(score_file),
                    "image": item.get("image", ""),
                    "image_anomaly_score": float(item["image_anomaly_score"]),
                }
            )

    if not records:
        print("No valid image_anomaly_score records found.")
        return

    records.sort(key=lambda x: x["image_anomaly_score"], reverse=True)
    topk = records[: args.topk]

    print(f"Found {len(records)} scored images from {len(score_files)} file(s).")
    print(f"Top {len(topk)} by image_anomaly_score:")
    for idx, rec in enumerate(topk, start=1):
        print(
            f"[{idx}] score={rec['image_anomaly_score']:.6f} | image={rec['image']} | source={rec['score_file']}"
        )


if __name__ == "__main__":
    main()
