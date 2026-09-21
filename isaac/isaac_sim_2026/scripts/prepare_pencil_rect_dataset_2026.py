#!/usr/bin/env python3
"""Create a YOLO rectangle dataset from cropped pencil images.

Every source image is already cropped to the object, so each label is a full
image box: class 0, center (0.5, 0.5), width 1.0, height 1.0.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        default="/home/u-zhuang/桌面/airobot",
        help="Directory containing cropped pencil screenshots.",
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "/home/u-zhuang/ros2_ws/src/isaac/"
            "grasp_demo_pkg/data/pencil_rect_yolo"
        ),
        help="Output YOLO dataset directory.",
    )
    parser.add_argument("--class-name", default="pencil")
    return parser.parse_args()


def safe_stem(path: Path, index: int) -> str:
    return f"{index:04d}_{path.stem.replace(' ', '_').replace(':', '-')}"


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    image_dir = output_dir / "images" / "train"
    label_dir = output_dir / "labels" / "train"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(
        p for p in source_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        raise SystemExit(f"no image files found in {source_dir}")

    rows = []
    for index, src in enumerate(paths, start=1):
        try:
            with Image.open(src) as image:
                width, height = image.size
                if width <= 0 or height <= 0:
                    raise ValueError("empty image")
                rgb = image.convert("RGB")
        except Exception as exc:
            print(f"skip {src}: {exc}")
            continue

        stem = safe_stem(src, index)
        dst_image = image_dir / f"{stem}.jpg"
        dst_label = label_dir / f"{stem}.txt"
        rgb.save(dst_image, quality=95)
        dst_label.write_text("0 0.5 0.5 1.0 1.0\n", encoding="utf-8")
        rows.append((src.name, dst_image.name, width, height))

    dataset_yaml = output_dir / "pencil_rect.yaml"
    dataset_yaml.write_text(
        "\n".join(
            [
                f"path: {output_dir}",
                "train: images/train",
                "val: images/train",
                "names:",
                f"  0: {args.class_name}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    manifest = output_dir / "manifest.tsv"
    manifest.write_text(
        "source\timage\twidth\theight\n"
        + "\n".join(f"{a}\t{b}\t{w}\t{h}" for a, b, w, h in rows)
        + "\n",
        encoding="utf-8",
    )

    shutil.copy2(dataset_yaml, output_dir / "data.yaml")
    print(f"wrote {len(rows)} samples to {output_dir}")
    print(f"dataset yaml: {dataset_yaml}")


if __name__ == "__main__":
    main()
