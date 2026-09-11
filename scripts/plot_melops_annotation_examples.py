#!/usr/bin/env python3
# ><(((o>  鱼类形态数据工具
"""Render a small QA plate from the released Melops CVAT annotations."""

from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
from PIL import Image


BODY_LABELS = {
    "snout",
    "bottom_opercular",
    "eyes_snout",
    "eyes_opercular",
    "pectoral_bottom",
    "pectoral_top",
    "papilla",
    "caudal_bottom",
    "caudal_top",
    "black_point",
    "eyes_pectoral",
    "eyes_top",
    "pelvic",
    "top_opercular",
    "medium_opercular",
    "anal_fin",
}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--xml",
        type=Path,
        default=root / "external/melops_v2/stage2/extracted/annotations.xml",
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=root / "external/melops_v2/stage2/manual_images",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "qa/melops_stage2_annotation_examples.png",
    )
    args = parser.parse_args()

    records = []
    for image_node in ET.parse(args.xml).getroot().findall("image"):
        points = []
        for point in image_node.findall("points"):
            label = point.get("label", "")
            if label not in BODY_LABELS:
                continue
            first_vertex = point.get("points", "").split(";")[0]
            x, y = (float(value) for value in first_vertex.split(","))
            points.append((label, x, y))
        if points:
            records.append((image_node.get("name", ""), points))

    # 等距选取示例，避免只展示一个采集批次。
    indices = [0, 84, 168, 252, 336, 420]
    selected = [records[index] for index in indices]
    fig, axes = plt.subplots(3, 2, figsize=(14, 13), constrained_layout=True)
    for axis, (name, points) in zip(axes.flat, selected):
        image = Image.open(args.images / name).convert("RGB")
        axis.imshow(image)
        for label, x, y in points:
            axis.scatter(x, y, s=18, color="#00A6D6", edgecolor="white", linewidth=0.5)
            axis.text(
                x + 8,
                y - 8,
                label,
                fontsize=6,
                color="black",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "pad": 0.4},
            )
        axis.set_title(name, fontsize=9)
        axis.set_axis_off()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
