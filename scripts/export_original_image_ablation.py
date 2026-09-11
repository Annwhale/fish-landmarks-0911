#!/usr/bin/env python3
# ><(((o>  鱼类形态数据工具
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def padded_box(points: np.ndarray, width: int, height: int, fraction: float = 0.08) -> tuple[float, float, float, float]:
    x0, y0 = points.min(axis=0)
    x1, y1 = points.max(axis=0)
    padding = fraction * max(x1 - x0, y1 - y0)
    return (
        max(0.0, float(x0 - padding)),
        max(0.0, float(y0 - padding)),
        min(float(width), float(x1 + padding)),
        min(float(height), float(y1 + padding)),
    )


if __name__ == "__main__":
    package = ROOT / "data" / "benchmark_v1"
    output = ROOT / "data" / "experiments" / "yolo_original_v1"
    samples = pd.read_parquet(package / "tables" / "benchmark_samples.parquet")
    samples = samples[samples["standard_split"].isin(["train", "validation", "test"])].copy()
    points = pd.read_parquet(package / "tables" / "benchmark_keypoints.parquet")
    point_lookup = {
        str(key): group.sort_values("canonical_id")[["x", "y"]].to_numpy(np.float64)
        for key, group in points.groupby("benchmark_id", sort=False)
    }
    rows = []
    for sample in samples.itertuples(index=False):
        split = str(sample.standard_split)
        image_dir = output / "images" / split
        label_dir = output / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        source = package / "images" / "original" / f"{sample.benchmark_id}.jpg"
        destination = image_dir / source.name
        if not destination.exists():
            try:
                os.link(source, destination)
            except OSError:
                destination.write_bytes(source.read_bytes())
        with Image.open(source) as image:
            width, height = image.size
        landmark = point_lookup[str(sample.benchmark_id)]
        x0, y0, x1, y1 = padded_box(landmark, width, height)
        values = [
            "0",
            f"{(x0 + x1) / (2 * width):.8f}",
            f"{(y0 + y1) / (2 * height):.8f}",
            f"{(x1 - x0) / width:.8f}",
            f"{(y1 - y0) / height:.8f}",
        ]
        for x, y in landmark:
            values.extend([f"{x / width:.8f}", f"{y / height:.8f}", "2"])
        (label_dir / f"{sample.benchmark_id}.txt").write_text(" ".join(values) + "\n", encoding="utf-8")
        rows.append({"benchmark_id": sample.benchmark_id, "split": split, "width": width, "height": height})
    yaml_path = output / "data.yaml"
    yaml_path.write_text(
        "\n".join([
            f"path: {output}", "train: images/train", "val: images/validation", "test: images/test",
            "names:", "  0: fish", "kpt_shape: [11, 3]",
            "flip_idx: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]", "",
        ]),
        encoding="utf-8",
    )
    pd.DataFrame(rows).to_csv(output / "sample_manifest_no_hash.csv", index=False)
    summary = {
        "images": len(rows),
        "representation": "unaligned original images with landmark-derived padded boxes",
        "storage": "hard links where supported; copies otherwise",
        "integrity_policy": "no additional content digest",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
