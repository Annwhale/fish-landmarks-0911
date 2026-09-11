#!/usr/bin/env python3
# ><(((o>  鱼类形态数据工具
"""Read one benchmark record and its 11 canonical-coordinate rows."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("package", type=Path, help="Path to the released benchmark_v1 directory")
    parser.add_argument("--benchmark-id", default=None)
    args = parser.parse_args()

    samples = pd.read_parquet(args.package / "tables/benchmark_samples.parquet")
    if args.benchmark_id is None:
        sample = samples.sort_values("benchmark_id").iloc[0]
    else:
        matches = samples[samples["benchmark_id"] == args.benchmark_id]
        if matches.empty:
            raise KeyError(f"benchmark_id not found: {args.benchmark_id}")
        sample = matches.iloc[0]

    benchmark_id = str(sample["benchmark_id"])
    keypoints = pd.read_parquet(args.package / "tables/benchmark_keypoints.parquet")
    keypoints = keypoints[keypoints["benchmark_id"] == benchmark_id].sort_values("canonical_id")
    image_path = args.package / "images/original" / f"{benchmark_id}.jpg"
    with Image.open(image_path) as image:
        print(f"benchmark_id={benchmark_id}")
        print(f"image={image_path.name}, size={image.size}, mode={image.mode}")
    print(f"semantic_status={sample['canonical_semantics_status']}")
    print(keypoints[["canonical_id", "x", "y", "canonical_name_provisional"]].to_string(index=False))


if __name__ == "__main__":
    main()
