#!/usr/bin/env python3
# ><(((o>  鱼类形态数据工具
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fishlandmark.metrics import evaluate_prediction, summarize_predictions


def record_path(value: str | Path) -> str:
    path = Path(value).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return f"external_local_weight/{path.name}"


def chunked(values: list[Path], size: int):
    """Yield explicit path batches.

    Ultralytics treats a Python list of image paths as one in-memory source
    batch, so its ``batch`` argument alone does not bound memory for this input
    form.  Explicit chunks keep long protocol evaluations memory-safe while
    preserving path order.
    """
    if size < 1:
        raise ValueError("batch must be at least 1")
    for start in range(0, len(values), size):
        yield values[start : start + size]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--representation", choices=["axial", "original"], default="axial")
    parser.add_argument(
        "--protocol-dir",
        default=None,
        help="Optional directory containing train.txt, validation.txt and test.txt from build_evaluation_protocols.py",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    package = ROOT / "data" / "benchmark_v1"
    all_samples = pd.read_parquet(package / "tables" / "benchmark_samples.parquet")
    protocol_paths: list[Path] | None = None
    if args.protocol_dir:
        protocol_file = Path(args.protocol_dir) / f"{args.split}.txt"
        protocol_paths = [Path(line) for line in protocol_file.read_text(encoding="utf-8").splitlines() if line]
        protocol_ids = [path.stem for path in protocol_paths]
        samples = all_samples.set_index("benchmark_id").loc[protocol_ids].reset_index()
    else:
        samples = all_samples[all_samples["standard_split"] == args.split].sort_values("benchmark_id")
    if args.representation == "axial":
        points = pd.read_parquet(package / "annotations" / "keypoints_crops_448.parquet")
        coordinate_columns = ["x_crop_px", "y_crop_px"]
        image_root = package / "yolo" / "images"
    else:
        points = pd.read_parquet(package / "tables" / "benchmark_keypoints.parquet")
        coordinate_columns = ["x", "y"]
        image_root = ROOT / "data" / "experiments" / "yolo_original_v1" / "images"
    if protocol_paths is not None:
        paths = protocol_paths
    else:
        paths = [image_root / args.split / f"{benchmark_id}.jpg" for benchmark_id in samples["benchmark_id"]]
    model = YOLO(args.weights)
    rows = []
    sample_by_id = samples.set_index("benchmark_id")
    for path_batch in chunked(paths, args.batch):
        results = model.predict(
            source=[str(path) for path in path_batch],
            imgsz=448,
            batch=args.batch,
            device=args.device,
            conf=0.001,
            iou=0.7,
            max_det=10,
            save=False,
            stream=True,
            verbose=False,
        )
        for path, result in zip(path_batch, results, strict=True):
            benchmark_id = path.stem
            sample = sample_by_id.loc[benchmark_id]
            target_rows = points[points["benchmark_id"] == benchmark_id].sort_values("canonical_id")
            target = target_rows[coordinate_columns].to_numpy(dtype=np.float64)
            prediction = None
            confidence = None
            if result.boxes is not None and len(result.boxes) and result.keypoints is not None:
                box_confidence = result.boxes.conf.detach().cpu().numpy()
                index = int(np.argmax(box_confidence))
                candidate = result.keypoints.xy[index].detach().cpu().numpy().astype(np.float64)
                if candidate.shape == (11, 2):
                    prediction = candidate
                    confidence = float(box_confidence[index])
            rows.append(
                evaluate_prediction(
                    benchmark_id,
                    target,
                    prediction,
                    taxon=str(sample["taxon_candidate"]),
                    origin=str(sample["origin_label"]),
                    confidence=confidence,
                )
            )
    output_root = ROOT / "models" / "evaluation" / args.name / args.split
    output_root.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_root / "per_image_metrics.csv", index=False)
    frame.to_parquet(output_root / "per_image_metrics.parquet", index=False)
    summary = summarize_predictions(frame)
    summary["model_weights"] = record_path(args.weights)
    summary["split"] = args.split
    summary["evaluation_seed"] = 20260825
    summary["representation"] = args.representation
    summary["target_annotation_assisted_representation"] = args.representation == "axial"
    summary["protocol_dir"] = record_path(args.protocol_dir) if args.protocol_dir else None
    summary["inference_configuration"] = {
        "imgsz": 448,
        "batch": args.batch,
        "confidence_threshold": 0.001,
        "iou_threshold": 0.7,
        "maximum_detections": 10,
        "selected_instance_rule": "highest box confidence",
    }
    summary["metric_definitions"] = {
        "nme_body_length": "legacy implementation field: mean point error divided by role-1-to-role-6 working axial span; not physical body length",
        "pck_all_images": "detection failures contribute zero",
        "contour_perimeter_relative_error": "relative error of ordered working polygon through roles 1-10; not the visible fish contour",
        "bootstrap_unit": "image; descriptive interval rather than biological-population inference",
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary["overall"], ensure_ascii=False, indent=2))
