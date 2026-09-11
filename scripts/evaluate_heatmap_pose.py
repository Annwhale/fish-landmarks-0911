#!/usr/bin/env python3
# ><(((o>  鱼类形态数据工具
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fishlandmark.heatmap_pose import FishLandmarkDataset, build_model, spatial_softmax_coordinates
from fishlandmark.metrics import evaluate_prediction, summarize_predictions


def record_path(value: str | Path) -> str:
    path = Path(value).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    configuration = checkpoint["configuration"]
    model = build_model(
        configuration["architecture"],
        dinov2_repo=configuration.get("dinov2_repo"),
        dinov2_weights=configuration.get("dinov2_weights"),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    data = FishLandmarkDataset(ROOT / "data" / "benchmark_v1", args.split, augment=False)
    loader = DataLoader(data, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)
    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                prediction = spatial_softmax_coordinates(model(image))
            prediction = prediction.float().cpu().numpy() * 447.0
            target = batch["points"].numpy() * 447.0
            for index in range(len(prediction)):
                rows.append(
                    evaluate_prediction(
                        str(batch["benchmark_id"][index]),
                        target[index].astype(np.float64),
                        prediction[index].astype(np.float64),
                        taxon=str(batch["taxon"][index]),
                        origin=str(batch["origin"][index]),
                        confidence=1.0,
                    )
                )
    output = ROOT / "models" / "evaluation" / args.name / args.split
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "per_image_metrics.csv", index=False)
    frame.to_parquet(output / "per_image_metrics.parquet", index=False)
    summary = summarize_predictions(frame)
    public_configuration = dict(configuration)
    for path_key in ("dinov2_repo", "dinov2_weights"):
        if public_configuration.get(path_key):
            public_configuration[path_key] = record_path(public_configuration[path_key])
    summary.update({
        "checkpoint": record_path(args.checkpoint),
        "split": args.split,
        "evaluation_seed": 20260825,
        "training_configuration": public_configuration,
    })
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["overall"], ensure_ascii=False, indent=2))
