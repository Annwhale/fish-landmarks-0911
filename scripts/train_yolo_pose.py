#!/usr/bin/env python3
# ><(((o>  鱼类形态数据工具
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=448)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--device", default="0")
    parser.add_argument("--data", default=str(ROOT / "config" / "yolo_training.yaml"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    project = ROOT / "models" / "yolo"
    run_root = project / args.name
    run_root.mkdir(parents=True, exist_ok=True)
    configuration = {
        **vars(args),
        "data": str(Path(args.data).resolve()),
        "seed": 20260825,
        "deterministic": True,
        "semantic_status": "canonical point names and dorsal-up convention provisional pending source confirmation",
    }
    (run_root / "locked_training_config.json").write_text(
        json.dumps(configuration, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    model = YOLO(args.weights)
    model.train(
        data=configuration["data"],
        project=str(project),
        name=args.name,
        exist_ok=True,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        patience=args.patience,
        workers=args.workers,
        fraction=args.fraction,
        device=args.device,
        seed=configuration["seed"],
        deterministic=True,
        amp=True,
        cache="ram",
        degrees=8.0,
        translate=0.04,
        scale=0.20,
        shear=2.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        erasing=0.15,
        hsv_h=0.01,
        hsv_s=0.25,
        hsv_v=0.20,
        plots=True,
        save=True,
        val=True,
        verbose=True,
    )
