#!/usr/bin/env python3
# ><(((o>  鱼类形态数据工具
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fishlandmark.heatmap_pose import FishLandmarkDataset, build_model, landmark_loss, spatial_softmax_coordinates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=["resnet34", "dinov2_vitl14_reg"], required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--geometry-weight", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dinov2-repo", default="~/.cache/torch/hub/facebookresearch_dinov2_main")
    parser.add_argument("--dinov2-weights", default="~/.cache/torch/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth")
    return parser.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


@torch.no_grad()
def validate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    nme_values: list[torch.Tensor] = []
    pck_values: list[torch.Tensor] = []
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        target = batch["points"].to(device, non_blocking=True)
        prediction = spatial_softmax_coordinates(model(image))
        body_axis = torch.linalg.vector_norm(target[:, 0] - target[:, 5], dim=-1).clamp_min(1e-4)
        error = torch.linalg.vector_norm(prediction - target, dim=-1) / body_axis[:, None]
        nme_values.append(error.mean(dim=1).cpu())
        pck_values.append((error <= 0.05).float().mean(dim=1).cpu())
    return {
        "nme_body_length": float(torch.cat(nme_values).mean()),
        "pck_0_05": float(torch.cat(pck_values).mean()),
    }


if __name__ == "__main__":
    args = parse_args()
    seed = 20260825
    seed_all(seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    package = ROOT / "data" / "benchmark_v1"
    train_data = FishLandmarkDataset(package, "train", augment=True, seed=seed)
    validation_data = FishLandmarkDataset(package, "validation", augment=False, seed=seed)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_data, batch_size=args.batch, shuffle=True, num_workers=args.workers,
        pin_memory=True, persistent_workers=args.workers > 0, generator=generator,
        worker_init_fn=seed_worker,
    )
    validation_loader = DataLoader(
        validation_data, batch_size=args.batch, shuffle=False, num_workers=args.workers,
        pin_memory=True, persistent_workers=args.workers > 0, worker_init_fn=seed_worker,
    )
    model = build_model(
        args.architecture,
        dinov2_repo=args.dinov2_repo,
        dinov2_weights=args.dinov2_weights,
    ).to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    output = ROOT / "models" / "heatmap" / args.name
    output.mkdir(parents=True, exist_ok=True)
    configuration = {
        **vars(args),
        "seed": seed,
        "train_images": len(train_data),
        "validation_images": len(validation_data),
        "image_size": 448,
        "heatmap_size": 112,
        "semantic_status": "canonical landmark names and dorsal-up convention remain provisional",
    }
    (output / "locked_training_config.json").write_text(json.dumps(configuration, indent=2), encoding="utf-8")
    history: list[dict[str, float | int]] = []
    best_nme = float("inf")
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        sums = {"total": 0.0, "heatmap": 0.0, "coordinate": 0.0, "geometry": 0.0}
        count = 0
        for batch in train_loader:
            image = batch["image"].to(device, non_blocking=True)
            target = batch["points"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                losses = landmark_loss(model(image), target, geometry_weight=args.geometry_weight)
            scaler.scale(losses.total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, 5.0)
            scaler.step(optimizer)
            scaler.update()
            batch_size = image.shape[0]
            count += batch_size
            for name in sums:
                sums[name] += float(getattr(losses, name).detach()) * batch_size
        scheduler.step()
        metrics = validate(model, validation_loader, device)
        row = {
            "epoch": epoch,
            **{f"train_{name}": value / count for name, value in sums.items()},
            **{f"val_{name}": value for name, value in metrics.items()},
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(json.dumps(row), flush=True)
        checkpoint = {"model": model.state_dict(), "configuration": configuration, "epoch": epoch, "metrics": metrics}
        torch.save(checkpoint, output / "last.pt")
        if metrics["nme_body_length"] < best_nme:
            best_nme = metrics["nme_body_length"]
            stale = 0
            torch.save(checkpoint, output / "best.pt")
        else:
            stale += 1
        if stale >= args.patience:
            break
