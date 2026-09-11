# ><(((o>  鱼类形态数据工具
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision.models import ResNet34_Weights, resnet34


IMAGE_SIZE = 448
HEATMAP_SIZE = 112
KEYPOINTS = 11


class FishLandmarkDataset(Dataset):
    """Locked benchmark reader with geometry-preserving image augmentation."""

    def __init__(
        self,
        package: str | Path,
        split: str,
        augment: bool = False,
        seed: int = 20260825,
    ) -> None:
        self.package = Path(package)
        samples = pd.read_parquet(self.package / "tables" / "benchmark_samples.parquet")
        self.samples = (
            samples[samples["standard_split"] == split]
            .sort_values("benchmark_id")
            .reset_index(drop=True)
        )
        points = pd.read_parquet(self.package / "annotations" / "keypoints_crops_448.parquet")
        self.points = {
            str(key): group.sort_values("canonical_id")[["x_crop_px", "y_crop_px"]].to_numpy(np.float32)
            for key, group in points.groupby("benchmark_id", sort=False)
        }
        self.split = split
        self.augment = augment
        self.seed = seed

    def __len__(self) -> int:
        return len(self.samples)

    def _augment(self, image: np.ndarray, points: np.ndarray, index: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.seed + index + np.random.randint(0, 1_000_000))
        angle = float(rng.uniform(-8.0, 8.0))
        scale = float(rng.uniform(0.85, 1.15))
        tx = float(rng.uniform(-0.04, 0.04) * IMAGE_SIZE)
        ty = float(rng.uniform(-0.04, 0.04) * IMAGE_SIZE)
        matrix = cv2.getRotationMatrix2D((IMAGE_SIZE / 2, IMAGE_SIZE / 2), angle, scale)
        matrix[:, 2] += (tx, ty)
        image = cv2.warpAffine(
            image,
            matrix,
            (IMAGE_SIZE, IMAGE_SIZE),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        points = np.concatenate([points, np.ones((KEYPOINTS, 1), np.float32)], axis=1) @ matrix.T
        if rng.random() < 0.5:
            image = image[:, ::-1].copy()
            points[:, 0] = (IMAGE_SIZE - 1) - points[:, 0]
        alpha = float(rng.uniform(0.80, 1.20))
        beta = float(rng.uniform(-18.0, 18.0))
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        if rng.random() < 0.20:
            width = int(rng.integers(24, 72))
            height = int(rng.integers(12, 44))
            x0 = int(rng.integers(0, IMAGE_SIZE - width))
            y0 = int(rng.integers(0, max(1, IMAGE_SIZE // 3 - height)))
            fill = np.median(image, axis=(0, 1)).astype(np.uint8)
            image[y0 : y0 + height, x0 : x0 + width] = fill
        return image, points

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.samples.iloc[index]
        benchmark_id = str(row["benchmark_id"])
        path = self.package / "yolo" / "images" / self.split / f"{benchmark_id}.jpg"
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        points = self.points[benchmark_id].copy()
        if self.augment:
            image, points = self._augment(image, points, index)
        points = np.clip(points, 0, IMAGE_SIZE - 1)
        tensor = torch.from_numpy(image.transpose(2, 0, 1)).float().div_(255.0)
        tensor = (tensor - torch.tensor([0.485, 0.456, 0.406])[:, None, None]) / torch.tensor(
            [0.229, 0.224, 0.225]
        )[:, None, None]
        return {
            "image": tensor,
            "points": torch.from_numpy(points / float(IMAGE_SIZE - 1)),
            "benchmark_id": benchmark_id,
            "taxon": str(row["taxon_candidate"]),
            "origin": str(row["origin_label"]),
        }


class UpsamplingHeatmapHead(nn.Module):
    def __init__(
        self,
        channels: int,
        keypoints: int = KEYPOINTS,
        *,
        hidden: int = 256,
        projection_kernel: int = 3,
    ) -> None:
        super().__init__()
        if projection_kernel not in {1, 3}:
            raise ValueError("projection_kernel must be 1 or 3")
        self.net = nn.Sequential(
            nn.Conv2d(
                channels,
                hidden,
                projection_kernel,
                padding=projection_kernel // 2,
                bias=False,
            ),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.ConvTranspose2d(hidden, hidden, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, keypoints, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        logits = self.net(features)
        return F.interpolate(logits, size=(HEATMAP_SIZE, HEATMAP_SIZE), mode="bilinear", align_corners=False)


class ResNetHeatmapPose(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        net = resnet34(weights=weights)
        self.backbone = nn.Sequential(
            net.conv1, net.bn1, net.relu, net.maxpool, net.layer1, net.layer2, net.layer3
        )
        self.head = UpsamplingHeatmapHead(256)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(image))


class DINOv2HeatmapPose(nn.Module):
    def __init__(self, repo: str | Path, weights: str | Path, freeze_backbone: bool = True) -> None:
        super().__init__()
        repo = Path(repo).expanduser()
        weights = Path(weights).expanduser()
        self.backbone = torch.hub.load(str(repo), "dinov2_vitl14_reg", source="local", pretrained=False)
        state = torch.load(str(weights), map_location="cpu", weights_only=True)
        self.backbone.load_state_dict(state, strict=True)
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
            self.backbone.eval()
        # 1×1瓶颈减少稠密特征投影开销，保留可训练空间头。
        self.head = UpsamplingHeatmapHead(1024, hidden=128, projection_kernel=1)

    def train(self, mode: bool = True) -> "DINOv2HeatmapPose":
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        context = torch.no_grad() if self.freeze_backbone else torch.enable_grad()
        with context:
            features = self.backbone.forward_features(image)["x_norm_patchtokens"]
        side = int(math.sqrt(features.shape[1]))
        features = features.transpose(1, 2).reshape(image.shape[0], features.shape[2], side, side)
        return self.head(features)


def spatial_softmax_coordinates(logits: torch.Tensor) -> torch.Tensor:
    batch, keypoints, height, width = logits.shape
    probability = logits.reshape(batch, keypoints, -1).softmax(dim=-1).reshape_as(logits)
    x_axis = torch.linspace(0.0, 1.0, width, device=logits.device, dtype=logits.dtype)
    y_axis = torch.linspace(0.0, 1.0, height, device=logits.device, dtype=logits.dtype)
    x = (probability.sum(dim=2) * x_axis).sum(dim=2)
    y = (probability.sum(dim=3) * y_axis).sum(dim=2)
    return torch.stack([x, y], dim=-1)


def gaussian_targets(points: torch.Tensor, size: int = HEATMAP_SIZE, sigma: float = 2.0) -> torch.Tensor:
    axis = torch.arange(size, device=points.device, dtype=points.dtype)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    x = points[..., 0, None, None] * (size - 1)
    y = points[..., 1, None, None] * (size - 1)
    heatmaps = torch.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * sigma**2))
    return heatmaps / heatmaps.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)


def normalized_pairwise_distances(
    points: torch.Tensor,
    scale: torch.Tensor | None = None,
) -> torch.Tensor:
    distances = torch.cdist(points, points)
    if scale is None:
        scale = torch.linalg.vector_norm(points[:, 0] - points[:, 5], dim=-1)
    return distances / scale[:, None, None].clamp_min(1e-4)


@dataclass
class LossBreakdown:
    total: torch.Tensor
    heatmap: torch.Tensor
    coordinate: torch.Tensor
    geometry: torch.Tensor


def landmark_loss(
    logits: torch.Tensor,
    target_points: torch.Tensor,
    geometry_weight: float = 0.0,
) -> LossBreakdown:
    if logits.shape[-2] != logits.shape[-1]:
        raise ValueError("heatmap logits must be square")
    target_heatmaps = gaussian_targets(target_points, size=logits.shape[-1])
    log_probability = logits.flatten(2).log_softmax(dim=-1).reshape_as(logits)
    heatmap = -(target_heatmaps * log_probability).sum(dim=(-2, -1)).mean()
    prediction = spatial_softmax_coordinates(logits)
    coordinate = F.smooth_l1_loss(prediction, target_points, beta=0.02)
    target_scale = torch.linalg.vector_norm(target_points[:, 0] - target_points[:, 5], dim=-1)
    pred_geometry = normalized_pairwise_distances(prediction, scale=target_scale)
    target_geometry = normalized_pairwise_distances(target_points, scale=target_scale)
    upper = torch.triu(torch.ones(KEYPOINTS, KEYPOINTS, dtype=torch.bool, device=logits.device), diagonal=1)
    geometry = F.smooth_l1_loss(pred_geometry[:, upper], target_geometry[:, upper], beta=0.02)
    total = heatmap + 10.0 * coordinate + float(geometry_weight) * geometry
    return LossBreakdown(total=total, heatmap=heatmap, coordinate=coordinate, geometry=geometry)


def build_model(
    architecture: str,
    *,
    dinov2_repo: str | Path | None = None,
    dinov2_weights: str | Path | None = None,
    pretrained: bool = True,
) -> nn.Module:
    if architecture == "resnet34":
        return ResNetHeatmapPose(pretrained=pretrained)
    if architecture == "dinov2_vitl14_reg":
        if dinov2_repo is None or dinov2_weights is None:
            raise ValueError("DINOv2 repo and weights are required")
        return DINOv2HeatmapPose(dinov2_repo, dinov2_weights, freeze_backbone=True)
    raise ValueError(f"unsupported architecture: {architecture}")
