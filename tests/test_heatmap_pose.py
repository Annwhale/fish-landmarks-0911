# ><(((o>  鱼类形态数据工具
import torch

from fishlandmark.heatmap_pose import (
    UpsamplingHeatmapHead,
    gaussian_targets,
    landmark_loss,
    normalized_pairwise_distances,
    spatial_softmax_coordinates,
)


def test_lightweight_dino_head_shape() -> None:
    head = UpsamplingHeatmapHead(1024, hidden=128, projection_kernel=1)
    output = head(torch.randn(2, 1024, 16, 16))
    assert output.shape == (2, 11, 112, 112)


def test_spatial_softmax_coordinates_returns_center_for_uniform_logits() -> None:
    logits = torch.zeros(2, 11, 9, 9)
    coordinates = spatial_softmax_coordinates(logits)
    assert coordinates.shape == (2, 11, 2)
    assert torch.allclose(coordinates, torch.full_like(coordinates, 0.5), atol=1e-6)


def test_gaussian_targets_are_normalized() -> None:
    points = torch.rand(3, 11, 2)
    targets = gaussian_targets(points, size=16, sigma=1.5)
    assert targets.shape == (3, 11, 16, 16)
    assert torch.allclose(targets.sum(dim=(-2, -1)), torch.ones(3, 11), atol=1e-5)


def test_geometry_is_scale_invariant() -> None:
    points = torch.rand(2, 11, 2)
    points[:, 5] = points[:, 0] + torch.tensor([0.5, 0.0])
    assert torch.allclose(normalized_pairwise_distances(points), normalized_pairwise_distances(points * 3), atol=1e-5)


def test_landmark_loss_is_finite() -> None:
    logits = torch.randn(2, 11, 16, 16, requires_grad=True)
    points = torch.rand(2, 11, 2)
    points[:, 5] = points[:, 0] + torch.tensor([0.4, 0.0])
    loss = landmark_loss(logits, points, geometry_weight=0.5)
    assert torch.isfinite(loss.total)
    loss.total.backward()
