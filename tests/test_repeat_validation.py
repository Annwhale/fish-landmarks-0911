# ><(((o>  鱼类形态数据工具
from __future__ import annotations

import numpy as np

from scripts.analyze_repeat_annotations import icc_a1, procrustes_rmse


def test_icc_a1_is_one_for_identical_repeats() -> None:
    values = np.array([[1.0, 1.0], [2.0, 2.0], [4.0, 4.0], [8.0, 8.0]])
    assert np.isclose(icc_a1(values), 1.0)


def test_procrustes_removes_translation_rotation_and_scale() -> None:
    points = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
    transformed = (points @ rotation) * 3.0 + np.array([10.0, -7.0])
    assert procrustes_rmse(points, transformed) < 1e-10
