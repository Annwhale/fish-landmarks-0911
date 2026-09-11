# ><(((o>  鱼类形态数据工具
from __future__ import annotations

import numpy as np

from scripts.evaluate_melops_external import mapping_target, parse_first_vertex


def test_parse_first_vertex_accepts_cvat_duplicate_encoding() -> None:
    value = parse_first_vertex("12.5,7.25;12.5,7.25")
    assert np.allclose(value, [12.5, 7.25])


def test_midpoint_mapping_uses_two_good_source_points() -> None:
    points = {
        "top": {"xy": np.array([2.0, 4.0]), "good": True},
        "bottom": {"xy": np.array([6.0, 8.0]), "good": True},
    }
    mapping = {"target_labels": ["top", "bottom"], "target_rule": "midpoint"}
    assert np.allclose(mapping_target(points, mapping), [4.0, 6.0])


def test_mapping_excludes_bad_source_landmark() -> None:
    points = {"point": {"xy": np.array([2.0, 4.0]), "good": False}}
    mapping = {"target_labels": ["point"], "target_rule": "direct"}
    assert mapping_target(points, mapping) is None
