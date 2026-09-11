# ><(((o>  鱼类形态数据工具
from pathlib import Path

import pandas as pd

from scripts.evaluate_locateanything_zero_shot import cluster_bootstrap_mean_ci, parse_point
from scripts.prepare_locateanything_protocol import LANDMARK_PROMPTS


def test_prompt_protocol_has_one_description_per_role() -> None:
    assert sorted(LANDMARK_PROMPTS) == list(range(1, 12))
    assert all(description.strip() for description in LANDMARK_PROMPTS.values())


def test_point_parser_scales_normalized_coordinates() -> None:
    point = parse_point("answer <box><250><750></box>", width=800, height=600)
    assert point == (200.0, 450.0)


def test_point_parser_rejects_missing_or_out_of_range_coordinates() -> None:
    assert parse_point("none", width=800, height=600) is None
    assert parse_point("<box><1001><500></box>", width=800, height=600) is None


def test_model_weights_are_outside_project_tree() -> None:
    project = Path(__file__).resolve().parents[1]
    assert not (project / "model_cache").exists()


def test_cluster_bootstrap_reports_query_and_image_counts() -> None:
    frame = pd.DataFrame({
        "benchmark_id": ["a", "a", "b", "b"],
        "metric": [0.0, 1.0, 0.0, 1.0],
    })
    result = cluster_bootstrap_mean_ci(frame, "metric", replicates=20)
    assert result["n"] == 4
    assert result["clusters"] == 2
    assert result["mean"] == 0.5
