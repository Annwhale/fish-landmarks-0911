# ><(((o>  鱼类形态数据工具
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"


def test_benchmark_has_one_row_per_content_id_and_complete_points():
    samples = pd.read_parquet(DERIVED / "benchmark_samples.parquet")
    points = pd.read_parquet(DERIVED / "benchmark_keypoints.parquet")
    assert samples["content_id_blake2b64"].is_unique
    assert points.groupby("benchmark_id").size().eq(11).all()
    assert set(points["benchmark_id"]) == set(samples["benchmark_id"])


def test_taxon_conflict_images_are_excluded():
    samples = pd.read_parquet(DERIVED / "benchmark_samples.parquet")
    exclusions = pd.read_parquet(DERIVED / "benchmark_exclusions.parquet")
    excluded_ids = set(exclusions["content_id_blake2b64"])
    assert excluded_ids
    assert excluded_ids.isdisjoint(set(samples["content_id_blake2b64"]))
    assert set(exclusions["exclusion_reason"]) == {"exact_image_conflicting_taxon_labels"}


def test_standard_split_is_disjoint_and_exhaustive():
    samples = pd.read_parquet(DERIVED / "benchmark_samples.parquet")
    split_sets = {
        split: set(samples.loc[samples["standard_split"] == split, "benchmark_id"])
        for split in ("train", "validation", "test", "review_holdout")
    }
    for left, right in ((left, right) for left in split_sets for right in split_sets if left < right):
        assert split_sets[left].isdisjoint(split_sets[right])
    assert set.union(*split_sets.values()) == set(samples["benchmark_id"])
    assert split_sets["review_holdout"] == set(
        samples.loc[samples["repeated_annotation_high_disagreement"], "benchmark_id"]
    )


def test_origin_conflicts_are_not_used_for_origin_transfer():
    samples = pd.read_parquet(DERIVED / "benchmark_samples.parquet")
    assert not samples.loc[samples["origin_conflict"], "cross_origin_eligible"].any()
