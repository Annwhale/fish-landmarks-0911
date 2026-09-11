# ><(((o>  鱼类形态数据工具
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "data" / "benchmark_v1"


def test_export_counts_and_formats_match():
    samples = pd.read_parquet(PACKAGE / "tables" / "benchmark_samples.parquet")
    crop_points = pd.read_parquet(PACKAGE / "annotations" / "keypoints_crops_448.parquet")
    coco = json.loads((PACKAGE / "annotations" / "coco_keypoints.json").read_text(encoding="utf-8"))
    assert len(coco["images"]) == len(samples)
    assert len(coco["annotations"]) == len(samples)
    assert len(crop_points) == 11 * len(samples)
    assert all(annotation["num_keypoints"] == 11 for annotation in coco["annotations"])


def test_yolo_labels_are_bounded_and_complete():
    labels = sorted((PACKAGE / "yolo" / "labels").glob("*/*.txt"))
    samples = pd.read_parquet(PACKAGE / "tables" / "benchmark_samples.parquet")
    assert len(labels) == len(samples)
    for label in labels:
        values = label.read_text(encoding="utf-8").split()
        assert len(values) == 5 + 11 * 3
        numeric = [float(value) for value in values[1:]]
        coordinates = numeric[:4] + [numeric[index] for index in range(4, len(numeric)) if (index - 4) % 3 != 2]
        assert all(0.0 <= value <= 1.0 for value in coordinates)


def test_review_holdout_is_not_in_validation_or_test_paths():
    samples = pd.read_parquet(PACKAGE / "tables" / "benchmark_samples.parquet")
    review_ids = set(samples.loc[samples["standard_split"] == "review_holdout", "benchmark_id"])
    exported_review = {path.stem for path in (PACKAGE / "yolo" / "images" / "review_holdout").glob("*.jpg")}
    assert review_ids == exported_review
