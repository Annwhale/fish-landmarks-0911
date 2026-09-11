# ><(((o>  鱼类形态数据工具
import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = spec_from_file_location(
    "near_duplicate_sensitivity", ROOT / "scripts" / "analyze_near_duplicate_sensitivity.py"
)
MODULE = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_cross_partition_rows_quarantines_only_test_member():
    pairs = pd.DataFrame(
        [
            {"benchmark_id_left": "a", "benchmark_id_right": "b", "distance": 1},
            {"benchmark_id_left": "c", "benchmark_id_right": "d", "distance": 2},
        ]
    )
    membership = {"a": "train", "b": "test", "c": "test", "d": "test"}
    result = MODULE.cross_partition_rows(pairs, membership, "demo")
    assert result["quarantined_test_id"].tolist() == ["b"]
    assert result["protocol"].tolist() == ["demo"]


def test_cross_origin_representations_are_explicit_and_separate():
    summary = json.loads(
        (ROOT / "data/experiments/protocols/protocol_summary.json").read_text(encoding="utf-8")
    )
    assisted = summary["cross_origin_annotation_assisted_axial"]
    original = summary["cross_origin_original_image"]
    assert assisted["representation"] == "annotation_assisted_axial"
    assert assisted["target_annotation_assisted"] is True
    assert original["representation"] == "original_image"
    assert original["target_annotation_assisted"] is False
    assert {key: assisted[key] for key in ("train", "validation", "test")} == {
        "train": 285,
        "validation": 60,
        "test": 1477,
    }
    assert {key: original[key] for key in ("train", "validation", "test")} == {
        "train": 285,
        "validation": 60,
        "test": 1477,
    }


def test_cross_origin_membership_accounts_for_all_eligible_rows():
    frame = pd.read_csv(ROOT / "data/experiments/protocols/cross_origin_membership_manifest.csv")
    assert len(frame) == 1888
    assert frame["protocol_role"].value_counts().to_dict() == {
        "test": 1477,
        "train": 285,
        "excluded": 66,
        "validation": 60,
    }
