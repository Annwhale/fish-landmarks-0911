#!/usr/bin/env python3
# ><(((o>  鱼类形态数据工具
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "data" / "benchmark_v1"
YOLO_ROOT = PACKAGE / "yolo"
OUTPUT = ROOT / "data" / "experiments" / "protocols"
ORIGINAL_ROOT = ROOT / "data" / "experiments" / "yolo_original_v1"


def image_paths(frame: pd.DataFrame, image_root: Path = YOLO_ROOT) -> list[str]:
    return [
        str((image_root / "images" / row.standard_split / f"{row.benchmark_id}.jpg").resolve())
        for row in frame.sort_values("benchmark_id").itertuples(index=False)
    ]


def write_protocol(
    name: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    *,
    image_root: Path = YOLO_ROOT,
    representation: str = "annotation_assisted_axial",
) -> dict[str, object]:
    root = OUTPUT / name
    root.mkdir(parents=True, exist_ok=True)
    for split, frame in (("train", train), ("validation", validation), ("test", test)):
        paths = image_paths(frame, image_root=image_root)
        missing = [path for path in paths if not Path(path).exists()]
        if missing:
            raise FileNotFoundError(f"{name}/{split} is missing {len(missing)} images; first={missing[0]}")
        (root / f"{split}.txt").write_text("\n".join(paths) + "\n", encoding="utf-8")
    yaml_path = root / "data.yaml"
    yaml_path.write_text(
        "\n".join([
            f"path: {image_root.resolve()}",
            f"train: {(root / 'train.txt').resolve()}",
            f"val: {(root / 'validation.txt').resolve()}",
            f"test: {(root / 'test.txt').resolve()}",
            "names:", "  0: fish", "kpt_shape: [11, 3]",
            "flip_idx: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]", "",
        ]),
        encoding="utf-8",
    )
    summary = {
        "name": name,
        "train": len(train),
        "validation": len(validation),
        "test": len(test),
        "train_taxa": sorted(train["taxon_candidate"].unique().tolist()),
        "test_taxa": sorted(test["taxon_candidate"].unique().tolist()),
        "train_origins": sorted(train["origin_label"].unique().tolist()),
        "test_origins": sorted(test["origin_label"].unique().tolist()),
        "representation": representation,
        "target_annotation_assisted": representation == "annotation_assisted_axial",
        "semantic_status": "taxon and origin labels remain subject to source-team confirmation",
    }
    (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    samples = pd.read_parquet(PACKAGE / "tables" / "benchmark_samples.parquet")
    eligible = samples[samples["technical_validation_eligible"]].copy()
    paired = eligible[eligible["cross_origin_eligible"]]
    cross_train = paired[(paired["origin_label"] == "farmed") & (paired["standard_split"] == "train")]
    cross_validation = paired[(paired["origin_label"] == "farmed") & (paired["standard_split"] == "validation")]
    cross_test = paired[paired["origin_label"] == "wild"]
    cross_origin = write_protocol(
        "cross_origin_farmed_to_wild",
        cross_train,
        cross_validation,
        cross_test,
    )
    cross_origin_original = write_protocol(
        "cross_origin_farmed_to_wild_original",
        cross_train,
        cross_validation,
        cross_test,
        image_root=ORIGINAL_ROOT,
        representation="original_image",
    )
    membership = samples[samples["cross_origin_eligible"]].copy()
    membership["protocol_role"] = "excluded"
    membership["exclusion_reason"] = ""
    membership.loc[
        membership["benchmark_id"].isin(cross_train["benchmark_id"]), "protocol_role"
    ] = "train"
    membership.loc[
        membership["benchmark_id"].isin(cross_validation["benchmark_id"]), "protocol_role"
    ] = "validation"
    membership.loc[
        membership["benchmark_id"].isin(cross_test["benchmark_id"]), "protocol_role"
    ] = "test"
    membership.loc[
        ~membership["technical_validation_eligible"], "exclusion_reason"
    ] = "review_holdout_not_technical_validation_eligible"
    membership.loc[
        (membership["protocol_role"] == "excluded")
        & membership["technical_validation_eligible"]
        & (membership["origin_label"] == "farmed")
        & (membership["standard_split"] == "test"),
        "exclusion_reason",
    ] = "candidate_farmed_standard_test_reserved_from_training_and_validation"
    membership_columns = [
        "benchmark_id", "taxon_candidate", "origin_label", "standard_split",
        "technical_validation_eligible", "protocol_role", "exclusion_reason",
    ]
    membership[membership_columns].sort_values("benchmark_id").to_csv(
        OUTPUT / "cross_origin_membership_manifest.csv", index=False
    )
    folds = {}
    for taxon in sorted(eligible["taxon_candidate"].unique()):
        slug = taxon.lower().replace(" ", "_").replace(".", "")
        fold = write_protocol(
            f"leave_one_taxon_{slug}",
            eligible[(eligible["taxon_candidate"] != taxon) & (eligible["standard_split"] == "train")],
            eligible[(eligible["taxon_candidate"] != taxon) & (eligible["standard_split"] == "validation")],
            eligible[eligible["taxon_candidate"] == taxon],
        )
        folds[taxon] = fold
    payload = {
        "cross_origin_annotation_assisted_axial": cross_origin,
        "cross_origin_original_image": cross_origin_original,
        "leave_one_taxon": folds,
    }
    (OUTPUT / "protocol_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
