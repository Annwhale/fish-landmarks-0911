#!/usr/bin/env python3
# ><(((o>  鱼类形态数据工具
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
STAGE1 = ROOT / "external" / "melops_v2" / "stage1"
MANAGEMENT = STAGE1 / "management" / "Image manipulation and management"


if __name__ == "__main__":
    metadata = pd.read_csv(STAGE1 / "Melops_metadata.txt", sep="\t")
    keypoints = pd.read_csv(MANAGEMENT / "keypoint_head.csv")
    white_card = pd.read_csv(MANAGEMENT / "keypoint_white_card.csv")
    bbox = pd.read_csv(MANAGEMENT / "Melops_bbox_coords.txt", sep=None, engine="python")
    keypoint_counts = keypoints.groupby("image").size()
    summary = {
        "record": {
            "version": "2.0",
            "doi": "10.5281/zenodo.17404087",
            "license": "CC-BY-4.0",
            "access_date": "2026-08-25",
        },
        "metadata": {
            "images": int(len(metadata)),
            "individual_ids": int(metadata["ID"].nunique()),
            "years": sorted(int(value) for value in metadata["year"].dropna().unique()),
            "sides": metadata["side"].value_counts(dropna=False).to_dict(),
            "length_mm_min": float(metadata["length"].min()),
            "length_mm_max": float(metadata["length"].max()),
            "length_mm_missing": int(metadata["length"].isna().sum()),
        },
        "management_outputs": {
            "bbox_images": int(len(bbox)),
            "predicted_keypoint_rows": int(len(keypoints)),
            "predicted_keypoint_images": int(keypoints["image"].nunique()),
            "predicted_keypoint_names": sorted(keypoints["keypoint_name"].unique().tolist()),
            "rows_per_predicted_image": {str(k): int(v) for k, v in keypoint_counts.value_counts().sort_index().items()},
            "white_card_rows": int(len(white_card)),
            "white_card_images": int(white_card["image"].nunique()),
        },
        "evidence_boundary": {
            "keypoint_head_csv": "model-generated management output, not the 505-image manual 11-point ground truth",
            "stage2_required_for_gt": True,
            "redistributed_in_project_release": False,
        },
    }
    output = ROOT / "evidence" / "melops_stage1_audit.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
