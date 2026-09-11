#!/usr/bin/env python3
# ><(((o>  鱼类形态数据工具
"""Quarantine sensitivity for unresolved perceptual-similarity candidates.

The candidate pairs are not asserted to be duplicates.  This analysis removes
test members whose paired candidate occurs in the corresponding train or
validation membership, then recomputes the existing image-level summaries.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fishlandmark.metrics import summarize_predictions


EVALUATIONS = {
    "standard_axial_yolo": ROOT / "models/evaluation/yolo26m_pose_v1_axial_seed20260825/test/per_image_metrics.csv",
    "standard_original_yolo": ROOT / "models/evaluation/yolo26m_pose_v1_original_seed20260825/test/per_image_metrics.csv",
    "standard_resnet34": ROOT / "models/evaluation/resnet34_heatmap_v1_seed20260825/test/per_image_metrics.csv",
    "standard_dinov2": ROOT / "models/evaluation/dinov2_vitl14_reg_heatmap_v1_seed20260825/test/per_image_metrics.csv",
    "standard_dinov2_geometry": ROOT / "models/evaluation/dinov2_vitl14_reg_geometry_v1_seed20260825/test/per_image_metrics.csv",
    "cross_origin_axial_diagnostic": ROOT / "models/evaluation/yolo26m_cross_origin_farmed_to_wild_seed20260825/test/per_image_metrics.csv",
    "cross_origin_original": ROOT / "models/evaluation/yolo26m_cross_origin_original_seed20260825/test/per_image_metrics.csv",
}


def cross_partition_rows(pairs: pd.DataFrame, membership: dict[str, str], protocol: str) -> pd.DataFrame:
    rows = []
    for row in pairs.itertuples(index=False):
        left_role = membership.get(row.benchmark_id_left, "not_in_protocol")
        right_role = membership.get(row.benchmark_id_right, "not_in_protocol")
        roles = {left_role, right_role}
        if "test" not in roles or not roles.intersection({"train", "validation"}):
            continue
        test_id = row.benchmark_id_left if left_role == "test" else row.benchmark_id_right
        rows.append({
            **row._asdict(),
            "protocol": protocol,
            "left_role": left_role,
            "right_role": right_role,
            "quarantined_test_id": test_id,
        })
    return pd.DataFrame(rows)


def core(summary: dict[str, object]) -> dict[str, object]:
    overall = summary["overall"]
    return {
        "images": overall["images"],
        "detected_images": overall["detected_images"],
        "detection_rate": overall["detection_rate"],
        "nme_working_axial_span_detected_only": overall["nme_body_length"],
        "pck_0_05_all_images": overall["pck_0_05_all_images"],
    }


def main() -> None:
    pairs = pd.read_csv(ROOT / "data/benchmark_v1/tables/near_duplicate_review_candidates.csv")
    samples = pd.read_parquet(ROOT / "data/benchmark_v1/tables/benchmark_samples.parquet")
    standard = dict(zip(samples.benchmark_id, samples.standard_split))
    membership_frame = pd.read_csv(ROOT / "data/experiments/protocols/cross_origin_membership_manifest.csv")
    cross = dict(zip(membership_frame.benchmark_id, membership_frame.protocol_role))

    standard_pairs = cross_partition_rows(pairs, standard, "standard")
    cross_pairs = cross_partition_rows(pairs, cross, "cross_origin")
    pair_audit = pd.concat([standard_pairs, cross_pairs], ignore_index=True)
    standard_exclude = set(standard_pairs.get("quarantined_test_id", pd.Series(dtype=str)))
    cross_exclude = set(cross_pairs.get("quarantined_test_id", pd.Series(dtype=str)))

    summaries: dict[str, object] = {}
    rows = []
    for name, path in EVALUATIONS.items():
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        excluded = cross_exclude if name.startswith("cross_origin") else standard_exclude
        retained = frame[~frame.benchmark_id.isin(excluded)].copy()
        full_summary = summarize_predictions(frame)
        retained_summary = summarize_predictions(retained)
        full_core = core(full_summary); retained_core = core(retained_summary)
        summaries[name] = {
            "evaluation_csv": path.relative_to(ROOT).as_posix(),
            "candidate_pairs_are_unresolved": True,
            "quarantined_test_ids": sorted(excluded.intersection(set(frame.benchmark_id))),
            "full": full_core,
            "after_candidate_quarantine": retained_core,
        }
        rows.append({
            "evaluation": name,
            "full_images": full_core["images"],
            "quarantined_test_images": full_core["images"] - retained_core["images"],
            "retained_images": retained_core["images"],
            "full_nme": full_core["nme_working_axial_span_detected_only"]["mean"],
            "retained_nme": retained_core["nme_working_axial_span_detected_only"]["mean"],
            "full_pck_0_05_all": full_core["pck_0_05_all_images"]["mean"],
            "retained_pck_0_05_all": retained_core["pck_0_05_all_images"]["mean"],
        })

    output = ROOT / "qa/near_duplicate_sensitivity"
    output.mkdir(parents=True, exist_ok=True)
    pair_audit.to_csv(output / "cross_partition_candidate_pairs.csv", index=False)
    pd.DataFrame(rows).to_csv(output / "model_sensitivity_summary.csv", index=False)
    payload = {
        "status": "sensitivity analysis; perceptual-similarity candidates are not confirmed duplicates",
        "candidate_pairs_total": int(len(pairs)),
        "standard_cross_partition_pairs": int(len(standard_pairs)),
        "standard_test_ids_quarantined": sorted(standard_exclude),
        "cross_origin_cross_partition_pairs": int(len(cross_pairs)),
        "cross_origin_test_ids_quarantined": sorted(cross_exclude),
        "models": summaries,
    }
    (output / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "models"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
