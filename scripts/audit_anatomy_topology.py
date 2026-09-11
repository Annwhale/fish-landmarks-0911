#!/usr/bin/env python3
# ><(((o>  鱼类形态数据工具
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "data" / "benchmark_v1"


if __name__ == "__main__":
    points = pd.read_parquet(PACKAGE / "annotations" / "keypoints_crops_448.parquet")
    samples = pd.read_parquet(PACKAGE / "tables" / "benchmark_samples.parquet")
    x = points.pivot(index="benchmark_id", columns="canonical_id", values="x_crop_px")
    y = points.pivot(index="benchmark_id", columns="canonical_id", values="y_crop_px")
    checks = {
        "snout_precedes_caudal_midpoint": x[1] < x[6],
        "nape_is_dorsal_of_body_axis": y[2] < ((y[1] + y[6]) / 2.0),
        "dorsal_origin_precedes_insertion": x[3] < x[4],
        "dorsal_insertion_precedes_upper_caudal": x[4] < x[5],
        "upper_caudal_near_or_before_midpoint": x[5] <= x[6] + 10.0,
        "lower_caudal_near_or_before_midpoint": x[7] <= x[6] + 10.0,
        "anal_origin_precedes_lower_caudal": x[8] < x[7],
        "pelvic_origin_precedes_anal_origin": x[9] < x[8],
        "pectoral_base_precedes_pelvic_origin": x[10] < x[9],
    }
    check_frame = pd.DataFrame(checks)
    check_frame.index.name = "benchmark_id"
    check_frame = check_frame.reset_index().merge(
        samples[["benchmark_id", "taxon_candidate", "origin_label", "standard_split"]],
        on="benchmark_id",
        how="left",
        validate="one_to_one",
    )
    output = ROOT / "qa"
    output.mkdir(parents=True, exist_ok=True)
    check_frame.to_csv(output / "anatomy_topology_checks.csv", index=False)
    check_frame.to_parquet(output / "anatomy_topology_checks.parquet", index=False)
    summary = {}
    for name in checks:
        passed = check_frame[name]
        summary[name] = {
            "images": int(len(passed)),
            "pass_count": int(passed.sum()),
            "pass_fraction": float(passed.mean()),
            "violation_count": int((~passed).sum()),
            "violation_ids": check_frame.loc[~passed, "benchmark_id"].tolist(),
        }
    payload = {
        "status": "geometry-based support for the working ontology; not a substitute for source-team semantics",
        "coordinate_system": "448 x 448 axial-normalized crops",
        "checks": summary,
    }
    (output / "anatomy_topology_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
