#!/usr/bin/env python3
# ><(((o>  鱼类形态数据工具
"""Write a compact, no-hash inventory of the reported evaluation artefacts."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "evidence/test_evaluation_freeze_manifest.csv"

RUNS = [
    ("standard_axial_yolo", "models/yolo/yolo26m_pose_v1_axial_seed20260825", "models/evaluation/yolo26m_pose_v1_axial_seed20260825/test"),
    ("standard_original_yolo", "models/yolo/yolo26m_pose_v1_original_seed20260825", "models/evaluation/yolo26m_pose_v1_original_seed20260825/test"),
    ("cross_origin_original_yolo", "models/yolo/yolo26m_cross_origin_original_seed20260825", "models/evaluation/yolo26m_cross_origin_original_seed20260825/test"),
    ("cross_origin_axial_diagnostic", "models/yolo/yolo26m_cross_origin_farmed_to_wild_seed20260825", "models/evaluation/yolo26m_cross_origin_farmed_to_wild_seed20260825/test"),
    ("resnet34_heatmap", "models/heatmap/resnet34_heatmap_v1_seed20260825", "models/evaluation/resnet34_heatmap_v1_seed20260825/test"),
    ("dinov2_heatmap", "models/heatmap/dinov2_vitl14_reg_heatmap_v1_seed20260825", "models/evaluation/dinov2_vitl14_reg_heatmap_v1_seed20260825/test"),
    ("dinov2_geometry", "models/heatmap/dinov2_vitl14_reg_geometry_v1_seed20260825", "models/evaluation/dinov2_vitl14_reg_geometry_v1_seed20260825/test"),
]

EVALUATION_ONLY = [
    (
        "locateanything_3b_zero_shot",
        [
            ("data/experiments/protocols/locateanything_3b_zero_shot/protocol.json", "frozen zero-shot protocol"),
            ("data/experiments/protocols/locateanything_3b_zero_shot/sample_manifest.csv", "frozen zero-shot cohort"),
            ("models/evaluation/locateanything_3b_zero_shot/summary.json", "reported evaluation summary"),
            ("models/evaluation/locateanything_3b_zero_shot/query_predictions.csv", "reported per-query evaluation"),
        ],
    ),
]


def add(rows: list[dict[str, object]], run: str, role: str, path: Path) -> None:
    if not path.exists():
        return
    stat = path.stat()
    rows.append({
        "run": run,
        "artefact_role": role,
        "relative_path": path.relative_to(ROOT).as_posix(),
        "bytes": stat.st_size,
        "modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        "cryptographic_hash": "not_computed_by_project_policy",
    })


def main() -> None:
    rows: list[dict[str, object]] = []
    for run, training_dir, evaluation_dir in RUNS:
        train = ROOT / training_dir
        evaluation = ROOT / evaluation_dir
        for filename, role in [
            ("locked_training_config.json", "training configuration"),
            ("results.csv", "training history"),
            ("history.json", "training history"),
            ("weights/best.pt", "local selected checkpoint; excluded from delivery archive"),
        ]:
            add(rows, run, role, train / filename)
        for filename, role in [
            ("summary.json", "reported evaluation summary"),
            ("per_image_metrics.csv", "reported per-image evaluation"),
        ]:
            add(rows, run, role, evaluation / filename)
    for run, artefacts in EVALUATION_ONLY:
        for relative_path, role in artefacts:
            add(rows, run, role, ROOT / relative_path)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{OUTPUT}: {len(rows)} artefacts; no cryptographic hashes computed")


if __name__ == "__main__":
    main()
