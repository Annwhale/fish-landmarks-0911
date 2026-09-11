#!/usr/bin/env python3
# ><(((o>  鱼类形态数据工具
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fishlandmark.metrics import bootstrap_mean_ci


def icc_a1(values: np.ndarray) -> float:
    """Two-way random-effects, absolute-agreement, single-measure ICC(A,1).

    Rows are targets and the two columns are technical-repeat coordinates. The
    repeats are not attributed to identified raters, so this is reported as a
    technical-repeat reliability statistic rather than an inter-rater ICC.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] < 2:
        return float("nan")
    n, k = array.shape
    grand = array.mean()
    row_mean = array.mean(axis=1)
    column_mean = array.mean(axis=0)
    ss_rows = k * np.sum((row_mean - grand) ** 2)
    ss_columns = n * np.sum((column_mean - grand) ** 2)
    residual = array - row_mean[:, None] - column_mean[None, :] + grand
    ss_error = np.sum(residual**2)
    ms_rows = ss_rows / (n - 1)
    ms_columns = ss_columns / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))
    denominator = ms_rows + (k - 1) * ms_error + k * (ms_columns - ms_error) / n
    return float((ms_rows - ms_error) / denominator) if abs(denominator) > 1e-12 else float("nan")


def procrustes_rmse(first: np.ndarray, second: np.ndarray) -> float:
    first = first - first.mean(axis=0, keepdims=True)
    second = second - second.mean(axis=0, keepdims=True)
    first_norm = np.linalg.norm(first)
    second_norm = np.linalg.norm(second)
    if first_norm <= 1e-12 or second_norm <= 1e-12:
        return float("nan")
    first = first / first_norm
    second = second / second_norm
    u, _, vt = np.linalg.svd(first.T @ second)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    aligned = first @ rotation
    return float(np.sqrt(np.mean(np.sum((aligned - second) ** 2, axis=1))))


def main() -> None:
    derived = ROOT / "data" / "derived"
    duplicates = pd.read_parquet(derived / "exact_duplicates.parquet")
    metadata = pd.read_parquet(derived / "metadata.parquet").set_index("sample_uid")
    points = pd.read_parquet(derived / "keypoints_canonical_long.parquet")
    points_by_sample = {
        str(sample_uid): group.sort_values("canonical_id")[["x", "y"]].to_numpy(np.float64)
        for sample_uid, group in points.groupby("sample_uid", sort=False)
        if len(group) == 11 and group["canonical_id"].tolist() == list(range(1, 12))
    }

    pair_rows: list[dict[str, object]] = []
    coordinate_pairs: dict[int, dict[str, list[list[float]]]] = {
        point_id: {"x": [], "y": []} for point_id in range(1, 12)
    }
    for content_id, group in duplicates.groupby("content_id_blake2b64", sort=True):
        sample_ids = [str(item) for item in group["sample_uid"] if str(item) in points_by_sample]
        taxa = {str(metadata.loc[sample_uid, "taxon_candidate"]) for sample_uid in sample_ids}
        if len(sample_ids) < 2 or len(taxa) != 1:
            continue
        for first_id, second_id in itertools.combinations(sorted(sample_ids), 2):
            first = points_by_sample[first_id]
            second = points_by_sample[second_id]
            width = float(metadata.loc[first_id, "width"])
            height = float(metadata.loc[first_id, "height"])
            diagonal = float(np.hypot(width, height))
            body_axis = 0.5 * (
                np.linalg.norm(first[0] - first[5]) + np.linalg.norm(second[0] - second[5])
            )
            errors = np.linalg.norm(first - second, axis=1)
            normalized_diagonal = errors / max(diagonal, 1e-12)
            normalized_body = errors / max(body_axis, 1e-12)
            row: dict[str, object] = {
                "content_id_blake2b64": str(content_id),
                "sample_uid_first": first_id,
                "sample_uid_second": second_id,
                "taxon_candidate": next(iter(taxa)),
                "image_diagonal_px": diagonal,
                "body_axis_px": body_axis,
                "mean_error_px": float(errors.mean()),
                "nme_image_diagonal": float(normalized_diagonal.mean()),
                "nme_body_axis": float(normalized_body.mean()),
                "pck_diagonal_0_01": float(np.mean(normalized_diagonal <= 0.01)),
                "pck_diagonal_0_02": float(np.mean(normalized_diagonal <= 0.02)),
                "pck_diagonal_0_05": float(np.mean(normalized_diagonal <= 0.05)),
                "procrustes_rmse": procrustes_rmse(first, second),
            }
            for point_id, (error, diag_nme, body_nme) in enumerate(
                zip(errors, normalized_diagonal, normalized_body), start=1
            ):
                row[f"kp{point_id}_error_px"] = float(error)
                row[f"kp{point_id}_nme_diagonal"] = float(diag_nme)
                row[f"kp{point_id}_nme_body_axis"] = float(body_nme)
                coordinate_pairs[point_id]["x"].append([float(first[point_id - 1, 0]), float(second[point_id - 1, 0])])
                coordinate_pairs[point_id]["y"].append([float(first[point_id - 1, 1]), float(second[point_id - 1, 1])])
            pair_rows.append(row)

    pairs = pd.DataFrame(pair_rows).sort_values(["content_id_blake2b64", "sample_uid_first"])
    point_rows: list[dict[str, object]] = []
    for point_id in range(1, 12):
        diagonal_summary = bootstrap_mean_ci(pairs[f"kp{point_id}_nme_diagonal"].to_numpy())
        body_summary = bootstrap_mean_ci(pairs[f"kp{point_id}_nme_body_axis"].to_numpy())
        point_rows.append(
            {
                "canonical_id": point_id,
                "technical_repeat_pairs": len(coordinate_pairs[point_id]["x"]),
                "icc_a1_x": icc_a1(np.asarray(coordinate_pairs[point_id]["x"])),
                "icc_a1_y": icc_a1(np.asarray(coordinate_pairs[point_id]["y"])),
                "mean_nme_image_diagonal": diagonal_summary["mean"],
                "nme_image_diagonal_ci95_low": diagonal_summary["ci95_low"],
                "nme_image_diagonal_ci95_high": diagonal_summary["ci95_high"],
                "mean_nme_body_axis": body_summary["mean"],
                "nme_body_axis_ci95_low": body_summary["ci95_low"],
                "nme_body_axis_ci95_high": body_summary["ci95_high"],
            }
        )
    per_point = pd.DataFrame(point_rows)

    output = ROOT / "qa" / "repeat_annotation_validation"
    output.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(output / "per_pair_metrics.csv", index=False)
    pairs.to_parquet(output / "per_pair_metrics.parquet", index=False)
    per_point.to_csv(output / "per_keypoint_reliability.csv", index=False)
    per_point.to_parquet(output / "per_keypoint_reliability.parquet", index=False)
    summary = {
        "status": "technical-repeat reliability; source annotator identities are unavailable",
        "unit_of_analysis": "exact-image annotation pair after excluding taxon-conflict groups",
        "pair_count": int(len(pairs)),
        "content_id_count": int(pairs["content_id_blake2b64"].nunique()),
        "nme_image_diagonal": bootstrap_mean_ci(pairs["nme_image_diagonal"].to_numpy()),
        "nme_body_axis": bootstrap_mean_ci(pairs["nme_body_axis"].to_numpy()),
        "pck_diagonal_0_01": bootstrap_mean_ci(pairs["pck_diagonal_0_01"].to_numpy()),
        "pck_diagonal_0_02": bootstrap_mean_ci(pairs["pck_diagonal_0_02"].to_numpy()),
        "pck_diagonal_0_05": bootstrap_mean_ci(pairs["pck_diagonal_0_05"].to_numpy()),
        "procrustes_rmse": bootstrap_mean_ci(pairs["procrustes_rmse"].to_numpy()),
        "icc_definition": "ICC(A,1), two-way random effects, absolute agreement, single measure",
        "icc_interpretation_boundary": "technical repeats are not assigned to identified raters",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
