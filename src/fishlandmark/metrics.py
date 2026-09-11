# ><(((o>  鱼类形态数据工具
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


MEASUREMENT_PAIRS = {
    "axial_span_1_6": (1, 6),
    "dorsal_base_3_4": (3, 4),
    "caudal_depth_5_7": (5, 7),
    "head_chord_2_10": (2, 10),
    "ventral_base_8_9": (8, 9),
}


def _procrustes_rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = prediction - prediction.mean(axis=0, keepdims=True)
    target = target - target.mean(axis=0, keepdims=True)
    pred_norm = np.linalg.norm(prediction)
    target_norm = np.linalg.norm(target)
    if pred_norm <= 1e-12 or target_norm <= 1e-12:
        return float("nan")
    prediction = prediction / pred_norm
    target = target / target_norm
    u, _, vt = np.linalg.svd(prediction.T @ target)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    aligned = prediction @ rotation
    return float(np.sqrt(np.mean(np.sum((aligned - target) ** 2, axis=1))))


def _contour_perimeter(points: np.ndarray) -> float:
    contour = points[:10]
    shifted = np.roll(contour, -1, axis=0)
    return float(np.linalg.norm(contour - shifted, axis=1).sum())


def evaluate_prediction(
    benchmark_id: str,
    target: np.ndarray,
    prediction: np.ndarray | None,
    taxon: str,
    origin: str,
    confidence: float | None = None,
) -> dict[str, Any]:
    body_length = float(np.linalg.norm(target[0] - target[5]))
    row: dict[str, Any] = {
        "benchmark_id": benchmark_id,
        "taxon_candidate": taxon,
        "origin_label": origin,
        "detected": prediction is not None,
        "confidence": confidence,
        "body_length_px": body_length,
    }
    if prediction is None or prediction.shape != target.shape:
        row.update(
            {
                "mean_error_px": np.nan,
                "rmse_px": np.nan,
                "nme_body_length": np.nan,
                "pck_0_02": 0.0,
                "pck_0_05": 0.0,
                "pck_0_10": 0.0,
                "procrustes_rmse": np.nan,
            }
        )
        return row
    errors = np.linalg.norm(prediction - target, axis=1)
    normalized = errors / max(body_length, 1e-12)
    row.update(
        {
            "mean_error_px": float(errors.mean()),
            "rmse_px": float(np.sqrt(np.mean(errors**2))),
            "nme_body_length": float(normalized.mean()),
            "pck_0_02": float(np.mean(normalized <= 0.02)),
            "pck_0_05": float(np.mean(normalized <= 0.05)),
            "pck_0_10": float(np.mean(normalized <= 0.10)),
            "procrustes_rmse": _procrustes_rmse(prediction, target),
        }
    )
    for index, (error, nme) in enumerate(zip(errors, normalized), start=1):
        row[f"kp{index}_error_px"] = float(error)
        row[f"kp{index}_nme"] = float(nme)
    for name, (left, right) in MEASUREMENT_PAIRS.items():
        target_value = float(np.linalg.norm(target[left - 1] - target[right - 1]))
        prediction_value = float(np.linalg.norm(prediction[left - 1] - prediction[right - 1]))
        row[f"{name}_absolute_error_px"] = abs(prediction_value - target_value)
        row[f"{name}_relative_error"] = abs(prediction_value - target_value) / max(target_value, 1e-12)
    target_perimeter = _contour_perimeter(target)
    pred_perimeter = _contour_perimeter(prediction)
    row["contour_perimeter_relative_error"] = abs(pred_perimeter - target_perimeter) / max(
        target_perimeter, 1e-12
    )
    return row


def bootstrap_mean_ci(
    values: np.ndarray,
    seed: int = 20260825,
    replicates: int = 2000,
) -> dict[str, float | int]:
    clean = np.asarray(values, dtype=np.float64)
    clean = clean[np.isfinite(clean)]
    if not len(clean):
        return {"n": 0, "mean": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(clean), size=(replicates, len(clean)))
    means = clean[indices].mean(axis=1)
    return {
        "n": int(len(clean)),
        "mean": float(clean.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
    }


def summarize_predictions(frame: pd.DataFrame, seed: int = 20260825) -> dict[str, Any]:
    detected = frame[frame["detected"]].copy()
    metric_columns = [
        "nme_body_length", "mean_error_px", "rmse_px", "pck_0_02", "pck_0_05",
        "pck_0_10", "procrustes_rmse", "contour_perimeter_relative_error",
    ]
    metric_columns.extend(
        column for column in frame.columns if column.endswith("_relative_error")
    )
    overall = {
        "images": int(len(frame)),
        "detected_images": int(frame["detected"].sum()),
        "detection_rate": float(frame["detected"].mean()),
    }
    for column in dict.fromkeys(metric_columns):
        if column in detected:
            overall[column] = bootstrap_mean_ci(detected[column].to_numpy(), seed=seed)
    for threshold in ("0_02", "0_05", "0_10"):
        column = f"pck_{threshold}"
        if column in frame:
            overall[f"{column}_all_images"] = bootstrap_mean_ci(frame[column].to_numpy(), seed=seed)
            overall[f"{column}_detected_only"] = bootstrap_mean_ci(detected[column].to_numpy(), seed=seed)
    per_taxon: dict[str, Any] = {}
    for taxon, group in frame.groupby("taxon_candidate", sort=True):
        valid = group[group["detected"]]
        per_taxon[str(taxon)] = {
            "images": int(len(group)),
            "detection_rate": float(group["detected"].mean()),
            "nme_body_length": bootstrap_mean_ci(valid["nme_body_length"].to_numpy(), seed=seed),
            "pck_0_05_all_images": bootstrap_mean_ci(group["pck_0_05"].to_numpy(), seed=seed),
            "pck_0_05_detected_only": bootstrap_mean_ci(valid["pck_0_05"].to_numpy(), seed=seed),
        }
    per_keypoint = {
        str(index): bootstrap_mean_ci(detected[f"kp{index}_nme"].to_numpy(), seed=seed)
        for index in range(1, 12)
        if f"kp{index}_nme" in detected
    }
    return {"overall": overall, "per_taxon": per_taxon, "per_keypoint": per_keypoint}
