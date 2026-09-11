#!/usr/bin/env python3
# ><(((o>  鱼类形态数据工具
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SEED = 20260825
REPLICATES = 2000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluation",
        action="append",
        required=True,
        metavar="NAME=CSV",
        help="Model label and per_image_metrics.csv path; repeat for each model",
    )
    parser.add_argument("--reference", required=True, help="Reference model label for paired differences")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--allow-id-intersection",
        action="store_true",
        help="Allow silently non-identical benchmark-ID sets; otherwise paired inputs must have the same IDs",
    )
    return parser.parse_args()


def paired_bootstrap(first: np.ndarray, second: np.ndarray) -> dict[str, float | int]:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    keep = np.isfinite(first) & np.isfinite(second)
    differences = first[keep] - second[keep]
    if not len(differences):
        return {"n": 0, "mean_difference": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}
    rng = np.random.default_rng(SEED)
    indices = rng.integers(0, len(differences), size=(REPLICATES, len(differences)))
    means = differences[indices].mean(axis=1)
    return {
        "n": int(len(differences)),
        "mean_difference": float(differences.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
    }


def mean_ci(values: pd.Series) -> dict[str, float | int]:
    array = values.to_numpy(np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"n": 0, "mean": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}
    rng = np.random.default_rng(SEED)
    indices = rng.integers(0, len(array), size=(REPLICATES, len(array)))
    means = array[indices].mean(axis=1)
    return {
        "n": int(len(array)),
        "mean": float(array.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
    }


def main() -> None:
    args = parse_args()
    frames: dict[str, pd.DataFrame] = {}
    for item in args.evaluation:
        name, path = item.split("=", 1)
        frame = pd.read_csv(path).sort_values("benchmark_id").set_index("benchmark_id")
        frames[name] = frame
    if args.reference not in frames:
        raise ValueError(f"reference {args.reference!r} is not among evaluations")
    id_sets = {name: set(frame.index) for name, frame in frames.items()}
    if not args.allow_id_intersection and any(ids != next(iter(id_sets.values())) for ids in id_sets.values()):
        raise ValueError(
            "paired evaluation ID sets differ; verify split/protocol and use --allow-id-intersection only with an explicit analysis justification"
        )
    common_ids = set.intersection(*(set(frame.index) for frame in frames.values()))
    if not common_ids:
        raise ValueError("evaluations have no common benchmark IDs")
    ordered_ids = sorted(common_ids)
    frames = {name: frame.loc[ordered_ids].copy() for name, frame in frames.items()}

    summary_rows: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "seed": SEED,
        "bootstrap_replicates": REPLICATES,
        "paired_images": len(ordered_ids),
        "reference": args.reference,
        "input_files": {name: item.split("=", 1)[1] for name, item in zip(frames, args.evaluation)},
        "comparison_contract": (
            "paired benchmark IDs were checked; representation, protocol and target-coordinate compatibility "
            "must be established from each evaluation summary and the caller's scientific design"
        ),
        "nme_boundary": "detected images only; paired NME differences use images detected by both models",
        "nme_normalizer": "role-1-to-role-6 working axial span; legacy field name nme_body_length is not physical body length",
        "pck_boundary": "all-image PCK includes detection failures as zero",
        "models": {},
        "paired_differences": {},
    }
    for name, frame in frames.items():
        detected = frame[frame["detected"].astype(bool)]
        model_summary = {
            "images": int(len(frame)),
            "detected_images": int(len(detected)),
            "detection_rate": float(frame["detected"].astype(bool).mean()),
            "nme_body_length_detected_only": mean_ci(detected["nme_body_length"]),
            "pck_0_05_all_images": mean_ci(frame["pck_0_05"]),
            "pck_0_05_detected_only": mean_ci(detected["pck_0_05"]),
            "contour_perimeter_relative_error_detected_only": mean_ci(
                detected["contour_perimeter_relative_error"]
            ),
        }
        summary["models"][name] = model_summary
        summary_rows.append(
            {
                "model": name,
                "images": len(frame),
                "detected_images": len(detected),
                "detection_rate": model_summary["detection_rate"],
                "nme_mean": model_summary["nme_body_length_detected_only"]["mean"],
                "nme_ci95_low": model_summary["nme_body_length_detected_only"]["ci95_low"],
                "nme_ci95_high": model_summary["nme_body_length_detected_only"]["ci95_high"],
                "pck_0_05_all_mean": model_summary["pck_0_05_all_images"]["mean"],
                "pck_0_05_all_ci95_low": model_summary["pck_0_05_all_images"]["ci95_low"],
                "pck_0_05_all_ci95_high": model_summary["pck_0_05_all_images"]["ci95_high"],
            }
        )

    reference = frames[args.reference]
    for name, frame in frames.items():
        if name == args.reference:
            continue
        both_detected = frame["detected"].astype(bool) & reference["detected"].astype(bool)
        summary["paired_differences"][name] = {
            "comparison": f"{name} minus {args.reference}",
            "nme_body_length_common_detected": paired_bootstrap(
                frame.loc[both_detected, "nme_body_length"].to_numpy(),
                reference.loc[both_detected, "nme_body_length"].to_numpy(),
            ),
            "pck_0_05_all_images": paired_bootstrap(
                frame["pck_0_05"].to_numpy(), reference["pck_0_05"].to_numpy()
            ),
        }

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(output / "model_summary.csv", index=False)
    wide = pd.DataFrame(index=ordered_ids)
    for name, frame in frames.items():
        for column in ("detected", "nme_body_length", "pck_0_05"):
            wide[f"{name}__{column}"] = frame[column]
    wide.index.name = "benchmark_id"
    wide.to_csv(output / "paired_core_metrics.csv")
    (output / "comparison_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
