# ><(((o>  鱼类形态数据工具
from __future__ import annotations

import json
import math
import random
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _write_table(frame: pd.DataFrame, output_root: Path, stem: str) -> None:
    frame.to_csv(output_root / f"{stem}.csv", index=False)
    frame.to_parquet(output_root / f"{stem}.parquet", index=False)


def _pipe(values: list[str]) -> str:
    return "|".join(sorted({str(value) for value in values}))


def _split_within_strata(
    samples: pd.DataFrame,
    seed: int,
    ratios: dict[str, float],
) -> pd.Series:
    """Deterministically assign train/validation/test inside taxon-origin strata."""
    assignments: dict[str, str] = {}
    for stratum, group in samples.groupby("split_stratum", sort=True):
        ids = sorted(group["benchmark_id"].tolist())
        stratum_seed = seed + sum((idx + 1) * ord(char) for idx, char in enumerate(stratum))
        rng = random.Random(stratum_seed)
        rng.shuffle(ids)
        n_items = len(ids)
        n_validation = max(1, int(round(n_items * ratios["validation"]))) if n_items >= 3 else 0
        n_test = max(1, int(round(n_items * ratios["test"]))) if n_items >= 3 else 0
        while n_validation + n_test >= n_items and (n_validation > 0 or n_test > 0):
            if n_validation >= n_test and n_validation > 0:
                n_validation -= 1
            elif n_test > 0:
                n_test -= 1
        for benchmark_id in ids[:n_test]:
            assignments[benchmark_id] = "test"
        for benchmark_id in ids[n_test : n_test + n_validation]:
            assignments[benchmark_id] = "validation"
        for benchmark_id in ids[n_test + n_validation :]:
            assignments[benchmark_id] = "train"
    return samples["benchmark_id"].map(assignments)


def _duplicate_disagreement(
    group: pd.DataFrame,
    canonical: pd.DataFrame,
) -> dict[str, float | int | None]:
    sample_uids = sorted(group["sample_uid"].tolist())
    point_maps: dict[str, dict[int, tuple[float, float]]] = {}
    for sample_uid in sample_uids:
        rows = canonical[canonical["sample_uid"] == sample_uid]
        point_maps[sample_uid] = {
            int(row.canonical_id): (float(row.x), float(row.y))
            for row in rows.itertuples(index=False)
        }
    pair_means: list[float] = []
    pair_maxima: list[float] = []
    normalized_means: list[float] = []
    complete_pairs = 0
    diagonal = math.hypot(float(group.iloc[0]["width"]), float(group.iloc[0]["height"]))
    for left, right in combinations(sample_uids, 2):
        common = sorted(set(point_maps[left]).intersection(point_maps[right]))
        if not common:
            continue
        errors = [
            math.hypot(
                point_maps[left][point_id][0] - point_maps[right][point_id][0],
                point_maps[left][point_id][1] - point_maps[right][point_id][1],
            )
            for point_id in common
        ]
        mean_error = float(np.mean(errors))
        pair_means.append(mean_error)
        pair_maxima.append(float(np.max(errors)))
        normalized_means.append(mean_error / diagonal if diagonal > 0 else float("nan"))
        complete_pairs += int(len(common) == 11)
    return {
        "record_count": len(sample_uids),
        "pair_count": len(pair_means),
        "complete_11_point_pair_count": complete_pairs,
        "pairwise_mean_error_px": float(np.mean(pair_means)) if pair_means else None,
        "pairwise_max_error_px": float(np.max(pair_maxima)) if pair_maxima else None,
        "pairwise_mean_error_image_diagonal": (
            float(np.mean(normalized_means)) if normalized_means else None
        ),
    }


def _near_duplicate_candidates(samples: pd.DataFrame) -> pd.DataFrame:
    """Review-only dHash candidates; never used for automatic exclusion or splitting."""
    rows: list[dict[str, Any]] = []
    working = samples.dropna(subset=["difference_hash_64", "width", "height", "file_bytes"])
    for _, group in working.groupby(["width", "height"], sort=True):
        records = list(group.itertuples(index=False))
        for left, right in combinations(records, 2):
            if left.content_id_blake2b64 == right.content_id_blake2b64:
                continue
            size_ratio = abs(left.file_bytes - right.file_bytes) / max(left.file_bytes, right.file_bytes)
            if size_ratio > 0.01:
                continue
            distance = (int(left.difference_hash_64, 16) ^ int(right.difference_hash_64, 16)).bit_count()
            if distance <= 2:
                rows.append(
                    {
                        "benchmark_id_left": left.benchmark_id,
                        "benchmark_id_right": right.benchmark_id,
                        "dhash_hamming_distance": distance,
                        "relative_file_size_difference": size_ratio,
                        "review_status": "manual_review_required",
                        "automatic_action": "none",
                    }
                )
    return pd.DataFrame(
        rows,
        columns=[
            "benchmark_id_left",
            "benchmark_id_right",
            "dhash_hamming_distance",
            "relative_file_size_difference",
            "review_status",
            "automatic_action",
        ],
    )


def build_benchmark(
    derived_root: Path,
    qa_root: Path,
    seed: int = 20260825,
    ratios: dict[str, float] | None = None,
    duplicate_disagreement_threshold: float = 0.02,
) -> dict[str, Any]:
    ratios = ratios or {"train": 0.70, "validation": 0.15, "test": 0.15}
    if not math.isclose(sum(ratios.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("standard split ratios must sum to one")
    derived_root.mkdir(parents=True, exist_ok=True)
    qa_root.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_parquet(derived_root / "metadata.parquet")
    canonical = pd.read_parquet(derived_root / "keypoints_canonical_long.parquet")
    eligible = metadata[
        metadata["fish_landmarks_1_to_11_complete"]
        & metadata["canonical_mapping_complete"]
        & metadata["image_decode_error"].isna()
        & metadata["content_id_blake2b64"].notna()
    ].copy()

    benchmark_rows: list[dict[str, Any]] = []
    point_rows: list[dict[str, Any]] = []
    agreement_rows: list[dict[str, Any]] = []
    exclusion_rows: list[dict[str, Any]] = []

    for digest, group in eligible.groupby("content_id_blake2b64", sort=True):
        taxa = sorted(group["taxon_candidate"].unique().tolist())
        if len(taxa) != 1:
            for row in group.itertuples(index=False):
                exclusion_rows.append(
                    {
                        "sample_uid": row.sample_uid,
                        "content_id_blake2b64": digest,
                        "source_group": row.source_group,
                        "taxon_candidate": row.taxon_candidate,
                        "exclusion_reason": "exact_image_conflicting_taxon_labels",
                    }
                )
            continue

        ordered = group.sort_values(["anomaly_count", "database_sample_id", "sample_uid"])
        representative = ordered.iloc[0]
        sample_uids = sorted(group["sample_uid"].tolist())
        source_groups = sorted(group["source_group"].unique().tolist())
        origins = sorted(group["origin_label_source"].unique().tolist())
        origin_label = origins[0] if len(origins) == 1 else "conflicted"
        benchmark_id = f"flm_{digest}"
        benchmark_rows.append(
            {
                "benchmark_id": benchmark_id,
                "content_id_blake2b64": digest,
                "representative_sample_uid": representative["sample_uid"],
                "source_record_count": len(group),
                "source_sample_uids": _pipe(sample_uids),
                "source_groups": _pipe(source_groups),
                "taxon_candidate": taxa[0],
                "origin_label": origin_label,
                "origin_conflict": len(origins) > 1,
                "representative_original_image": representative["original_image"],
                "representative_rendered_image": representative["rendered_image"],
                "representative_csv_annotation": representative["csv_annotation"],
                "representative_yolo_annotation": representative["yolo_annotation"],
                "width": int(representative["width"]),
                "height": int(representative["height"]),
                "file_bytes": int(representative["file_bytes"]),
                "difference_hash_64": representative["difference_hash_64"],
                "scale_endpoints_1_to_2_complete": bool(
                    group["scale_endpoints_1_to_2_complete"].all()
                ),
                "canonical_mapping_modes": _pipe(group["canonical_mapping_mode"].tolist()),
                "canonical_mapping_reasons": _pipe(group["canonical_mapping_reason"].tolist()),
                "provenance_status": "source_team_confirmation_pending",
                "canonical_semantics_status": "provisional_source_confirmation_pending",
            }
        )

        duplicate_points = canonical[canonical["sample_uid"].isin(sample_uids)]
        for canonical_id, point_group in duplicate_points.groupby("canonical_id", sort=True):
            point_rows.append(
                {
                    "benchmark_id": benchmark_id,
                    "canonical_id": int(canonical_id),
                    "canonical_name_provisional": point_group["canonical_name"].iloc[0],
                    "x": float(point_group["x"].median()),
                    "y": float(point_group["y"].median()),
                    "consensus_record_count": int(len(point_group)),
                    "x_mad_px": float((point_group["x"] - point_group["x"].median()).abs().median()),
                    "y_mad_px": float((point_group["y"] - point_group["y"].median()).abs().median()),
                }
            )
        if len(group) > 1:
            agreement_rows.append(
                {
                    "benchmark_id": benchmark_id,
                    "content_id_blake2b64": digest,
                    "taxon_candidate": taxa[0],
                    "origin_conflict": len(origins) > 1,
                    **_duplicate_disagreement(group, canonical),
                }
            )

    samples = pd.DataFrame(benchmark_rows).sort_values("benchmark_id").reset_index(drop=True)
    points = pd.DataFrame(point_rows).sort_values(["benchmark_id", "canonical_id"]).reset_index(drop=True)
    agreement = pd.DataFrame(agreement_rows).sort_values("benchmark_id").reset_index(drop=True)
    exclusions = pd.DataFrame(exclusion_rows).sort_values(
        ["content_id_blake2b64", "sample_uid"]
    ).reset_index(drop=True)

    point_counts = points.groupby("benchmark_id").size()
    bad_ids = point_counts[point_counts != 11].index.tolist()
    if bad_ids:
        raise RuntimeError(f"curated benchmark contains incomplete canonical points: {bad_ids[:5]}")

    taxa_with_both_origins = {
        taxon
        for taxon, group in samples[~samples["origin_conflict"]].groupby("taxon_candidate")
        if {"farmed", "wild"}.issubset(set(group["origin_label"]))
    }
    samples["cross_origin_eligible"] = (
        samples["taxon_candidate"].isin(taxa_with_both_origins) & ~samples["origin_conflict"]
    )
    disagreement_by_id = agreement.set_index("benchmark_id")[
        "pairwise_mean_error_image_diagonal"
    ].to_dict()
    samples["repeated_annotation_mean_error_image_diagonal"] = samples["benchmark_id"].map(
        disagreement_by_id
    )
    samples["repeated_annotation_high_disagreement"] = (
        samples["repeated_annotation_mean_error_image_diagonal"].fillna(0.0)
        > duplicate_disagreement_threshold
    )
    samples["technical_validation_eligible"] = ~samples[
        "repeated_annotation_high_disagreement"
    ]
    samples["technical_validation_status"] = np.where(
        samples["technical_validation_eligible"],
        "eligible",
        "review_holdout_repeated_annotation_disagreement",
    )
    samples["split_stratum"] = samples["taxon_candidate"] + "|" + samples["origin_label"]
    split_assignments = _split_within_strata(
        samples[samples["technical_validation_eligible"]].copy(), seed, ratios
    )
    samples["standard_split"] = samples["benchmark_id"].map(
        dict(zip(
            samples.loc[samples["technical_validation_eligible"], "benchmark_id"],
            split_assignments,
        ))
    ).fillna("review_holdout")
    samples["leave_one_taxon_fold"] = samples["taxon_candidate"]

    near_duplicates = _near_duplicate_candidates(samples)

    _write_table(samples, derived_root, "benchmark_samples")
    _write_table(points, derived_root, "benchmark_keypoints")
    _write_table(agreement, derived_root, "duplicate_annotation_agreement")
    _write_table(exclusions, derived_root, "benchmark_exclusions")
    _write_table(near_duplicates, derived_root, "near_duplicate_review_candidates")

    standard_split = {
        split: sorted(samples.loc[samples["standard_split"] == split, "benchmark_id"].tolist())
        for split in ("train", "validation", "test", "review_holdout")
    }
    leave_one_taxon = {
        taxon: {
            "train": sorted(samples.loc[samples["taxon_candidate"] != taxon, "benchmark_id"].tolist()),
            "test": sorted(samples.loc[samples["taxon_candidate"] == taxon, "benchmark_id"].tolist()),
        }
        for taxon in sorted(samples["taxon_candidate"].unique())
    }
    with (derived_root / "splits_standard.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {"seed": seed, "ratios": ratios, "stratification": "taxon_candidate|origin_label", **standard_split},
            handle,
            ensure_ascii=False,
            indent=2,
        )
    with (derived_root / "splits_leave_one_taxon.json").open("w", encoding="utf-8") as handle:
        json.dump(leave_one_taxon, handle, ensure_ascii=False, indent=2)

    summary = {
        "source_records": int(len(metadata)),
        "eligible_source_records_before_duplicate_resolution": int(len(eligible)),
        "benchmark_unique_images": int(len(samples)),
        "benchmark_keypoint_records": int(len(points)),
        "taxa": int(samples["taxon_candidate"].nunique()),
        "excluded_taxon_conflict_content_ids": int(exclusions["content_id_blake2b64"].nunique()),
        "excluded_taxon_conflict_source_records": int(len(exclusions)),
        "same_taxon_duplicate_consensus_groups": int(len(agreement)),
        "repeated_annotation_high_disagreement_images": int(
            samples["repeated_annotation_high_disagreement"].sum()
        ),
        "repeated_annotation_disagreement_threshold_image_diagonal": (
            duplicate_disagreement_threshold
        ),
        "technical_validation_eligible_images": int(
            samples["technical_validation_eligible"].sum()
        ),
        "origin_conflict_images": int(samples["origin_conflict"].sum()),
        "cross_origin_eligible_images": int(samples["cross_origin_eligible"].sum()),
        "standard_split_counts": {
            str(key): int(value) for key, value in samples["standard_split"].value_counts().items()
        },
        "counts_by_taxon": {
            str(key): int(value) for key, value in samples["taxon_candidate"].value_counts().items()
        },
        "near_duplicate_review_candidates": int(len(near_duplicates)),
        "near_duplicate_policy": "review only; no automatic exclusion or split grouping",
        "semantic_status": "canonical names and scale-side convention remain provisional pending source-team confirmation",
        "physical_measurement_status": "not available until scale length and unit are confirmed",
    }
    with (qa_root / "curation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return summary
