#!/usr/bin/env python3
# ><(((o>  鱼类形态数据工具
"""Materialise plot-ready figure sources without importing a graphics stack.

This separation lets the parquet-capable analysis environment produce CSV
source data while the frozen plotting environment remains Python/matplotlib
only.  No digest is computed and no image content is altered.
"""
from __future__ import annotations

from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "data" / "benchmark_v1"
TABLES = PACKAGE / "tables"
ANN = PACKAGE / "annotations"
OUT = ROOT / "figures" / "source_data"
ZERO_RESULTS = ROOT / "models" / "evaluation" / "locateanything_3b_zero_shot" / "query_predictions.csv"
THRESHOLD = 0.02
USED_IDS: set[str] = set()
REGISTRY_ROWS: list[dict[str, object]] = []
TAXON_SHORT = {
    "Carassius auratus": "C. auratus",
    "Coilia nasus": "C. nasus",
    "Ctenopharyngodon idella": "C. idella",
    "Culter alburnus": "C. alburnus",
    "Hypophthalmichthys molitrix": "H. molitrix",
    "Hypophthalmichthys nobilis": "H. nobilis",
    "Megalobrama amblycephala": "M. amblycephala",
    "Mylopharyngodon piceus": "M. piceus",
    "Siniperca chuatsi": "S. chuatsi",
}
POINT_NAMES = {
    1: "snout tip", 2: "nape dorsal outline", 3: "dorsal fin origin",
    4: "dorsal fin insertion", 5: "upper caudal fin insertion",
    6: "caudal peduncle midpoint", 7: "lower caudal fin insertion",
    8: "anal fin origin", 9: "pelvic fin origin", 10: "pectoral fin base",
    11: "posterior opercular margin",
}


def source_display(taxon: str, origin: str) -> str:
    prefix = {"farmed": "F", "wild": "W", "conflicted": "C"}.get(str(origin), "?")
    return f"{prefix} {TAXON_SHORT.get(str(taxon), str(taxon))}"


def available(frame: pd.DataFrame) -> pd.DataFrame:
    """Return rows whose benchmark image is not used by another main-figure slot."""
    return frame[~frame["benchmark_id"].astype(str).isin(USED_IDS)].copy()


def reserve(row: pd.Series, figure: str, panel: str, slot: str, rule: str) -> pd.Series:
    benchmark_id = str(row["benchmark_id"])
    if benchmark_id in USED_IDS:
        raise RuntimeError(f"duplicate main-figure image reservation: {benchmark_id}")
    USED_IDS.add(benchmark_id)
    REGISTRY_ROWS.append({
        "figure": figure,
        "panel": panel,
        "slot": slot,
        "benchmark_id": benchmark_id,
        "source_groups": row.get("source_groups", ""),
        "taxon_candidate": row.get("taxon_candidate", ""),
        "origin_label": row.get("origin_label", ""),
        "selection_rule": rule,
    })
    return row


def save(frame: pd.DataFrame, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / name, index=False)


def prepare_figure5_selection() -> None:
    frame = pd.read_csv(ZERO_RESULTS)
    per_image = frame.groupby("benchmark_id", sort=True).agg(
        mean_nme=("nme_role1_role6", "mean"),
        source_groups=("source_groups", "first"),
        taxon_candidate=("taxon_candidate", "first"),
        origin_label=("origin_label", "first"),
    ).reset_index()
    per_image["origin_order"] = per_image["origin_label"].map({"farmed": 0, "wild": 1}).fillna(2)
    picks = []
    ordered_groups = per_image[["source_groups", "taxon_candidate", "origin_order"]].drop_duplicates().sort_values(
        ["taxon_candidate", "origin_order", "source_groups"]
    )
    for slot_index, group_row in enumerate(ordered_groups.itertuples(index=False), start=1):
        group = per_image[per_image["source_groups"] == group_row.source_groups].copy()
        target = float(group["mean_nme"].median())
        pick = group.assign(_distance=(group["mean_nme"] - target).abs()).sort_values(
            ["_distance", "benchmark_id"]
        ).iloc[0]
        picks.append({
            "benchmark_id": pick["benchmark_id"],
            "source_groups": pick["source_groups"],
            "taxon_candidate": pick["taxon_candidate"],
            "origin_label": pick["origin_label"],
            "source_display": source_display(pick["taxon_candidate"], pick["origin_label"]),
            "mean_nme": float(pick["mean_nme"]),
            "selection_rule": "nearest to within-source-group median zero-shot mean NME",
        })
    selected = pd.DataFrame(picks)
    selected["origin_order"] = selected["origin_label"].map({"farmed": 0, "wild": 1}).fillna(2)
    selected = selected.sort_values(["taxon_candidate", "origin_order", "source_groups"]).drop(
        columns="origin_order"
    ).reset_index(drop=True)
    if len(selected) != 13:
        raise RuntimeError(f"expected 13 non-conflicted zero-shot source groups, found {len(selected)}")
    save(selected, "figure5_selected_images.csv")

    # 展示六个不同样例；完整13组来源仍由数据表保留。
    display_ids = [
        "flm_0210f67ad1497c7f",  # wild H. molitrix
        "flm_48e75883e252235a",  # farmed H. nobilis
        "flm_009d9c9b591675ff",  # wild C. nasus
        "flm_08e88d7fcddf132c",  # farmed C. idella
        "flm_0e1305877a8ced63",  # farmed S. chuatsi
        "flm_3e0cf2609cc04388",  # farmed M. amblycephala
    ]
    display = selected.set_index("benchmark_id").loc[display_ids].reset_index()
    display["selection_rule"] = (
        "prespecified six-taxon display spanning the observed zero-shot error range"
    )
    for slot_index, (_, item) in enumerate(display.iterrows(), start=1):
        reserve(item, "Figure 5", "c", f"display-{slot_index:02d}",
                str(item["selection_rule"]))
    save(display, "figure5_display_images.csv")


def prepare_figure1() -> None:
    samples = pd.read_parquet(TABLES / "benchmark_samples.parquet")
    meta = pd.read_parquet(TABLES / "metadata.parquet")
    points = pd.read_parquet(TABLES / "benchmark_keypoints.parquet")
    crops = pd.read_parquet(ANN / "keypoints_crops_448.parquet")
    transforms = pd.read_parquet(ANN / "crop_transforms.parquet")

    merged = samples.merge(transforms[["benchmark_id", "body_axis_length_px"]],
                           on="benchmark_id", how="left")
    merged = merged[merged["origin_label"].isin(["farmed", "wild"])].copy()
    merged["origin_order"] = merged["origin_label"].map({"farmed": 0, "wild": 1})

    # 图1保留两张实拍图，来源覆盖用计数表示。
    workflow_pool = available(merged[merged["taxon_candidate"] == "Coilia nasus"])
    workflow_target = float(workflow_pool["body_axis_length_px"].median())
    workflow_pick = workflow_pool.assign(
        _distance=(workflow_pool["body_axis_length_px"] - workflow_target).abs()
    ).sort_values(["_distance", "benchmark_id"]).iloc[0]
    workflow_pick = reserve(
        workflow_pick, "Figure 1", "a", "workflow-specimen",
        "unused image nearest to taxon median body-axis length",
    )
    workflow_sample = pd.DataFrame([{
        "benchmark_id": workflow_pick["benchmark_id"],
        "source_groups": workflow_pick["source_groups"],
        "source_display": source_display(workflow_pick["taxon_candidate"], workflow_pick["origin_label"]),
        "taxon_candidate": workflow_pick["taxon_candidate"],
        "origin_label": workflow_pick["origin_label"],
        "width": int(workflow_pick["width"]), "height": int(workflow_pick["height"]),
        "representative_original_image": workflow_pick["representative_original_image"],
    }])
    save(workflow_sample, "figure1_workflow_sample.csv")
    workflow_id = str(workflow_pick["benchmark_id"])
    workflow_points = crops[crops["benchmark_id"] == workflow_id].merge(
        points[["benchmark_id", "canonical_id", "x", "y"]],
        on=["benchmark_id", "canonical_id"], how="left",
    ).rename(columns={"x": "x_source_px", "y": "y_source_px"})
    save(workflow_points[[
        "benchmark_id", "canonical_id", "canonical_name_provisional", "x_source_px",
        "y_source_px", "x_crop_px", "y_crop_px", "visibility",
    ]].sort_values(["benchmark_id", "canonical_id"]), "figure1_workflow_points.csv")

    trace_pool = available(merged[merged["taxon_candidate"] == "Siniperca chuatsi"])
    trace_target = float(trace_pool["body_axis_length_px"].median())
    trace_pick = trace_pool.assign(_distance=(trace_pool["body_axis_length_px"] - trace_target).abs()).sort_values(
        ["_distance", "benchmark_id"]
    ).iloc[0]
    trace_pick = reserve(trace_pick, "Figure 1", "d", "coordinate-trace",
                         "unused image nearest to taxon median body-axis length")
    trace_sample = pd.DataFrame([{
        "benchmark_id": trace_pick["benchmark_id"],
        "source_groups": trace_pick["source_groups"],
        "source_display": source_display(trace_pick["taxon_candidate"], trace_pick["origin_label"]),
        "taxon_candidate": trace_pick["taxon_candidate"],
        "origin_label": trace_pick["origin_label"],
        "width": int(trace_pick["width"]), "height": int(trace_pick["height"]),
        "representative_original_image": trace_pick["representative_original_image"],
    }])
    save(trace_sample, "figure1_trace_sample.csv")
    trace_id = str(trace_pick["benchmark_id"])
    trace = crops[crops["benchmark_id"] == trace_id].merge(
        points[["benchmark_id", "canonical_id", "x", "y"]],
        on=["benchmark_id", "canonical_id"], how="left"
    ).rename(columns={"x": "x_source_px", "y": "y_source_px"})
    save(trace[[
        "benchmark_id", "canonical_id", "canonical_name_provisional", "x_source_px",
        "y_source_px", "x_crop_px", "y_crop_px", "visibility",
    ]].sort_values(["benchmark_id", "canonical_id"]), "figure1_trace_points.csv")

    transform_columns = [
        "benchmark_id", "affine_00", "affine_01", "affine_02", "affine_10",
        "affine_11", "affine_12", "source_width", "source_height", "output_size",
        "body_axis_length_px",
    ]
    transform_ids = [workflow_id, trace_id]
    save(transforms[transforms["benchmark_id"].isin(transform_ids)][transform_columns].sort_values("benchmark_id"),
         "figure1_representative_transforms.csv")

    eligible = int((meta["fish_landmarks_1_to_11_complete"] & meta["canonical_mapping_complete"]).sum())
    benchmark = int(len(samples))
    validation = int(samples["technical_validation_eligible"].fillna(False).sum())
    holdout = benchmark - validation
    flow = pd.DataFrame([
        {"stage": "source records", "count": int(len(meta)), "excluded_from_previous": 0,
         "denominator_note": "all audited source records"},
        {"stage": "role-aware eligible records", "count": eligible,
         "excluded_from_previous": int(len(meta) - eligible),
         "denominator_note": "11 fish roles and complete canonical mapping"},
        {"stage": "unique benchmark images", "count": benchmark,
         "excluded_from_previous": int(eligible - benchmark),
         "denominator_note": "duplicate consensus and taxon-conflict resolution"},
        {"stage": "technical validation eligible", "count": validation,
         "excluded_from_previous": holdout,
         "denominator_note": "review holdout retained but not in locked technical cohort"},
        {"stage": "review holdout", "count": holdout, "excluded_from_previous": 0,
         "denominator_note": "high repeated-annotation disagreement"},
    ])
    save(flow, "figure1_flow.csv")
    split = samples.assign(split_display=samples["standard_split"].replace(
        {"review_holdout": "review holdout"}
    )).groupby(["taxon_candidate", "split_display"], dropna=False).size().reset_index(name="count")
    split["taxon_display"] = split["taxon_candidate"].map(TAXON_SHORT).fillna(split["taxon_candidate"])
    save(split[["taxon_candidate", "taxon_display", "split_display", "count"]],
         "figure1_split_taxon.csv")

    source_coverage = samples.groupby(
        ["source_groups", "taxon_candidate", "origin_label"], dropna=False
    ).size().reset_index(name="unique_images")
    source_coverage["taxon_display"] = source_coverage["taxon_candidate"].map(TAXON_SHORT).fillna(
        source_coverage["taxon_candidate"]
    )
    save(source_coverage.sort_values(["taxon_candidate", "origin_label", "source_groups"]),
         "figure1_source_coverage.csv")

    database_specs = [
        ("Source audit", "metadata", "sample_uid"),
        ("Source audit", "keypoints_raw_long", "sample_uid"),
        ("Source audit", "scale_endpoints_long", "sample_uid"),
        ("Curation and QC", "anomalies", "sample_uid"),
        ("Curation and QC", "exact_duplicates", "content_id / sample_uid"),
        ("Curation and QC", "duplicate_annotation_agreement", "content_id"),
        ("Curation and QC", "near_duplicate_review_candidates", "benchmark_id pair"),
        ("Curation and QC", "benchmark_exclusions", "sample_uid"),
        ("Benchmark core", "benchmark_samples", "benchmark_id"),
        ("Benchmark core", "crop_transforms", "benchmark_id"),
        ("Coordinate records", "benchmark_keypoints_original", "benchmark_id / role"),
        ("Coordinate records", "benchmark_keypoints_crops_448", "benchmark_id / role"),
    ]
    database_path = PACKAGE / "fish_landmarks.sqlite"
    with sqlite3.connect(f"file:{database_path}?mode=ro", uri=True) as connection:
        database_rows = [
            {
                "layer": layer,
                "table_name": table,
                "row_count": int(connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]),
                "logical_key": key,
            }
            for layer, table, key in database_specs
        ]
    save(pd.DataFrame(database_rows), "figure1_database_tables.csv")


def diverse_quantile_examples(valid: pd.DataFrame, count: int = 8) -> pd.DataFrame:
    valid = valid.sort_values(["pairwise_mean_error_image_diagonal", "benchmark_id"]).reset_index(drop=True)
    targets = np.linspace(0.0, 1.0, count)
    selected_rows: list[pd.Series] = []
    used_taxa: set[str] = set()
    used_ids: set[str] = set()
    for target in targets:
        target_rank = target * (len(valid) - 1)
        candidates = valid.assign(_distance=(valid.index.to_numpy() - target_rank).astype(float).__abs__())
        pool = candidates[
            ~candidates["benchmark_id"].astype(str).isin(used_ids | USED_IDS)
        ].copy()
        if target not in {0.0, 1.0}:
            pool["_diversity_penalty"] = np.where(
                pool["taxon_candidate"].astype(str).isin(used_taxa), float(len(valid)), 0.0
            )
        else:
            pool["_diversity_penalty"] = 0.0
        pool["_selection_cost"] = pool["_distance"] + pool["_diversity_penalty"]
        pick = pool.sort_values(["_selection_cost", "_distance", "benchmark_id"]).iloc[0].copy()
        pick["selection"] = f"q{int(round(target * 100)):02d}"
        pick["target_quantile"] = float(target)
        selected_rows.append(pick)
        used_taxa.add(str(pick["taxon_candidate"]))
        used_ids.add(str(pick["benchmark_id"]))
    return pd.DataFrame(selected_rows).drop(
        columns=["_distance", "_diversity_penalty", "_selection_cost"], errors="ignore"
    )


def prepare_figure2() -> None:
    meta = pd.read_parquet(TABLES / "metadata.parquet")
    agreement = pd.read_parquet(TABLES / "duplicate_annotation_agreement.parquet").drop(
        columns=["content_id_blake2b64"], errors="ignore"
    )
    near = pd.read_parquet(TABLES / "near_duplicate_review_candidates.parquet")
    samples = pd.read_parquet(TABLES / "benchmark_samples.parquet")
    keypoints = pd.read_parquet(TABLES / "benchmark_keypoints.parquet")
    canonical = pd.read_parquet(TABLES / "keypoints_canonical_long.parquet")

    discrepancy = meta[[
        "sample_uid", "source_group_code", "taxon_candidate", "origin_label_source",
        "csv_yolo_max_error_px",
    ]].rename(columns={"csv_yolo_max_error_px": "stored_coordinate_discrepancy"})
    discrepancy = discrepancy[discrepancy["stored_coordinate_discrepancy"].notna()].copy()
    save(discrepancy, "figure2_csv_yolo_discrepancy.csv")
    grouped = discrepancy.groupby([
        "source_group_code", "taxon_candidate", "origin_label_source",
    ])["stored_coordinate_discrepancy"].agg(
        count="count", minimum="min", q1=lambda x: x.quantile(0.25), median="median",
        q3=lambda x: x.quantile(0.75), maximum="max",
    ).reset_index()
    grouped["display_label"] = grouped.apply(
        lambda r: ("F " if r.origin_label_source == "farmed" else "W ") +
                  TAXON_SHORT.get(r.taxon_candidate, r.taxon_candidate), axis=1
    )
    save(grouped, "figure2_discrepancy_by_group.csv")

    agreement["review_threshold"] = THRESHOLD
    agreement["above_threshold"] = agreement["pairwise_mean_error_image_diagonal"] >= THRESHOLD
    save(agreement, "figure2_duplicate_annotation_agreement.csv")

    repeat = keypoints[keypoints["consensus_record_count"] > 1].copy()
    repeat["radial_mad_px"] = np.hypot(repeat["x_mad_px"].astype(float), repeat["y_mad_px"].astype(float))
    repeat["point_label"] = repeat["canonical_id"].map(POINT_NAMES)
    save(repeat[[
        "benchmark_id", "canonical_id", "point_label", "consensus_record_count",
        "x_mad_px", "y_mad_px", "radial_mad_px",
    ]], "figure2_point_uncertainty.csv")

    reliability = pd.read_parquet(
        ROOT / "qa" / "repeat_annotation_validation" / "per_keypoint_reliability.parquet"
    ).rename(columns={
        "technical_repeat_pairs": "pair_count",
        "nme_image_diagonal_ci95_low": "ci95_low",
        "nme_image_diagonal_ci95_high": "ci95_high",
    })
    reliability["point_label"] = reliability["canonical_id"].map(POINT_NAMES)
    save(reliability, "figure2_repeat_point_summary.csv")
    modes = meta["canonical_mapping_mode"].fillna("missing").value_counts().rename_axis(
        "mapping_mode"
    ).reset_index(name="count")
    save(modes, "figure2_mapping_counts.csv")
    save(near, "figure2_near_duplicate_candidates.csv")
    save(meta[["sample_uid", "original_image", "width", "height"]], "figure2_image_crosswalk.csv")


def prepare_figure4() -> None:
    samples = pd.read_parquet(TABLES / "benchmark_samples.parquet")
    points = pd.read_parquet(TABLES / "benchmark_keypoints.parquet")
    transforms = pd.read_parquet(ANN / "crop_transforms.parquet")
    cross = pd.read_csv(
        ROOT / "models" / "evaluation" / "yolo26m_cross_origin_original_seed20260825" /
        "test" / "per_image_metrics.csv"
    )
    taxa = sorted(cross["taxon_candidate"].dropna().unique())
    merged = samples[samples["taxon_candidate"].isin(taxa)].merge(
        transforms[["benchmark_id", "body_axis_length_px"]], on="benchmark_id", how="left"
    )
    examples = []
    for taxon in taxa:
        for origin in ["farmed", "wild"]:
            group = available(merged[(merged["taxon_candidate"] == taxon) & (merged["origin_label"] == origin)])
            if group.empty:
                continue
            median_axis = float(group["body_axis_length_px"].median())
            pick = group.assign(_distance=(group["body_axis_length_px"] - median_axis).abs()).sort_values(
                ["_distance", "benchmark_id"]
            ).iloc[0]
            pick = reserve(pick, "Figure 4", "a", f"{taxon}-{origin}",
                           "nearest unused image to taxon-origin median body-axis length")
            examples.append({
                "benchmark_id": pick["benchmark_id"], "taxon_candidate": taxon,
                "taxon_display": TAXON_SHORT.get(taxon, taxon), "origin_label": origin,
                "body_axis_length_px": float(pick["body_axis_length_px"]),
                "taxon_origin_median_body_axis_length_px": median_axis,
                "selection_rule": "nearest to taxon-origin median body-axis length",
            })
    example_frame = pd.DataFrame(examples).sort_values(["taxon_candidate", "origin_label"])
    save(example_frame, "figure4_domain_examples.csv")
    selected_points = points[points["benchmark_id"].isin(example_frame["benchmark_id"])].copy()
    save(selected_points[["benchmark_id", "canonical_id", "x", "y"]].sort_values(
        ["benchmark_id", "canonical_id"]
    ), "figure4_domain_example_points.csv")


def prepare_figure3() -> None:
    samples = pd.read_parquet(TABLES / "benchmark_samples.parquet")
    points = pd.read_parquet(TABLES / "benchmark_keypoints.parquet")
    transforms = pd.read_parquet(ANN / "crop_transforms.parquet")
    merged = samples.merge(transforms[["benchmark_id", "body_axis_length_px"]], on="benchmark_id", how="left")
    specs = [
        ("axial", "axial span 1–6", 1, 6, "Carassius auratus"),
        ("dorsal", "dorsal base 3–4", 3, 4, "Siniperca chuatsi"),
        ("caudal", "caudal depth 5–7", 5, 7, "Coilia nasus"),
        ("head", "head chord 2–10", 2, 10, "Megalobrama amblycephala"),
        ("ventral", "ventral base 8–9", 8, 9, "Ctenopharyngodon idella"),
    ]
    rows = []
    for slot_index, (key, label, first, second, taxon) in enumerate(specs, start=1):
        group = available(merged[(merged["taxon_candidate"] == taxon) &
                                 merged["technical_validation_eligible"].fillna(False)])
        target = float(group["body_axis_length_px"].median())
        pick = group.assign(_distance=(group["body_axis_length_px"] - target).abs()).sort_values(
            ["_distance", "benchmark_id"]
        ).iloc[0]
        pick = reserve(pick, "Figure 3", "a", f"geometry-{slot_index:02d}",
                       "unused taxon-specific image nearest to median body-axis length")
        rows.append({
            "geometry_key": key, "geometry_label": label,
            "first_role": first, "second_role": second,
            "benchmark_id": pick["benchmark_id"],
            "source_groups": pick["source_groups"],
            "source_display": source_display(pick["taxon_candidate"], pick["origin_label"]),
            "taxon_candidate": pick["taxon_candidate"], "origin_label": pick["origin_label"],
            "representative_original_image": pick["representative_original_image"],
            "width": int(pick["width"]), "height": int(pick["height"]),
        })
    examples = pd.DataFrame(rows)
    save(examples, "figure3_geometry_examples.csv")
    selected_points = points[points["benchmark_id"].isin(examples["benchmark_id"])][
        ["benchmark_id", "canonical_id", "x", "y"]
    ]
    save(selected_points.sort_values(["benchmark_id", "canonical_id"]),
         "figure3_geometry_example_points.csv")


def main() -> None:
    USED_IDS.clear(); REGISTRY_ROWS.clear()
    prepare_figure5_selection()
    prepare_figure1()
    prepare_figure2()
    registry = pd.DataFrame(REGISTRY_ROWS)
    if registry["benchmark_id"].duplicated().any():
        raise RuntimeError("main-figure image registry contains repeated benchmark IDs")
    save(registry, "figure_image_registry.csv")
    print(f"Prepared plot-ready sources with {len(registry)} mutually exclusive main-figure images")


if __name__ == "__main__":
    main()
