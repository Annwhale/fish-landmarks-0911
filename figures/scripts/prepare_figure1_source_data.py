#!/usr/bin/env python3
# ><(((o>  鱼类形态数据工具
"""Materialise Figure 1 source tables directly from the read-only SQLite copy.

This narrow builder avoids a Parquet dependency and does not compute hashes.
It records the two deterministic specimen selections used by Figure 1.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATABASE = ROOT / "data" / "benchmark_v1" / "fish_landmarks.sqlite"
OUT = ROOT / "figures" / "source_data"


def save(frame: pd.DataFrame, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / name, index=False)


def select_median_specimen(connection: sqlite3.Connection, taxon: str,
                           excluded: set[str]) -> pd.Series:
    query = """
        SELECT s.*, t.body_axis_length_px
        FROM benchmark_samples AS s
        JOIN crop_transforms AS t USING (benchmark_id)
        WHERE s.taxon_candidate = ? AND s.origin_label IN ('farmed', 'wild')
          AND s.width > s.height
    """
    frame = pd.read_sql_query(query, connection, params=(taxon,))
    frame = frame[~frame["benchmark_id"].isin(excluded)].copy()
    target = float(frame["body_axis_length_px"].median())
    return frame.assign(
        _distance=(frame["body_axis_length_px"] - target).abs()
    ).sort_values(["_distance", "benchmark_id"]).iloc[0]


def point_frame(connection: sqlite3.Connection, benchmark_id: str) -> pd.DataFrame:
    query = """
        SELECT o.benchmark_id, o.canonical_id, o.canonical_name_provisional,
               o.x AS x_source_px, o.y AS y_source_px,
               c.x_crop_px, c.y_crop_px, c.visibility
        FROM benchmark_keypoints_original AS o
        JOIN benchmark_keypoints_crops_448 AS c
          ON o.benchmark_id = c.benchmark_id AND o.canonical_id = c.canonical_id
        WHERE o.benchmark_id = ?
        ORDER BY o.canonical_id
    """
    return pd.read_sql_query(query, connection, params=(benchmark_id,))


def sample_frame(row: pd.Series) -> pd.DataFrame:
    return pd.DataFrame([{
        "benchmark_id": row["benchmark_id"],
        "source_groups": row["source_groups"],
        "source_display": f"{row['origin_label']} · {row['taxon_candidate']}",
        "taxon_candidate": row["taxon_candidate"],
        "origin_label": row["origin_label"],
        "width": int(row["width"]),
        "height": int(row["height"]),
        "representative_original_image": row["representative_original_image"],
    }])


def main() -> None:
    with sqlite3.connect(f"file:{DATABASE}?mode=ro", uri=True) as connection:
        workflow = select_median_specimen(connection, "Coilia nasus", set())
        trace = select_median_specimen(connection, "Siniperca chuatsi", {str(workflow["benchmark_id"])})
        save(sample_frame(workflow), "figure1_workflow_sample.csv")
        save(point_frame(connection, str(workflow["benchmark_id"])), "figure1_workflow_points.csv")
        save(sample_frame(trace), "figure1_trace_sample.csv")
        save(point_frame(connection, str(trace["benchmark_id"])), "figure1_trace_points.csv")

        ids = [str(workflow["benchmark_id"]), str(trace["benchmark_id"])]
        placeholders = ",".join("?" for _ in ids)
        transforms = pd.read_sql_query(
            f"SELECT * FROM crop_transforms WHERE benchmark_id IN ({placeholders}) ORDER BY benchmark_id",
            connection, params=ids,
        )
        save(transforms, "figure1_representative_transforms.csv")

        source_records = int(connection.execute("SELECT COUNT(*) FROM metadata").fetchone()[0])
        role_ready = int(connection.execute(
            "SELECT COUNT(*) FROM metadata WHERE fish_landmarks_1_to_11_complete = 1 AND canonical_mapping_complete = 1"
        ).fetchone()[0])
        benchmark = int(connection.execute("SELECT COUNT(*) FROM benchmark_samples").fetchone()[0])
        validation = int(connection.execute(
            "SELECT COUNT(*) FROM benchmark_samples WHERE technical_validation_eligible = 1"
        ).fetchone()[0])
        holdout = benchmark - validation
        flow = pd.DataFrame([
            {"stage": "source records", "count": source_records, "excluded_from_previous": 0},
            {"stage": "role-aware eligible records", "count": role_ready,
             "excluded_from_previous": source_records - role_ready},
            {"stage": "unique benchmark images", "count": benchmark,
             "excluded_from_previous": role_ready - benchmark},
            {"stage": "technical validation eligible", "count": validation,
             "excluded_from_previous": holdout},
            {"stage": "review holdout", "count": holdout, "excluded_from_previous": 0},
        ])
        save(flow, "figure1_flow.csv")

        split = pd.read_sql_query(
            """
            SELECT taxon_candidate,
                   CASE WHEN standard_split = 'review_holdout' THEN 'review holdout'
                        ELSE standard_split END AS split_display,
                   COUNT(*) AS count
            FROM benchmark_samples
            GROUP BY taxon_candidate, split_display
            ORDER BY taxon_candidate, split_display
            """,
            connection,
        )
        save(split, "figure1_split_taxon.csv")

        coverage = pd.read_sql_query(
            """
            SELECT source_groups, taxon_candidate, origin_label, COUNT(*) AS unique_images
            FROM benchmark_samples
            GROUP BY source_groups, taxon_candidate, origin_label
            ORDER BY taxon_candidate, origin_label, source_groups
            """,
            connection,
        )
        save(coverage, "figure1_source_coverage.csv")

        specs = [
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
        database_rows = [{
            "layer": layer,
            "table_name": table,
            "row_count": int(connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]),
            "logical_key": key,
        } for layer, table, key in specs]
        save(pd.DataFrame(database_rows), "figure1_database_tables.csv")

    registry_path = OUT / "figure_image_registry.csv"
    registry = pd.read_csv(registry_path) if registry_path.exists() else pd.DataFrame()
    if not registry.empty:
        registry = registry[registry["figure"] != "Figure 1"].copy()
    new_rows = pd.DataFrame([
        {"figure": "Figure 1", "panel": "a", "slot": "workflow-specimen",
         "benchmark_id": workflow["benchmark_id"], "source_groups": workflow["source_groups"],
         "taxon_candidate": workflow["taxon_candidate"], "origin_label": workflow["origin_label"],
         "selection_rule": "taxon median body-axis length"},
        {"figure": "Figure 1", "panel": "d", "slot": "coordinate-trace",
         "benchmark_id": trace["benchmark_id"], "source_groups": trace["source_groups"],
         "taxon_candidate": trace["taxon_candidate"], "origin_label": trace["origin_label"],
         "selection_rule": "taxon median body-axis length"},
    ])
    save(pd.concat([registry, new_rows], ignore_index=True), "figure_image_registry.csv")
    print(f"Figure 1 source data prepared from SQLite: {workflow['benchmark_id']}, {trace['benchmark_id']}")


if __name__ == "__main__":
    main()
