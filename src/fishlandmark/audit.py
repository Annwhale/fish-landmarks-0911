# ><(((o>  鱼类形态数据工具
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageFile

from .io import Point, parse_keypoint_csv, parse_yolo_pose, select_fish_and_scale_rows

ImageFile.LOAD_TRUNCATED_IMAGES = False


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def content_id(path: Path) -> str:
    digest = hashlib.blake2b(digest_size=8)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def difference_hash(image: Image.Image) -> str:
    gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray, dtype=np.int16)
    bits = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in bits.ravel():
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def polygon_area(points: list[Point]) -> float | None:
    by_id = {point.point_id: point for point in points}
    if not all(point_id in by_id for point_id in range(1, 11)):
        return None
    contour = [by_id[point_id] for point_id in range(1, 11)]
    return 0.5 * sum(
        contour[idx].x * contour[(idx + 1) % 10].y
        - contour[(idx + 1) % 10].x * contour[idx].y
        for idx in range(10)
    )


def match_yolo_role_points(
    row: Any | None,
    csv_points: list[Point],
    width: int | None,
    height: int | None,
    tolerance_px: float = 1.5,
) -> list[dict[str, Any]]:
    """Assign YOLO row positions to CSV coordinates without trusting raw CSV IDs.

    Some source exports contain 10 fish points followed by scale points with raw
    IDs 11 and 12, whereas the dominant format contains 11 fish points followed
    by scale points 12 and 13. The YOLO rows preserve the object role, so role
    IDs are derived from row position and matched back to exact CSV pixels.
    """
    if row is None or width is None or height is None:
        return []
    unused = {point.point_id: point for point in csv_points}
    matched: list[dict[str, Any]] = []
    for role_id, (x_norm, y_norm, visibility) in enumerate(row.keypoints, start=1):
        if x_norm == 0.0 and y_norm == 0.0:
            continue
        x_px = float(x_norm * width)
        y_px = float(y_norm * height)
        nearest_id = None
        nearest_error = float("inf")
        for point_id, point in unused.items():
            error = math.hypot(point.x - x_px, point.y - y_px)
            if error < nearest_error:
                nearest_id = point_id
                nearest_error = error
        source_point = unused.pop(nearest_id) if nearest_id is not None and nearest_error <= tolerance_px else None
        matched.append(
            {
                "role_id": role_id,
                "source_csv_point_id": source_point.point_id if source_point is not None else None,
                "x": source_point.x if source_point is not None else x_px,
                "y": source_point.y if source_point is not None else y_px,
                "visibility": visibility,
                "csv_yolo_error_px": nearest_error if source_point is not None else None,
                "matched_within_tolerance": source_point is not None,
            }
        )
    return matched


def mapping_assessment(points: list[Point]) -> tuple[str, str, float | None]:
    """Conservative orientation assessment for contour-ordered raw points.

    When the photographic scale endpoints are present, their side of the
    snout-to-tail axis supplies a rotation-invariant acquisition cue. Source
    images place the scale on the visually dorsal side of the specimen; this
    convention still requires provenance confirmation and is therefore stored
    as a mapping reason rather than silently assumed. Horizontal specimens
    without scale points fall back to image-up geometry. Strongly oblique
    specimens without a scale remain unresolved.
    """
    by_id = {point.point_id: point for point in points}
    required = {1, 2, 6, 10}
    if not required.issubset(by_id):
        return "unresolved", "missing_required_points", None
    snout, tail = by_id[1], by_id[6]
    dx, dy = tail.x - snout.x, tail.y - snout.y
    angle = math.degrees(math.atan2(dy, dx))
    if {12, 13}.issubset(by_id):
        scale_x = 0.5 * (by_id[12].x + by_id[13].x)
        scale_y = 0.5 * (by_id[12].y + by_id[13].y)
        branch2_cross = dx * (by_id[2].y - snout.y) - dy * (by_id[2].x - snout.x)
        branch10_cross = dx * (by_id[10].y - snout.y) - dy * (by_id[10].x - snout.x)
        scale_cross = dx * (scale_y - snout.y) - dy * (scale_x - snout.x)
        if abs(scale_cross) > 1e-6 and branch2_cross * branch10_cross < 0:
            mode = "direct" if branch2_cross * scale_cross > 0 else "reverse"
            return mode, "scale_side_geometry_unconfirmed_convention", angle
    horizontal_score = abs(dx) / max(math.hypot(dx, dy), 1e-9)
    if horizontal_score < 0.70:
        return "ambiguous", "oblique_or_vertical_body_axis", angle
    mode = "direct" if by_id[2].y < by_id[10].y else "reverse"
    return mode, "horizontal_branch_geometry", angle


def canonicalize(points: list[Point], mode: str, config: dict[str, Any]) -> list[dict[str, Any]]:
    by_id = {point.point_id: point for point in points}
    if mode not in {"direct", "reverse"}:
        return []
    mapping = {int(key): int(value) for key, value in config["mapping"][mode].items()}
    names = {int(key): value for key, value in config["canonical_landmarks"].items()}
    output = []
    for canonical_id, raw_id in mapping.items():
        point = by_id.get(raw_id)
        if point is None:
            continue
        output.append(
            {
                "canonical_id": canonical_id,
                "canonical_name": names[canonical_id],
                "raw_id": raw_id,
                "x": point.x,
                "y": point.y,
            }
        )
    return output


def database_ids(database: Path) -> dict[tuple[str, str], int]:
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            """
            SELECT s.species_name, fs.sample_name, fs.id
            FROM fish_samples fs JOIN species s ON s.id = fs.species_id
            """
        ).fetchall()
    finally:
        connection.close()
    return {(str(group), str(name)): int(sample_id) for group, name, sample_id in rows}


def run_audit(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    dataset = config["dataset"]
    source_root = Path(dataset["source_root"])
    output_root = Path(dataset["output_root"])
    qa_root = Path(dataset["qa_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    qa_root.mkdir(parents=True, exist_ok=True)
    db_ids = database_ids(Path(dataset["source_database"]))
    identity_cache: dict[tuple[str, int, int, int], tuple[str | None, str | None]] = {}
    prior_metadata_path = output_root / "metadata.parquet"
    if prior_metadata_path.exists():
        prior = pd.read_parquet(prior_metadata_path)
        required_cache_columns = {
            "original_image", "file_bytes", "width", "height",
            "content_id_blake2b64", "difference_hash_64",
        }
        if required_cache_columns.issubset(prior.columns):
            for row in prior.itertuples(index=False):
                if pd.notna(row.file_bytes) and pd.notna(row.width) and pd.notna(row.height):
                    identity_cache[(
                        str(row.original_image), int(row.file_bytes), int(row.width), int(row.height)
                    )] = (
                        row.content_id_blake2b64 if pd.notna(row.content_id_blake2b64) else None,
                        row.difference_hash_64 if pd.notna(row.difference_hash_64) else None,
                    )

    metadata: list[dict[str, Any]] = []
    raw_points: list[dict[str, Any]] = []
    canonical_points: list[dict[str, Any]] = []
    scale_points: list[dict[str, Any]] = []
    anomalies: list[dict[str, Any]] = []

    groups = config["source_groups"]
    for group_name in sorted(groups):
        group_dir = source_root / group_name
        group_meta = groups[group_name]
        csv_paths = sorted(group_dir.glob("*_result.csv"), key=lambda item: item.name)
        for ordinal, csv_path in enumerate(csv_paths, start=1):
            base = csv_path.name[: -len("_result.csv")]
            sample_db_id = db_ids.get((group_name, base))
            sample_uid = f"{group_meta['code']}__{ordinal:05d}"
            original = group_dir / f"{base}_0.jpg"
            rendered = group_dir / f"{base}_result.jpg"
            yolo_path = group_dir / f"{base}_0.txt"
            points = parse_keypoint_csv(csv_path)
            yolo_rows = parse_yolo_pose(yolo_path) if yolo_path.exists() else []
            fish_row, scale_row = select_fish_and_scale_rows(yolo_rows)

            image_error = None
            width = height = None
            digest = dhash = None
            try:
                with Image.open(original) as image:
                    image.load()
                    width, height = image.size
                    relative_original = str(original.relative_to(source_root))
                    cache_key = (relative_original, original.stat().st_size, width, height)
                    cached_digest, cached_dhash = identity_cache.get(cache_key, (None, None))
                    dhash = cached_dhash or difference_hash(image)
                digest = cached_digest or content_id(original)
            except Exception as exc:  # recorded, never hidden
                image_error = f"{type(exc).__name__}: {exc}"

            point_ids = {point.point_id for point in points}
            fish_matches = match_yolo_role_points(fish_row, points, width, height)
            scale_matches = match_yolo_role_points(scale_row, points, width, height)
            fish_points = [Point(int(item["role_id"]), float(item["x"]), float(item["y"])) for item in fish_matches]
            normalized_scale_points = [
                Point(11 + int(item["role_id"]), float(item["x"]), float(item["y"]))
                for item in scale_matches[:2]
            ]
            mapping_mode, mapping_reason, axis_angle = mapping_assessment(
                fish_points + normalized_scale_points
            )
            canonical = canonicalize(fish_points, mapping_mode, config)
            fish_role_to_raw = {
                int(item["role_id"]): item["source_csv_point_id"] for item in fish_matches
            }
            for item in canonical:
                item["fish_source_order_id"] = item.pop("raw_id")
                item["source_csv_point_id"] = fish_role_to_raw.get(item["fish_source_order_id"])
            area = polygon_area(fish_points)
            matched_errors = [
                float(item["csv_yolo_error_px"])
                for item in fish_matches + scale_matches
                if item["csv_yolo_error_px"] is not None
            ]
            fish_scale_agreement = max(matched_errors) if matched_errors else None

            raw_roles: dict[int, tuple[str, int]] = {}
            for item in fish_matches:
                if item["source_csv_point_id"] is not None:
                    raw_roles[int(item["source_csv_point_id"])] = (
                        "fish_landmark_source_order", int(item["role_id"])
                    )
            for item in scale_matches:
                if item["source_csv_point_id"] is not None:
                    raw_roles[int(item["source_csv_point_id"])] = (
                        "scale_endpoint", int(item["role_id"])
                    )

            for point in points:
                point_role, role_point_id = raw_roles.get(
                    point.point_id, ("unassigned_raw_point", point.point_id)
                )
                raw_points.append(
                    {
                        "sample_uid": sample_uid,
                        "source_group": group_name,
                        "source_csv_point_id": point.point_id,
                        "point_role": point_role,
                        "role_point_id": role_point_id,
                        "x": point.x,
                        "y": point.y,
                    }
                )
            for point in canonical:
                canonical_points.append({"sample_uid": sample_uid, **point})
            for point in scale_matches[:2]:
                scale_points.append(
                    {
                        "sample_uid": sample_uid,
                        "scale_endpoint_id": int(point["role_id"]),
                        "source_csv_point_id": point["source_csv_point_id"],
                        "x": float(point["x"]),
                        "y": float(point["y"]),
                        "scale_length_value": None,
                        "scale_unit": None,
                    }
                )

            reasons = []
            if len(points) != 13:
                reasons.append(f"csv_point_count_{len(points)}")
            if len(fish_matches) != 11:
                reasons.append(f"fish_landmark_count_{len(fish_matches)}")
            if len(scale_matches) != 2:
                reasons.append(f"scale_endpoint_count_{len(scale_matches)}")
            if any(not item["matched_within_tolerance"] for item in fish_matches + scale_matches):
                reasons.append("yolo_csv_role_match_incomplete")
            if len(fish_matches) == 10 and [item["source_csv_point_id"] for item in scale_matches[:2]] == [11, 12]:
                reasons.append("source_variant_10_fish_plus_raw_11_12_scale")
            if len(yolo_rows) != 2:
                reasons.append(f"yolo_row_count_{len(yolo_rows)}")
            if fish_row is None:
                reasons.append("missing_fish_pose_row")
            if scale_row is None:
                reasons.append("missing_scale_row")
            if image_error:
                reasons.append("image_decode_error")
            if fish_scale_agreement is not None and fish_scale_agreement > 1.5:
                reasons.append("csv_yolo_coordinate_disagreement")
            if mapping_mode == "ambiguous":
                reasons.append("canonical_mapping_ambiguous")
            if area is not None and area <= 0:
                reasons.append("nonpositive_raw_contour_orientation")

            metadata.append(
                {
                    "sample_uid": sample_uid,
                    "database_sample_id": sample_db_id,
                    "source_group": group_name,
                    "source_group_code": group_meta["code"],
                    "taxon_candidate": group_meta["taxon"],
                    "origin_label_source": group_meta["origin_label"],
                    "source_basename": base,
                    "original_image": str(original.relative_to(source_root)),
                    "rendered_image": str(rendered.relative_to(source_root)),
                    "csv_annotation": str(csv_path.relative_to(source_root)),
                    "yolo_annotation": str(yolo_path.relative_to(source_root)),
                    "width": width,
                    "height": height,
                    "file_bytes": original.stat().st_size if original.exists() else None,
                    "content_id_blake2b64": digest,
                    "difference_hash_64": dhash,
                    "raw_point_count": len(points),
                    "fish_landmarks_1_to_10_complete": len(fish_matches) >= 10,
                    "fish_landmarks_1_to_11_complete": len(fish_matches) == 11,
                    "scale_endpoints_1_to_2_complete": len(scale_matches) == 2,
                    "legacy_raw_ids_1_to_11_present": set(range(1, 12)).issubset(point_ids),
                    "legacy_raw_ids_12_to_13_present": {12, 13}.issubset(point_ids),
                    "yolo_row_count": len(yolo_rows),
                    "fish_yolo_nonzero_points": fish_row.nonzero_keypoints if fish_row else None,
                    "scale_yolo_nonzero_points": scale_row.nonzero_keypoints if scale_row else None,
                    "csv_yolo_max_error_px": fish_scale_agreement,
                    "raw_polygon_signed_area": area,
                    "body_axis_angle_deg": axis_angle,
                    "canonical_mapping_mode": mapping_mode,
                    "canonical_mapping_reason": mapping_reason,
                    "canonical_mapping_complete": len(fish_matches) == 11 and len(canonical) == 11,
                    "image_decode_error": image_error,
                    "anomaly_count": len(reasons),
                    "anomaly_codes": ";".join(reasons),
                }
            )
            for reason in reasons:
                anomalies.append(
                    {
                        "sample_uid": sample_uid,
                        "source_group": group_name,
                        "source_basename": base,
                        "anomaly_code": reason,
                    }
                )

    metadata_df = pd.DataFrame(metadata)
    raw_df = pd.DataFrame(raw_points)
    canonical_df = pd.DataFrame(canonical_points)
    scale_df = pd.DataFrame(scale_points)
    anomaly_df = pd.DataFrame(anomalies)

    duplicate_map: dict[str, list[str]] = defaultdict(list)
    for row in metadata:
        if row["content_id_blake2b64"]:
            duplicate_map[row["content_id_blake2b64"]].append(row["sample_uid"])
    duplicate_groups = []
    duplicate_group_by_sample: dict[str, str] = {}
    group_counter = 0
    for digest, sample_uids in sorted(duplicate_map.items()):
        if len(sample_uids) < 2:
            continue
        group_counter += 1
        group_id = f"exact_{group_counter:04d}"
        for sample_uid in sample_uids:
            duplicate_group_by_sample[sample_uid] = group_id
            duplicate_groups.append(
                {
                    "duplicate_group_id": group_id,
                    "content_id_blake2b64": digest,
                    "sample_uid": sample_uid,
                    "group_size": len(sample_uids),
                }
            )
    metadata_df["exact_duplicate_group_id"] = metadata_df["sample_uid"].map(duplicate_group_by_sample)
    metadata_df["is_exact_duplicate_member"] = metadata_df["exact_duplicate_group_id"].notna()
    duplicate_df = pd.DataFrame(duplicate_groups)

    for name, frame in {
        "metadata": metadata_df,
        "keypoints_raw_long": raw_df,
        "keypoints_canonical_long": canonical_df,
        "scale_endpoints_long": scale_df,
        "anomalies": anomaly_df,
        "exact_duplicates": duplicate_df,
    }.items():
        frame.to_csv(output_root / f"{name}.csv", index=False)
        frame.to_parquet(output_root / f"{name}.parquet", index=False)

    counts_by_group = metadata_df.groupby("source_group", sort=True).size().to_dict()
    summary = {
        "source_records": int(len(metadata_df)),
        "source_groups": int(metadata_df["source_group"].nunique()),
        "taxon_candidates": int(metadata_df["taxon_candidate"].nunique()),
        "raw_keypoint_records": int(len(raw_df)),
        "complete_fish_landmarks_1_to_11_role_aware": int(
            metadata_df["fish_landmarks_1_to_11_complete"].sum()
        ),
        "complete_scale_endpoints_1_to_2_role_aware": int(
            metadata_df["scale_endpoints_1_to_2_complete"].sum()
        ),
        "source_variant_10_fish_plus_raw_11_12_scale": int(
            metadata_df["anomaly_codes"].str.contains(
                "source_variant_10_fish_plus_raw_11_12_scale", regex=False
            ).sum()
        ),
        "canonical_mapping_complete": int(metadata_df["canonical_mapping_complete"].sum()),
        "canonical_mapping_ambiguous": int((metadata_df["canonical_mapping_mode"] == "ambiguous").sum()),
        "unique_content_ids": int(metadata_df["content_id_blake2b64"].nunique(dropna=True)),
        "exact_duplicate_groups": int(group_counter),
        "exact_duplicate_members": int(metadata_df["is_exact_duplicate_member"].sum()),
        "image_decode_errors": int(metadata_df["image_decode_error"].notna().sum()),
        "counts_by_source_group": {str(key): int(value) for key, value in counts_by_group.items()},
        "anomaly_counts": {
            str(key): int(value)
            for key, value in Counter(anomaly_df["anomaly_code"] if not anomaly_df.empty else []).items()
        },
        "digest_policy": "one BLAKE2b-64 content identifier per original image, reused from the prior table when path, size and dimensions are unchanged; no dataset SHA256 scan",
    }
    with (qa_root / "dataset_audit_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return summary
