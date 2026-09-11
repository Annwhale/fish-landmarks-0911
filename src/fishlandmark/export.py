# ><(((o>  鱼类形态数据工具
from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import cv2
from PIL import Image

from .io import parse_yolo_pose, select_fish_and_scale_rows


SKELETON_EDGES = [
    (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
    (6, 7), (7, 8), (8, 9), (9, 10), (10, 1),
    (1, 11), (2, 11), (10, 11),
]


def _safe_bbox_from_pose(path: Path, width: int, height: int, points: np.ndarray) -> tuple[float, float, float, float]:
    rows = parse_yolo_pose(path) if path.exists() else []
    fish, _ = select_fish_and_scale_rows(rows)
    if fish is not None:
        x0 = (fish.bbox_cx - fish.bbox_w / 2.0) * width
        y0 = (fish.bbox_cy - fish.bbox_h / 2.0) * height
        x1 = (fish.bbox_cx + fish.bbox_w / 2.0) * width
        y1 = (fish.bbox_cy + fish.bbox_h / 2.0) * height
    else:
        x0, y0 = points.min(axis=0)
        x1, y1 = points.max(axis=0)
    return max(0.0, x0), max(0.0, y0), min(float(width), x1), min(float(height), y1)


def _padded_crop_box(
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
    padding_fraction: float,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    pad = padding_fraction * max(x1 - x0, y1 - y0)
    return (
        max(0, int(np.floor(x0 - pad))),
        max(0, int(np.floor(y0 - pad))),
        min(width, int(np.ceil(x1 + pad))),
        min(height, int(np.ceil(y1 + pad))),
    )


def _letterbox(
    image: Image.Image,
    crop_box: tuple[int, int, int, int],
    output_size: int,
) -> tuple[Image.Image, float, int, int]:
    crop = image.crop(crop_box).convert("RGB")
    scale = min(output_size / crop.width, output_size / crop.height)
    resized_width = max(1, int(round(crop.width * scale)))
    resized_height = max(1, int(round(crop.height * scale)))
    resized = crop.resize((resized_width, resized_height), Image.Resampling.LANCZOS)
    pad_x = (output_size - resized_width) // 2
    pad_y = (output_size - resized_height) // 2
    canvas = Image.new("RGB", (output_size, output_size), color=(114, 114, 114))
    canvas.paste(resized, (pad_x, pad_y))
    return canvas, scale, pad_x, pad_y


def _transform_points(
    points: np.ndarray,
    crop_box: tuple[int, int, int, int],
    scale: float,
    pad_x: int,
    pad_y: int,
) -> np.ndarray:
    transformed = points.copy().astype(np.float64)
    transformed[:, 0] = (transformed[:, 0] - crop_box[0]) * scale + pad_x
    transformed[:, 1] = (transformed[:, 1] - crop_box[1]) * scale + pad_y
    return transformed


def _axial_normalize(
    image: Image.Image,
    points: np.ndarray,
    output_size: int,
) -> tuple[Image.Image, np.ndarray, dict[str, float]]:
    """Rotate to snout-left/dorsal-up and crop using landmark-derived extents.

    The transform is fully recorded and invertible. Point 1 is the source snout
    anchor, point 6 the caudal-peduncle anchor, and canonical point 2 selects
    the dorsal half-plane. The last convention remains provisional until the
    source team confirms the contour-order protocol.
    """
    snout = points[0]
    caudal = points[5]
    axis = caudal - snout
    body_length = float(np.linalg.norm(axis))
    if body_length <= 1e-6:
        raise ValueError("snout and caudal anchors coincide")
    u = axis / body_length
    normal = np.asarray([-u[1], u[0]], dtype=np.float64)
    # 图像纵轴向下，2号点应位于体轴上方。
    if float(np.dot(points[1] - snout, normal)) > 0:
        v_down = -normal
    else:
        v_down = normal
    relative = points - snout
    u_values = relative @ u
    v_values = relative @ v_down
    u_min = min(float(u_values.min()) - 0.03 * body_length, -0.05 * body_length)
    u_max = max(float(u_values.max()) + 0.05 * body_length, 1.25 * body_length)
    v_min = float(v_values.min()) - 0.08 * body_length
    v_max = float(v_values.max()) + 0.08 * body_length
    inset = 8.0
    scale = min(
        (output_size - 2 * inset) / (u_max - u_min),
        (output_size - 2 * inset) / (v_max - v_min),
    )
    content_width = (u_max - u_min) * scale
    content_height = (v_max - v_min) * scale
    pad_x = (output_size - content_width) / 2.0
    pad_y = (output_size - content_height) / 2.0
    matrix = np.asarray(
        [
            [
                scale * u[0],
                scale * u[1],
                pad_x - scale * (float(np.dot(snout, u)) + u_min),
            ],
            [
                scale * v_down[0],
                scale * v_down[1],
                pad_y - scale * (float(np.dot(snout, v_down)) + v_min),
            ],
        ],
        dtype=np.float64,
    )
    warped = cv2.warpAffine(
        np.asarray(image.convert("RGB")),
        matrix,
        (output_size, output_size),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(114, 114, 114),
    )
    transformed = cv2.transform(points.reshape(1, -1, 2).astype(np.float64), matrix)[0]
    metadata = {
        "body_axis_length_px": body_length,
        "axis_unit_x": float(u[0]),
        "axis_unit_y": float(u[1]),
        "down_unit_x": float(v_down[0]),
        "down_unit_y": float(v_down[1]),
        "u_min": u_min,
        "u_max": u_max,
        "v_min": v_min,
        "v_max": v_max,
        "affine_00": float(matrix[0, 0]),
        "affine_01": float(matrix[0, 1]),
        "affine_02": float(matrix[0, 2]),
        "affine_10": float(matrix[1, 0]),
        "affine_11": float(matrix[1, 1]),
        "affine_12": float(matrix[1, 2]),
        "content_x0": pad_x,
        "content_y0": pad_y,
        "content_x1": pad_x + content_width,
        "content_y1": pad_y + content_height,
        "resize_scale": scale,
    }
    return Image.fromarray(warped), transformed, metadata


def export_benchmark_package(
    source_root: Path,
    derived_root: Path,
    output_root: Path,
    crop_size: int = 448,
    bbox_padding_fraction: float = 0.08,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    images_original_root = output_root / "images" / "original"
    yolo_root = output_root / "yolo"
    table_root = output_root / "tables"
    annotation_root = output_root / "annotations"
    split_root = output_root / "splits"
    for path in (images_original_root, table_root, annotation_root, split_root):
        path.mkdir(parents=True, exist_ok=True)

    samples = pd.read_parquet(derived_root / "benchmark_samples.parquet")
    points_long = pd.read_parquet(derived_root / "benchmark_keypoints.parquet")
    transformed_point_rows: list[dict[str, Any]] = []
    transform_rows: list[dict[str, Any]] = []
    coco_images: list[dict[str, Any]] = []
    coco_annotations: list[dict[str, Any]] = []
    landmark_names = (
        points_long[["canonical_id", "canonical_name_provisional"]]
        .drop_duplicates()
        .sort_values("canonical_id")["canonical_name_provisional"]
        .tolist()
    )

    for image_id, sample in enumerate(samples.itertuples(index=False), start=1):
        split = sample.standard_split
        image_dir = yolo_root / "images" / split
        label_dir = yolo_root / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        source_image = source_root / sample.representative_original_image
        original_target = images_original_root / f"{sample.benchmark_id}.jpg"
        crop_target = image_dir / f"{sample.benchmark_id}.jpg"
        label_target = label_dir / f"{sample.benchmark_id}.txt"

        if not original_target.exists() or original_target.stat().st_size != source_image.stat().st_size:
            shutil.copy2(source_image, original_target)
        point_group = points_long[points_long["benchmark_id"] == sample.benchmark_id].sort_values(
            "canonical_id"
        )
        points = point_group[["x", "y"]].to_numpy(dtype=np.float64)
        if points.shape != (11, 2):
            raise RuntimeError(f"expected 11 points for {sample.benchmark_id}, got {points.shape}")
        with Image.open(source_image) as image:
            image.load()
            crop, crop_points, axial_meta = _axial_normalize(image, points, crop_size)
            crop.save(crop_target, format="JPEG", quality=95, subsampling=0, optimize=True)
        bx0, by0 = axial_meta["content_x0"], axial_meta["content_y0"]
        bx1, by1 = axial_meta["content_x1"], axial_meta["content_y1"]
        bx0, by0 = max(0.0, bx0), max(0.0, by0)
        bx1, by1 = min(float(crop_size), bx1), min(float(crop_size), by1)
        bbox_w, bbox_h = max(1e-6, bx1 - bx0), max(1e-6, by1 - by0)

        yolo_values = [
            "0",
            f"{(bx0 + bx1) / (2 * crop_size):.8f}",
            f"{(by0 + by1) / (2 * crop_size):.8f}",
            f"{bbox_w / crop_size:.8f}",
            f"{bbox_h / crop_size:.8f}",
        ]
        coco_keypoints: list[float] = []
        for canonical_id, (x, y) in enumerate(crop_points, start=1):
            x = float(np.clip(x, 0, crop_size - 1e-6))
            y = float(np.clip(y, 0, crop_size - 1e-6))
            yolo_values.extend([f"{x / crop_size:.8f}", f"{y / crop_size:.8f}", "2"])
            coco_keypoints.extend([x, y, 2])
            transformed_point_rows.append(
                {
                    "benchmark_id": sample.benchmark_id,
                    "canonical_id": canonical_id,
                    "canonical_name_provisional": landmark_names[canonical_id - 1],
                    "x_crop_px": x,
                    "y_crop_px": y,
                    "x_crop_normalized": x / crop_size,
                    "y_crop_normalized": y / crop_size,
                    "visibility": 2,
                }
            )
        label_target.write_text(" ".join(yolo_values) + "\n", encoding="utf-8")

        transform_rows.append(
            {
                "benchmark_id": sample.benchmark_id,
                "source_width": sample.width,
                "source_height": sample.height,
                "normalization_mode": "snout_left_dorsal_up_landmark_affine",
                "orientation_semantic_status": "provisional_pending_source_confirmation",
                "output_size": crop_size,
                **axial_meta,
            }
        )
        coco_images.append(
            {
                "id": image_id,
                "file_name": f"yolo/images/{split}/{sample.benchmark_id}.jpg",
                "width": crop_size,
                "height": crop_size,
                "benchmark_id": sample.benchmark_id,
                "split": split,
                "taxon_candidate": sample.taxon_candidate,
                "origin_label": sample.origin_label,
            }
        )
        coco_annotations.append(
            {
                "id": image_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [bx0, by0, bbox_w, bbox_h],
                "area": bbox_w * bbox_h,
                "iscrowd": 0,
                "num_keypoints": 11,
                "keypoints": coco_keypoints,
            }
        )

    transformed_points = pd.DataFrame(transformed_point_rows)
    transforms = pd.DataFrame(transform_rows)
    transformed_points.to_csv(annotation_root / "keypoints_crops_448.csv", index=False)
    transformed_points.to_parquet(annotation_root / "keypoints_crops_448.parquet", index=False)
    transforms.to_csv(annotation_root / "crop_transforms.csv", index=False)
    transforms.to_parquet(annotation_root / "crop_transforms.parquet", index=False)

    coco = {
        "info": {
            "description": "Cross-taxon fish external-morphology landmark benchmark with invertible axial crops",
            "version": "0.1-provisional",
            "semantic_status": "landmark names pending source-team point-by-point confirmation",
            "license": "CC BY 4.0; project-generated primary data, author-confirmed for public release",
        },
        "licenses": [{"id": 1, "name": "CC BY 4.0", "url": "https://creativecommons.org/licenses/by/4.0/"}],
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": [
            {
                "id": 1,
                "name": "fish",
                "supercategory": "animal",
                "keypoints": landmark_names,
                "skeleton": [list(edge) for edge in SKELETON_EDGES],
                "semantic_status": "provisional",
            }
        ],
    }
    (annotation_root / "coco_keypoints.json").write_text(
        json.dumps(coco, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for path in sorted(derived_root.iterdir()):
        if path.suffix in {".csv", ".parquet"}:
            shutil.copy2(path, table_root / path.name)
        elif path.name.startswith("splits_") and path.suffix == ".json":
            shutil.copy2(path, split_root / path.name)

    yolo_yaml = "\n".join(
        [
            "path: .",
            "train: images/train",
            "val: images/validation",
            "test: images/test",
            "names:",
            "  0: fish",
            "kpt_shape: [11, 3]",
            "flip_idx: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]",
            "",
        ]
    )
    (yolo_root / "data.yaml").write_text(yolo_yaml, encoding="utf-8")

    database_path = output_root / "fish_landmarks.sqlite"
    if database_path.exists():
        database_path.unlink()
    connection = sqlite3.connect(database_path)
    try:
        samples.to_sql("benchmark_samples", connection, index=False)
        points_long.to_sql("benchmark_keypoints_original", connection, index=False)
        transformed_points.to_sql("benchmark_keypoints_crops_448", connection, index=False)
        transforms.to_sql("crop_transforms", connection, index=False)
        for stem in (
            "metadata", "keypoints_raw_long", "scale_endpoints_long", "anomalies",
            "exact_duplicates", "benchmark_exclusions", "duplicate_annotation_agreement",
            "near_duplicate_review_candidates",
        ):
            table = pd.read_parquet(derived_root / f"{stem}.parquet")
            table.to_sql(stem, connection, index=False)
    finally:
        connection.close()

    file_rows = []
    for path in sorted(item for item in output_root.rglob("*") if item.is_file()):
        file_rows.append({"relative_path": str(path.relative_to(output_root)), "bytes": path.stat().st_size})
    manifest = pd.DataFrame(file_rows)
    manifest.to_csv(output_root / "file_manifest_no_hash.csv", index=False)
    summary = {
        "unique_images": int(len(samples)),
        "technical_validation_eligible": int(samples["technical_validation_eligible"].sum()),
        "review_holdout": int((samples["standard_split"] == "review_holdout").sum()),
        "canonical_keypoint_records": int(len(points_long)),
        "crop_size": crop_size,
        "coco_annotations": len(coco_annotations),
        "package_files_before_manifest": len(file_rows),
        "package_bytes_before_manifest": int(manifest["bytes"].sum()),
        "integrity_policy": "file list, size, ZIP CRC/test-open at packaging; no repeated SHA256 scan",
    }
    (output_root / "package_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
