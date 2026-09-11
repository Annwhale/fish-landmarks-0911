#!/usr/bin/env python3
# ><(((o>  鱼类形态数据工具
"""External-domain stress test against candidate Melops correspondences.

This is deliberately not a merged-ontology evaluation. The project and Melops
landmark protocols differ, so each correspondence is retained with a tier and
the aggregate metrics are labelled exploratory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fishlandmark.metrics import bootstrap_mean_ci


def record_path(value: str | Path) -> str:
    path = Path(value).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


MAPPINGS = [
    {
        "project_id": 1,
        "project_name": "snout_tip",
        "melops_definition": "snout",
        "tier": "anchor",
        "target_labels": ["snout"],
        "target_rule": "direct",
    },
    {
        "project_id": 5,
        "project_name": "upper_caudal_fin_insertion",
        "melops_definition": "caudal_top",
        "tier": "anchor",
        "target_labels": ["caudal_top"],
        "target_rule": "direct",
    },
    {
        "project_id": 6,
        "project_name": "posterior_caudal_peduncle_midpoint",
        "melops_definition": "midpoint(caudal_top, caudal_bottom)",
        "tier": "derived",
        "target_labels": ["caudal_top", "caudal_bottom"],
        "target_rule": "midpoint",
    },
    {
        "project_id": 7,
        "project_name": "lower_caudal_fin_insertion",
        "melops_definition": "caudal_bottom",
        "tier": "anchor",
        "target_labels": ["caudal_bottom"],
        "target_rule": "direct",
    },
    {
        "project_id": 8,
        "project_name": "anal_fin_origin",
        "melops_definition": "anal_fin",
        "tier": "regional",
        "target_labels": ["anal_fin"],
        "target_rule": "direct",
    },
    {
        "project_id": 9,
        "project_name": "pelvic_fin_origin",
        "melops_definition": "pelvic",
        "tier": "regional",
        "target_labels": ["pelvic"],
        "target_rule": "direct",
    },
    {
        "project_id": 10,
        "project_name": "pectoral_fin_base",
        "melops_definition": "midpoint(pectoral_top, pectoral_bottom)",
        "tier": "derived",
        "target_labels": ["pectoral_top", "pectoral_bottom"],
        "target_rule": "midpoint",
    },
    {
        "project_id": 11,
        "project_name": "posterior_opercular_margin",
        "melops_definition": "medium_opercular",
        "tier": "regional",
        "target_labels": ["medium_opercular"],
        "target_rule": "direct",
    },
]


def point_attribute(point: ET.Element, name: str) -> str | None:
    for attribute in point.findall("attribute"):
        if attribute.get("name") == name:
            return (attribute.text or "").strip()
    return None


def parse_first_vertex(value: str) -> np.ndarray:
    first = value.split(";")[0]
    x, y = (float(item) for item in first.split(","))
    return np.array([x, y], dtype=np.float64)


def load_targets(xml_path: Path) -> dict[str, dict[str, dict[str, object]]]:
    records: dict[str, dict[str, dict[str, object]]] = {}
    for image in ET.parse(xml_path).getroot().findall("image"):
        points: dict[str, dict[str, object]] = {}
        for point in image.findall("points"):
            label = point.get("label", "")
            if label.startswith("stand_"):
                continue
            points[label] = {
                "xy": parse_first_vertex(point.get("points", "")),
                "good": point_attribute(point, "good_landmark") != "false",
            }
        if points:
            records[image.get("name", "")] = points
    return records


def mapping_target(points: dict[str, dict[str, object]], mapping: dict) -> np.ndarray | None:
    components = []
    for label in mapping["target_labels"]:
        record = points.get(label)
        if record is None or not bool(record["good"]):
            return None
        components.append(np.asarray(record["xy"], dtype=np.float64))
    if mapping["target_rule"] == "direct":
        return components[0]
    if mapping["target_rule"] == "midpoint":
        return np.mean(components, axis=0)
    raise ValueError(mapping["target_rule"])


def summarize(frame: pd.DataFrame) -> dict:
    detected = frame[frame["detected"]]
    result = {
        "images": int(len(frame)),
        "detected_images": int(frame["detected"].sum()),
        "detection_rate": float(frame["detected"].mean()),
        "normalization": "distance from Melops snout to midpoint(caudal_top, caudal_bottom)",
        "claim_boundary": (
            "candidate-correspondence external stress test; not an exact merged-ontology benchmark"
        ),
        "tiers": {},
        "per_correspondence": {},
    }
    for tier in ("anchor", "regional", "derived", "all_candidate"):
        prefix = tier
        result["tiers"][tier] = {
            "nme_detected": bootstrap_mean_ci(detected[f"{prefix}_nme"].to_numpy()),
            "pck_0_05_all_images": bootstrap_mean_ci(frame[f"{prefix}_pck_0_05"].to_numpy()),
            "pck_0_05_detected": bootstrap_mean_ci(detected[f"{prefix}_pck_0_05"].to_numpy()),
            "pck_0_10_all_images": bootstrap_mean_ci(frame[f"{prefix}_pck_0_10"].to_numpy()),
        }
    for mapping in MAPPINGS:
        project_id = mapping["project_id"]
        column = f"kp{project_id}_candidate_nme"
        result["per_correspondence"][str(project_id)] = {
            **mapping,
            "nme_detected": bootstrap_mean_ci(detected[column].to_numpy()),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument(
        "--xml",
        type=Path,
        default=ROOT / "external/melops_v2/stage2/extracted/annotations.xml",
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=ROOT / "external/melops_v2/stage2/manual_images",
    )
    args = parser.parse_args()

    targets = load_targets(args.xml)
    paths = [args.images / name for name in sorted(targets)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} external images are missing; first: {missing[0]}")

    model = YOLO(args.weights)
    predictions = model.predict(
        source=[str(path) for path in paths],
        imgsz=448,
        batch=args.batch,
        device=args.device,
        conf=0.001,
        iou=0.7,
        max_det=10,
        save=False,
        stream=True,
        verbose=False,
    )

    rows = []
    for path, result in zip(paths, predictions):
        source_points = targets[path.name]
        snout = np.asarray(source_points["snout"]["xy"], dtype=np.float64)
        caudal_mid = np.mean(
            [source_points["caudal_top"]["xy"], source_points["caudal_bottom"]["xy"]], axis=0
        )
        body_axis = float(np.linalg.norm(snout - caudal_mid))
        prediction = None
        confidence = np.nan
        if result.boxes is not None and len(result.boxes) and result.keypoints is not None:
            confidence_values = result.boxes.conf.detach().cpu().numpy()
            best = int(np.argmax(confidence_values))
            candidate = result.keypoints.xy[best].detach().cpu().numpy().astype(np.float64)
            if candidate.shape == (11, 2):
                prediction = candidate
                confidence = float(confidence_values[best])

        row: dict[str, object] = {
            "image_name": path.name,
            "detected": prediction is not None,
            "confidence": confidence,
            "melops_body_axis_px": body_axis,
        }
        tier_values: dict[str, list[float]] = {"anchor": [], "regional": [], "derived": []}
        for mapping in MAPPINGS:
            target = mapping_target(source_points, mapping)
            project_id = mapping["project_id"]
            value = np.nan
            if prediction is not None and target is not None:
                value = float(np.linalg.norm(prediction[project_id - 1] - target) / max(body_axis, 1e-12))
                tier_values[mapping["tier"]].append(value)
            row[f"kp{project_id}_candidate_nme"] = value

        for tier, values in tier_values.items():
            row[f"{tier}_nme"] = float(np.mean(values)) if values else np.nan
            row[f"{tier}_pck_0_05"] = float(np.mean(np.asarray(values) <= 0.05)) if values else 0.0
            row[f"{tier}_pck_0_10"] = float(np.mean(np.asarray(values) <= 0.10)) if values else 0.0
        all_values = [value for values in tier_values.values() for value in values]
        row["all_candidate_nme"] = float(np.mean(all_values)) if all_values else np.nan
        row["all_candidate_pck_0_05"] = (
            float(np.mean(np.asarray(all_values) <= 0.05)) if all_values else 0.0
        )
        row["all_candidate_pck_0_10"] = (
            float(np.mean(np.asarray(all_values) <= 0.10)) if all_values else 0.0
        )
        rows.append(row)

    frame = pd.DataFrame(rows)
    output = ROOT / "models/evaluation" / args.name / "melops_external"
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "per_image_metrics.csv", index=False)
    frame.to_parquet(output / "per_image_metrics.parquet", index=False)
    report = summarize(frame)
    report.update(
        {
            "model_weights": record_path(args.weights),
            "source_xml": record_path(args.xml),
            "source_images": record_path(args.images),
            "melops_version_doi": "10.5281/zenodo.17404087",
            "mappings": MAPPINGS,
            "evaluation_seed": 20260825,
        }
    )
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
