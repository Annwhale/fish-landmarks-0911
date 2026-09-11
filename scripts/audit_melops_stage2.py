#!/usr/bin/env python3
# ><(((o>  鱼类形态数据工具
"""Audit the official Melops CVAT export without hashing or modifying it.

The output separates the labels declared by the CVAT task, labels actually
placed on images, and labels represented in the supplied model-error table.
This distinction is important because those three sets are not identical.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET


ARTICLE_ELEVEN = {
    "snout",
    "eyes_snout",
    "eyes_opercular",
    "eyes_pectoral",
    "top_opercular",
    "medium_opercular",
    "bottom_opercular",
    "anal_fin",
    "caudal_top",
    "caudal_bottom",
    "black_point",
}
CARD_LABELS = {"stand_1", "stand_2", "stand_3", "stand_4"}


def point_attribute(point: ET.Element, name: str) -> str | None:
    for attr in point.findall("attribute"):
        if attr.get("name") == name:
            return (attr.text or "").strip()
    return None


def element_attributes(element: ET.Element) -> dict[str, str]:
    return {
        attr.get("name", ""): (attr.text or "").strip()
        for attr in element.findall("attribute")
    }


def parse_point_string(value: str) -> tuple[float | None, float | None, int, bool]:
    """Return first coordinate and report CVAT multi-vertex point strings."""
    vertices = [item for item in value.split(";") if item]
    parsed = []
    for vertex in vertices:
        parts = vertex.split(",")
        if len(parts) != 2:
            continue
        parsed.append((float(parts[0]), float(parts[1])))
    if not parsed:
        return None, None, len(vertices), False
    return parsed[0][0], parsed[0][1], len(parsed), len(set(parsed)) == 1


def audit_xml(path: Path) -> tuple[dict, list[dict], list[dict]]:
    root = ET.parse(path).getroot()
    task = root.find("./meta/task")
    declared = []
    if task is not None:
        for label in task.findall("./labels/label"):
            declared.append(
                {
                    "name": label.findtext("name"),
                    "type": label.findtext("type"),
                }
            )

    image_rows: list[dict] = []
    point_rows: list[dict] = []
    label_frequency: Counter[str] = Counter()
    good_frequency: dict[str, Counter[str]] = defaultdict(Counter)
    body_set_frequency: Counter[tuple[str, ...]] = Counter()
    all_set_frequency: Counter[tuple[str, ...]] = Counter()
    multi_vertex_count = 0
    repeated_identical_vertex_count = 0
    body_attribute_frequency: dict[str, Counter[str]] = defaultdict(Counter)

    for image in root.findall("image"):
        body_boxes = [box for box in image.findall("box") if box.get("label") == "Fish body"]
        body_attributes = element_attributes(body_boxes[0]) if body_boxes else {}
        for name, value in body_attributes.items():
            body_attribute_frequency[name][value] += 1
        labels = []
        body_labels = []
        card_labels = []
        for point in image.findall("points"):
            label = point.get("label", "")
            labels.append(label)
            label_frequency[label] += 1
            good_value = point_attribute(point, "good_landmark")
            good_frequency[label][good_value if good_value is not None else "missing"] += 1
            if label in CARD_LABELS:
                card_labels.append(label)
            else:
                body_labels.append(label)
            x, y, vertex_count, vertices_identical = parse_point_string(point.get("points", ""))
            if vertex_count > 1:
                multi_vertex_count += 1
                repeated_identical_vertex_count += int(vertices_identical)
            point_rows.append(
                {
                    "image_id": image.get("id"),
                    "image_name": image.get("name"),
                    "width": image.get("width"),
                    "height": image.get("height"),
                    "label": label,
                    "x": "" if x is None else x,
                    "y": "" if y is None else y,
                    "vertex_count": vertex_count,
                    "vertices_identical": vertices_identical,
                    "good_landmark": good_value or "",
                }
            )

        body_set = tuple(sorted(set(body_labels)))
        all_set = tuple(sorted(set(labels)))
        if body_set:
            body_set_frequency[body_set] += 1
        if all_set:
            all_set_frequency[all_set] += 1
        image_rows.append(
            {
                "image_id": image.get("id"),
                "image_name": image.get("name"),
                "width": image.get("width"),
                "height": image.get("height"),
                "box_count": len(image.findall("box")),
                "body_box_count": len(body_boxes),
                "orientation": body_attributes.get("orientation", ""),
                "species": body_attributes.get("species", ""),
                "sex": body_attributes.get("sex", ""),
                "whole_fish": body_attributes.get("whole fish", ""),
                "blurry": body_attributes.get("blurry", ""),
                "jumping": body_attributes.get("jumping", ""),
                "overexposed": body_attributes.get("overexposed", ""),
                "point_count": len(labels),
                "body_point_count": len(body_labels),
                "card_point_count": len(card_labels),
                "unique_body_label_count": len(body_set),
                "has_article_eleven": ARTICLE_ELEVEN.issubset(body_set),
                "body_labels": "|".join(body_set),
                "card_labels": "|".join(sorted(set(card_labels))),
            }
        )

    images_with_points = sum(int(row["point_count"] > 0) for row in image_rows)
    images_with_body_points = sum(int(row["body_point_count"] > 0) for row in image_rows)
    images_with_all_article_eleven = sum(int(row["has_article_eleven"]) for row in image_rows)
    body_count_distribution = Counter(str(row["unique_body_label_count"]) for row in image_rows)
    point_count_distribution = Counter(str(row["point_count"]) for row in image_rows)

    summary = {
        "source_xml": str(path),
        "cvat_task_size": int(task.findtext("size")) if task is not None else None,
        "image_elements": len(image_rows),
        "images_with_any_points": images_with_points,
        "images_with_body_points": images_with_body_points,
        "images_with_all_article_eleven": images_with_all_article_eleven,
        "declared_labels": declared,
        "placed_point_labels": dict(sorted(label_frequency.items())),
        "multi_vertex_point_elements": multi_vertex_count,
        "multi_vertex_elements_with_identical_vertices": repeated_identical_vertex_count,
        "fish_body_attribute_frequency": {
            name: dict(sorted(values.items()))
            for name, values in sorted(body_attribute_frequency.items())
        },
        "good_landmark_values_by_label": {
            label: dict(sorted(values.items())) for label, values in sorted(good_frequency.items())
        },
        "unique_body_label_count_distribution": dict(sorted(body_count_distribution.items(), key=lambda x: int(x[0]))),
        "total_point_count_distribution": dict(sorted(point_count_distribution.items(), key=lambda x: int(x[0]))),
        "body_label_set_frequency": [
            {"count": count, "labels": list(labels)}
            for labels, count in body_set_frequency.most_common()
        ],
        "all_point_label_set_frequency": [
            {"count": count, "labels": list(labels)}
            for labels, count in all_set_frequency.most_common()
        ],
    }
    return summary, image_rows, point_rows


def audit_mne(path: Path) -> dict:
    keypoint_ids: dict[str, set[str]] = defaultdict(set)
    image_names = set()
    row_count = 0
    annotated_count = 0
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            row_count += 1
            image_names.add(row["image"])
            keypoint_ids[row["keypoint_name"]].add(row["keypoint_id"])
            if row.get("x_annotated") not in {None, "", "NA"}:
                annotated_count += 1
    labels = sorted(keypoint_ids)
    return {
        "source_csv": str(path),
        "rows": row_count,
        "images": len(image_names),
        "annotated_coordinate_rows": annotated_count,
        "keypoint_count": len(labels),
        "keypoint_names": labels,
        "keypoint_ids_by_name": {name: sorted(ids, key=int) for name, ids in sorted(keypoint_ids.items())},
        "article_eleven_missing_from_mne": sorted(ARTICLE_ELEVEN - set(labels)),
        "mne_labels_outside_article_eleven": sorted(set(labels) - ARTICLE_ELEVEN),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--xml",
        type=Path,
        default=root / "external/melops_v2/stage2/extracted/annotations.xml",
    )
    parser.add_argument(
        "--mne",
        type=Path,
        default=root / "external/melops_v2/stage2/extracted/MNE_result.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=root / "evidence/melops_stage2_audit"
    )
    args = parser.parse_args()

    xml_summary, image_rows, point_rows = audit_xml(args.xml)
    mne_summary = audit_mne(args.mne)
    report = {
        "audit_scope": "read-only structural audit; no image or annotation hashes computed",
        "article_reported_body_landmarks": sorted(ARTICLE_ELEVEN),
        "xml": xml_summary,
        "mne_result": mne_summary,
        "cross_source_observation": {
            "xml_placed_body_labels": sorted(
                label for label in xml_summary["placed_point_labels"] if label not in CARD_LABELS
            ),
            "mne_labels_not_placed_in_xml": sorted(
                set(mne_summary["keypoint_names"])
                - set(xml_summary["placed_point_labels"])
            ),
            "xml_body_labels_not_in_mne": sorted(
                (
                    set(xml_summary["placed_point_labels"])
                    - CARD_LABELS
                    - set(mne_summary["keypoint_names"])
                )
            ),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "audit_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_csv(args.output_dir / "image_inventory.csv", image_rows)
    write_csv(args.output_dir / "point_inventory.csv", point_rows)
    manual_members = [
        f"task_0/data/{row['image_name']}"
        for row in image_rows
        if row["body_point_count"] > 0
    ]
    (args.output_dir / "manual_image_members.txt").write_text(
        "\n".join(manual_members) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
