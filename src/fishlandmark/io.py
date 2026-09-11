# ><(((o>  鱼类形态数据工具
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Point:
    point_id: int
    x: float
    y: float


@dataclass(frozen=True)
class YoloRow:
    class_id: int
    bbox_cx: float
    bbox_cy: float
    bbox_w: float
    bbox_h: float
    keypoints: tuple[tuple[float, float, int], ...]
    raw: str

    @property
    def nonzero_keypoints(self) -> int:
        return sum(1 for x, y, _ in self.keypoints if x != 0.0 or y != 0.0)


def parse_keypoint_csv(path: Path) -> list[Point]:
    points: list[Point] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            raw_id = (row.get("ID") or "").strip().rstrip(",")
            raw_x = (row.get("X") or "").strip().rstrip(",")
            raw_y = (row.get("Y") or "").strip().rstrip(",")
            if not raw_id or not raw_x or not raw_y:
                continue
            try:
                point = Point(int(raw_id), float(raw_x), float(raw_y))
            except ValueError:
                continue
            if point.x == 0.0 and point.y == 0.0:
                continue
            points.append(point)
    return points


def parse_yolo_pose(path: Path) -> list[YoloRow]:
    rows: list[YoloRow] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        fields = raw.split()
        if len(fields) < 5:
            continue
        remainder = fields[5:]
        triples = []
        for idx in range(0, len(remainder) - 2, 3):
            try:
                triples.append(
                    (float(remainder[idx]), float(remainder[idx + 1]), int(float(remainder[idx + 2])))
                )
            except ValueError:
                break
        rows.append(
            YoloRow(
                class_id=int(float(fields[0])),
                bbox_cx=float(fields[1]),
                bbox_cy=float(fields[2]),
                bbox_w=float(fields[3]),
                bbox_h=float(fields[4]),
                keypoints=tuple(triples),
                raw=raw,
            )
        )
    return rows


def select_fish_and_scale_rows(rows: list[YoloRow]) -> tuple[YoloRow | None, YoloRow | None]:
    """Select rows by populated pose points, not by unstable class identifiers."""
    if not rows:
        return None, None
    ranked = sorted(rows, key=lambda row: (row.nonzero_keypoints, row.bbox_w * row.bbox_h), reverse=True)
    fish = ranked[0] if ranked[0].nonzero_keypoints >= 10 else None
    scale_candidates = [row for row in ranked[1:] if 1 <= row.nonzero_keypoints <= 2]
    scale = scale_candidates[0] if scale_candidates else None
    return fish, scale
