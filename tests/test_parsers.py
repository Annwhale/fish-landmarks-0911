# ><(((o>  鱼类形态数据工具
from pathlib import Path
import os

from fishlandmark.audit import mapping_assessment, match_yolo_role_points
from fishlandmark.io import Point, parse_keypoint_csv, parse_yolo_pose, select_fish_and_scale_rows


SOURCE = Path(os.environ.get('FISH_SOURCE_ROOT', '/path/to/source_fish_landmark_data'))


def test_complete_sample_parsing():
    folder = SOURCE / "草鱼"
    points = parse_keypoint_csv(folder / "caoyu (1)_result.csv")
    rows = parse_yolo_pose(folder / "caoyu (1)_0.txt")
    fish, scale = select_fish_and_scale_rows(rows)
    assert len(points) == 13
    assert fish is not None and fish.nonzero_keypoints == 11
    assert scale is not None and scale.nonzero_keypoints == 2


def test_horizontal_orientation_modes():
    left_points = parse_keypoint_csv(SOURCE / "草鱼" / "caoyu (1)_result.csv")
    right_points = parse_keypoint_csv(
        SOURCE / "野生翘嘴鲌" / "yeshengbo     (1)_result.csv"
    )
    assert mapping_assessment(left_points)[0] == "direct"
    assert mapping_assessment(right_points)[0] == "reverse"


def test_oblique_mapping_is_not_forced():
    points = [Point(1, 0, 0), Point(2, 1, 1), Point(6, 1, 10), Point(10, -1, 1)]
    assert mapping_assessment(points)[0] == "ambiguous"


def test_vertical_sample_uses_scale_side_geometry():
    points = parse_keypoint_csv(SOURCE / "刀鲚" / "2239169_result.csv")
    mode, reason, _ = mapping_assessment(points)
    assert mode == "direct"
    assert reason == "scale_side_geometry_unconfirmed_convention"


def test_ten_point_source_variant_does_not_promote_scale_to_fish_landmark():
    folder = SOURCE / "草鱼"
    points = parse_keypoint_csv(folder / "caoyu (57)_result.csv")
    rows = parse_yolo_pose(folder / "caoyu (57)_0.txt")
    fish, scale = select_fish_and_scale_rows(rows)
    fish_matches = match_yolo_role_points(fish, points, width=800, height=450)
    scale_matches = match_yolo_role_points(scale, points, width=800, height=450)
    assert len(fish_matches) == 10
    assert [item["source_csv_point_id"] for item in scale_matches] == [11, 12]
