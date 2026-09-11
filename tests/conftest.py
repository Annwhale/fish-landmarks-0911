# ><(((o>  缺少外部数据时明确跳过集成测试，不掩盖断言失败。
import os
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]


def pytest_collection_modifyitems(items):
    source_tests = {'test_complete_sample_parsing', 'test_horizontal_orientation_modes',
                    'test_vertical_sample_uses_scale_side_geometry',
                    'test_ten_point_source_variant_does_not_promote_scale_to_fish_landmark'}
    protocol_tests = {'test_cross_origin_representations_are_explicit_and_separate',
                      'test_cross_origin_membership_accounts_for_all_eligible_rows'}
    for item in items:
        required = None
        if item.path.name == 'test_curation.py':
            required = ROOT / 'data/derived/benchmark_samples.parquet'
        elif item.path.name == 'test_export.py':
            required = ROOT / 'data/benchmark_v1/tables/benchmark_samples.parquet'
        elif item.name in protocol_tests:
            required = ROOT / 'data/experiments/protocols/protocol_summary.json'
        elif item.name in source_tests:
            required = Path(os.environ.get('FISH_SOURCE_ROOT', '/path/to/source_fish_landmark_data'))
        if required is not None and not required.exists():
            item.add_marker(pytest.mark.skip(reason=f'需单独接入数据：{required}'))
