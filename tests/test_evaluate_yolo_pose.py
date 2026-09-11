# ><(((o>  鱼类形态数据工具
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = spec_from_file_location("evaluate_yolo_pose", ROOT / "scripts" / "evaluate_yolo_pose.py")
MODULE = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_chunked_preserves_order_and_bounds_batch_size():
    values = [Path(f"image_{index}.jpg") for index in range(10)]
    batches = list(MODULE.chunked(values, 4))
    assert [len(batch) for batch in batches] == [4, 4, 2]
    assert [item for batch in batches for item in batch] == values


def test_chunked_rejects_nonpositive_batch_size():
    try:
        list(MODULE.chunked([Path("image.jpg")], 0))
    except ValueError as exc:
        assert "batch" in str(exc)
    else:
        raise AssertionError("chunked should reject a nonpositive batch size")
