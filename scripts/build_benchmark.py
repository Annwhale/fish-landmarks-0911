# ><(((o>  鱼类形态数据工具
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fishlandmark.audit import load_config
from fishlandmark.curate import build_benchmark


if __name__ == "__main__":
    config = load_config(ROOT / "config" / "dataset.yaml")
    dataset = config["dataset"]
    summary = build_benchmark(
        Path(dataset["output_root"]),
        Path(dataset["qa_root"]),
        seed=int(dataset["benchmark_seed"]),
        ratios={key: float(value) for key, value in dataset["standard_split_ratios"].items()},
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
