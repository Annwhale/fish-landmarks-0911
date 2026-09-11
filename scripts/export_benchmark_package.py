#!/usr/bin/env python3
# ><(((o>  鱼类形态数据工具
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fishlandmark.audit import load_config
from fishlandmark.export import export_benchmark_package


if __name__ == "__main__":
    config = load_config(ROOT / "config" / "dataset.yaml")
    dataset = config["dataset"]
    result = export_benchmark_package(
        source_root=Path(dataset["source_root"]),
        derived_root=Path(dataset["output_root"]),
        output_root=Path(dataset["benchmark_package"]),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
