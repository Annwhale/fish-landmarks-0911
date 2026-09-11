#!/usr/bin/env python3
# ><(((o>  鱼类形态数据工具
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fishlandmark.audit import run_audit


if __name__ == "__main__":
    result = run_audit(ROOT / "config" / "dataset.yaml")
    print(json.dumps(result, ensure_ascii=False, indent=2))
