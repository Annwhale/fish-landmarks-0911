#!/usr/bin/env python3
# ><(((o>  鱼类形态数据工具
"""Report pairwise text-span intersections in one or more PDF files."""
from __future__ import annotations

import argparse
from pathlib import Path

import fitz


def intersections(path: Path, page_numbers: set[int] | None = None,
                  *, show_details: bool = False) -> dict[int, int]:
    document = fitz.open(path)
    results: dict[int, int] = {}
    for page_index, page in enumerate(document, start=1):
        if page_numbers is not None and page_index not in page_numbers:
            continue
        spans: list[tuple[fitz.Rect, str, int, int]] = []
        page_dict = page.get_text("dict")
        for block_index, block in enumerate(page_dict.get("blocks", [])):
            if block.get("type") != 0:
                continue
            for line_index, line in enumerate(block.get("lines", [])):
                for span in line.get("spans", []):
                    text = str(span.get("text", "")).strip()
                    if text:
                        spans.append((fitz.Rect(span["bbox"]), text, block_index, line_index))
        hits: list[tuple[str, str, fitz.Rect]] = []
        for left_index, (left_box, left_text, left_block, left_line) in enumerate(spans):
            for right_box, right_text, right_block, right_line in spans[left_index + 1:]:
                if left_block == right_block and left_line == right_line:
                    continue
                overlap = left_box & right_box
                if overlap.is_empty:
                    continue
                # 两方向重叠均达到0.75点才报告，避免字体边界接触误报。
                if overlap.width > 0.75 and overlap.height > 0.75:
                    hits.append((left_text, right_text, overlap))
        results[page_index] = len(hits)
        if show_details:
            for left_text, right_text, overlap in hits:
                print(f"  page {page_index}: {left_text!r} <> {right_text!r} at {tuple(round(v, 2) for v in overlap)}")
    document.close()
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", nargs="+", type=Path)
    parser.add_argument("--pages", help="comma-separated one-based page numbers")
    parser.add_argument("--details", action="store_true")
    args = parser.parse_args()
    selected = {int(value) for value in args.pages.split(",")} if args.pages else None
    failed = False
    for path in args.pdf:
        result = intersections(path, selected, show_details=args.details)
        total = sum(result.values())
        print(f"{path}: {total} intersections; per page {result}")
        failed = failed or total > 0
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
