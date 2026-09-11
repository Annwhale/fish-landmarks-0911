#!/usr/bin/env python3
# ><(((o>  鱼类形态数据工具
"""Shared visual primitives for the Scientific Data figure set."""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# 深色高对比配色，兼顾缩放后的可读性。
INK = "#172A3A"
SLATE = "#526270"
PALE = "#F4F5F3"
TEAL = "#087E73"
CYAN = "#2563A6"
GOLD = "#B7791F"
CORAL = "#B9483D"
VIOLET = "#6B4EA0"
MAGENTA = "#A43B68"
HAIRLINE = "#AEB9C1"
GRID = "#D5DDE2"


def apply_publication_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 8.3,
        "axes.titlesize": 8.9,
        "axes.labelsize": 8.1,
        "axes.linewidth": 0.65,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.fontsize": 7.3,
        "legend.frameon": False,
        "xtick.labelsize": 7.3,
        "ytick.labelsize": 7.3,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def style_axis(ax: mpl.axes.Axes, *, facecolor: str = "white", grid_axis: str | None = None) -> None:
    ax.set_facecolor(facecolor)
    ax.spines["left"].set_color("#8999A5")
    ax.spines["bottom"].set_color("#8999A5")
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(width=0.5, length=2.4, colors=INK)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.45, zorder=0)
        ax.set_axisbelow(True)


def panel_label(ax: mpl.axes.Axes, value: str, *, x: float = -0.035, y: float = 1.025) -> None:
    ax.text(x, y, value, transform=ax.transAxes, fontsize=10.4,
            fontweight="bold", color="#111111", va="bottom", ha="left")


def clean_image_axis(ax: mpl.axes.Axes, *, border: bool = True) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(border)
        if border:
            spine.set_color(HAIRLINE)
            spine.set_linewidth(0.55)


def packed_row_boxes(
    parent_ax: mpl.axes.Axes,
    row_aspects: list[list[float]],
    *,
    top: float = 0.90,
    bottom: float = 0.01,
    horizontal_gap: float = 0.007,
    vertical_gap: float = 0.018,
    label_space: float = 0.027,
) -> list[list[tuple[float, float, float, float]]]:
    """Return tightly packed inset boxes without changing image aspect ratios.

    Equal-height subplot grids leave large vertical voids when specimen crops
    range from deep-bodied to strongly elongate fish.  This layout computes a
    common physical height for each row from the actual crop aspect ratios,
    then packs rows from the top.  If the natural stack is taller than the
    available panel, all row boxes are uniformly reduced and centred; images
    are never stretched.
    """
    bbox = parent_ax.get_position()
    fig_width, fig_height = parent_ax.figure.get_size_inches()
    parent_ratio = (bbox.width * fig_width) / max(bbox.height * fig_height, 1e-9)
    clean_rows = [[max(float(value), 0.25) for value in row] for row in row_aspects]
    natural_heights = [
        max(0.01, (1.0 - horizontal_gap * max(0, len(row) - 1)) * parent_ratio / sum(row))
        for row in clean_rows
    ]
    natural_total = (
        sum(natural_heights)
        + len(clean_rows) * label_space
        + max(0, len(clean_rows) - 1) * vertical_gap
    )
    available = max(0.05, top - bottom)
    scale = min(1.0, available / max(natural_total, 1e-9))
    heights = [value * scale for value in natural_heights]
    effective_label = label_space * scale
    effective_vgap = vertical_gap * scale

    packed: list[list[tuple[float, float, float, float]]] = []
    cursor = top
    for aspects, height in zip(clean_rows, heights):
        cursor -= effective_label
        widths = [height * aspect / parent_ratio for aspect in aspects]
        row_width = sum(widths) + horizontal_gap * max(0, len(widths) - 1)
        x = (1.0 - row_width) / 2.0
        y = cursor - height
        row_boxes = []
        for width in widths:
            row_boxes.append((x, y, width, height))
            x += width + horizontal_gap
        packed.append(row_boxes)
        cursor = y - effective_vgap
    return packed


def crop_limits(xs: np.ndarray, ys: np.ndarray, width: int, height: int,
                *, margin_x: float = 0.12, margin_y: float = 0.28) -> tuple[float, float, float, float]:
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    valid = np.isfinite(xs) & np.isfinite(ys)
    if valid.sum() < 2:
        return 0.0, float(width), float(height), 0.0
    xmin, xmax = float(xs[valid].min()), float(xs[valid].max())
    ymin, ymax = float(ys[valid].min()), float(ys[valid].max())
    dx = max(24.0, (xmax - xmin) * margin_x)
    dy = max(24.0, (ymax - ymin) * margin_y)
    return max(0.0, xmin - dx), min(float(width), xmax + dx), min(float(height), ymax + dy), max(0.0, ymin - dy)


def show_image_with_crop(ax: mpl.axes.Axes, image: Image.Image | np.ndarray,
                         xs: np.ndarray, ys: np.ndarray, *, interpolation: str = "nearest") -> None:
    array = np.asarray(image)
    ax.imshow(array, interpolation=interpolation)
    h, w = array.shape[:2]
    x0, x1, y1, y0 = crop_limits(xs, ys, w, h)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    clean_image_axis(ax)


def save_bundle(fig: mpl.figure.Figure, stem: Path, *, dpi: int = 600) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".tiff"), dpi=dpi, bbox_inches="tight",
                facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_half_violin(ax: mpl.axes.Axes, values: np.ndarray, y: float, color: str,
                     *, width: float = 0.34, side: str = "top") -> None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return
    violin = ax.violinplot([values], positions=[y], vert=False, widths=width * 2,
                           showmeans=False, showmedians=False, showextrema=False)
    for body in violin["bodies"]:
        verts = body.get_paths()[0].vertices
        if side == "top":
            verts[:, 1] = np.maximum(verts[:, 1], y)
        else:
            verts[:, 1] = np.minimum(verts[:, 1], y)
        body.set_facecolor(color)
        body.set_edgecolor("none")
        body.set_alpha(0.52)
