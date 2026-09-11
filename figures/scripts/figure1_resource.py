#!/usr/bin/env python3
# ><(((o>  鱼类形态数据工具
"""Scientific Data Figure 1: workflow, relational architecture and traceability.

The figure is drawn deterministically from project records, documented source
images and original vector geometry. All numerical labels and database row
totals come from materialised source-data ledgers.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec, patheffects
from matplotlib.path import Path as MplPath
from matplotlib.patches import (
    Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, PathPatch, Polygon, Rectangle,
)
import numpy as np
import pandas as pd
from PIL import Image

from visual_style import (
    CORAL, CYAN, GOLD, INK, MAGENTA, PALE, SLATE, TEAL, VIOLET,
    apply_publication_style, clean_image_axis, crop_limits, panel_label,
)


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "data" / "benchmark_v1"
SOURCE_ROOT = Path("/path/to/source_fish_landmark_data")
SOURCE_DATA = ROOT / "figures" / "source_data"
GENERATED = ROOT / "figures" / "generated"
FIG_WIDTH_MM = 183.0
WIDTH_IN = 7.20472440945

ROLE_COLORS = {
    1: CYAN, 2: CYAN, 10: CYAN, 11: CYAN,
    3: VIOLET, 4: VIOLET,
    5: GOLD, 6: GOLD, 7: GOLD,
    8: CORAL, 9: CORAL,
}
ROLE_MARKERS = {
    1: "o", 2: "o", 10: "o", 11: "o",
    3: "D", 4: "D",
    5: "s", 6: "s", 7: "s",
    8: "^", 9: "^",
}


def save_figure(fig: mpl.figure.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight",
                facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def image_path(relative: str) -> Path:
    return SOURCE_ROOT / Path(relative)


def load_original(sample: pd.Series) -> Image.Image:
    path = PACKAGE / "images/original" / f"{sample['benchmark_id']}.jpg"
    if not path.exists():
        path = image_path(str(sample["representative_original_image"]))
    if not path.exists():
        raise FileNotFoundError(path)
    return Image.open(path).convert("RGB")


def affine_normalize(image: Image.Image, transform: pd.Series) -> Image.Image:
    matrix = np.array([
        [float(transform.affine_00), float(transform.affine_01), float(transform.affine_02)],
        [float(transform.affine_10), float(transform.affine_11), float(transform.affine_12)],
        [0.0, 0.0, 1.0],
    ])
    inverse = np.linalg.inv(matrix)
    coefficients = tuple(float(value) for value in (
        inverse[0, 0], inverse[0, 1], inverse[0, 2],
        inverse[1, 0], inverse[1, 1], inverse[1, 2],
    ))
    return image.transform(
        (448, 448), Image.Transform.AFFINE, coefficients,
        resample=Image.Resampling.BILINEAR, fillcolor=(255, 255, 255),
    )


def rounded_box(ax: mpl.axes.Axes, x: float, y: float, width: float, height: float,
                *, face: str = "white", edge: str = "#D6E0E5", linewidth: float = 0.8,
                radius: float = 0.018, zorder: int = 1) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
        transform=ax.transAxes, facecolor=face, edgecolor=edge,
        linewidth=linewidth, clip_on=False, zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def plot_points(ax: mpl.axes.Axes, points: pd.DataFrame, *, x_col: str, y_col: str,
                annotate_ids: set[int] | None = None, size: float = 23) -> None:
    order = points.sort_values("canonical_id")
    ax.plot(order[x_col], order[y_col], color="white", linewidth=1.7, alpha=0.86, zorder=2)
    ax.plot(order[x_col], order[y_col], color=INK, linewidth=0.62, alpha=0.76, zorder=2)
    for marker, group in order.groupby(order["canonical_id"].map(ROLE_MARKERS)):
        colors = [ROLE_COLORS[int(role)] for role in group["canonical_id"]]
        ax.scatter(group[x_col], group[y_col], s=size, c=colors, marker=marker,
                   edgecolors="white", linewidths=0.65, zorder=3)
    if annotate_ids:
        for _, point in order[order["canonical_id"].isin(annotate_ids)].iterrows():
            label = ax.text(
                float(point[x_col]) + 3.0, float(point[y_col]) - 3.0,
                str(int(point["canonical_id"])), fontsize=6.8,
                fontweight="bold", color=INK, zorder=4,
            )
            label.set_path_effects([patheffects.withStroke(linewidth=1.4, foreground="white")])


def show_fish(ax: mpl.axes.Axes, image: Image.Image, points: pd.DataFrame | None,
              *, x_col: str, y_col: str, width: int, height: int,
              margin_x: float = 0.10, margin_y: float = 0.24,
              annotate_ids: set[int] | None = None, draw_landmarks: bool = True) -> None:
    ax.imshow(image, interpolation="nearest")
    if points is not None and not points.empty:
        if draw_landmarks:
            plot_points(ax, points, x_col=x_col, y_col=y_col, annotate_ids=annotate_ids)
        x0, x1, y1, y0 = crop_limits(
            points[x_col].to_numpy(), points[y_col].to_numpy(), width, height,
            margin_x=margin_x, margin_y=margin_y,
        )
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
    clean_image_axis(ax)


def draw_database_icon(ax: mpl.axes.Axes, centre_x: float, centre_y: float,
                       width: float, height: float) -> None:
    x0 = centre_x - width / 2
    y0 = centre_y - height / 2
    ax.add_patch(Rectangle((x0, y0), width, height, transform=ax.transAxes,
                           facecolor="#E8F3F1", edgecolor=TEAL, linewidth=0.9, zorder=3))
    ax.add_patch(Ellipse((centre_x, y0 + height), width, height * 0.24,
                         transform=ax.transAxes, facecolor="#D2E8E4",
                         edgecolor=TEAL, linewidth=0.9, zorder=4))
    ax.add_patch(Ellipse((centre_x, y0), width, height * 0.24,
                         transform=ax.transAxes, facecolor="#E8F3F1",
                         edgecolor=TEAL, linewidth=0.9, zorder=4))
    for fraction in (0.34, 0.67):
        ax.plot([x0, x0 + width], [y0 + height * fraction] * 2,
                transform=ax.transAxes, color=TEAL, linewidth=0.6, alpha=0.75, zorder=5)


def draw_anatomical_fish(ax: mpl.axes.Axes, x: float, y: float,
                         width: float, height: float,
                         *, show_role_ids: bool = False) -> None:
    """Draw an original vector fish with the 11 working role positions.

    The composition was informed by a public-domain 1905 fish-topography plate,
    but the pixels from that plate are not embedded in the research figure.
    """
    def p(px: float, py: float) -> tuple[float, float]:
        return x + px * width, y + py * height

    ax.add_patch(Ellipse(p(0.49, 0.50), width * 1.06, height * 1.18,
                         transform=ax.transAxes, facecolor="#EEF3F6",
                         edgecolor="none", alpha=0.92, zorder=2))
    body_vertices = [
        p(0.04, 0.52),
        p(0.10, 0.22), p(0.40, 0.12), p(0.76, 0.27),
        p(0.82, 0.34), p(0.84, 0.42), p(0.84, 0.50),
        p(0.84, 0.58), p(0.76, 0.73), p(0.40, 0.88),
        p(0.10, 0.78), p(0.04, 0.62), p(0.04, 0.52),
        p(0.04, 0.52),
    ]
    body_codes = [MplPath.MOVETO] + [MplPath.CURVE4] * 12 + [MplPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MplPath(body_vertices, body_codes), transform=ax.transAxes,
                           facecolor="#CBD8E7", edgecolor=INK, lw=1.15, zorder=3))
    ax.add_patch(Polygon([p(0.82, 0.42), p(1.00, 0.10), p(0.96, 0.50),
                          p(1.00, 0.90), p(0.82, 0.58)], closed=True,
                         transform=ax.transAxes, facecolor="#F5C96B",
                         edgecolor=INK, lw=1.05, zorder=3))
    fin_specs = [
        ([p(0.36, 0.20), p(0.47, -0.03), p(0.59, 0.23)], "#F5C96B"),
        ([p(0.54, 0.79), p(0.64, 1.02), p(0.70, 0.75)], "#F0A38E"),
        ([p(0.38, 0.75), p(0.44, 0.98), p(0.51, 0.79)], "#F0A38E"),
        ([p(0.27, 0.58), p(0.39, 0.72), p(0.31, 0.48)], "#91CEC9"),
    ]
    for vertices, fin_colour in fin_specs:
        ax.add_patch(Polygon(vertices, closed=True, transform=ax.transAxes,
                             facecolor=fin_colour, edgecolor=INK, lw=0.85, zorder=3))
    ax.add_patch(Circle(p(0.145, 0.43), height * 0.035, transform=ax.transAxes,
                        facecolor=INK, edgecolor="white", lw=0.65, zorder=5))
    ax.add_patch(mpl.patches.Arc(p(0.255, 0.52), width * 0.12, height * 0.48,
                                 theta1=95, theta2=265, transform=ax.transAxes,
                                 color=CORAL, lw=1.15, zorder=5))
    ax.plot([p(0.12, 0.57)[0], p(0.77, 0.57)[0]],
            [p(0.12, 0.57)[1], p(0.77, 0.57)[1]], transform=ax.transAxes,
            color="#7DA29C", lw=0.75, ls=(0, (2, 2)), zorder=4)
    for sx in np.linspace(0.30, 0.70, 6):
        ax.add_patch(mpl.patches.Arc(p(sx, 0.48), width * 0.055, height * 0.14,
                                     theta1=80, theta2=280, transform=ax.transAxes,
                                     color="#89AAA4", lw=0.45, alpha=0.75, zorder=4))

    positions = {
        1: (0.045, 0.52), 2: (0.23, 0.27), 3: (0.38, 0.20), 4: (0.59, 0.23),
        5: (0.82, 0.31), 6: (0.84, 0.50), 7: (0.82, 0.69), 8: (0.64, 0.76),
        9: (0.46, 0.79), 10: (0.35, 0.69), 11: (0.235, 0.50),
    }
    for role_id, (px, py) in positions.items():
        cx, cy = p(px, py)
        marker = ROLE_MARKERS[role_id]
        ax.scatter([cx], [cy], s=46, marker=marker, transform=ax.transAxes,
                   facecolor=INK, edgecolor="white", linewidth=0.65,
                   clip_on=False, zorder=7)
        ax.scatter([cx], [cy], s=25, marker=marker, transform=ax.transAxes,
                   facecolor=ROLE_COLORS[role_id], edgecolor="none",
                   clip_on=False, zorder=8)
        if show_role_ids:
            ax.text(cx, cy, str(role_id), transform=ax.transAxes, ha="center", va="center",
                    fontsize=6.8, color="white", fontweight="bold", zorder=9)


def draw_data_lattice(ax: mpl.axes.Axes, x: float, y: float,
                      width: float, height: float) -> None:
    """Deterministic relational lattice used by the submission-safe workflow."""
    ax.add_patch(Ellipse((x + width * 0.50, y + height * 0.50), width * 1.12,
                         height * 1.06, transform=ax.transAxes, facecolor="#E8F3F1",
                         edgecolor="none", alpha=0.90, zorder=1))
    cols, rows = 5, 4
    coords: dict[tuple[int, int], tuple[float, float]] = {}
    for i in range(cols):
        for j in range(rows):
            px = x + width * (0.08 + 0.21 * i + 0.018 * np.sin(j + i))
            py = y + height * (0.12 + 0.25 * j + 0.025 * np.cos(2 * i + j))
            coords[(i, j)] = (px, py)
    for (i, j), (px, py) in coords.items():
        for target in ((i + 1, j), (i, j + 1), (i + 1, j + (1 if (i + j) % 2 else -1))):
            if target in coords:
                qx, qy = coords[target]
                ax.plot([px, qx], [py, qy], transform=ax.transAxes,
                        color="#7FA9B4", lw=0.48, alpha=0.72, zorder=2)
    node_colours = [CYAN, TEAL, VIOLET, GOLD]
    for (i, j), (px, py) in coords.items():
        ax.add_patch(Circle((px, py), height * (0.018 + 0.004 * ((i + j) % 3)),
                            transform=ax.transAxes, facecolor=node_colours[j],
                            edgecolor="white", lw=0.45, zorder=3))


def draw_release_stack(ax: mpl.axes.Axes, x: float, y: float,
                       width: float, height: float) -> None:
    colours = [CYAN, TEAL, VIOLET, GOLD, CORAL]
    for index, colour in enumerate(colours):
        y0 = y + index * height * 0.115
        shift = index * width * 0.012
        sheet = Polygon([
            (x + shift, y0), (x + width * 0.82 + shift, y0 + height * 0.02),
            (x + width + shift, y0 + height * 0.20),
            (x + width * 0.16 + shift, y0 + height * 0.18),
        ], closed=True, transform=ax.transAxes, facecolor="white", edgecolor=colour,
           lw=0.72, zorder=4 + index)
        ax.add_patch(sheet)
        for row in range(3):
            yy = y0 + height * (0.052 + row * 0.038)
            ax.plot([x + width * 0.20 + shift, x + width * 0.74 + shift],
                    [yy, yy], transform=ax.transAxes, color=colour,
                    lw=0.40, alpha=0.72, zorder=5 + index)


def draw_validation_seal(ax: mpl.axes.Axes, x: float, y: float,
                         radius: float) -> None:
    ax.add_patch(Circle((x, y), radius * 1.36, transform=ax.transAxes,
                        facecolor="#FBE9E4", edgecolor="none", alpha=0.85, zorder=2))
    ax.add_patch(Circle((x, y), radius, transform=ax.transAxes,
                        facecolor="white", edgecolor=CORAL, lw=1.05, zorder=4))
    ax.plot([x - radius * 0.48, x - radius * 0.10, x + radius * 0.58],
            [y - radius * 0.02, y - radius * 0.36, y + radius * 0.42],
            transform=ax.transAxes, color=CORAL, lw=1.9,
            solid_capstyle="round", zorder=6)


def panel_a_vector(ax: mpl.axes.Axes, sample: pd.Series, points: pd.DataFrame,
                   database: pd.DataFrame) -> None:
    """Submission-safe workflow drawn from source records and vector geometry."""
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.01, 0.985, "From source pixels to a reusable validation resource",
            transform=ax.transAxes, fontsize=9.35, fontweight="bold", va="top", color=INK)

    flow_vertices = [
        (0.055, 0.525), (0.18, 0.62), (0.29, 0.48), (0.42, 0.55),
        (0.55, 0.63), (0.69, 0.48), (0.82, 0.56), (0.95, 0.53),
    ]
    flow_codes = [MplPath.MOVETO] + [MplPath.CURVE4] * 6 + [MplPath.LINETO]
    ax.add_patch(PathPatch(MplPath(flow_vertices, flow_codes), transform=ax.transAxes,
                           facecolor="none", edgecolor="#B9D4D7", lw=12.0,
                           alpha=0.30, capstyle="round", zorder=0))
    ax.add_patch(PathPatch(MplPath(flow_vertices, flow_codes), transform=ax.transAxes,
                           facecolor="none", edgecolor="#7EA4AF", lw=0.85,
                           alpha=0.90, zorder=1))

    original = load_original(sample)
    ax.add_patch(Ellipse((0.120, 0.515), 0.205, 0.395, transform=ax.transAxes,
                         facecolor="#EDF5F6", edgecolor="none", zorder=1))
    source_ax = ax.inset_axes([0.035, 0.365, 0.165, 0.300], zorder=5)
    show_fish(source_ax, original, points, x_col="x_source_px", y_col="y_source_px",
              width=int(sample["width"]), height=int(sample["height"]),
              margin_x=0.08, margin_y=0.23, draw_landmarks=False)
    source_ax.patch.set_edgecolor(CYAN); source_ax.patch.set_linewidth(0.8)

    draw_anatomical_fish(ax, 0.235, 0.335, 0.325, 0.355, show_role_ids=False)
    draw_data_lattice(ax, 0.595, 0.335, 0.165, 0.360)
    draw_release_stack(ax, 0.785, 0.375, 0.095, 0.285)
    draw_validation_seal(ax, 0.948, 0.525, 0.042)

    for start, end, colour in [
        ((0.195, 0.525), (0.232, 0.525), CYAN),
        ((0.560, 0.525), (0.595, 0.525), VIOLET),
        ((0.760, 0.525), (0.785, 0.525), TEAL),
        ((0.885, 0.525), (0.907, 0.525), GOLD),
    ]:
        ax.add_patch(FancyArrowPatch(start, end, transform=ax.transAxes,
                                     arrowstyle="-|>", mutation_scale=8,
                                     color=colour, lw=0.85, zorder=10))
    ax.text(0.832, 0.294, "CSV · Parquet · SQLite · COCO · YOLO",
            transform=ax.transAxes, ha="center", va="center", fontsize=6.7,
            fontweight="bold", color=SLATE)
    ax.text(0.50, 0.095,
            "2,699 source records  →  11 anatomical landmarks  →  12 linked tables  →  five portable formats  →  2,555 validation images",
            transform=ax.transAxes, ha="center", va="center", fontsize=7.0,
            color=INK, fontweight="bold")


def database_lines(group: pd.DataFrame) -> list[tuple[str, int]]:
    display = {
        "metadata": "metadata",
        "keypoints_raw_long": "raw keypoints",
        "scale_endpoints_long": "scale endpoints",
        "anomalies": "anomalies",
        "exact_duplicates": "exact duplicates",
        "duplicate_annotation_agreement": "repeat agreement",
        "near_duplicate_review_candidates": "near-duplicate review",
        "benchmark_exclusions": "excluded records",
        "benchmark_samples": "images",
        "crop_transforms": "transforms",
        "benchmark_keypoints_original": "source coordinates",
        "benchmark_keypoints_crops_448": "axial coordinates",
    }
    return [
        (display[row.table_name], int(row.row_count))
        for row in group.itertuples(index=False)
    ]


def panel_b(ax: mpl.axes.Axes, database: pd.DataFrame) -> None:
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.01, 0.985, "Relational topology · 12 traceable tables",
            transform=ax.transAxes, fontsize=9.15, fontweight="bold", va="top", color=INK)
    ax.text(0.99, 0.978, "node labels are table contents · values are actual row counts",
            transform=ax.transAxes, fontsize=6.85, va="top", ha="right", color=SLATE)

    layers = ["Source audit", "Curation and QC", "Benchmark core", "Coordinate records"]
    colours = [CYAN, VIOLET, TEAL, GOLD]
    node_x = [0.035, 0.260, 0.585, 0.795]
    count_x = [0.205, 0.535, 0.765, 0.985]
    hubs = [(0.105, "sample_uid"), (0.350, "content_id"),
            (0.665, "benchmark_id"), (0.875, "canonical_role")]
    y_map = {
        2: [0.615, 0.445],
        3: [0.680, 0.525, 0.370],
        5: [0.720, 0.620, 0.520, 0.420, 0.320],
    }

    for index, (layer, colour, nx, cx) in enumerate(zip(layers, colours, node_x, count_x)):
        lines = database_lines(database[database["layer"] == layer])
        ys = y_map[len(lines)]
        halo_centre = float(np.mean(ys))
        halo_width = (cx - nx) + 0.020
        halo_height = max(ys) - min(ys) + 0.175
        ax.add_patch(Ellipse((nx + halo_width * 0.49, halo_centre), halo_width,
                             halo_height, transform=ax.transAxes,
                             facecolor=colour, edgecolor="none", alpha=0.065, zorder=0))
        ax.text(nx, 0.835, layer, transform=ax.transAxes, ha="left", va="center",
                fontsize=7.35, fontweight="bold", color=colour)
        ax.text(cx, 0.835, f"{len(lines)}", transform=ax.transAxes, ha="right", va="center",
                fontsize=7.35, fontweight="bold", color=colour)

        hub_x, _ = hubs[index]
        for (label, count), row_y in zip(lines, ys):
            ax.add_patch(FancyArrowPatch(
                (nx + 0.002, row_y), (hub_x, 0.185), transform=ax.transAxes,
                arrowstyle="-", connectionstyle="arc3,rad=0.12",
                linewidth=0.48, color=colour, alpha=0.48, zorder=1,
            ))
            ax.add_patch(Circle((nx, row_y), 0.0105, transform=ax.transAxes,
                                facecolor=colour, edgecolor="white", lw=0.55, zorder=4))
            ax.text(nx + 0.017, row_y, label, transform=ax.transAxes,
                    ha="left", va="center", fontsize=6.55, color=INK, zorder=5)
            ax.text(cx, row_y, f"{count:,}", transform=ax.transAxes,
                    ha="right", va="center", fontsize=6.65,
                    fontweight="bold", color=colour, zorder=5)

    for index, ((hub_x, key), colour) in enumerate(zip(hubs, colours)):
        ax.add_patch(Circle((hub_x, 0.165), 0.0145, transform=ax.transAxes,
                            facecolor="white", edgecolor=colour, lw=1.1, zorder=7))
        ax.add_patch(Circle((hub_x, 0.165), 0.0052, transform=ax.transAxes,
                            facecolor=colour, edgecolor="none", zorder=8))
        ax.text(hub_x, 0.085, key, transform=ax.transAxes, ha="center", va="center",
                fontsize=6.65, family="monospace", fontweight="bold", color=INK)
        if index < len(hubs) - 1:
            next_x = hubs[index + 1][0]
            ax.add_patch(FancyArrowPatch(
                (hub_x + 0.016, 0.165), (next_x - 0.016, 0.165),
                transform=ax.transAxes, arrowstyle="-|>", mutation_scale=8,
                linewidth=1.15, color="#617887", zorder=6,
            ))


def panel_c(ax: mpl.axes.Axes, flow: pd.DataFrame, split: pd.DataFrame) -> None:
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.01, 0.985, "Curation flow and locked evaluation partitions",
            transform=ax.transAxes, fontsize=9.15, fontweight="bold", va="top", color=INK)
    ax.text(0.99, 0.978, "branch widths encode split counts · loss labels are exact",
            transform=ax.transAxes, fontsize=6.75, va="top", ha="right", color=SLATE)
    indexed = flow.set_index("stage")
    stages = [
        ("Source records", "source records", INK),
        ("Role-ready", "role-aware eligible records", VIOLET),
        ("Unique images", "unique benchmark images", TEAL),
        ("Validation-eligible", "technical validation eligible", CORAL),
    ]
    centres = [0.055, 0.235, 0.415, 0.595]
    base_count = int(indexed.loc["source records", "count"])
    for index in range(len(centres) - 1):
        x0, x1 = centres[index], centres[index + 1]
        key = stages[index][1]
        count = int(indexed.loc[key, "count"])
        next_colour = stages[index + 1][2]
        ax.add_patch(FancyArrowPatch(
            (x0, 0.515), (x1, 0.515), transform=ax.transAxes,
            arrowstyle="-", connectionstyle=f"arc3,rad={0.035 * (-1) ** index}",
            linewidth=18.0 * count / base_count, color=next_colour,
            alpha=0.30, capstyle="round", zorder=1,
        ))
        ax.add_patch(FancyArrowPatch(
            (x0, 0.515), (x1, 0.515), transform=ax.transAxes,
            arrowstyle="-|>", mutation_scale=7.5,
            connectionstyle=f"arc3,rad={0.035 * (-1) ** index}",
            linewidth=0.80, color=next_colour, zorder=3,
        ))

    for (label, key, colour), centre in zip(stages, centres):
        count = int(indexed.loc[key, "count"])
        ax.add_patch(Circle((centre, 0.515), 0.016, transform=ax.transAxes,
                            facecolor="white", edgecolor=colour, lw=1.05, zorder=6))
        ax.add_patch(Circle((centre, 0.515), 0.006, transform=ax.transAxes,
                            facecolor=colour, edgecolor="none", zorder=7))
        ax.text(centre, 0.755, label, transform=ax.transAxes,
                ha="center", va="center", fontsize=6.75, fontweight="bold", color=SLATE)
        ax.text(centre, 0.660, f"{count:,}", transform=ax.transAxes,
                ha="center", va="center", fontsize=8.8, fontweight="bold", color=colour)

    losses = [
        (0.145, 0.235, "8 role-parsing\nexclusions", 8),
        (0.325, 0.175, "127-record reduction\nto unique images", 127),
        (0.505, 0.225, "Review holdout · 9", 9),
    ]
    for x, target_y, label, count in losses:
        linewidth = 0.85 + 1.75 * np.sqrt(count / 127.0)
        ax.add_patch(FancyArrowPatch(
            (x, 0.485), (x + 0.045, target_y + 0.045), transform=ax.transAxes,
            arrowstyle="-|>", mutation_scale=7.5, connectionstyle="arc3,rad=0.20",
            linewidth=linewidth, color=CORAL, alpha=0.88, zorder=5,
        ))
        ax.text(x + 0.050, target_y, label, transform=ax.transAxes,
                ha="center", va="top", fontsize=6.45, fontweight="bold",
                color=CORAL, linespacing=1.32)

    split_counts = split.groupby("split_display")["count"].sum().to_dict()
    train_count = int(split_counts.get("train", 0))
    val_count = int(split_counts.get("validation", 0))
    test_count = int(split_counts.get("test", 0))
    validation_total = train_count + val_count + test_count
    ax.text(0.790, 0.790, f"Fixed split · n = {validation_total:,}", transform=ax.transAxes,
            ha="center", va="center", fontsize=7.0, fontweight="bold", color=INK)
    branches = [
        ("Train", train_count, INK, 0.675),
        ("Validation", val_count, CYAN, 0.500),
        ("Test", test_count, GOLD, 0.345),
    ]
    for label, count, colour, endpoint_y in branches:
        branch_width = max(1.25, 17.0 * count / max(validation_total, 1))
        ax.add_patch(FancyArrowPatch(
            (0.615, 0.515), (0.845, endpoint_y), transform=ax.transAxes,
            arrowstyle="-", connectionstyle="arc3,rad=0.10",
            linewidth=branch_width, color=colour, alpha=0.82,
            capstyle="round", zorder=2,
        ))
        ax.add_patch(Circle((0.850, endpoint_y), 0.0085, transform=ax.transAxes,
                            facecolor=colour, edgecolor="white", lw=0.45, zorder=5))
        ax.text(0.870, endpoint_y, label, transform=ax.transAxes,
                ha="left", va="center", fontsize=6.65, color=SLATE, fontweight="bold")
        ax.text(0.985, endpoint_y, f"{count:,}", transform=ax.transAxes,
                ha="right", va="center", fontsize=7.05, color=colour, fontweight="bold")


def panel_d(ax: mpl.axes.Axes, sample: pd.Series, points: pd.DataFrame,
            transform: pd.Series) -> None:
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.01, 0.985, "Reversible source-to-axial coordinate trace",
            transform=ax.transAxes, fontsize=9.0, fontweight="bold", va="top", color=INK)
    ax.text(0.99, 0.985, "same benchmark_id · stored affine map · no local image enhancement",
            transform=ax.transAxes, fontsize=7.0, va="top", ha="right", color=SLATE)

    original = load_original(sample)
    normalized = affine_normalize(original, transform)
    source_ax = ax.inset_axes([0.015, 0.14, 0.345, 0.70], zorder=3)
    normalized_ax = ax.inset_axes([0.640, 0.14, 0.345, 0.70], zorder=3)
    show_fish(source_ax, original, points, x_col="x_source_px", y_col="y_source_px",
              width=int(sample["width"]), height=int(sample["height"]),
              margin_x=0.10, margin_y=0.28, annotate_ids={1, 2, 6})
    show_fish(normalized_ax, normalized, points, x_col="x_crop_px", y_col="y_crop_px",
              width=448, height=448, margin_x=0.10, margin_y=0.28,
              annotate_ids={1, 2, 6})
    source_ax.set_title("Original pixels", fontsize=7.5, fontweight="bold", color=CORAL, pad=2)
    normalized_ax.set_title("448 × 448 axial coordinates", fontsize=7.5,
                            fontweight="bold", color=TEAL, pad=2)

    rounded_box(ax, 0.385, 0.16, 0.230, 0.64, face="#F5F8FA",
                edge="#B8C7CF", linewidth=0.85, radius=0.018)
    ax.text(0.500, 0.710, "annotation-assisted affine map", transform=ax.transAxes,
            ha="center", va="center", fontsize=7.2, fontweight="bold", color=INK)
    ax.text(0.500, 0.595, "x′ = A x + b", transform=ax.transAxes,
            ha="center", va="center", fontsize=9.2, fontweight="bold", color=TEAL)
    ax.text(0.500, 0.480, "anchors 1 · 6 · 2", transform=ax.transAxes,
            ha="center", va="center", fontsize=7.1, color=SLATE)
    ax.text(0.500, 0.370, "crop_transforms", transform=ax.transAxes,
            ha="center", va="center", fontsize=7.0, family="monospace", color=VIOLET)
    ax.text(0.500, 0.285, "original  ↔  crops_448", transform=ax.transAxes,
            ha="center", va="center", fontsize=6.9, family="monospace", color=INK)
    ax.text(0.500, 0.205, "shared benchmark ID", transform=ax.transAxes,
            ha="center", va="center", fontsize=6.8, color=SLATE)
    ax.add_patch(FancyArrowPatch((0.355, 0.50), (0.385, 0.50), transform=ax.transAxes,
                                 arrowstyle="-|>", mutation_scale=9, color=CORAL, lw=0.9))
    ax.add_patch(FancyArrowPatch((0.615, 0.50), (0.645, 0.50), transform=ax.transAxes,
                                 arrowstyle="-|>", mutation_scale=9, color=TEAL, lw=0.9))
    ax.add_patch(FancyArrowPatch((0.645, 0.42), (0.615, 0.42), transform=ax.transAxes,
                                 arrowstyle="-|>", mutation_scale=8, color="#8A9AA5", lw=0.7))


def main() -> None:
    SOURCE_DATA.mkdir(parents=True, exist_ok=True)
    GENERATED.mkdir(parents=True, exist_ok=True)
    workflow_sample = pd.read_csv(SOURCE_DATA / "figure1_workflow_sample.csv").iloc[0]
    workflow_points = pd.read_csv(SOURCE_DATA / "figure1_workflow_points.csv")
    trace_sample = pd.read_csv(SOURCE_DATA / "figure1_trace_sample.csv").iloc[0]
    trace_points = pd.read_csv(SOURCE_DATA / "figure1_trace_points.csv")
    transforms = pd.read_csv(SOURCE_DATA / "figure1_representative_transforms.csv")
    flow = pd.read_csv(SOURCE_DATA / "figure1_flow.csv")
    split = pd.read_csv(SOURCE_DATA / "figure1_split_taxon.csv")
    database = pd.read_csv(SOURCE_DATA / "figure1_database_tables.csv")
    trace_transform = transforms[transforms["benchmark_id"] == trace_sample["benchmark_id"]].iloc[0]

    apply_publication_style()
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 8.3,
    })
    fig = plt.figure(figsize=(WIDTH_IN, 7.70), facecolor="white")
    grid = gridspec.GridSpec(
        4, 12, figure=fig, height_ratios=[1.15, 1.08, 0.78, 1.18],
        hspace=0.105, wspace=0.48, left=0.040, right=0.992,
        top=0.985, bottom=0.025,
    )
    ax_a = fig.add_subplot(grid[0, :])
    ax_b = fig.add_subplot(grid[1, :])
    ax_c = fig.add_subplot(grid[2, :])
    ax_d = fig.add_subplot(grid[3, :])
    panel_label(ax_a, "a", x=-0.036, y=1.00)
    panel_label(ax_b, "b", x=-0.036, y=1.00)
    panel_label(ax_c, "c", x=-0.036, y=1.00)
    panel_label(ax_d, "d", x=-0.036, y=1.00)
    panel_a_vector(ax_a, workflow_sample, workflow_points, database)
    panel_b(ax_b, database)
    panel_c(ax_c, flow, split)
    panel_d(ax_d, trace_sample, trace_points, trace_transform)
    output_stem = "figure1_resource"
    save_figure(fig, GENERATED / output_stem)
    print(json.dumps({
        "figure": output_stem,
        "mode": "submission-safe deterministic",
        "workflow_sample": str(workflow_sample["benchmark_id"]),
        "trace_sample": str(trace_sample["benchmark_id"]),
        "database_tables": int(len(database)),
        "output": str(GENERATED),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
