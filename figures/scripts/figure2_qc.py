#!/usr/bin/env python3
# ><(((o>  鱼类形态数据工具
"""Scientific Data Figure 2: multi-view annotation-quality audit."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Rectangle
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage, leaves_list

from visual_style import (
    CORAL, CYAN, GOLD, GRID, INK, MAGENTA, PALE, SLATE, TEAL, VIOLET,
    apply_publication_style, panel_label, style_axis,
)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 8.3,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})

ROOT = Path(__file__).resolve().parents[2]
SOURCE_OUT = ROOT / "figures" / "source_data"
GENERATED = ROOT / "figures" / "generated"
FIG_WIDTH_MM = 183.0
WIDTH_IN = FIG_WIDTH_MM / 25.4
THRESHOLD = 0.02
TAXON_SHORT = {
    "Carassius auratus": "C. auratus", "Coilia nasus": "C. nasus",
    "Ctenopharyngodon idella": "C. idella", "Culter alburnus": "C. alburnus",
    "Hypophthalmichthys molitrix": "H. molitrix",
    "Hypophthalmichthys nobilis": "H. nobilis",
    "Megalobrama amblycephala": "M. amblycephala",
    "Mylopharyngodon piceus": "M. piceus", "Siniperca chuatsi": "S. chuatsi",
}


def read_data() -> dict[str, pd.DataFrame]:
    return {
        "reliability": pd.read_csv(SOURCE_OUT / "figure2_repeat_point_summary.csv"),
        "agreement": pd.read_csv(SOURCE_OUT / "figure2_duplicate_annotation_agreement.csv"),
        "discrepancy": pd.read_csv(SOURCE_OUT / "figure2_discrepancy_by_group.csv"),
    }


def save_figure(fig: mpl.figure.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight",
                facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    span = np.nanmax(values) - np.nanmin(values)
    return np.zeros_like(values) if span <= 1e-12 else (values - np.nanmin(values)) / span


def panel_a(ax: mpl.axes.Axes, frame: pd.DataFrame) -> None:
    """Clustered omics-style reliability matrix."""
    ax.axis("off")
    display = np.column_stack([
        frame["mean_nme_image_diagonal"].to_numpy(float) * 100,
        (frame["ci95_high"] - frame["ci95_low"]).to_numpy(float) * 100,
        frame["icc_a1_x"].to_numpy(float),
        frame["icc_a1_y"].to_numpy(float),
    ])
    risk = np.column_stack([
        minmax(display[:, 0]), minmax(display[:, 1]),
        minmax(1 - display[:, 2]), minmax(1 - display[:, 3]),
    ])
    standardized = (risk - risk.mean(axis=0)) / np.where(risk.std(axis=0) == 0, 1, risk.std(axis=0))
    tree = linkage(standardized, method="average", metric="euclidean")
    order = leaves_list(tree)

    # 可靠性矩阵占满上排，为标签与数值保留空间。
    dendro_ax = ax.inset_axes([0.015, 0.12, 0.105, 0.72])
    dendrogram(tree, orientation="left", no_labels=True, color_threshold=0,
               above_threshold_color=SLATE, ax=dendro_ax)
    for collection in dendro_ax.collections:
        collection.set_linewidth(0.9)
    dendro_ax.axis("off")

    heat_ax = ax.inset_axes([0.285, 0.12, 0.445, 0.72])
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "qc_risk", ["#FFF8E8", GOLD, CORAL, "#7C2146"]
    )
    heat_ax.imshow(risk[order], aspect="auto", cmap=cmap, vmin=0, vmax=1,
                   interpolation="nearest")
    short_labels = {
        1: "1  snout", 2: "2  nape", 3: "3  dorsal origin",
        4: "4  dorsal insertion", 5: "5  upper caudal",
        6: "6  caudal midpoint", 7: "7  lower caudal",
        8: "8  anal origin", 9: "9  pelvic origin",
        10: "10  pectoral base", 11: "11  opercular margin",
    }
    labels = [short_labels[int(frame.iloc[i].canonical_id)] for i in order]
    heat_ax.set_yticks(np.arange(len(order)), labels, fontsize=7.35)
    heat_ax.set_xticks(np.arange(4), ["mean NME\n(%)", "CI width\n(pp)", "ICC x", "ICC y"],
                       fontsize=7.25)
    heat_ax.xaxis.tick_top()
    heat_ax.tick_params(axis="x", length=0, pad=3)
    heat_ax.tick_params(axis="y", length=0, pad=3)
    for row_pos, source_index in enumerate(order):
        for column in range(4):
            value = display[source_index, column]
            text_value = f"{value:.2f}" if column < 2 else f"{value:.3f}"
            heat_ax.text(column, row_pos, text_value, ha="center", va="center",
                         fontsize=7.05, fontweight="bold" if risk[source_index, column] > 0.72 else "normal",
                         color="white" if risk[source_index, column] > 0.56 else INK)
    for spine in heat_ax.spines.values():
        spine.set_visible(False)
    ax.text(0.0, 1.01, "Role reliability clustered across four QC dimensions",
            transform=ax.transAxes, fontsize=9.0, fontweight="bold", color=INK, va="bottom")
    ax.text(0.067, 0.075, "average-linkage\nrole tree", transform=ax.transAxes,
            ha="center", va="top", fontsize=6.9, color=SLATE)

    key_ax = ax.inset_axes([0.770, 0.15, 0.220, 0.65])
    key_ax.set_xlim(0, 1); key_ax.set_ylim(0, 1); key_ax.axis("off")
    key_ax.text(0.02, 0.97, "Four complementary checks", ha="left", va="top",
                fontsize=7.6, fontweight="bold", color=INK)
    key_rows = [
        ("Repeat error", "mean image-diagonal NME", CORAL),
        ("Uncertainty", "bootstrap interval width", GOLD),
        ("Horizontal", "absolute-agreement ICC x", CYAN),
        ("Vertical", "absolute-agreement ICC y", VIOLET),
    ]
    for i, (heading, detail, colour) in enumerate(key_rows):
        yy = 0.78 - i * 0.19
        key_ax.add_patch(Rectangle((0.02, yy - 0.045), 0.055, 0.105,
                                   facecolor=colour, edgecolor="none"))
        key_ax.text(0.105, yy + 0.025, heading, ha="left", va="center",
                    fontsize=7.15, fontweight="bold", color=INK)
        key_ax.text(0.105, yy - 0.045, detail, ha="left", va="center",
                    fontsize=6.85, color=SLATE)
    key_ax.text(0.02, 0.02, "Darker cells indicate higher\nwithin-metric review risk.",
                ha="left", va="bottom", fontsize=6.9, color=INK)


def classify_intersections(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["above 2%"] = result["pairwise_mean_error_image_diagonal"] >= THRESHOLD
    result["origin conflict"] = result["origin_conflict"].astype(bool)
    result[">2 records"] = result["record_count"].astype(int) > 2
    return result


def panel_b(ax: mpl.axes.Axes, frame: pd.DataFrame) -> None:
    """Exact UpSet of repeated-image QC flags."""
    ax.axis("off")
    flags = ["above 2%", "origin conflict", ">2 records"]
    patterns = frame[flags].astype(bool).value_counts().reset_index(name="count")
    patterns = patterns.sort_values(["count", *flags], ascending=[False, False, False, False]).reset_index(drop=True)

    bar = ax.inset_axes([0.36, 0.46, 0.61, 0.43])
    xs = np.arange(len(patterns))
    colors = [CORAL if row["above 2%"] else VIOLET if row["origin conflict"] else TEAL
              for _, row in patterns.iterrows()]
    bar.bar(xs, patterns["count"], color=colors, width=0.68)
    bar.set_xlim(-0.6, len(patterns) - 0.4)
    bar.set_ylim(0, float(patterns["count"].max()) * 1.22)
    bar.set_ylabel("intersection size")
    bar.set_xticks([])
    style_axis(bar, grid_axis="y")
    for x, count in zip(xs, patterns["count"]):
        bar.text(x, count + patterns["count"].max() * 0.025, str(int(count)),
                 ha="center", va="bottom", fontsize=7.15, fontweight="bold", color=INK)

    matrix = ax.inset_axes([0.36, 0.08, 0.61, 0.29])
    matrix.set_xlim(-0.6, len(patterns) - 0.4)
    matrix.set_ylim(-0.6, len(flags) - 0.4)
    for x, (_, row) in enumerate(patterns.iterrows()):
        active = [i for i, flag in enumerate(flags) if bool(row[flag])]
        if active:
            matrix.plot([x, x], [min(active), max(active)], color=INK, lw=1.05, zorder=1)
        for y, flag in enumerate(flags):
            matrix.scatter(x, y, s=30, color=INK if bool(row[flag]) else "#D8DEE2",
                           edgecolor="white", linewidth=0.45, zorder=2)
    sizes = [int(frame[flag].astype(bool).sum()) for flag in flags]
    matrix.set_yticks(range(len(flags)),
                      [f"{flag}  ({size})" for flag, size in zip(flags, sizes)], fontsize=6.9)
    matrix.set_xticks(xs, ["clean" if not any(bool(row[f]) for f in flags) else str(i + 1)
                           for i, (_, row) in enumerate(patterns.iterrows())], fontsize=6.8)
    matrix.invert_yaxis()
    matrix.tick_params(length=0, pad=3)
    for spine in matrix.spines.values():
        spine.set_visible(False)

    ax.text(0.0, 1.01, "Repeated-image UpSet audit",
            transform=ax.transAxes, fontsize=8.9, fontweight="bold", color=INK, va="bottom")


def ribbon(ax: mpl.axes.Axes, x0: float, x1: float, y0a: float, y0b: float,
           y1a: float, y1b: float, color: str, alpha: float = 0.62) -> None:
    dx = (x1 - x0) * 0.45
    vertices = [
        (x0, y0a), (x0 + dx, y0a), (x1 - dx, y1a), (x1, y1a),
        (x1, y1b), (x1 - dx, y1b), (x0 + dx, y0b), (x0, y0b), (x0, y0a),
    ]
    codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
             MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
             MplPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MplPath(vertices, codes), facecolor=color,
                           edgecolor="none", alpha=alpha))


def node_ranges(counts: list[int], bottom: float = 0.05, top: float = 0.95,
                gap: float = 0.012) -> list[tuple[float, float]]:
    available = top - bottom - gap * max(0, len(counts) - 1)
    total = max(sum(counts), 1)
    ranges = []
    cursor = top
    for count in counts:
        height = available * count / total
        ranges.append((cursor - height, cursor))
        cursor -= height + gap
    return ranges


def panel_c(ax: mpl.axes.Axes, frame: pd.DataFrame) -> None:
    """Taxon-to-QC-outcome alluvial with exact record widths."""
    frame = classify_intersections(frame)
    frame["outcome"] = np.select(
        [frame["above 2%"] & frame[">2 records"], frame["above 2%"], frame["origin conflict"]],
        ["above 2% + >2 records", "above 2%", "origin conflict"], default="clean"
    )
    taxa = frame["taxon_candidate"].value_counts().sort_values(ascending=False).index.tolist()
    outcomes = ["clean", "origin conflict", "above 2%", "above 2% + >2 records"]
    table = pd.crosstab(frame["taxon_candidate"], frame["outcome"]).reindex(
        index=taxa, columns=outcomes, fill_value=0
    )
    left_counts = table.sum(axis=1).astype(int).tolist()
    right_counts = table.sum(axis=0).astype(int).tolist()
    left_ranges = node_ranges(left_counts, gap=0.009)
    right_ranges = node_ranges(right_counts, gap=0.018)
    outcome_colors = {"clean": TEAL, "origin conflict": VIOLET,
                      "above 2%": CORAL, "above 2% + >2 records": MAGENTA}
    left_cursor = {taxon: left_ranges[i][0] for i, taxon in enumerate(taxa)}
    right_cursor = {outcome: right_ranges[i][0] for i, outcome in enumerate(outcomes)}
    left_scale = {taxon: (left_ranges[i][1] - left_ranges[i][0]) / max(left_counts[i], 1)
                  for i, taxon in enumerate(taxa)}
    right_scale = {outcome: (right_ranges[i][1] - right_ranges[i][0]) / max(right_counts[i], 1)
                   for i, outcome in enumerate(outcomes)}
    for taxon in taxa:
        for outcome in outcomes:
            count = int(table.loc[taxon, outcome])
            if count == 0:
                continue
            y0a = left_cursor[taxon]; y0b = y0a + count * left_scale[taxon]
            y1a = right_cursor[outcome]; y1b = y1a + count * right_scale[outcome]
            ribbon(ax, 0.10, 0.88, y0a, y0b, y1a, y1b, outcome_colors[outcome], 0.56)
            left_cursor[taxon] = y0b; right_cursor[outcome] = y1b
    for (taxon, count, (low, high)) in zip(taxa, left_counts, left_ranges):
        ax.add_patch(Rectangle((0.075, low), 0.025, high - low, color=INK, ec="none"))
    label_ys = np.linspace(0.91, 0.09, len(taxa))
    for taxon, count, (low, high), label_y in zip(taxa, left_counts, left_ranges, label_ys):
        center = (low + high) / 2
        ax.plot([0.02, 0.073], [label_y, center], color=SLATE, lw=0.45)
        ax.text(0.015, label_y, f"{TAXON_SHORT.get(taxon, taxon)}  {count}",
                ha="right", va="center", fontsize=6.85, color=INK, fontstyle="italic")
    for outcome, count, (low, high) in zip(outcomes, right_counts, right_ranges):
        ax.add_patch(Rectangle((0.88, low), 0.025, high - low,
                               color=outcome_colors[outcome], ec="none"))
        ax.text(0.915, (low + high) / 2, f"{outcome}  {count}", ha="left", va="center",
                fontsize=6.85, color=INK)
    ax.set_xlim(-0.18, 1.22); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("Taxon → review outcome", loc="left",
                 fontsize=8.9, fontweight="bold", color=INK, pad=4)
    ax.text(0.50, -0.025, "ribbon width = exact repeated-image count (n=104)",
            transform=ax.transAxes, ha="center", va="top", fontsize=6.8, color=SLATE)


def panel_d(ax: mpl.axes.Axes, frame: pd.DataFrame) -> None:
    """Source-group precision heatmap with count annotation."""
    ax.axis("off")
    ordered = frame.sort_values(["maximum", "median"], ascending=False).reset_index(drop=True)
    matrix = ordered[["median", "maximum"]].to_numpy(float) * 1e4
    heat = ax.inset_axes([0.31, 0.07, 0.43, 0.83])
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "precision", ["#FFF7E1", GOLD, CORAL, "#6C1D45"]
    )
    norm = mpl.colors.Normalize(vmin=float(matrix.min()), vmax=float(matrix.max()))
    heat.imshow(matrix, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    heat.set_xticks([0, 1], ["median", "maximum"], fontsize=7.0)
    heat.set_yticks(np.arange(len(ordered)), ordered["display_label"], fontsize=6.85)
    heat.tick_params(length=0, pad=3)
    for i in range(len(ordered)):
        for j in range(2):
            value = matrix[i, j]
            heat.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=6.75,
                      color="white" if norm(value) > 0.52 else INK)
    for spine in heat.spines.values():
        spine.set_visible(False)
    counts = ax.inset_axes([0.80, 0.07, 0.18, 0.83])
    ypos = np.arange(len(ordered))
    counts.barh(ypos, ordered["count"], color=CYAN, height=0.64)
    counts.set_ylim(len(ordered) - 0.5, -0.5)
    counts.set_yticks([]); counts.set_xlabel("records", fontsize=6.9)
    counts.tick_params(axis="x", labelsize=6.7)
    style_axis(counts, grid_axis="x")
    for y, value in zip(ypos, ordered["count"]):
        counts.text(value, y, f" {int(value)}", va="center", ha="left", fontsize=6.7, color=INK)
    ax.text(0.0, 1.01, "Serialization precision",
            transform=ax.transAxes, fontsize=8.9, fontweight="bold", color=INK, va="bottom")
    ax.text(0.98, 0.95, "heat values x 10^-4 px",
            transform=ax.transAxes, fontsize=6.8, color=SLATE, ha="right")


def main() -> None:
    data = read_data()
    apply_publication_style()
    fig = plt.figure(figsize=(WIDTH_IN, 6.90), facecolor="white")
    gs = gridspec.GridSpec(2, 12, figure=fig, height_ratios=[1.08, 0.94],
                           hspace=0.25, wspace=0.95, left=0.065, right=0.985,
                           top=0.955, bottom=0.065)
    axes = [fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, :4]),
            fig.add_subplot(gs[1, 4:8]), fig.add_subplot(gs[1, 8:])]
    for axis, letter, x in zip(axes, "abcd", [-0.045, -0.14, -0.14, -0.14]):
        panel_label(axis, letter, x=x, y=1.02)
    panel_a(axes[0], data["reliability"])
    panel_b(axes[1], classify_intersections(data["agreement"]))
    panel_c(axes[2], data["agreement"])
    panel_d(axes[3], data["discrepancy"])
    save_figure(fig, GENERATED / "figure2_qc")
    agreement = data["agreement"]
    print(json.dumps({
        "figure": "figure2_qc", "agreement_n": int(len(agreement)),
        "above_threshold": int((agreement.pairwise_mean_error_image_diagonal >= THRESHOLD).sum()),
        "origin_conflicts": int(agreement.origin_conflict.astype(bool).sum()),
        "multi_record": int((agreement.record_count > 2).sum()),
    }, indent=2))


if __name__ == "__main__":
    main()
