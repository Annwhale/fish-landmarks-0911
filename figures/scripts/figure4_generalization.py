#!/usr/bin/env python3
# ><(((o>  鱼类形态数据工具
"""Scientific Data Figure 4: source-shift and ontology-boundary validation."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.path import Path as MplPath
from matplotlib.patches import Circle, PathPatch, Rectangle
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from visual_style import (
    CORAL, CYAN, GOLD, GRID, INK, MAGENTA, SLATE, TEAL, VIOLET,
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
STANDARD = ROOT / "models/evaluation/yolo26m_pose_v1_original_seed20260825/test/per_image_metrics.csv"
CROSS = ROOT / "models/evaluation/yolo26m_cross_origin_original_seed20260825/test/per_image_metrics.csv"
MELOPS = ROOT / "models/evaluation/yolo26m_pose_v1_original_seed20260825/melops_external/per_image_metrics.csv"
SEED = 20260825
REPLICATES = 2000
FIG_WIDTH_MM = 183.0
WIDTH_IN = FIG_WIDTH_MM / 25.4
COLORS = {"standard original-image": CYAN, "cross-origin original-image": CORAL,
          "anchor": TEAL, "regional": GOLD, "derived": MAGENTA}
TAXON_SHORT = {
    "Carassius auratus": "C. auratus", "Coilia nasus": "C. nasus",
    "Ctenopharyngodon idella": "C. idella", "Culter alburnus": "C. alburnus",
    "Hypophthalmichthys molitrix": "H. molitrix",
    "Hypophthalmichthys nobilis": "H. nobilis",
    "Mylopharyngodon piceus": "M. piceus",
}
POINTS = [
    ("kp1_candidate_nme", 1, "snout", "1 snout", "anchor", "candidate"),
    ("kp5_candidate_nme", 5, "caudal_top", "5 caudal top", "anchor", "candidate"),
    ("kp7_candidate_nme", 7, "caudal_bottom", "7 caudal bottom", "anchor", "candidate"),
    ("kp8_candidate_nme", 8, "anal_fin", "8 anal-fin region", "regional", "candidate"),
    ("kp9_candidate_nme", 9, "pelvic", "9 pelvic region", "regional", "candidate"),
    ("kp11_candidate_nme", 11, "medium_opercular", "11 opercular region", "regional", "candidate"),
    ("kp6_candidate_nme", 6, "midpoint(caudal_top, caudal_bottom)", "6 caudal midpoint", "derived", "derived candidate"),
    ("kp10_candidate_nme", 10, "midpoint(pectoral_top, pectoral_bottom)", "10 pectoral midpoint", "derived", "derived candidate"),
]


def save_figure(fig: mpl.figure.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight",
                facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def mean_ci(values: pd.Series | np.ndarray) -> tuple[int, float, float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    rng = np.random.Generator(np.random.PCG64(SEED))
    indices = rng.integers(0, len(array), size=(REPLICATES, len(array)))
    means = array[indices].mean(axis=1)
    return len(array), float(array.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def prepare() -> dict[str, pd.DataFrame]:
    standard = pd.read_csv(STANDARD)
    cross = pd.read_csv(CROSS)
    melops = pd.read_csv(MELOPS)
    rows = []
    for protocol_name, frame in [("standard original-image", standard),
                                 ("cross-origin original-image", cross)]:
        n, mean, lo, hi = mean_ci(frame["nme_body_length"])
        _, pmean, plo, phi = mean_ci(frame["pck_0_05"].fillna(0))
        rows.append({
            "protocol": protocol_name, "representation": "original source image",
            "cohort_comparison": "separate protocols; not a paired comparison",
            "images": len(frame), "detected": int(frame["detected"].astype(bool).sum()),
            "nme_normalizer": "role-1-to-role-6 working axial span",
            "nme_n": n, "nme_mean": mean, "nme_ci95_low": lo, "nme_ci95_high": hi,
            "pck_mean_all_images": pmean, "pck_ci95_low": plo, "pck_ci95_high": phi,
        })
    protocol = pd.DataFrame(rows)
    taxa = []
    for taxon_name, frame in cross.groupby("taxon_candidate", sort=True):
        n, mean, lo, hi = mean_ci(frame["nme_body_length"])
        taxa.append({
            "taxon_candidate": taxon_name, "display": TAXON_SHORT.get(taxon_name, taxon_name),
            "images": len(frame), "detected": int(frame["detected"].astype(bool).sum()),
            "nme_n": n, "nme_mean": mean, "nme_ci95_low": lo, "nme_ci95_high": hi,
        })
    taxon = pd.DataFrame(taxa).sort_values("nme_mean")
    points = []
    for column, role_id, source_label, point_name, tier, mapping_status in POINTS:
        n, mean, lo, hi = mean_ci(melops[column])
        points.append({
            "column": column, "project_role_id": role_id, "melops_source_label": source_label,
            "point": point_name, "tier": tier, "mapping_status": mapping_status,
            "target_rule": "midpoint" if tier == "derived" else "direct",
            "evaluated_images": n, "valid_point_pairs": n,
            "excluded_source_values": "two images lack an eligible source value"
            if source_label == "medium_opercular" and n == 503 else "none",
            "nme_normalizer": "Melops snout-to-caudal-midpoint span",
            "nme_mean": mean, "nme_ci95_low": lo, "nme_ci95_high": hi,
        })
    point = pd.DataFrame(points)
    SOURCE_OUT.mkdir(parents=True, exist_ok=True)
    protocol.to_csv(SOURCE_OUT / "figure4_protocol_summary.csv", index=False)
    taxon.to_csv(SOURCE_OUT / "figure4_cross_origin_by_taxon.csv", index=False)
    point.to_csv(SOURCE_OUT / "figure4_melops_point_summary.csv", index=False)
    return {"protocol": protocol, "taxon": taxon, "point": point,
            "cross_frame": cross, "melops_frame": melops}


def ribbon(ax: mpl.axes.Axes, x0: float, x1: float, y0a: float, y0b: float,
           y1a: float, y1b: float, color: str, alpha: float) -> None:
    dx = (x1 - x0) * 0.42
    vertices = [(x0, y0a), (x0 + dx, y0a), (x1 - dx, y1a), (x1, y1a),
                (x1, y1b), (x1 - dx, y1b), (x0 + dx, y0b), (x0, y0b), (x0, y0a)]
    codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
             MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
             MplPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MplPath(vertices, codes), facecolor=color,
                           edgecolor="none", alpha=alpha))


def ranges(counts: list[int], gap: float, bottom: float = 0.08,
           top: float = 0.92) -> list[tuple[float, float]]:
    usable = top - bottom - gap * max(0, len(counts) - 1)
    total = max(sum(counts), 1)
    output = []
    cursor = top
    for count in counts:
        height = usable * count / total
        output.append((cursor - height, cursor)); cursor -= height + gap
    return output


def panel_a(ax: mpl.axes.Axes, frame: pd.DataFrame) -> None:
    """Three-stage source-protocol to taxon to detection alluvial."""
    working = frame.copy()
    working["detected_label"] = np.where(
        working["detected"].astype(str).str.lower().eq("true"), "detected", "missed"
    )
    taxa = working["taxon_candidate"].value_counts().sort_values(ascending=False).index.tolist()
    outcomes = ["detected", "missed"]
    table = pd.crosstab(working.taxon_candidate, working.detected_label).reindex(
        index=taxa, columns=outcomes, fill_value=0
    )
    total = len(working)
    taxon_counts = table.sum(axis=1).astype(int).tolist()
    outcome_counts = table.sum(axis=0).astype(int).tolist()
    source_range = ranges([total], 0)[0]
    taxon_ranges = ranges(taxon_counts, 0.025)
    outcome_ranges = ranges(outcome_counts, 0.025)
    taxon_colors = [CYAN, GOLD, VIOLET, CORAL]

    source_cursor = source_range[0]
    for i, (taxon, count) in enumerate(zip(taxa, taxon_counts)):
        y0a = source_cursor; y0b = y0a + (source_range[1] - source_range[0]) * count / total
        y1a, y1b = taxon_ranges[i]
        ribbon(ax, 0.10, 0.47, y0a, y0b, y1a, y1b, taxon_colors[i], 0.60)
        source_cursor = y0b
    outcome_cursor = {name: outcome_ranges[i][0] for i, name in enumerate(outcomes)}
    outcome_scale = {name: (outcome_ranges[i][1] - outcome_ranges[i][0]) / max(outcome_counts[i], 1)
                     for i, name in enumerate(outcomes)}
    for i, taxon in enumerate(taxa):
        taxon_cursor = taxon_ranges[i][0]
        taxon_scale = (taxon_ranges[i][1] - taxon_ranges[i][0]) / max(taxon_counts[i], 1)
        for outcome in outcomes:
            count = int(table.loc[taxon, outcome])
            if count == 0:
                continue
            y0a = taxon_cursor; y0b = y0a + count * taxon_scale
            y1a = outcome_cursor[outcome]; y1b = y1a + count * outcome_scale[outcome]
            ribbon(ax, 0.50, 0.88, y0a, y0b, y1a, y1b,
                   TEAL if outcome == "detected" else CORAL, 0.58)
            taxon_cursor = y0b; outcome_cursor[outcome] = y1b

    ax.add_patch(Rectangle((0.075, source_range[0]), 0.025,
                           source_range[1] - source_range[0], color=INK, ec="none"))
    ax.text(0.065, sum(source_range) / 2, f"cross-origin\nprotocol\nn={total:,}",
            ha="right", va="center", fontsize=7.15, color=INK, fontweight="bold")
    for i, (taxon, count, (low, high)) in enumerate(zip(taxa, taxon_counts, taxon_ranges)):
        ax.add_patch(Rectangle((0.47, low), 0.03, high - low, color=taxon_colors[i], ec="none"))
        ax.text(0.455, (low + high) / 2, f"{TAXON_SHORT.get(taxon, taxon)}  {count:,}",
                ha="right", va="center", fontsize=7.1, color=INK, fontstyle="italic")
    for outcome, count, (low, high) in zip(outcomes, outcome_counts, outcome_ranges):
        colour = TEAL if outcome == "detected" else CORAL
        # 未命中节点保留轮廓，避免窄条不可见。
        ax.add_patch(Rectangle((0.88, low), 0.025, max(high - low, 0.003),
                               facecolor=colour, edgecolor=INK if outcome == "missed" else "none",
                               lw=0.65))
        ax.text(0.915, (low + high) / 2, f"{outcome}  {count:,}",
                ha="left", va="center", fontsize=7.1, color=INK,
                fontweight="bold" if outcome == "missed" else "normal")
    ax.set_xlim(-0.05, 1.08); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("Cross-origin evaluation flow", loc="left", fontsize=9.0,
                 fontweight="bold", color=INK, pad=4)
    ax.text(0.99, 1.02, "all ribbon widths use exact image counts",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.0, color=SLATE)


def panel_b(ax: mpl.axes.Axes, data: pd.DataFrame) -> None:
    style_axis(ax)
    points = {}
    for row in data.itertuples():
        x = row.nme_mean * 100; xlo = row.nme_ci95_low * 100; xhi = row.nme_ci95_high * 100
        y = row.pck_mean_all_images; ylo = row.pck_ci95_low; yhi = row.pck_ci95_high
        color = COLORS[row.protocol]
        ax.errorbar(x, y, xerr=[[x - xlo], [xhi - x]], yerr=[[y - ylo], [yhi - y]],
                    fmt="o", ms=6.0, color=color, ecolor=SLATE, capsize=2.2, lw=0.9, zorder=3)
        points[row.protocol] = (x, y)
        label_text = "standard" if row.protocol.startswith("standard") else "cross-origin"
        if label_text == "standard":
            ax.annotate(f"{label_text}\nn={row.images:,}", (x, y), xytext=(5, 0),
                        textcoords="offset points", fontsize=7.05, color=color,
                        va="center", ha="left")
        else:
            # 说明文字留在b面板内，避免与c面板刻度相撞。
            ax.annotate(f"{label_text}\nn={row.images:,}", (x, y),
                        xytext=(0.42, 0.08), textcoords=ax.transAxes,
                        fontsize=7.05, color=color, va="bottom", ha="center",
                        arrowprops={"arrowstyle": "-", "color": color, "lw": 0.65})
    start = points["standard original-image"]; end = points["cross-origin original-image"]
    ax.annotate("", xy=end, xytext=start,
                arrowprops={"arrowstyle": "-|>", "color": CORAL, "lw": 1.2,
                            "connectionstyle": "arc3,rad=-0.10"})
    ax.set_xlabel("detected-image NME (% span)")
    ax.set_ylabel("all-image PCK@0.05")
    ax.set_title("Domain-shift vector", loc="left", fontweight="bold", color=INK)
    ax.grid(color=GRID, lw=0.45); ax.set_axisbelow(True)


def panel_c(ax: mpl.axes.Axes, frame: pd.DataFrame) -> None:
    """Ridgeline distributions retain full per-image evidence."""
    taxa = sorted(frame["taxon_candidate"].unique())
    palette = [TEAL, GOLD, CORAL, VIOLET]
    finite_all = frame["nme_body_length"].to_numpy(float)
    finite_all = finite_all[np.isfinite(finite_all)] * 100
    xgrid = np.linspace(0, float(np.quantile(finite_all, 0.995)), 280)
    for i, (taxon, color) in enumerate(zip(taxa, palette)):
        values = frame.loc[frame.taxon_candidate == taxon, "nme_body_length"].to_numpy(float)
        values = values[np.isfinite(values)] * 100
        density = gaussian_kde(values)(xgrid)
        density = density / max(density.max(), 1e-12) * 0.72
        ax.fill_between(xgrid, i, i - density, color=color, alpha=0.72, lw=0)
        ax.plot(xgrid, i - density, color=color, lw=0.9)
        median = float(np.median(values))
        ax.plot([median, median], [i - 0.72, i + 0.06], color=INK, lw=0.75)
        ax.text(median, i - 0.78, f"{median:.1f}", ha="center", va="top", fontsize=6.85, color=INK)
    labels = [f"{TAXON_SHORT.get(t, t)}  n={int((frame.taxon_candidate == t).sum()):,}" for t in taxa]
    ax.set_yticks(range(len(taxa)), labels, fontsize=7.0)
    ax.set_ylim(len(taxa) - 0.4, -1.0); ax.set_xlim(xgrid.min(), xgrid.max())
    ax.set_xlabel("cross-origin NME (% span)")
    ax.set_title("Taxon-stratified error ridgelines", loc="left", fontweight="bold", color=INK)
    style_axis(ax, grid_axis="x")


def panel_d(ax: mpl.axes.Axes, data: pd.DataFrame) -> None:
    """Concentric ontology-crosswalk network with true circular nodes."""
    ordered = data.sort_values("project_role_id").reset_index(drop=True)
    angles = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, len(ordered), endpoint=False)
    role_pos = {int(row.project_role_id): np.array([np.cos(angle), np.sin(angle)])
                for angle, row in zip(angles, ordered.itertuples())}
    tier_pos = {"anchor": np.array([-0.32, 0.18]), "regional": np.array([0.32, 0.18]),
                "derived": np.array([0.0, -0.34])}
    values = ordered.nme_mean.to_numpy(float) * 100
    norm = mpl.colors.Normalize(vmin=float(values.min()), vmax=float(values.max()))
    cmap = mpl.colors.LinearSegmentedColormap.from_list("ontology_error", [TEAL, GOLD, CORAL, "#7C2146"])
    for row in ordered.itertuples():
        start = role_pos[int(row.project_role_id)]; end = tier_pos[row.tier]
        control = (start + end) * 0.36
        path = MplPath([start, control, end], [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3])
        value = float(row.nme_mean) * 100
        ax.add_patch(PathPatch(path, facecolor="none", edgecolor=cmap(norm(value)),
                               lw=0.9 + 2.0 * norm(value), alpha=0.72, zorder=1))
    short = {1: "snout", 5: "caudal top", 7: "caudal bottom", 8: "anal",
             9: "pelvic", 11: "opercular", 6: "caudal mid", 10: "pectoral mid"}
    for row in ordered.itertuples():
        point = role_pos[int(row.project_role_id)]
        colour = COLORS[row.tier]
        ax.add_patch(Circle(point, 0.080, facecolor=colour, edgecolor="white", lw=0.9, zorder=3))
        align = "left" if point[0] >= 0 else "right"
        ax.text(point[0] + (0.10 if point[0] >= 0 else -0.10), point[1],
                f"{int(row.project_role_id)} {short[int(row.project_role_id)]}",
                ha=align, va="center", fontsize=6.85, color=INK)
    for tier, point in tier_pos.items():
        ax.add_patch(Circle(point, 0.125, facecolor="white", edgecolor=COLORS[tier], lw=1.6, zorder=4))
        ax.text(point[0], point[1], tier, ha="center", va="center", fontsize=6.95,
                color=COLORS[tier], fontweight="bold", zorder=5)
    ax.set_xlim(-1.55, 1.55); ax.set_ylim(-1.27, 1.25)
    ax.set_aspect("equal", adjustable="box"); ax.axis("off")
    ax.set_title("External ontology crosswalk network", loc="left", fontweight="bold", color=INK, pad=4)
    ax.text(0.50, -0.01, f"edge colour/width = candidate NME ({values.min():.1f}–{values.max():.1f}%) · 505 images",
            transform=ax.transAxes, ha="center", va="top", fontsize=6.85, color=SLATE)


def main() -> None:
    data = prepare(); apply_publication_style()
    fig = plt.figure(figsize=(WIDTH_IN, 6.25), facecolor="white")
    gs = gridspec.GridSpec(2, 24, figure=fig, height_ratios=[0.74, 1.0],
                           hspace=0.24, wspace=0.95, left=0.065, right=0.980,
                           top=0.955, bottom=0.075)
    axes = [fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, :7]),
            fig.add_subplot(gs[1, 7:15]), fig.add_subplot(gs[1, 15:])]
    for axis, letter, x in zip(axes, "abcd", [-0.045, -0.16, -0.13, -0.10]):
        panel_label(axis, letter, x=x, y=1.03)
    panel_a(axes[0], data["cross_frame"])
    panel_b(axes[1], data["protocol"])
    panel_c(axes[2], data["cross_frame"])
    panel_d(axes[3], data["point"])
    save_figure(fig, GENERATED / "figure4_generalization")
    print(json.dumps({"figure": "figure4_generalization",
                      "standard_images": int(data["protocol"].iloc[0].images),
                      "cross_origin_images": int(data["protocol"].iloc[1].images),
                      "melops_images": 505, "actual_image_panels": 0}, indent=2))


if __name__ == "__main__":
    main()
