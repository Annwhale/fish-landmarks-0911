#!/usr/bin/env python3
# ><(((o>  鱼类形态数据工具
"""Scientific Data Figure 3: model utility and geometry-specific failure modes."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.path import Path as MplPath
from matplotlib.patches import Circle, PathPatch
import numpy as np
import pandas as pd

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
SEED = 20260825
REPLICATES = 2000
FIG_WIDTH_MM = 183.0
WIDTH_IN = FIG_WIDTH_MM / 25.4
MODELS = {
    "YOLO pose": ROOT / "models/evaluation/yolo26m_pose_v1_axial_seed20260825/test/per_image_metrics.csv",
    "ResNet-34": ROOT / "models/evaluation/resnet34_heatmap_v1_seed20260825/test/per_image_metrics.csv",
    "DINOv2": ROOT / "models/evaluation/dinov2_vitl14_reg_heatmap_v1_seed20260825/test/per_image_metrics.csv",
    "DINOv2 + geometry": ROOT / "models/evaluation/dinov2_vitl14_reg_geometry_v1_seed20260825/test/per_image_metrics.csv",
}
ORIGINAL = ROOT / "models/evaluation/yolo26m_pose_v1_original_seed20260825/test/per_image_metrics.csv"
COLORS = {"YOLO pose": CYAN, "ResNet-34": TEAL, "DINOv2": GOLD,
          "DINOv2 + geometry": MAGENTA}
GEOMETRY = [
    ("axial_span_1_6_relative_error", "axial span 1–6"),
    ("dorsal_base_3_4_relative_error", "dorsal base 3–4"),
    ("caudal_depth_5_7_relative_error", "caudal depth 5–7"),
    ("head_chord_2_10_relative_error", "head chord 2–10"),
    ("ventral_base_8_9_relative_error", "ventral base 8–9"),
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


def paired_ci(first: pd.Series, second: pd.Series) -> tuple[int, float, float, float]:
    a = np.asarray(first, dtype=float); b = np.asarray(second, dtype=float)
    keep = np.isfinite(a) & np.isfinite(b); delta = a[keep] - b[keep]
    rng = np.random.Generator(np.random.PCG64(SEED))
    indices = rng.integers(0, len(delta), size=(REPLICATES, len(delta)))
    means = delta[indices].mean(axis=1)
    return len(delta), float(delta.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def prepare() -> dict[str, object]:
    frames = {name: pd.read_csv(path).sort_values("benchmark_id").set_index("benchmark_id")
              for name, path in MODELS.items()}
    original = pd.read_csv(ORIGINAL).sort_values("benchmark_id").set_index("benchmark_id")
    common = sorted(set.intersection(*(set(frame.index) for frame in [*frames.values(), original])))
    frames = {name: frame.loc[common].copy() for name, frame in frames.items()}
    original = original.loc[common].copy()

    summary_rows = []
    for name, frame in frames.items():
        n, mean, lo, hi = mean_ci(frame.nme_body_length)
        _, pmean, plo, phi = mean_ci(frame.pck_0_05.fillna(0))
        summary_rows.append({
            "model": name, "images": len(frame), "detected": int(frame.detected.astype(bool).sum()),
            "representation": "annotation-assisted axial",
            "nme_normalizer": "role-1-to-role-6 working axial span",
            "nme_n": n, "nme_mean": mean, "nme_ci95_low": lo, "nme_ci95_high": hi,
            "pck_mean_all_images": pmean, "pck_ci95_low": plo, "pck_ci95_high": phi,
        })
    summary = pd.DataFrame(summary_rows)

    paired_rows = []
    for label, first, second in [
        ("original − assisted axial YOLO", original, frames["YOLO pose"]),
        ("geometry − DINOv2", frames["DINOv2 + geometry"], frames["DINOv2"]),
    ]:
        n, mean, lo, hi = paired_ci(first.nme_body_length, second.nme_body_length)
        _, pmean, plo, phi = paired_ci(first.pck_0_05.fillna(0), second.pck_0_05.fillna(0))
        paired_rows.append({
            "comparison": label, "n": n, "nme_delta": mean,
            "nme_ci95_low": lo, "nme_ci95_high": hi,
            "pck_delta": pmean, "pck_ci95_low": plo, "pck_ci95_high": phi,
        })
    paired = pd.DataFrame(paired_rows)

    point_rows = []
    for name, frame in frames.items():
        for point in range(1, 12):
            point_rows.append({"model": name, "canonical_id": point,
                               "mean_nme": float(frame[f"kp{point}_nme"].mean())})
    point = pd.DataFrame(point_rows)
    geometry_rows = []
    for name, frame in frames.items():
        for column, display in GEOMETRY:
            geometry_rows.append({"model": name, "metric": display,
                                  "mean_relative_error": float(frame[column].mean())})
    geometry = pd.DataFrame(geometry_rows)

    SOURCE_OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SOURCE_OUT / "figure3_model_summary.csv", index=False)
    paired.to_csv(SOURCE_OUT / "figure3_paired_effects.csv", index=False)
    point.to_csv(SOURCE_OUT / "figure3_per_landmark_nme.csv", index=False)
    geometry.to_csv(SOURCE_OUT / "figure3_geometry_errors.csv", index=False)
    return {"frames": frames, "original": original, "summary": summary,
            "paired": paired, "point": point, "geometry": geometry}


def normalize_risk(values: np.ndarray, *, higher_is_better: bool = False) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    span = values.max() - values.min()
    normalized = np.zeros_like(values) if span <= 1e-12 else (values - values.min()) / span
    return 1 - normalized if higher_is_better else normalized


def panel_a(ax: mpl.axes.Axes, summary: pd.DataFrame, point: pd.DataFrame,
            geometry: pd.DataFrame) -> None:
    """Multi-track bubble matrix inspired by omics dot heatmaps."""
    models = list(MODELS)
    model_summary = summary.set_index("model").loc[models]
    values = np.column_stack([
        model_summary.nme_mean.to_numpy(float) * 100,
        model_summary.pck_mean_all_images.to_numpy(float),
        point.groupby("model").mean_nme.median().reindex(models).to_numpy(float) * 100,
        geometry.groupby("model").mean_relative_error.mean().reindex(models).to_numpy(float) * 100,
    ])
    risk = np.column_stack([
        normalize_risk(values[:, 0]), normalize_risk(values[:, 1], higher_is_better=True),
        normalize_risk(values[:, 2]), normalize_risk(values[:, 3]),
    ])
    labels = ["overall NME\n(%)", "all-image\nPCK@0.05",
              "median role\nNME (%)", "mean geometry\nerror (%)"]
    cmap = mpl.colors.LinearSegmentedColormap.from_list("model_risk", [TEAL, GOLD, CORAL, "#7C2146"])
    ax.set_xlim(-0.7, len(labels) - 0.3); ax.set_ylim(len(models) - 0.5, -0.5)
    for i, model in enumerate(models):
        for j in range(len(labels)):
            score = risk[i, j]
            ax.scatter(j, i, s=95 + 170 * score, color=cmap(score),
                       edgecolor="white", linewidth=0.8, zorder=2)
            value_text = f"{values[i, j]:.2f}" if j != 1 else f"{values[i, j]:.3f}"
            ax.text(j + 0.20, i, value_text, ha="left", va="center", fontsize=7.15,
                    color=INK, fontweight="bold" if score < 0.18 else "normal")
    ax.set_xticks(range(len(labels)), labels, fontsize=7.35)
    ax.set_yticks(range(len(models)), models, fontsize=7.45)
    ax.tick_params(length=0, pad=4)
    ax.grid(color=GRID, linewidth=0.45)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Four-model technical-utility dot matrix", loc="left",
                 fontsize=9.0, fontweight="bold", color=INK, pad=4)
    ax.text(0.99, 1.03, "larger/darker = greater within-metric disadvantage",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.0, color=SLATE)


def panel_b(ax: mpl.axes.Axes, data: pd.DataFrame) -> None:
    style_axis(ax)
    short_names = {
        "YOLO pose": "YOLO", "ResNet-34": "ResNet",
        "DINOv2": "DINO", "DINOv2 + geometry": "+geometry",
    }
    offsets = {
        "YOLO pose": (8, -9), "ResNet-34": (8, 9),
        "DINOv2": (13, -18), "DINOv2 + geometry": (13, 17),
    }
    for row in data.itertuples():
        x = row.nme_mean * 100; xlo = row.nme_ci95_low * 100; xhi = row.nme_ci95_high * 100
        y = row.pck_mean_all_images; ylo = row.pck_ci95_low; yhi = row.pck_ci95_high
        ax.errorbar(x, y, xerr=[[x - xlo], [xhi - x]], yerr=[[y - ylo], [yhi - y]],
                    fmt="o", ms=6.0, color=COLORS[row.model], ecolor=SLATE,
                    capsize=2.3, lw=0.9, zorder=3)
        ax.annotate(short_names[row.model], (x, y), xytext=offsets[row.model], textcoords="offset points",
                    fontsize=7.1, color=COLORS[row.model], va="center")
    ax.set_xlabel("detected-image NME (% span)")
    ax.set_ylabel("all-image PCK@0.05")
    ax.set_title("Bivariate performance frontier", loc="left", fontweight="bold", color=INK)
    ax.grid(color=GRID, lw=0.45); ax.set_axisbelow(True)


def forest_track(axis: mpl.axes.Axes, data: pd.DataFrame, value: str, low: str,
                 high: str, title: str, xlabel: str, scale: float) -> None:
    y = np.arange(len(data))
    colors = [CYAN, MAGENTA]
    for i, row in enumerate(data.itertuples()):
        center = float(getattr(row, value)) * scale
        lo = float(getattr(row, low)) * scale
        hi = float(getattr(row, high)) * scale
        axis.errorbar(center, i, xerr=[[center - lo], [hi - center]], fmt="o",
                      ms=5.2, color=colors[i], ecolor=colors[i], capsize=2.3, lw=1.0)
        label_y = i + 0.18 if i == 0 else i - 0.18
        axis.text(center, label_y, f"{center:+.2f}", va="center", ha="center",
                  fontsize=6.9, color=INK)
    axis.axvline(0, color=SLATE, lw=0.8, ls="--")
    axis.set_yticks(y, ["original − axial", "geometry − DINOv2"], fontsize=6.95)
    axis.invert_yaxis(); axis.set_xlabel(xlabel); axis.set_title(title, loc="left", fontweight="bold")
    style_axis(axis, grid_axis="x")


def panel_c(ax: mpl.axes.Axes, paired: pd.DataFrame) -> None:
    ax.axis("off")
    top = ax.inset_axes([0.05, 0.59, 0.93, 0.32])
    bottom = ax.inset_axes([0.05, 0.08, 0.93, 0.32])
    forest_track(top, paired, "nme_delta", "nme_ci95_low", "nme_ci95_high",
                 "Paired NME effects", "", 100)
    forest_track(bottom, paired, "pck_delta", "pck_ci95_low", "pck_ci95_high",
                 "Paired PCK effects", "difference (percentage points)", 100)
    ax.text(0.01, 1.02, "Dual-track paired-effect forest",
            transform=ax.transAxes, fontsize=8.9, fontweight="bold", color=INK, va="bottom")


def panel_d(ax: mpl.axes.Axes, geometry: pd.DataFrame) -> None:
    """True-circle Circos-inspired model-to-geometry error graph."""
    models = list(MODELS)
    metrics = [display for _, display in GEOMETRY]
    pivot = geometry.pivot(index="model", columns="metric", values="mean_relative_error").loc[models, metrics]
    model_angles = np.deg2rad([115, 155, 205, 245])
    metric_angles = np.deg2rad([-68, -34, 0, 34, 68])
    model_pos = {model: np.array([np.cos(angle), np.sin(angle)]) for model, angle in zip(models, model_angles)}
    metric_pos = {metric: np.array([np.cos(angle), np.sin(angle)]) for metric, angle in zip(metrics, metric_angles)}
    values = pivot.to_numpy(float) * 100
    norm = mpl.colors.Normalize(vmin=float(values.min()), vmax=float(values.max()))
    cmap = mpl.colors.LinearSegmentedColormap.from_list("geometry_error", [TEAL, GOLD, CORAL, "#7C2146"])
    for model in models:
        for metric in metrics:
            value = float(pivot.loc[model, metric]) * 100
            start = model_pos[model]; end = metric_pos[metric]
            path = MplPath([start, (0.0, 0.0), end],
                           [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3])
            ax.add_patch(PathPatch(path, facecolor="none", edgecolor=cmap(norm(value)),
                                   lw=0.65 + 2.0 * norm(value), alpha=0.55, zorder=1))
    for model, point in model_pos.items():
        ax.add_patch(Circle(point, 0.082, facecolor=COLORS[model], edgecolor="white", lw=0.9, zorder=4))
        ax.text(point[0] - 0.11, point[1], model.replace(" + ", "\n+ "), ha="right", va="center",
                fontsize=6.9, color=INK)
    metric_short = {"axial span 1–6": "axial 1–6", "dorsal base 3–4": "dorsal 3–4",
                    "caudal depth 5–7": "caudal 5–7", "head chord 2–10": "head 2–10",
                    "ventral base 8–9": "ventral 8–9"}
    for metric, point in metric_pos.items():
        ax.add_patch(Circle(point, 0.072, facecolor="white", edgecolor=INK, lw=1.0, zorder=4))
        ax.text(point[0] + 0.10, point[1], metric_short[metric], ha="left", va="center",
                fontsize=6.9, color=INK)
    ax.set_xlim(-1.55, 1.65); ax.set_ylim(-1.25, 1.25)
    ax.set_aspect("equal", adjustable="box"); ax.axis("off")
    ax.set_title("Geometry-error chord map", loc="left", fontweight="bold", color=INK, pad=4)
    ax.text(0.50, -0.02, f"edge colour/width: {values.min():.1f}–{values.max():.1f}% mean relative error",
            transform=ax.transAxes, ha="center", va="top", fontsize=6.9, color=SLATE)


def main() -> None:
    data = prepare(); apply_publication_style()
    fig = plt.figure(figsize=(WIDTH_IN, 6.05), facecolor="white")
    gs = gridspec.GridSpec(2, 12, figure=fig, height_ratios=[0.66, 1.0],
                           hspace=0.25, wspace=0.86, left=0.065, right=0.980,
                           top=0.955, bottom=0.075)
    axes = [fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, :4]),
            fig.add_subplot(gs[1, 4:8]), fig.add_subplot(gs[1, 8:])]
    for axis, letter, x in zip(axes, "abcd", [-0.045, -0.13, -0.13, -0.13]):
        panel_label(axis, letter, x=x, y=1.03)
    panel_a(axes[0], data["summary"], data["point"], data["geometry"])
    panel_b(axes[1], data["summary"])
    panel_c(axes[2], data["paired"])
    panel_d(axes[3], data["geometry"])
    save_figure(fig, GENERATED / "figure3_models")
    print(json.dumps({"figure": "figure3_models", "models": list(MODELS),
                      "test_images": 382, "actual_image_panels": 0}, indent=2))


if __name__ == "__main__":
    main()
