#!/usr/bin/env python3
# ><(((o>  鱼类形态数据工具
"""Scientific Data Figure 5: zero-shot semantic landmark grounding."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from PIL import Image

from visual_style import (
    CORAL, CYAN, GOLD, GRID, INK, MAGENTA, SLATE, TEAL, VIOLET,
    apply_publication_style, clean_image_axis, crop_limits, packed_row_boxes,
    panel_label,
)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 8.3,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "models/evaluation/locateanything_3b_zero_shot/query_predictions.csv"
SUMMARY = ROOT / "models/evaluation/locateanything_3b_zero_shot/summary.json"
IMAGES = ROOT / "data/benchmark_v1/images/original"
SOURCE_OUT = ROOT / "figures/source_data"
GENERATED = ROOT / "figures/generated"
FIG_WIDTH_MM = 183.0
WIDTH_IN = FIG_WIDTH_MM / 25.4
ROLE_NAMES = {
    1: "snout tip", 2: "nape outline", 3: "dorsal origin", 4: "dorsal insertion",
    5: "upper caudal", 6: "peduncle midpoint", 7: "lower caudal", 8: "anal origin",
    9: "pelvic origin", 10: "pectoral base", 11: "opercular margin",
}


def save_figure(fig: mpl.figure.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight",
                facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def prepare() -> dict[str, object]:
    frame = pd.read_csv(RESULTS)
    frame["output_available"] = as_bool(frame["output_available"])
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    overall = pd.DataFrame([{
        "model": summary["model_id"], "images": summary["images"],
        "queries": summary["queries"], "outputs": summary["outputs"],
        "output_rate": summary["output_rate"],
        "complete_11_point_images": summary["complete_11_point_images"],
        "complete_11_point_image_rate": summary["complete_11_point_image_rate"],
        "nme_detected_mean": summary["nme_detected_only"]["mean"],
        "nme_detected_ci95_low": summary["nme_detected_only"]["ci95_low"],
        "nme_detected_ci95_high": summary["nme_detected_only"]["ci95_high"],
        "pck_0_05_all_queries": summary["pck_0_05_all_queries"]["mean"],
        "pck_0_05_ci95_low": summary["pck_0_05_all_queries"]["ci95_low"],
        "pck_0_05_ci95_high": summary["pck_0_05_all_queries"]["ci95_high"],
        "normalizer": "role-1-to-role-6 working axial span",
        "selection": "three lexicographically first locked-test images per non-conflicted source group",
    }])
    role_rows = []
    for key, item in summary["per_role"].items():
        role_id = int(key)
        role_rows.append({
            "role_id": role_id, "display": f"{role_id} {ROLE_NAMES[role_id]}",
            "queries": item["queries"], "outputs": item["outputs"],
            "output_rate": item["output_rate"],
            "nme_mean": item["nme_detected_only"]["mean"],
            "nme_ci95_low": item["nme_detected_only"]["ci95_low"],
            "nme_ci95_high": item["nme_detected_only"]["ci95_high"],
            "pck_0_05_mean_all_queries": item["pck_0_05_all_queries"]["mean"],
            "pck_0_05_ci95_low": item["pck_0_05_all_queries"]["ci95_low"],
            "pck_0_05_ci95_high": item["pck_0_05_all_queries"]["ci95_high"],
        })
    role = pd.DataFrame(role_rows).sort_values("role_id")

    full_selection = pd.read_csv(SOURCE_OUT / "figure5_selected_images.csv")
    display_selection = pd.read_csv(SOURCE_OUT / "figure5_display_images.csv")
    selected_ids = display_selection["benchmark_id"].astype(str).tolist()
    display_labels = display_selection["source_display"].astype(str).tolist()
    examples = frame[frame["benchmark_id"].astype(str).isin(selected_ids)].copy()
    examples["source_display"] = examples["benchmark_id"].map(
        dict(zip(selected_ids, display_labels))
    )
    examples["selection_rule"] = (
        "prespecified six-taxon display spanning the observed zero-shot error range"
    )

    group_order = full_selection["source_groups"].astype(str).tolist()
    group_labels = full_selection["source_display"].astype(str).tolist()
    query_matrix = np.zeros((len(group_order), 33), dtype=int)
    image_matrix = np.zeros((3, len(group_order)), dtype=int)
    image_order_rows = []
    for group_index, group_name in enumerate(group_order):
        group = frame[frame["source_groups"].astype(str) == group_name].copy()
        image_ids = sorted(group["benchmark_id"].astype(str).unique())
        if len(image_ids) != 3:
            raise RuntimeError(f"expected three zero-shot images for {group_name}, found {len(image_ids)}")
        for slot, benchmark_id in enumerate(image_ids):
            image_rows = group[group["benchmark_id"].astype(str) == benchmark_id].sort_values("role_id")
            if len(image_rows) != 11:
                raise RuntimeError(f"expected 11 role queries for {benchmark_id}")
            availability = image_rows["output_available"].to_numpy(bool).astype(int)
            query_matrix[group_index, slot * 11:(slot + 1) * 11] = availability
            image_matrix[slot, group_index] = int(availability.all())
            image_order_rows.append({"source_groups": group_name, "source_display": group_labels[group_index],
                                     "image_slot": slot + 1, "benchmark_id": benchmark_id,
                                     "complete_11_roles": bool(availability.all())})

    SOURCE_OUT.mkdir(parents=True, exist_ok=True)
    overall.to_csv(SOURCE_OUT / "figure5_overall.csv", index=False)
    role.to_csv(SOURCE_OUT / "figure5_per_role.csv", index=False)
    examples.to_csv(SOURCE_OUT / "figure5_examples.csv", index=False)
    pd.DataFrame(image_order_rows).to_csv(SOURCE_OUT / "figure5_matrix_image_order.csv", index=False)
    return {"frame": frame, "summary": summary, "overall": overall, "role": role,
            "examples": examples, "selected_ids": selected_ids,
            "display_labels": display_labels, "group_labels": group_labels,
            "query_matrix": query_matrix, "image_matrix": image_matrix}


def panel_a(ax: mpl.axes.Axes, data: dict[str, object]) -> None:
    """Workflow plus exact query/image completion matrices."""
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    summary = data["summary"]
    ax.text(0.01, 0.98, "Frozen semantic-grounding route and completion structure",
            transform=ax.transAxes, fontsize=9.0, fontweight="bold", color=INK, va="top")
    xs = [0.07, 0.34, 0.64, 0.93]
    colors = [CYAN, GOLD, VIOLET, TEAL]
    labels = ["39 images\n13 groups", "11 fixed\nrole prompts",
              "LocateAnything-3B\nzero-shot", "428/429\noutputs"]
    route_y = 0.76
    ax.plot(xs, [route_y] * 4, color="#93A4AF", lw=1.2, zorder=1)
    for index, (x, color, text_value) in enumerate(zip(xs, colors, labels)):
        ax.scatter([x], [route_y], s=110 if index == 2 else 78, color=color,
                   edgecolor="white", linewidth=0.9, zorder=3)
        ax.text(x, 0.89 if index % 2 == 0 else 0.64, text_value,
                ha="center", va="center", fontsize=7.05,
                fontweight="bold" if index >= 2 else "normal", color=INK)
        if index < 3:
            ax.annotate("", xy=(xs[index + 1] - 0.035, route_y), xytext=(x + 0.035, route_y),
                        arrowprops={"arrowstyle": "-|>", "color": SLATE, "lw": 0.9})

    cmap = ListedColormap([CORAL, TEAL])
    query_ax = ax.inset_axes([0.12, 0.04, 0.52, 0.51])
    query_ax.imshow(data["query_matrix"], aspect="auto", cmap=cmap, vmin=0, vmax=1,
                    interpolation="nearest")
    query_ax.set_yticks(np.arange(13), data["group_labels"], fontsize=6.65)
    query_ax.set_xticks([5, 16, 27], ["image 1", "image 2", "image 3"], fontsize=6.9)
    query_ax.tick_params(length=0, pad=2)
    for boundary in [10.5, 21.5]:
        query_ax.axvline(boundary, color="white", lw=1.0)
    for spine in query_ax.spines.values():
        spine.set_visible(False)
    query_ax.set_title(f"query outputs  {summary['outputs']}/{summary['queries']}",
                       loc="left", fontsize=7.35, fontweight="bold", color=INK, pad=2)

    image_ax = ax.inset_axes([0.76, 0.08, 0.22, 0.42])
    image_ax.imshow(data["image_matrix"], aspect="auto", cmap=cmap, vmin=0, vmax=1,
                    interpolation="nearest")
    image_ax.set_yticks(range(3), ["image 1", "image 2", "image 3"], fontsize=6.8)
    shown_roles = np.arange(0, 13, 2)
    image_ax.set_xticks(shown_roles, [str(i + 1) for i in shown_roles], fontsize=6.7)
    image_ax.tick_params(length=0, pad=2)
    for spine in image_ax.spines.values():
        spine.set_visible(False)
    image_ax.set_title(f"complete images  {summary['complete_11_point_images']}/{summary['images']}",
                       loc="left", fontsize=7.35, fontweight="bold", color=INK, pad=2)


def panel_b(ax: mpl.axes.Axes, role: pd.DataFrame) -> None:
    """True-circle dual-ring role heatmap."""
    ax.axis("off")
    polar = ax.figure.add_subplot(ax.get_subplotspec(), projection="polar")
    ordered = role.sort_values("role_id")
    theta = np.linspace(0, 2 * np.pi, len(ordered), endpoint=False)
    width = 2 * np.pi / len(ordered) * 0.93
    nme = ordered.nme_mean.to_numpy(float) * 100
    pck = ordered.pck_0_05_mean_all_queries.to_numpy(float)
    nme_norm = mpl.colors.Normalize(vmin=float(nme.min()), vmax=float(nme.max()))
    pck_norm = mpl.colors.Normalize(vmin=0, vmax=1)
    nme_cmap = mpl.colors.LinearSegmentedColormap.from_list("nme_ring", ["#CFE8E6", CYAN, MAGENTA])
    pck_cmap = mpl.colors.LinearSegmentedColormap.from_list("pck_ring", [CORAL, GOLD, TEAL])
    polar.bar(theta, np.full_like(theta, 0.72), width=width, bottom=1.00,
              color=[pck_cmap(pck_norm(value)) for value in pck], edgecolor="white", linewidth=0.7)
    polar.bar(theta, np.full_like(theta, 0.72), width=width, bottom=1.80,
              color=[nme_cmap(nme_norm(value)) for value in nme], edgecolor="white", linewidth=0.7)
    polar.set_theta_zero_location("N"); polar.set_theta_direction(-1)
    polar.set_xticks(theta, [str(int(value)) for value in ordered.role_id], fontsize=7.05)
    polar.tick_params(axis="x", pad=2)
    polar.set_yticks([]); polar.set_ylim(0, 2.72); polar.grid(False)
    polar.spines["polar"].set_visible(False)
    polar.text(0, 0, "inner  PCK\nouter  NME", ha="center", va="center",
               fontsize=7.0, color=INK, fontweight="bold")
    polar.set_title("Dual-ring role fingerprint", fontsize=8.9, fontweight="bold",
                    color=INK, pad=12)


def panel_c(ax: mpl.axes.Axes, data: dict[str, object]) -> None:
    ax.axis("off")
    examples = data["examples"]
    items = []
    for benchmark_id, display_label in zip(data["selected_ids"], data["display_labels"]):
        image = Image.open(IMAGES / f"{benchmark_id}.jpg").convert("RGB")
        group = examples[examples["benchmark_id"].astype(str) == benchmark_id].sort_values("role_id")
        pred_x = group.prediction_x.to_numpy(float); pred_y = group.prediction_y.to_numpy(float)
        x_all = np.concatenate([group.target_x.to_numpy(float), pred_x[np.isfinite(pred_x)]])
        y_all = np.concatenate([group.target_y.to_numpy(float), pred_y[np.isfinite(pred_y)]])
        x0, x1, y1, y0 = crop_limits(x_all, y_all, image.size[0], image.size[1],
                                      margin_x=0.08, margin_y=0.18)
        left, top = int(np.floor(x0)), int(np.floor(y0))
        right, bottom = int(np.ceil(x1)), int(np.ceil(y1))
        tile = image.crop((left, top, right, bottom))
        plotted = group.copy()
        for xcol, ycol in [("target_x", "target_y"), ("prediction_x", "prediction_y")]:
            plotted[xcol] -= left; plotted[ycol] -= top
        if tile.height > tile.width:
            old_height = tile.height
            tile = tile.transpose(Image.Transpose.ROTATE_270)
            for xcol, ycol in [("target_x", "target_y"), ("prediction_x", "prediction_y")]:
                old_x = plotted[xcol].copy()
                plotted[xcol] = old_height - 1 - plotted[ycol]
                plotted[ycol] = old_x
        items.append({"tile": tile, "points": plotted,
                      "label": f"{display_label}  ·  {group.nme_role1_role6.mean() * 100:.1f}%"})

    rows = [items[:3], items[3:]]
    boxes = packed_row_boxes(
        ax, [[item["tile"].width / max(item["tile"].height, 1) for item in row] for row in rows],
        top=0.90, bottom=0.005, horizontal_gap=0.012, vertical_gap=0.025,
        label_space=0.035,
    )
    for row, row_boxes in zip(rows, boxes):
        for item, box in zip(row, row_boxes):
            image_ax = ax.inset_axes(box)
            image_ax.imshow(item["tile"], interpolation="nearest")
            for point in item["points"].itertuples():
                image_ax.scatter(point.target_x, point.target_y, s=16, marker="o",
                                 facecolors="none", edgecolors=CYAN, linewidths=1.0, zorder=4)
                if bool(point.output_available):
                    image_ax.plot([point.target_x, point.prediction_x],
                                  [point.target_y, point.prediction_y],
                                  color="white", lw=1.35, alpha=0.88, zorder=2)
                    image_ax.plot([point.target_x, point.prediction_x],
                                  [point.target_y, point.prediction_y],
                                  color=MAGENTA, lw=0.55, alpha=0.9, zorder=3)
                    image_ax.scatter(point.prediction_x, point.prediction_y, s=14, marker="x",
                                     c=MAGENTA, linewidths=0.95, zorder=5)
            clean_image_axis(image_ax)
            ax.text(box[0] + box[2] / 2, box[1] + box[3] + 0.005, item["label"],
                    transform=ax.transAxes, ha="center", va="bottom", fontsize=6.95,
                    fontstyle="italic", color=INK)
    ax.text(0.01, 1.015, "Six-taxon zero-shot localisation examples",
            transform=ax.transAxes, fontsize=9.0, fontweight="bold", va="bottom", color=INK)
    ax.text(0.99, 1.015, "target ○   prediction ×   error vector —",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.9, color=SLATE)


def main() -> None:
    data = prepare(); apply_publication_style()
    fig = plt.figure(figsize=(WIDTH_IN, 6.55), facecolor="white")
    gs = gridspec.GridSpec(2, 12, figure=fig, height_ratios=[1.10, 1.0],
                           hspace=0.20, wspace=0.82, left=0.06, right=0.98,
                           top=0.95, bottom=0.065)
    axes = [fig.add_subplot(gs[0, :7]), fig.add_subplot(gs[0, 7:]),
            fig.add_subplot(gs[1, :])]
    for axis, letter, x in zip(axes, "abc", [-0.08, -0.12, -0.045]):
        panel_label(axis, letter, x=x, y=1.03)
    panel_a(axes[0], data); panel_b(axes[1], data["role"]); panel_c(axes[2], data)
    save_figure(fig, GENERATED / "figure5_zero_shot_grounding")
    print(json.dumps({"figure": "figure5_zero_shot_grounding", "images": 39,
                      "display_images": len(data["selected_ids"]), "queries": 429,
                      "model": "nvidia/LocateAnything-3B"}, indent=2))


if __name__ == "__main__":
    main()
