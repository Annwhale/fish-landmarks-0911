#!/usr/bin/env python3
# ><(((o>  核对用编号图；原坐标不作修改。
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

root = Path(__file__).resolve().parents[1]
package = root / 'data/benchmark_v1'
samples = pd.read_parquet(package / 'tables/benchmark_samples.parquet')
points = pd.read_parquet(package / 'tables/benchmark_keypoints.parquet').rename(columns={'x': 'x_crop_px', 'y': 'y_crop_px'})
print(list(points.columns))
out = root / 'qa/anatomy-review-0911'
out.mkdir(exist_ok=True)
for taxon, group in samples.groupby('taxon_candidate'):
    chosen = group[group.technical_validation_eligible].iloc[:3]
    fig, axes = plt.subplots(1, len(chosen), figsize=(15, 5), squeeze=False)
    for ax, (_, row) in zip(axes.flat, chosen.iterrows()):
        path = package / 'images/original' / (row.benchmark_id + '.jpg')
        ax.imshow(Image.open(path))
        subset = points[points.benchmark_id == row.benchmark_id]
        for _, p in subset.iterrows():
            ax.plot(p.x_crop_px, p.y_crop_px, 'o', color='yellow', markersize=4)
            ax.annotate(str(int(p.canonical_id)), (p.x_crop_px, p.y_crop_px), xytext=(0, 9 if p.canonical_id <= 5 else -13), textcoords='offset points', ha='center', color='black', fontsize=11, bbox=dict(facecolor='white', alpha=.8, pad=.2, edgecolor='none'))
        ax.set_title(row.benchmark_id, fontsize=10)
        ax.axis('off')
    fig.suptitle(taxon)
    fig.tight_layout()
    fig.savefig(out / (taxon.replace(' ', '-') + '.png'), dpi=160)
    plt.close(fig)
