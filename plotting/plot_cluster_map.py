"""
Cluster map — 2×4 grid of subplots, one per cluster (1–8).
Only the monitoring points belonging to each cluster are shown.
A horizontal legend sits at the bottom of the figure.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── GDAL / PROJ ────────────────────────────────────────────────────────────
# Only needed if your GDAL/PROJ install doesn't already expose these via its
# own environment activation - most geopandas installs won't need this at all.
_conda = os.environ.get("CONDA_PREFIX")
if _conda:
    if not os.environ.get("GDAL_DATA"):
        os.environ["GDAL_DATA"] = str(Path(_conda) / "Library" / "share" / "gdal")
    if not os.environ.get("PROJ_LIB"):
        os.environ["PROJ_LIB"] = str(Path(_conda) / "Library" / "share" / "proj")

# ── paths ──────────────────────────────────────────────────────────────────
HERE      = Path(__file__).parent
REPO_ROOT = HERE.parent

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--metadata-csv", type=Path, default=REPO_ROOT / "data" / "EUGM_mp.csv",
                     help="Monitoring point metadata with id_mp/xutm/yutm.")
parser.add_argument("--trends-cluster-csv", type=Path, default=REPO_ROOT / "data" / "trends_cluster.csv",
                     help="Trend/cluster table with id_mp/cluster_number.")
parser.add_argument("--boundary-shp", type=Path, default=REPO_ROOT / "data" / "_aux" / "SHP" / "Europe_map_GSEU_Partners.shp",
                     help="Optional country-boundary shapefile for map context (needs a 'Country' "
                          "column; any source works, e.g. Natural Earth's admin-0 countries layer). "
                          "Not bundled with this release - if not found at the default path, the "
                          "map is drawn without country outlines rather than failing.")
parser.add_argument("--out-dir", type=Path, default=HERE, help="Output directory for the figure.")
args = parser.parse_args()

gdf_bound = None
if args.boundary_shp.exists():
    gdf_bound = gpd.read_file(args.boundary_shp)
    gdf_bound["Country"] = gdf_bound["Country"].replace("Belgium", "Belgium- Flanders")
else:
    print(f"Note: boundary shapefile not found at {args.boundary_shp} - plotting points without country outlines.")

# ── load & merge ───────────────────────────────────────────────────────────
df_clusters = pd.read_csv(args.trends_cluster_csv, usecols=["id_mp", "cluster_number"])
df_clusters = df_clusters.rename(columns={"cluster_number": "Cluster number"})
df_meta     = pd.read_csv(args.metadata_csv, usecols=["id_mp", "xutm", "yutm"])
df          = df_clusters.merge(df_meta, on="id_mp", how="left")

gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df.xutm, df.yutm),
    crs="EPSG:3035",
).dropna(subset=["xutm", "yutm"])

# ── colours — one per cluster ──────────────────────────────────────────────
N_CLUSTERS = 8
COLORS = [
    "#4e79a7",  # 1 – steel blue
    "#f28e2b",  # 2 – orange
    "#59a14f",  # 3 – green
    "#e15759",  # 4 – red
    "#76b7b2",  # 5 – teal
    "#edc948",  # 6 – yellow
    "#b07aa1",  # 7 – purple
    "#ff9da7",  # 8 – rose
]

# ── map extent (EPSG:3035) ─────────────────────────────────────────────────
XLIM = [2_500_000, 6_200_000]
YLIM = [1_300_000, 5_500_000]

# ── figure ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(
    2, 4,
    figsize=(8, 5),
    gridspec_kw=dict(wspace=0.01, hspace=0.10),
)
axes = axes.flatten()

for idx, cluster_id in enumerate(range(1, N_CLUSTERS + 1)):
    ax    = axes[idx]
    color = COLORS[idx]

    # country boundaries (optional)
    if gdf_bound is not None:
        gdf_bound.plot(ax=ax, color="white", edgecolor="#999999",
                       linewidth=0.15, zorder=1)

    # cluster points
    sub = gdf[gdf["Cluster number"] == cluster_id]
    n   = len(sub)
    ax.scatter(
        sub.geometry.x, sub.geometry.y,
        c=color, s=1, linewidths=0,
        zorder=2, alpha=0.75,
        rasterized=True,
    )

    ax.set_xlim(XLIM)
    ax.set_ylim(YLIM)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"Cluster {cluster_id}  (n = {n:,})", fontsize=6.5, pad=2)

# ── horizontal legend at the bottom ───────────────────────────────────────
handles = [
    mpatches.Patch(facecolor=COLORS[i], edgecolor="0.3", linewidth=0.4,
                   label=f"Cluster {i + 1}")
    for i in range(N_CLUSTERS)
]
fig.legend(
    handles=handles,
    loc="lower center",
    ncol=N_CLUSTERS,
    fontsize=7,
    bbox_to_anchor=(0.5, 0.005),
    frameon=False,
    handlelength=1.2,
    handleheight=0.7,
    columnspacing=0.8,
)

fig.subplots_adjust(bottom=0.08)

out = args.out_dir / "cluster_map.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")
