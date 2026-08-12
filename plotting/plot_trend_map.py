"""
Significant groundwater level trend map.

Produces:
  trend_map.png               – full Europe overview
  trend_map_zoom_<region>.png – 5 regional zoom figures (same layout)

Layout per figure (left→right): negative-trend map | histogram | positive-trend map

Symbols:
  grey circle  – non-significant (or absent) trend
  ▼ coloured   – significant negative trend (colour = trend value)
  ▲ coloured   – significant positive trend (colour = trend value)

Colormap: RdBu clipped at ±20 cm/yr  (red = negative, blue = positive).
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import TwoSlopeNorm

# ── GDAL/PROJ ──────────────────────────────────────────────────────────────
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
                     help="Trend table with id_mp/tac_slope_sig_mon_RP.")
parser.add_argument("--boundary-shp", type=Path, default=REPO_ROOT / "data" / "_aux" / "SHP" / "Europe_map_GSEU_Partners.shp",
                     help="Optional country-boundary shapefile for map context (needs a 'Country' "
                          "column; any source works, e.g. Natural Earth's admin-0 countries layer). "
                          "Not bundled with this release - if not found at the default path, the "
                          "map is drawn without country outlines rather than failing.")
parser.add_argument("--out-dir", type=Path, default=HERE, help="Output directory for the figures.")
args = parser.parse_args()

gdf_bound = None
if args.boundary_shp.exists():
    gdf_bound = gpd.read_file(args.boundary_shp)
    gdf_bound["Country"] = gdf_bound["Country"].replace("Belgium", "Belgium- Flanders")
else:
    print(f"Note: boundary shapefile not found at {args.boundary_shp} - plotting points without country outlines.")

# ── load & merge ───────────────────────────────────────────────────────────
df_trends = pd.read_csv(args.trends_cluster_csv, usecols=["id_mp", "tac_slope_sig_mon_RP"])
df_meta   = pd.read_csv(args.metadata_csv, usecols=["id_mp", "xutm", "yutm"])
df        = df_trends.merge(df_meta, on="id_mp", how="left")

# tac_slope_sig_mon_RP is NaN (not a sentinel) where a trend isn't significant.
df["slope_sig_cmyr"] = df["tac_slope_sig_mon_RP"] * 100
df["is_sig"]     = df["slope_sig_cmyr"].notna()
df["is_neg_sig"] = df["is_sig"] & (df["slope_sig_cmyr"] < 0)
df["is_pos_sig"] = df["is_sig"] & (df["slope_sig_cmyr"] > 0)

gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df.xutm, df.yutm),
    crs="EPSG:3035",
).dropna(subset=["xutm", "yutm"])

# ── shared colormap ────────────────────────────────────────────────────────
CLIP_CM = 20.0
CMAP    = plt.get_cmap("RdBu")
NORM    = TwoSlopeNorm(vmin=-CLIP_CM, vcenter=0.0, vmax=CLIP_CM)
COL_NEG = CMAP(NORM(-CLIP_CM))
COL_POS = CMAP(NORM(+CLIP_CM))

# ── style ──────────────────────────────────────────────────────────────────
GREY     = "#b8b8b8"
S_NS     = 3
S_SIG    = 9
ALPHA_NS = 0.35

# ── global histogram (same across all figures) ─────────────────────────────
sig_vals    = gdf["slope_sig_cmyr"].dropna()
sig_clip    = sig_vals.clip(-CLIP_CM, CLIP_CM)
n_bins      = 60
bins        = np.linspace(-CLIP_CM, CLIP_CM, n_bins + 1)
bin_w       = bins[1] - bins[0]
bin_centers = 0.5 * (bins[:-1] + bins[1:])
counts, _   = np.histogram(sig_clip, bins=bins)
mean_val    = sig_vals.mean()

# ── gradient triangle images (built once, reused) ──────────────────────────
n_px = 80
tri_upper = np.zeros((n_px, n_px, 4))
for i in range(n_px):
    y_norm = i / (n_px - 1)
    rgba   = CMAP(NORM(20.0 * (1.0 - y_norm)))
    for j in range(n_px):
        x_norm = j / (n_px - 1)
        if abs(x_norm - 0.5) <= y_norm * 0.5 + 0.008:
            tri_upper[i, j] = (rgba[0], rgba[1], rgba[2], 1.0)

tri_lower = np.zeros((n_px, n_px, 4))
for i in range(n_px):
    y_norm = i / (n_px - 1)
    rgba   = CMAP(NORM(-20.0 * y_norm))
    for j in range(n_px):
        x_norm = j / (n_px - 1)
        if abs(x_norm - 0.5) <= (1.0 - y_norm) * 0.5 + 0.008:
            tri_lower[i, j] = (rgba[0], rgba[1], rgba[2], 1.0)

# ── legend handles (same across all figures) ──────────────────────────────
n_total = len(gdf)
n_ns    = int((~gdf["is_sig"]).sum())
n_neg   = int(gdf["is_neg_sig"].sum())
n_pos   = int(gdf["is_pos_sig"].sum())
pct_ns  = 100 * n_ns / n_total

legend_handles = [
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=GREY,
               markersize=8, label=f"Non-significant  (n = {n_ns:,}, {pct_ns:.0f}%)"),
    plt.Line2D([0], [0], marker="v", color="w", markerfacecolor=COL_NEG,
               markersize=9, label=f"Significant negative  (n = {n_neg:,})"),
    plt.Line2D([0], [0], marker="^", color="w", markerfacecolor=COL_POS,
               markersize=9, label=f"Significant positive  (n = {n_pos:,})"),
]

# ── figures to produce ─────────────────────────────────────────────────────
XLIM_FULL = [2_500_000, 7_500_000]
YLIM_FULL = [1_300_000, 5_500_000]

FIGURES = [
    ("trend_map",                 XLIM_FULL,                    YLIM_FULL),
    # Regional zoom variants below are not referenced in the manuscript
    # (only the Europe-wide trend_map.png is Fig 7) - left here, disabled,
    # in case regional detail is useful for review/QA.
    # ("trend_map_zoom_northern",   [3_500_000, 5_700_000],       [3_700_000, 5_500_000]),
    # ("trend_map_zoom_western",    [2_500_000, 4_100_000],       [1_700_000, 3_700_000]),
    # ("trend_map_zoom_central",    [3_800_000, 5_200_000],       [2_600_000, 3_800_000]),
    # ("trend_map_zoom_southern",   [3_000_000, 5_200_000],       [1_300_000, 2_700_000]),
    # ("trend_map_zoom_eastern",    [4_900_000, 6_500_000],       [2_200_000, 3_700_000]),
]


# ── figure factory ─────────────────────────────────────────────────────────
def make_figure(xlim, ylim, out_path):

    fig = plt.figure(figsize=(18, 7))
    gs  = fig.add_gridspec(
        3, 3,
        width_ratios=[5, 2.8, 5],
        height_ratios=[1, 2, 0.3],
        wspace=0.05, hspace=0.0,
    )
    ax_neg  = fig.add_subplot(gs[:, 0])
    ax_hist = fig.add_subplot(gs[1, 1])
    ax_pos  = fig.add_subplot(gs[:, 2])
    ax_hist.set_facecolor("white")
    ax_hist.set_zorder(5)
    ax_hist.patch.set_alpha(1.0)

    # ── maps ──────────────────────────────────────────────────────────────
    def draw_map(ax, highlight):
        if gdf_bound is not None:
            gdf_bound.plot(ax=ax, color="white", edgecolor="#888888",
                           linewidth=0.5, zorder=1)
        ns = gdf[~gdf["is_sig"]]
        ax.scatter(ns.geometry.x, ns.geometry.y,
                   c=GREY, s=S_NS, marker="o", linewidths=0, zorder=2,
                   alpha=ALPHA_NS)
        if highlight == "neg":
            opp = gdf[gdf["is_pos_sig"]]
            ax.scatter(opp.geometry.x, opp.geometry.y,
                       c=GREY, s=S_NS, marker="o", linewidths=0, zorder=3,
                       alpha=ALPHA_NS)
            sel  = gdf[gdf["is_neg_sig"]]
            vals = sel["slope_sig_cmyr"].clip(-CLIP_CM, CLIP_CM).values
            ax.scatter(sel.geometry.x, sel.geometry.y,
                       c=vals, cmap=CMAP, norm=NORM,
                       s=S_SIG, marker="v", linewidths=0, zorder=4, alpha=0.9)
        else:
            opp = gdf[gdf["is_neg_sig"]]
            ax.scatter(opp.geometry.x, opp.geometry.y,
                       c=GREY, s=S_NS, marker="o", linewidths=0, zorder=3,
                       alpha=ALPHA_NS)
            sel  = gdf[gdf["is_pos_sig"]]
            vals = sel["slope_sig_cmyr"].clip(-CLIP_CM, CLIP_CM).values
            ax.scatter(sel.geometry.x, sel.geometry.y,
                       c=vals, cmap=CMAP, norm=NORM,
                       s=S_SIG, marker="^", linewidths=0, zorder=4, alpha=0.9)
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_aspect("equal"); ax.axis("off")

    draw_map(ax_neg, "neg")
    draw_map(ax_pos, "pos")

    # ── histogram (global) ────────────────────────────────────────────────
    for cnt, bc in zip(counts, bin_centers):
        ax_hist.bar(bc, cnt, width=bin_w, color=CMAP(NORM(bc)), edgecolor="none")

    ax_hist.axvline(mean_val, color="0.25", linestyle="--", linewidth=1.3, zorder=5)
    ax_hist.text(mean_val - 0.6, max(counts) * 0.97, f"Mean: {mean_val:.1f}",
                 color="0.25", fontsize=8.5, va="top", ha="right")

    stats_text = (
        f"n = {len(sig_vals):,}\n"
        f"Mean = {mean_val:.1f} cm/yr\n"
        f"Std  = {sig_vals.std():.1f} cm/yr\n"
        f"Min  = {sig_vals.min():.1f} cm/yr\n"
        f"Max  = {sig_vals.max():.1f} cm/yr"
    )
    ax_hist.text(0.97, 0.97, stats_text, transform=ax_hist.transAxes,
                 fontsize=7.5, va="top", ha="right",
                 bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                           alpha=0.75, edgecolor="0.75", linewidth=0.5))
    ax_hist.set_title("Histogram of significant trends", fontsize=10, pad=6)
    ax_hist.set_xlabel("Significant trend (cm/yr)", fontsize=10)
    ax_hist.set_ylabel("Frequency", fontsize=10)
    ax_hist.set_xlim(-CLIP_CM, CLIP_CM)
    ax_hist.spines[["top", "right"]].set_visible(False)
    ax_hist.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=5))
    ax_hist.xaxis.set_major_locator(ticker.MultipleLocator(5))

    # ── legend ────────────────────────────────────────────────────────────
    fig.subplots_adjust(bottom=0.10)
    fig.legend(handles=legend_handles, loc="lower center", ncol=3,
               fontsize=10, bbox_to_anchor=(0.40, 0.032), frameon=False)

    # ── gradient triangles ────────────────────────────────────────────────
    fig.canvas.draw()
    renderer  = fig.canvas.get_renderer()
    leg_bbox  = fig.legends[0].get_window_extent(renderer=renderer)
    fig_w_px  = fig.get_figwidth()  * fig.dpi
    fig_h_px  = fig.get_figheight() * fig.dpi
    leg_left  = leg_bbox.x0 / fig_w_px
    leg_right = leg_bbox.x1 / fig_w_px
    leg_bot   = leg_bbox.y0 / fig_h_px
    leg_top   = leg_bbox.y1 / fig_h_px
    col_w     = (leg_right - leg_left) / 3

    tri_h    = 0.022
    tri_w    = 0.010
    cx       = leg_right + col_w / 2
    y_centre = (leg_bot + leg_top) / 2

    ax_tri_up = fig.add_axes([cx - tri_w / 2, y_centre + 0.002, tri_w, tri_h])
    ax_tri_up.imshow(tri_upper, origin="upper", interpolation="bilinear",
                     aspect="auto")
    ax_tri_up.axis("off")

    ax_tri_lo = fig.add_axes([cx - tri_w / 2, y_centre - tri_h - 0.002,
                               tri_w, tri_h])
    ax_tri_lo.imshow(tri_lower, origin="upper", interpolation="bilinear",
                     aspect="auto")
    ax_tri_lo.axis("off")

    fig.text(cx + tri_w / 2 + 0.007, y_centre + tri_h / 2 + 0.002,
             "+20 cm/yr", fontsize=9, va="center", ha="left", color="0.3")
    fig.text(cx + tri_w / 2 + 0.007, y_centre - tri_h / 2 - 0.002,
             "−20 cm/yr", fontsize=9, va="center", ha="left", color="0.3")

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ── produce all figures ────────────────────────────────────────────────────
print("Generating figures...")
for fname, xlim, ylim in FIGURES:
    make_figure(xlim=xlim, ylim=ylim, out_path=args.out_dir / f"{fname}.png")

print(f"\nDone — {len(FIGURES)} figures written to {args.out_dir}")
