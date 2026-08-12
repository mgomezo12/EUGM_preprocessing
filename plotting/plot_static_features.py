#!/usr/bin/env python3
"""
Plot static features from data/static.csv using the shared map-plus-histogram
utilities, with composition helpers for Figure 5.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


repo_root = Path(__file__).resolve().parents[1]  # .../EUGM_preprocessing
sys.path.insert(0, str(repo_root))

from gseu_preprocessing.utils_geo import (
    plot_categorical_map_and_histogram_per_column,
    plot_numeric_map_with_histogram,
)


# Column names match data/static.csv (the published, renamed schema) - if
# pointed at the older pre-rename merged_static.csv via --static-csv, pass
# the old names (fault_binary, AgeName, LEVEL1-5, SALTINTRUS, AQUIF_NAME)
# via --columns explicitly instead.
CATEGORICAL_COLUMNS = {
    "igme_structure",
    "igme_age",
    "ihme_level1",
    "ihme_level2",
    "ihme_level3",
    "ihme_level4",
    "ihme_level5",
    "ihme_saltintrus",
    "ihme_aquif_name",
    "corine_level1_1990",
    "corine_level1_2000",
    "corine_level1_2006",
    "corine_level1_2012",
    "corine_level1_2018",
    "corine_level3_1990",
    "corine_level3_2000",
    "corine_level3_2006",
    "corine_level3_2012",
    "corine_level3_2018",
}

AQUIF_LABEL_MAP = {
    "Highly productive porous aquifers": "High prod. porous",
    "Low and moderately productive porous aquifers": "Low/mod porous",
    "Locally aquiferous rocks, porous or fissured": "Local aquiferous",
    "Highly productive fissured aquifers (including karstified rocks)": "High prod. fiss./karst",
    "Practically non-aquiferous rocks, porous or fissured": "Non-aquiferous",
    "Low and moderately productive fissured aquifers (including karstified rocks)": "Low/mod fiss./karst",
    "Inland water": "Inland water",
    "NaN": "Unknown",
}

AQUIF_DESCRIPTION = (
    "Abbrev.: High prod. porous = highly productive porous aquifers; "
    "Low/mod porous = low and moderately productive porous aquifers; "
    "Local aquiferous = locally aquiferous rocks, porous or fissured; "
    "High prod. fiss./karst = highly productive fissured aquifers incl. karstified rocks; "
    "Non-aquiferous = practically non-aquiferous rocks, porous or fissured; "
    "Low/mod fiss./karst = low and moderately productive fissured aquifers incl. karstified rocks."
)


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", name)


def _is_categorical(series: pd.Series, column: str) -> bool:
    if column in CATEGORICAL_COLUMNS:
        return True
    if series.dtype == object or pd.api.types.is_bool_dtype(series):
        return True
    return False


def _load_static_with_geometry(static_csv: Path, metadata_csv: Path) -> gpd.GeoDataFrame:
    static_df = pd.read_csv(static_csv)
    if "id_mp" not in static_df.columns:
        raise KeyError(f"Missing id_mp in {static_csv}")
    meta = pd.read_csv(metadata_csv, usecols=["id_mp", "xutm", "yutm"])
    meta = meta.drop_duplicates(subset="id_mp")
    merged = static_df.merge(meta, on="id_mp", how="left")
    missing_geom = merged[["xutm", "yutm"]].isna().any(axis=1).sum()
    if missing_geom:
        print(f"[warn] {missing_geom} rows are missing coordinates and may not plot correctly.")
    return gpd.GeoDataFrame(
        merged,
        geometry=gpd.points_from_xy(merged.xutm, merged.yutm),
        crs="EPSG:3035",
    )


def _plot_numeric_feature(gdf_points: gpd.GeoDataFrame, gdf_boundary: gpd.GeoDataFrame, column: str, out_dir: Path) -> None:
    values = pd.to_numeric(gdf_points[column], errors="coerce")
    finite = values[np.isfinite(values)]
    if finite.empty:
        print(f"Skipping {column}: no finite numeric values")
        return
    cap_max = float(np.nanpercentile(finite, 99)) if finite.size > 1 else float(finite.iloc[0])
    if not np.isfinite(cap_max) or cap_max == 0:
        cap_max = float(np.nanmax(finite)) if finite.size else 1.0
    stats_decimals = 2 if np.nanmax(np.abs(finite)) < 100 else 1
    plot_numeric_map_with_histogram(
        gdf_points=gdf_points.assign(**{column: values}),
        gdf_boundary=gdf_boundary,
        column=column,
        label=column,
        cap_max=cap_max,
        cap_min=None,
        cmap_name="viridis",
        tick_format=None,
        stats_decimals=stats_decimals,
        show_overflow_label=True,
        save_path=out_dir / f"map_{_safe_name(column)}.png",
    )


def _plot_aquif_name_feature(gdf_points: gpd.GeoDataFrame, gdf_boundary: gpd.GeoDataFrame, out_dir: Path) -> None:
    column = "ihme_aquif_name"
    plot_df = gdf_points.copy()
    plot_df[column] = plot_df[column].fillna("NaN")
    plot_df["AQUIF_NAME_SHORT"] = plot_df[column].map(AQUIF_LABEL_MAP).fillna(plot_df[column])
    ordered_categories = [
        "Highly productive porous aquifers",
        "Low and moderately productive porous aquifers",
        "Locally aquiferous rocks, porous or fissured",
        "Highly productive fissured aquifers (including karstified rocks)",
        "Practically non-aquiferous rocks, porous or fissured",
        "Low and moderately productive fissured aquifers (including karstified rocks)",
        "Inland water",
        "NaN",
    ]
    ordered_short = [AQUIF_LABEL_MAP[c] for c in ordered_categories if c in plot_df[column].unique()]
    palette = sns.color_palette("Set2", n_colors=max(len(ordered_short), 3))
    color_dict = dict(zip(ordered_short, palette[:len(ordered_short)]))

    fig, ax_map = plt.subplots(figsize=(14, 9.5))
    if gdf_boundary is not None:
        gdf_boundary.boundary.plot(ax=ax_map, color="grey", linewidth=0.5)
    plot_df["color"] = plot_df["AQUIF_NAME_SHORT"].map(color_dict).fillna("#d3d3d3")
    ax_map.scatter(plot_df.geometry.x, plot_df.geometry.y, c=plot_df["color"], s=12, alpha=0.85, edgecolor="none")
    ax_map.set_xlim([2500000, 7500000])
    ax_map.set_ylim([1300000, 5500000])
    for spine in ax_map.spines.values():
        spine.set_visible(False)
    ax_map.set_xticks([])
    ax_map.set_yticks([])
    ax_map.grid(False)

    counts = plot_df["AQUIF_NAME_SHORT"].value_counts().reindex(ordered_short).dropna()
    inset_ax = fig.add_axes([0.70, 0.58, 0.24, 0.25], facecolor="white")
    sns.barplot(x=counts.values, y=counts.index, palette=[color_dict[k] for k in counts.index], ax=inset_ax)
    inset_ax.set_xlabel("Number of MPs", fontsize=11)
    inset_ax.set_ylabel("")
    inset_ax.tick_params(axis="both", labelsize=10)
    inset_ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.6)
    inset_ax.spines["top"].set_visible(False)
    inset_ax.spines["right"].set_visible(False)

    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color_dict[name], markersize=7, label=name) for name in counts.index]
    ax_map.legend(handles=handles, title="AQUIF_NAME", loc="upper left", bbox_to_anchor=(0.70, 0.52), fontsize=10, title_fontsize=11, frameon=False, borderaxespad=0.0, labelspacing=0.5)
    ax_map.text(0.02, 0.02, AQUIF_DESCRIPTION, transform=ax_map.transAxes, fontsize=9, color="#444444", ha="left", va="bottom", wrap=True, bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))

    out_path = out_dir / "cat_ihme_aquif_name_ihme_aquif_name.png"
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_level5_feature(gdf_points: gpd.GeoDataFrame, gdf_boundary: gpd.GeoDataFrame, out_dir: Path) -> None:
    column = "ihme_level5"
    plot_df = gdf_points.copy()
    plot_df[column] = plot_df[column].fillna("NaN")
    label_map = {
        "Unconsolidated materials": "Unconsolidated",
        "Consolidated materials": "Consolidated",
        "Partly consolidated materials": "Partly consolidated",
        "Inland water": "Inland water",
        "NaN": "Unknown",
    }
    plot_df["LEVEL5_SHORT"] = plot_df[column].map(label_map).fillna(plot_df[column])
    categories = list(plot_df["LEVEL5_SHORT"].value_counts().index)
    palette = sns.color_palette("tab20", n_colors=max(len(categories), 3))
    color_dict = dict(zip(categories, palette[:len(categories)]))

    fig, ax_map = plt.subplots(figsize=(13.5, 9.5))
    if gdf_boundary is not None:
        gdf_boundary.boundary.plot(ax=ax_map, color="grey", linewidth=0.5)
    plot_df["color"] = plot_df["LEVEL5_SHORT"].map(color_dict).fillna("#d3d3d3")
    ax_map.scatter(plot_df.geometry.x, plot_df.geometry.y, c=plot_df["color"], s=10, alpha=0.85, edgecolor="none")
    ax_map.set_xlim([2500000, 7500000])
    ax_map.set_ylim([1300000, 5500000])
    for spine in ax_map.spines.values():
        spine.set_visible(False)
    ax_map.set_xticks([])
    ax_map.set_yticks([])
    ax_map.grid(False)

    counts = plot_df["LEVEL5_SHORT"].value_counts()
    inset_ax = fig.add_axes([0.69, 0.58, 0.24, 0.26], facecolor="white")
    sns.barplot(x=counts.values, y=counts.index, palette=[color_dict[k] for k in counts.index], ax=inset_ax)
    inset_ax.set_xlabel("Number of MPs", fontsize=16)
    inset_ax.set_ylabel("")
    inset_ax.tick_params(axis="both", labelsize=14)
    inset_ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.6)
    inset_ax.spines["top"].set_visible(False)
    inset_ax.spines["right"].set_visible(False)

    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color_dict[name], markersize=8, label=str(name)) for name in counts.index]
    legend = ax_map.legend(
        handles=handles,
        title="Classification level 5",
        loc="upper left",
        bbox_to_anchor=(0.67, 0.47),
        fontsize=16,
        title_fontsize=18,
        frameon=True,
        borderaxespad=0.0,
        labelspacing=0.5,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(0.95)
    legend.get_frame().set_edgecolor("#d0d0d0")
    legend.get_frame().set_linewidth(0.8)

    out_path = out_dir / "cat_ihme_level5_ihme_level5.png"
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_categorical_feature(gdf_points: gpd.GeoDataFrame, gdf_boundary: gpd.GeoDataFrame, column: str, out_dir: Path) -> None:
    if column == "ihme_aquif_name":
        _plot_aquif_name_feature(gdf_points, gdf_boundary, out_dir)
        return
    if column == "ihme_level5":
        _plot_level5_feature(gdf_points, gdf_boundary, out_dir)
        return
    base_path = out_dir / f"cat_{_safe_name(column)}"
    plot_categorical_map_and_histogram_per_column(gdf_points=gdf_points.copy(), gdf=gdf_boundary, column=column, save_path=base_path)


def compose_selected_static_features(
    output_dir: Path,
    output_name: str = "static_features_panel_abcd.png",
    panel_label_fontsize: float = 14.0,
    panel_d_fontsize: float | None = None,
) -> Path:
    refs = [
        ("a", output_dir / "map_clay.png"),
        ("b", output_dir / "map_sand.png"),
        ("c", output_dir / "map_nearest_river_strahler.png"),
        ("d", output_dir / "cat_ihme_level5_ihme_level5.png"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for ax, (letter, path) in zip(axes, refs):
        ax.axis("off")
        if path.exists():
            ax.imshow(mpimg.imread(path))
        fontsize = panel_d_fontsize if letter == "d" and panel_d_fontsize is not None else panel_label_fontsize
        ax.text(
            0.02,
            0.98,
            letter,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=fontsize,
            fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
        )
    plt.tight_layout()
    out_path = output_dir / output_name
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot static features from data/static.csv")
    parser.add_argument("--static-csv", type=Path, default=repo_root / "data" / "static.csv", help="Input static feature table")
    parser.add_argument("--metadata-csv", type=Path, default=repo_root / "data" / "EUGM_mp.csv", help="Monitoring point metadata with id_mp/xutm/yutm")
    parser.add_argument("--boundary-shp", type=Path, default=repo_root / "data" / "_aux" / "SHP" / "Europe_map_GSEU_Partners.shp",
                         help="Optional country-boundary shapefile for map context (needs a 'Country' "
                              "column). Not bundled with this release - if not found at the default "
                              "path, maps are drawn without country outlines rather than failing.")
    parser.add_argument("--output-dir", type=Path, default=repo_root / "outputs" / "figures" / "static_features", help="Output folder for static feature plots")
    parser.add_argument("--columns", nargs="*", default=["clay", "sand", "nearest_river_strahler", "ihme_level5"],
                         help="Columns to plot. Defaults to just the 4 that make up Figure 5's panel; "
                              "pass any other data/static.csv column name(s) to plot those instead.")
    parser.add_argument("--compose-panel", action="store_true", default=True,
                         help="Compose the clay/sand/nearest_river_strahler/ihme_level5 panel figure "
                              "after plotting (Figure 5). On by default.")
    parser.add_argument("--no-compose-panel", action="store_false", dest="compose_panel")
    parser.add_argument("--compose-only", action="store_true", help="Only compose the combined panel from existing component images")
    parser.add_argument("--panel-output-name", default="static_features_panel_abcd.png", help="Filename for the combined panel output")
    parser.add_argument("--panel-label-fontsize", type=float, default=14.0, help="Base fontsize for panel labels")
    parser.add_argument("--panel-d-fontsize", type=float, default=None, help="Optional override fontsize for panel d")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.compose_only:
        gdf_points = _load_static_with_geometry(args.static_csv, args.metadata_csv)
        gdf_boundary = None
        if args.boundary_shp.exists():
            gdf_boundary = gpd.read_file(args.boundary_shp)
            gdf_boundary["Country"] = gdf_boundary["Country"].replace("Belgium", "Belgium- Flanders")
        else:
            print(f"Note: boundary shapefile not found at {args.boundary_shp} - plotting points without country outlines.")

        for column in args.columns:
            if column not in gdf_points.columns:
                print(f"Skipping {column}: not found")
                continue
            series = gdf_points[column]
            if _is_categorical(series, column):
                print(f"Plotting categorical feature: {column}")
                _plot_categorical_feature(gdf_points, gdf_boundary, column, args.output_dir)
            else:
                print(f"Plotting numeric feature: {column}")
                _plot_numeric_feature(gdf_points, gdf_boundary, column, args.output_dir)

    if args.compose_panel or args.compose_only:
        panel = compose_selected_static_features(
            args.output_dir,
            output_name=args.panel_output_name,
            panel_label_fontsize=args.panel_label_fontsize,
            panel_d_fontsize=args.panel_d_fontsize,
        )
        print(f"Saved combined panel to {panel}")

    print(f"Saved static feature plots to {args.output_dir}")


if __name__ == "__main__":
    main()
