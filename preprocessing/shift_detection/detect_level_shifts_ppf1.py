#!/usr/bin/env python3
"""
Detect persistent level shifts in EUGM_TS_ppf1 using rolling pre/post medians.

This script is standalone and does not modify the preprocessing pipeline.
It reads the final processed time-series and monitoring-point outputs, detects
the single strongest candidate shift per station, classifies flagged stations,
saves diagnostic figures for flagged stations, and creates Europe-wide maps of
detected shifts.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path


def _configure_gdal_env(prefix_text: str | None = None) -> None:
    prefix = Path(prefix_text or os.environ.get("CONDA_PREFIX", sys.prefix))
    candidates = {
        "GDAL_DATA": [prefix / "Library/share/gdal", prefix / "share/gdal"],
        "PROJ_LIB": [prefix / "Library/share/proj", prefix / "share/proj"],
    }
    for env_name, paths in candidates.items():
        if os.environ.get(env_name):
            continue
        for candidate in paths:
            if candidate.exists():
                os.environ[env_name] = str(candidate)
                break


_configure_gdal_env()

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Pipeline outputs now live in a snapshot-specific folder (see io.py's
# `Paths.data_processed`/`Paths.snapshot`, derived from EUGM_ICGC's trailing date).
# This script is standalone and doesn't load the config, so update this tag manually
# when the active snapshot changes.
_SNAPSHOT = "20260707"
DEFAULT_TS = Path(f"data/EUGM_processed/{_SNAPSHOT}/EUGM_TS_ppf1_{_SNAPSHOT}.parquet")
DEFAULT_MP = Path(f"data/EUGM_processed/{_SNAPSHOT}/EUGM_MP_ppf1_{_SNAPSHOT}.csv")
DEFAULT_BACKGROUND = Path("data/_aux/SHP/Europe_map_GSEU_Partners.shp")
DEFAULT_OUTPUT_DIR = Path("outputs/shift_detection")


@dataclass
class Thresholds:
    score_review: float
    score_likely: float
    raw_step_review: float
    raw_step_likely: float


def detect_value_column(columns: list[str]) -> str:
    for cand in ["Value", "Value_imp", "Imputed_value", "Value_org"]:
        if cand in columns:
            return cand
    raise ValueError(f"No supported value column found in columns: {columns}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect persistent level shifts in EUGM_TS_ppf1."
    )
    parser.add_argument("--ts-parquet", type=Path, default=DEFAULT_TS)
    parser.add_argument("--mp-csv", type=Path, default=DEFAULT_MP)
    parser.add_argument("--background-shp", type=Path, default=DEFAULT_BACKGROUND)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--window", type=int, default=12, help="Pre/post median window in months.")
    parser.add_argument(
        "--min-valid-side",
        type=int,
        default=6,
        help="Minimum number of valid months required on each side of a candidate.",
    )
    parser.add_argument(
        "--persistence-months",
        type=int,
        default=6,
        help="Months used to validate persistence after a candidate break.",
    )
    parser.add_argument(
        "--missing-tolerance-near-break",
        type=int,
        default=2,
        help="Maximum missing values allowed in each side window.",
    )
    parser.add_argument(
        "--score-threshold-review",
        type=float,
        default=8.0,
        help="Minimum normalized score to classify as needs_manual_review.",
    )
    parser.add_argument(
        "--score-threshold-likely",
        type=float,
        default=20.0,
        help="Minimum normalized score to classify as likely_artificial_shift.",
    )
    parser.add_argument(
        "--raw-step-threshold-review",
        type=float,
        default=1.0,
        help="Minimum raw median shift for needs_manual_review.",
    )
    parser.add_argument(
        "--raw-step-threshold-likely",
        type=float,
        default=5.0,
        help="Minimum raw median shift for likely_artificial_shift.",
    )
    parser.add_argument(
        "--min-obs",
        type=int,
        default=60,
        help="Minimum number of valid observations required per station.",
    )
    parser.add_argument(
        "--max-plots",
        type=int,
        default=0,
        help="Limit on number of per-station diagnostic figures (one per flagged "
             "station - can be hundreds). Defaults to 0 (score tables only, no "
             "figures); pass e.g. --max-plots 20 for a manageable sample, or a "
             "large number for the full set.",
    )
    parser.add_argument(
        "--station-id",
        type=str,
        default=None,
        help="If provided, only analyze a single id_ts.",
    )
    parser.add_argument(
        "--render-only-results",
        type=Path,
        default=None,
        help="If provided, skip detection and render outputs from an existing all-stations CSV.",
    )
    return parser.parse_args()


def load_timeseries(ts_path: Path) -> pd.DataFrame:
    if ts_path.suffix.lower() == ".parquet":
        cols = ["id_ts", "id_mp", "TimeInstant", "Value"]
        return pd.read_parquet(ts_path, columns=cols)

    cols = pd.read_csv(ts_path, nrows=0).columns.tolist()
    value_col = detect_value_column(cols)
    usecols = ["id_ts", "id_mp", "TimeInstant", value_col]
    df = pd.read_csv(ts_path, usecols=usecols, parse_dates=["TimeInstant"])
    if value_col != "Value":
        df = df.rename(columns={value_col: "Value"})
    return df


def robust_scale_from_diffs(values: pd.Series) -> float:
    diffs = values.diff()
    diffs = diffs[np.isfinite(diffs)]
    if len(diffs) < 6:
        return np.nan
    mad = np.median(np.abs(diffs - np.median(diffs)))
    scale = 1.4826 * mad
    return float(scale) if np.isfinite(scale) and scale > 0 else np.nan


def window_stats(values: np.ndarray, center_idx: int, window: int) -> tuple[dict, dict] | tuple[None, None]:
    left = values[max(0, center_idx - window):center_idx]
    right = values[center_idx + 1:center_idx + 1 + window]
    left_valid = left[np.isfinite(left)]
    right_valid = right[np.isfinite(right)]
    if len(left_valid) == 0 or len(right_valid) == 0:
        return None, None
    left_stats = {
        "median": float(np.median(left_valid)),
        "count": int(len(left_valid)),
        "missing": int(len(left) - len(left_valid)),
    }
    right_stats = {
        "median": float(np.median(right_valid)),
        "count": int(len(right_valid)),
        "missing": int(len(right) - len(right_valid)),
    }
    return left_stats, right_stats


def persistence_metrics(values: np.ndarray, center_idx: int, step: float, pre_median: float, post_median: float, persistence_months: int) -> dict:
    if persistence_months <= 0:
        return {"after_count": 0, "after_fraction": np.nan, "before_fraction": np.nan}
    sign = np.sign(post_median - pre_median)
    if sign == 0:
        return {"after_count": 0, "after_fraction": 0.0, "before_fraction": 0.0}

    span = persistence_months * 2
    after = values[center_idx + 1:center_idx + 1 + span]
    before = values[max(0, center_idx - span):center_idx]
    after_valid = after[np.isfinite(after)]
    before_valid = before[np.isfinite(before)]
    half_step = abs(step) * 0.5

    if len(after_valid) > 0:
        after_hits = ((after_valid - pre_median) * sign) >= half_step
        after_fraction = float(after_hits.mean())
    else:
        after_fraction = np.nan

    if len(before_valid) > 0:
        before_hits = ((post_median - before_valid) * sign) >= half_step
        before_fraction = float(before_hits.mean())
    else:
        before_fraction = np.nan

    return {
        "after_count": int(len(after_valid)),
        "before_count": int(len(before_valid)),
        "after_fraction": after_fraction,
        "before_fraction": before_fraction,
    }


def detect_station_shift(
    station: pd.DataFrame,
    window: int,
    min_valid_side: int,
    persistence_months: int,
    missing_tolerance_near_break: int,
    min_obs: int,
) -> dict:
    station = station.sort_values("TimeInstant").copy()
    station["TimeInstant"] = pd.to_datetime(station["TimeInstant"], errors="coerce")
    station["Value"] = pd.to_numeric(station["Value"], errors="coerce")
    station = station.dropna(subset=["TimeInstant"])

    values = station["Value"].to_numpy(dtype=float)
    dates = station["TimeInstant"].to_numpy()
    valid_obs = int(np.isfinite(values).sum())
    if valid_obs < min_obs:
        return {
            "id_ts": station["id_ts"].iloc[0],
            "id_mp": station["id_mp"].iloc[0],
            "n_total": int(len(station)),
            "n_valid": valid_obs,
            "status": "insufficient_data",
        }

    robust_scale = robust_scale_from_diffs(station["Value"])
    fallback_scale = float(np.nanstd(np.diff(values[np.isfinite(values)]))) if valid_obs >= 3 else np.nan
    if (not np.isfinite(robust_scale)) or robust_scale <= 0:
        robust_scale = fallback_scale
    if (not np.isfinite(robust_scale)) or robust_scale <= 0:
        robust_scale = np.nan

    best: dict | None = None
    for idx in range(1, len(values) - 1):
        left_stats, right_stats = window_stats(values, idx, window)
        if left_stats is None or right_stats is None:
            continue
        if left_stats["count"] < min_valid_side or right_stats["count"] < min_valid_side:
            continue
        if left_stats["missing"] > missing_tolerance_near_break or right_stats["missing"] > missing_tolerance_near_break:
            continue

        raw_step = abs(right_stats["median"] - left_stats["median"])
        persistence = persistence_metrics(
            values=values,
            center_idx=idx,
            step=raw_step,
            pre_median=left_stats["median"],
            post_median=right_stats["median"],
            persistence_months=persistence_months,
        )
        if persistence["after_count"] < persistence_months or persistence["before_count"] < persistence_months:
            continue

        score = raw_step / robust_scale if np.isfinite(robust_scale) and robust_scale > 0 else np.nan
        row = {
            "id_ts": station["id_ts"].iloc[0],
            "id_mp": station["id_mp"].iloc[0],
            "n_total": int(len(station)),
            "n_valid": valid_obs,
            "shift_time": pd.Timestamp(dates[idx]),
            "pre_median": left_stats["median"],
            "post_median": right_stats["median"],
            "raw_step": float(raw_step),
            "score": float(score) if np.isfinite(score) else np.nan,
            "robust_scale": float(robust_scale) if np.isfinite(robust_scale) else np.nan,
            "left_count": left_stats["count"],
            "right_count": right_stats["count"],
            "left_missing": left_stats["missing"],
            "right_missing": right_stats["missing"],
            "after_fraction": persistence["after_fraction"],
            "before_fraction": persistence["before_fraction"],
            "after_count": persistence["after_count"],
            "before_count": persistence["before_count"],
            "status": "evaluated",
        }

        if best is None:
            best = row
            continue

        best_score = best["score"] if np.isfinite(best["score"]) else -np.inf
        row_score = row["score"] if np.isfinite(row["score"]) else -np.inf
        if (row_score > best_score) or (
            np.isclose(row_score, best_score, equal_nan=True) and row["raw_step"] > best["raw_step"]
        ):
            best = row

    if best is None:
        return {
            "id_ts": station["id_ts"].iloc[0],
            "id_mp": station["id_mp"].iloc[0],
            "n_total": int(len(station)),
            "n_valid": valid_obs,
            "status": "no_candidate",
            "robust_scale": float(robust_scale) if np.isfinite(robust_scale) else np.nan,
        }

    return best


def classify_shift(row: pd.Series, thresholds: Thresholds) -> str:
    score = row.get("score", np.nan)
    raw_step = row.get("raw_step", np.nan)
    after_fraction = row.get("after_fraction", np.nan)
    before_fraction = row.get("before_fraction", np.nan)

    persistence_ok = (
        np.isfinite(after_fraction)
        and np.isfinite(before_fraction)
        and after_fraction >= 0.67
        and before_fraction >= 0.67
    )
    likely_ok = (
        persistence_ok
        and np.isfinite(score)
        and score >= thresholds.score_likely
        and np.isfinite(raw_step)
        and raw_step >= thresholds.raw_step_likely
    )
    review_ok = (
        persistence_ok
        and np.isfinite(score)
        and score >= thresholds.score_review
        and np.isfinite(raw_step)
        and raw_step >= thresholds.raw_step_review
    )

    if likely_ok:
        return "likely_artificial_shift"
    if review_ok:
        return "needs_manual_review"
    return "not_flagged"


def analyze_all_stations(ts: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    records: list[dict] = []
    grouped = ts.groupby("id_ts", sort=False)
    total = grouped.ngroups
    for idx, (_, station) in enumerate(grouped, start=1):
        if idx % 1000 == 0 or idx == total:
            print(f"Processed {idx}/{total} stations")
        records.append(
            detect_station_shift(
                station=station,
                window=args.window,
                min_valid_side=args.min_valid_side,
                persistence_months=args.persistence_months,
                missing_tolerance_near_break=args.missing_tolerance_near_break,
                min_obs=args.min_obs,
            )
        )
    return pd.DataFrame.from_records(records)


def add_metadata(results: pd.DataFrame, mp: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "id_ts",
        "id_mp",
        "region",
        "bhName",
        "relatedParty",
        "MPType",
        "RelatedAquifer",
        "xutm",
        "yutm",
    ]
    keep_cols = [c for c in keep_cols if c in mp.columns]
    mp_small = mp[keep_cols].drop_duplicates(subset=["id_ts"], keep="first")
    return results.merge(mp_small, on=["id_ts", "id_mp"], how="left")


def save_tables(results: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_path = output_dir / "shift_scores_all_stations.csv"
    flagged_path = output_dir / "shift_scores_flagged.csv"
    likely_path = output_dir / "shift_scores_likely_artificial.csv"
    review_path = output_dir / "shift_scores_manual_review.csv"

    results_sorted = results.sort_values(
        by=["class", "score", "raw_step"],
        ascending=[True, False, False],
        na_position="last",
    ).copy()
    results_sorted.to_csv(all_path, index=False)

    flagged = results_sorted[results_sorted["class"].isin(["likely_artificial_shift", "needs_manual_review"])].copy()
    flagged.to_csv(flagged_path, index=False)

    likely = flagged[flagged["class"] == "likely_artificial_shift"].copy()
    review = flagged[flagged["class"] == "needs_manual_review"].copy()
    likely.to_csv(likely_path, index=False)
    review.to_csv(review_path, index=False)
    return flagged, likely, review


def safe_filename(text: str) -> str:
    safe = str(text)
    for ch in ['\\', '/', ':', '*', '?', '"', '<', '>', '|']:
        safe = safe.replace(ch, "_")
    return safe


def plot_station_figure(
    station: pd.DataFrame,
    result_row: pd.Series,
    fig_path: Path,
    window: int,
) -> None:
    station = station.sort_values("TimeInstant").copy()
    station["TimeInstant"] = pd.to_datetime(station["TimeInstant"], errors="coerce")
    station["Value"] = pd.to_numeric(station["Value"], errors="coerce")
    station["pre_roll_median"] = station["Value"].rolling(window, min_periods=max(4, window // 2)).median().shift(1)
    station["post_roll_median"] = (
        station["Value"][::-1].rolling(window, min_periods=max(4, window // 2)).median()[::-1].shift(-1)
    )
    station["step_score"] = (station["post_roll_median"] - station["pre_roll_median"]).abs()
    scale = result_row.get("robust_scale", np.nan)
    station["step_score_norm"] = station["step_score"] / scale if np.isfinite(scale) and scale > 0 else np.nan
    station["diff"] = station["Value"].diff()

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True, constrained_layout=True)
    shift_time = pd.to_datetime(result_row["shift_time"])

    axes[0].plot(station["TimeInstant"], station["Value"], color="#1d4ed8", linewidth=1.4, label="Value")
    axes[0].plot(station["TimeInstant"], station["pre_roll_median"], color="#0f766e", linewidth=1.2, alpha=0.9, label="Pre median")
    axes[0].plot(station["TimeInstant"], station["post_roll_median"], color="#b45309", linewidth=1.2, alpha=0.9, label="Post median")
    axes[0].axvline(shift_time, color="#dc2626", linestyle="--", linewidth=1.3)
    axes[0].set_ylabel("Value")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best")

    axes[1].bar(station["TimeInstant"], station["diff"], width=25, color="#64748b")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].axvline(shift_time, color="#dc2626", linestyle="--", linewidth=1.3)
    axes[1].set_ylabel("Diff")
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(station["TimeInstant"], station["step_score_norm"], color="#7c3aed", linewidth=1.4)
    axes[2].axvline(shift_time, color="#dc2626", linestyle="--", linewidth=1.3)
    axes[2].set_ylabel("Norm. step score")
    axes[2].set_xlabel("Time")
    axes[2].grid(True, alpha=0.25)

    title = (
        f"{result_row['id_ts']} | {result_row.get('class', 'unknown')} | shift {shift_time.date()}\n"
        f"raw_step={result_row.get('raw_step', np.nan):.3f} | score={result_row.get('score', np.nan):.2f} | "
        f"provider={result_row.get('relatedParty', 'NA')}"
    )
    axes[0].set_title(title)

    text = (
        f"id_mp={result_row.get('id_mp', 'NA')}\n"
        f"region={result_row.get('region', 'NA')}\n"
        f"pre={result_row.get('pre_median', np.nan):.3f} post={result_row.get('post_median', np.nan):.3f}\n"
        f"before_fraction={result_row.get('before_fraction', np.nan):.2f} "
        f"after_fraction={result_row.get('after_fraction', np.nan):.2f}\n"
        f"valid={int(result_row.get('n_valid', 0))}"
    )
    axes[0].text(
        0.01,
        0.02,
        text,
        transform=axes[0].transAxes,
        fontsize=9,
        va="bottom",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#cbd5e1"},
    )

    fig.savefig(fig_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_likely_station_figure(
    station: pd.DataFrame,
    result_row: pd.Series,
    fig_path: Path,
    window: int,
) -> None:
    station = station.sort_values("TimeInstant").copy()
    station["TimeInstant"] = pd.to_datetime(station["TimeInstant"], errors="coerce")
    station["Value"] = pd.to_numeric(station["Value"], errors="coerce")
    pre_roll = station["Value"].rolling(window, min_periods=max(4, window // 2)).median().shift(1)
    post_roll = station["Value"][::-1].rolling(window, min_periods=max(4, window // 2)).median()[::-1].shift(-1)
    station["step_score"] = (post_roll - pre_roll).abs()
    scale = result_row.get("robust_scale", np.nan)
    station["step_score_norm"] = station["step_score"] / scale if np.isfinite(scale) and scale > 0 else np.nan
    station["diff"] = station["Value"].diff()

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True, constrained_layout=True)
    shift_time = pd.to_datetime(result_row["shift_time"])

    axes[0].plot(station["TimeInstant"], station["Value"], color="#1d4ed8", linewidth=1.5)
    axes[0].axvline(shift_time, color="#dc2626", linestyle="--", linewidth=1.3)
    axes[0].set_ylabel("Value")
    axes[0].grid(True, alpha=0.25)

    axes[1].bar(station["TimeInstant"], station["diff"], width=25, color="#64748b")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].axvline(shift_time, color="#dc2626", linestyle="--", linewidth=1.3)
    axes[1].set_ylabel("Diff")
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(station["TimeInstant"], station["step_score_norm"], color="#7c3aed", linewidth=1.4)
    axes[2].axvline(shift_time, color="#dc2626", linestyle="--", linewidth=1.3)
    axes[2].set_ylabel("Norm. step score")
    axes[2].set_xlabel("Time")
    axes[2].grid(True, alpha=0.25)

    title = (
        f"{result_row['id_ts']} | shift {shift_time.date()}\n"
        f"raw_step={result_row.get('raw_step', np.nan):.3f} | score={result_row.get('score', np.nan):.2f} | "
        f"provider={result_row.get('relatedParty', 'NA')}"
    )
    axes[0].set_title(title)

    text = (
        f"id_mp={result_row.get('id_mp', 'NA')}\n"
        f"region={result_row.get('region', 'NA')}\n"
        f"before_fraction={result_row.get('before_fraction', np.nan):.2f} "
        f"after_fraction={result_row.get('after_fraction', np.nan):.2f}\n"
        f"valid={int(result_row.get('n_valid', 0))}"
    )
    axes[0].text(
        0.01,
        0.02,
        text,
        transform=axes[0].transAxes,
        fontsize=9,
        va="bottom",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#cbd5e1"},
    )

    fig.savefig(fig_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def create_station_figures(
    ts: pd.DataFrame,
    flagged: pd.DataFrame,
    figures_dir: Path,
    window: int,
    max_plots: int | None,
) -> None:
    if flagged.empty:
        return
    wanted = flagged.copy()
    if max_plots is not None:
        wanted = wanted.head(max_plots).copy()
    wanted_ids = set(wanted["id_ts"])
    ts_subset = ts[ts["id_ts"].isin(wanted_ids)].copy()
    for _, row in wanted.iterrows():
        station = ts_subset[ts_subset["id_ts"] == row["id_ts"]].copy()
        fig_path = figures_dir / f"{safe_filename(row['id_ts'])}_shift.png"
        if fig_path.exists():
            continue
        plot_station_figure(station, row, fig_path, window=window)


def create_likely_station_figures(
    ts: pd.DataFrame,
    likely: pd.DataFrame,
    figures_dir: Path,
    window: int,
    max_plots: int | None,
) -> None:
    if likely.empty:
        return
    wanted = likely.copy()
    if max_plots is not None:
        wanted = wanted.head(max_plots).copy()
    wanted_ids = set(wanted["id_ts"])
    ts_subset = ts[ts["id_ts"].isin(wanted_ids)].copy()
    for _, row in wanted.iterrows():
        station = ts_subset[ts_subset["id_ts"] == row["id_ts"]].copy()
        fig_path = figures_dir / f"{safe_filename(row['id_ts'])}_shift.png"
        if fig_path.exists():
            continue
        plot_likely_station_figure(station, row, fig_path, window=window)


def plot_shift_map(
    flagged: pd.DataFrame,
    background_shp: Path,
    output_png: Path,
    classes: list[str] | None = None,
) -> None:
    to_plot = flagged.copy()
    if classes is not None:
        to_plot = to_plot[to_plot["class"].isin(classes)].copy()
    to_plot = to_plot.dropna(subset=["xutm", "yutm"])
    if to_plot.empty:
        return

    europe = gpd.read_file(background_shp)
    gdf = gpd.GeoDataFrame(
        to_plot,
        geometry=gpd.points_from_xy(to_plot["xutm"], to_plot["yutm"]),
        crs="EPSG:3035",
    )
    colors = {
        "likely_artificial_shift": "#dc2626",
        "needs_manual_review": "#f59e0b",
    }

    fig, ax = plt.subplots(figsize=(14, 10), constrained_layout=True)
    europe.plot(ax=ax, color="#f2efe8", edgecolor="#c8c2b8", linewidth=0.4, zorder=1)
    for cls, group in gdf.groupby("class"):
        group.plot(
            ax=ax,
            color=colors.get(cls, "#475569"),
            markersize=np.clip(group["score"].fillna(1).to_numpy() * 3, 12, 90),
            alpha=0.8,
            edgecolor="#0f172a",
            linewidth=0.2,
            label=cls,
            zorder=2,
        )
    ax.set_axis_off()
    ax.legend(loc="lower left")
    ax.set_title(f"Detected level shifts across Europe ({len(gdf)} stations)")
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    figures_dir = output_dir / "figures"
    likely_figures_dir = output_dir / "figures_likely_artificial_shift"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    likely_figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading time series: {args.ts_parquet}")
    ts = load_timeseries(args.ts_parquet)
    if args.station_id:
        ts = ts[ts["id_ts"] == args.station_id].copy()
        if ts.empty:
            raise ValueError(f"Station {args.station_id} not found in {args.ts_parquet}")

    print(f"Loading metadata: {args.mp_csv}")
    mp = pd.read_csv(args.mp_csv)

    if args.render_only_results is not None:
        print(f"Loading precomputed results: {args.render_only_results}")
        results = pd.read_csv(args.render_only_results, parse_dates=["shift_time"], low_memory=False)
    else:
        print("Detecting strongest shift per station")
        results = analyze_all_stations(ts, args)
        results = add_metadata(results, mp)

        thresholds = Thresholds(
            score_review=args.score_threshold_review,
            score_likely=args.score_threshold_likely,
            raw_step_review=args.raw_step_threshold_review,
            raw_step_likely=args.raw_step_threshold_likely,
        )
        results["class"] = results.apply(classify_shift, axis=1, thresholds=thresholds)

    flagged, likely, review = save_tables(results, output_dir)
    print(f"Flagged stations: {len(flagged)}")
    print(f"Likely artificial shifts: {len(likely)}")
    print(f"Needs manual review: {len(review)}")

    print("Saving flagged-station figures")
    create_station_figures(ts, flagged, figures_dir, window=args.window, max_plots=args.max_plots)
    print("Saving likely-artificial-shift figures")
    create_likely_station_figures(ts, likely, likely_figures_dir, window=args.window, max_plots=args.max_plots)

    print("Saving Europe maps")
    plot_shift_map(flagged, args.background_shp, output_dir / "map_shift_detected_points.png")
    plot_shift_map(likely, args.background_shp, output_dir / "map_likely_artificial_points.png")
    plot_shift_map(review, args.background_shp, output_dir / "map_manual_review_points.png")

    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
