from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import pastas as ps
import geopandas as gpd
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from .after_filter_pipeline import _load_imputed_ts  # reuse robust loader/value selection
from .utils_geo import plot_numeric_map_with_histogram

# Signatures that require (near-)daily data or produce unstable results on monthly series
EXCLUDE_FOR_MONTHLY = {
    "low_pulse_count",
    "high_pulse_count",
    "low_pulse_duration",
    "high_pulse_duration",
    "rise_rate",
    "fall_rate",
    "cv_rise_rate",
    "cv_fall_rate",
    "reversals_avg",
    "reversals_cv",
    "recession_constant",
    "recovery_constant",
    "richards_pathlength",
    "baselevel_index",
    "baselevel_stability",
}
REPLACE_WITH_MONTHLY = {"date_min", "date_max", "cv_date_min", "cv_date_max"}


def _autocorr_time_fixed(series: pd.Series, cutoff: float = 0.8, max_lags_days: float = 365) -> float:
    """
    Version-independent replacement for pastas' autocorr_time on regularly-
    spaced monthly data. Some pastas versions (confirmed regressed by
    1.14.0, confirmed correct in 1.7.0) build ccf()'s lag array as literal
    day values but still feed them into a row-position-shift correlation
    function, so a nominal "30-day lag" actually shifts by 30 rows (~900
    days for monthly data) instead of 30 days - collapsing the result to
    ~1 month for nearly every station regardless of the real data. This
    reimplements the originally-intended algorithm directly: correlate the
    series against itself shifted by 1, 2, 3, ... integer steps (each step
    = one real sampling interval), and return the first step where
    correlation drops below the cutoff - already expressed in steps
    (months for monthly data), no separate day/month conversion needed.
    Validated 2026-08-04 against pastas==1.7.0 (pre-regression) on a
    200-station sample: matching median/mean (2.0/2.3ish on both).
    """
    s = series.dropna()
    if len(s) < 2:
        return np.nan
    dt_mu = s.index.to_series().diff().dropna().dt.total_seconds().median() / (3600 * 24)
    if not dt_mu or np.isnan(dt_mu) or dt_mu <= 0:
        return np.nan
    max_steps = int(max_lags_days // dt_mu)
    x = s.to_numpy()
    n = len(x)
    for step in range(1, max_steps + 1):
        if step >= n:
            return np.nan
        c = np.corrcoef(x[:-step], x[step:])[0, 1]
        if c < cutoff:
            return float(step)
    return np.nan


def _monthly_circular_stats(dates: pd.DatetimeIndex) -> tuple[float, float]:
    """Compute circular mean month and circular std (in months) from a datetime index."""
    if dates is None or len(dates) == 0:
        return np.nan, np.nan
    dates = pd.DatetimeIndex(dates)
    months = dates.month.to_numpy(float)
    m = 12.0
    two_pi = 2 * np.pi
    thetas = months * two_pi / m
    c = np.cos(thetas).sum()
    s = np.sin(thetas).sum()
    r = np.sqrt(c**2 + s**2) / months.size

    if s > 0 and c > 0:
        mean_theta = np.arctan(s / c)
    elif c < 0:
        mean_theta = np.arctan(s / c) + np.pi
    elif s < 0 and c > 0:
        mean_theta = np.arctan(s / c) + two_pi
    else:
        mean_theta = 0.0

    mu = mean_theta * m / two_pi
    std = np.sqrt(-2 * np.log(r)) * m / two_pi
    return mu, std


def _monthly_date_signatures(series: pd.Series) -> dict[str, float]:
    """Monthly equivalents of date_min/date_max and their CVs using months instead of days."""
    if series.empty:
        return {}
    series = series.dropna()
    if series.empty:
        return {}
    # resample to monthly start just to be sure
    series.index = series.index.to_period("M").to_timestamp(how="start")
    annual = series.groupby(series.index.year)
    min_dates = annual.idxmin(skipna=True).dropna()
    max_dates = annual.idxmax(skipna=True).dropna()

    sigs = {}
    if not min_dates.empty:
        mu_min, std_min = _monthly_circular_stats(pd.DatetimeIndex(min_dates.values))
        sigs["date_min"] = mu_min  # month (1-12 scale)
        sigs["cv_date_min"] = std_min / mu_min if mu_min and not np.isnan(mu_min) else np.nan
    if not max_dates.empty:
        mu_max, std_max = _monthly_circular_stats(pd.DatetimeIndex(max_dates.values))
        sigs["date_max"] = mu_max
        sigs["cv_date_max"] = std_max / mu_max if mu_max and not np.isnan(mu_max) else np.nan
    return sigs


def _colwell_monthly(series: pd.Series, bins: int = 11) -> tuple[float, float]:
    """Monthly version of Colwell constancy and contingency to avoid isocalendar issues."""
    s = series.sort_index().dropna()
    if s.empty:
        return np.nan, np.nan
    s = s.resample("ME").mean().dropna()
    if s.empty:
        return np.nan, np.nan

    # Normalize to 0-1
    s_norm = (s - s.min()) / (s.max() - s.min()) if s.max() != s.min() else s.copy()
    binned = pd.cut(s_norm, bins=bins, right=False, include_lowest=True, labels=range(bins))
    df = pd.DataFrame({"state": binned})
    df["time"] = df.index.month
    df["values"] = 1.0
    table = df.pivot_table(columns="state", index="time", values="values", aggfunc="sum")

    # counts
    x = table.sum(axis=1)  # time
    y = table.sum(axis=0)  # state
    z = s.size

    with np.errstate(divide="ignore", invalid="ignore"):
        hx = -(x / z * np.log(x / z)).sum()
        hy = -(y / z * np.log(y / z)).sum()
        hxy = -(table / z * np.log(table / z)).sum().sum()

    p = 1 - (hxy - hx) / np.log(bins)  # predictability
    c = 1 - hy / np.log(bins)  # constancy
    m = (hx + hy - hxy) / np.log(bins)  # contingency
    return c, m


def compute_signatures_dataframe(
    ts_df: pd.DataFrame,
    value_col: str,
    signatures: list[str],
    max_points: int | None = None,
    show_progress: bool = False,
) -> pd.DataFrame:
    """
    Compute Pastas signatures for each monitoring point and return a tidy DataFrame.
    """
    records = []
    grouped = list(ts_df.groupby("id_mp"))
    if max_points is not None:
        grouped = grouped[:max_points]

    iterator = grouped
    if show_progress and tqdm is not None:
        iterator = tqdm(grouped, desc="Computing signatures", unit="mp")
    elif show_progress:
        total = len(grouped)
        print(f"Computing signatures for {total} monitoring points...")

    for idx, (id_mp, group) in enumerate(iterator):
        series = pd.Series(group[value_col].values, index=group["TimeInstant"])
        series.name = id_mp
        try:
            sigs_df = ps.stats.signatures.summary(series, signatures=signatures)
        except Exception as e:
            print(f"Skipping {id_mp}: {e}")
            continue
        # Drop non-finite signature values to avoid downstream artifacts
        sigs_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        # Replace Pastas magnitude with a robust percentile-based magnitude (p95 - p5)
        series_vals = series.dropna().to_numpy()
        if series_vals.size > 0:
            p95 = float(np.nanpercentile(series_vals, 95))
            p5 = float(np.nanpercentile(series_vals, 5))
            mag_val = p95 - p5
            if "magnitude" in sigs_df.index:
                sigs_df.loc["magnitude", id_mp] = mag_val
            else:
                sigs_df.loc["magnitude"] = mag_val
        # Drop extreme cv_period_mean when mean is near zero (unstable CV)
        if "cv_period_mean" in sigs_df.index:
            cv_val = sigs_df.loc["cv_period_mean", id_mp]
            series_mean = series.mean()
            series_std = series.std(ddof=0)
            if (
                pd.notna(cv_val)
                and pd.notna(series_mean)
                and pd.notna(series_std)
                and series_std > 0
                and abs(series_mean) < 0.01 * series_std
            ):
                print(
                    f"Skipping cv_period_mean for {id_mp}: mean {series_mean} too small vs std {series_std}"
                )
                sigs_df.loc["cv_period_mean", id_mp] = np.nan
        # autocorr_time is NOT taken from pastas' own computation - see
        # _autocorr_time_fixed's docstring for why (pastas version
        # regression for regularly-spaced monthly data).
        if "autocorr_time" in sigs_df.index:
            sigs_df.loc["autocorr_time", id_mp] = _autocorr_time_fixed(series)
        # Recompute Colwell constancy/contingency on monthly data explicitly using safer helper
        if "colwell_constancy" in sigs_df.index or "colwell_contingency" in sigs_df.index:
            try:
                monthly_series = pd.Series(series.values, index=pd.DatetimeIndex(series.index)).sort_index().dropna()
                cc, cont = _colwell_monthly(monthly_series)
                if "colwell_constancy" in sigs_df.index:
                    sigs_df.loc["colwell_constancy", id_mp] = cc
                if "colwell_contingency" in sigs_df.index:
                    sigs_df.loc["colwell_contingency", id_mp] = cont
            except Exception as e:
                print(f"Colwell monthly failed for {id_mp}: {e}")
        sigs_series = sigs_df[id_mp]
        sigs_series.name = "value"
        sigs_series = sigs_series.reset_index()
        sigs_series.columns = ["signature", "value"]
        sigs_series["id_mp"] = id_mp
        # Add monthly variants of date_min/date_max and their CVs
        monthly_dates = _monthly_date_signatures(series)
        if monthly_dates:
            for k, v in monthly_dates.items():
                sigs_series = pd.concat([sigs_series, pd.DataFrame([{"signature": k, "value": v, "id_mp": id_mp}])], ignore_index=True)
        records.append(sigs_series)

    if not records:
        return pd.DataFrame(columns=["id_mp", "signature", "value"])
    return pd.concat(records, ignore_index=True)


def plot_signature_violins(sig_df: pd.DataFrame, fig_dir: Path) -> None:
    """
    Create a single figure with per-signature violin plots (separate y-scales), each with stats.
    """
    if sig_df.empty:
        print("No signatures to plot.")
        return

    fig_dir.mkdir(parents=True, exist_ok=True)
    preferred_order = [
        "date_min",
        "date_max",
        "cv_date_min",
        "cv_date_max",
        "autocorr_time",
        "colwell_constancy",
        "colwell_contingency",
        "cv_period_mean",
        "parde_seasonality",
        "avg_seasonal_fluctuation",
        "interannual_variation",
        "bimodality_coefficient",
        "mean_annual_maximum",
        "duration_curve_slope",
        "duration_curve_ratio",
        "magnitude",
    ]
    sigs_present = list(sig_df["signature"].unique())
    sigs_ordered = [s for s in preferred_order if s in sigs_present] + [
        s for s in sigs_present if s not in preferred_order
    ]
    n = len(sigs_ordered)
    if n == 0:
        print("No signatures to plot.")
        return
    rows = 3
    cols = int(np.ceil(n / rows))
    palette = sns.color_palette("husl", n_colors=n)
    color_map = dict(zip(sigs_ordered, palette))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.0), squeeze=False)
    axes = axes.flatten()
    letters = [chr(ord("a") + i) for i in range(n)]

    for ax, sig_name, letter in zip(axes, sigs_ordered, letters):
        sub = sig_df[sig_df["signature"] == sig_name]
        sns.violinplot(data=sub, y="value", color=color_map[sig_name], inner="box", cut=0, ax=ax)
        sns.stripplot(data=sub, y="value", color="k", size=1.5, alpha=0.2, ax=ax)

        stats = sub["value"].describe()
        stats_text = (
            f"mean: {stats['mean']:.3f}\n"
            f"std: {stats['std']:.3f}\n"
            f"min: {stats['min']:.3f}\n"
            f"25%: {stats['25%']:.3f}\n"
            f"50%: {stats['50%']:.3f}\n"
            f"75%: {stats['75%']:.3f}\n"
            f"max: {stats['max']:.3f}"
        )
        ax.text(
            0.98,
            0.95,
            stats_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
        )
        ax.set_title(f"{letter}. {sig_name}", fontsize=11)
        ax.set_xlabel("")
        ax.set_ylabel("Value", fontsize=9)
        ax.tick_params(axis="both", labelsize=9)
        ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Turn off any unused axes
    for ax in axes[n:]:
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(fig_dir / "signatures_violins.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_example_timeseries_by_signature(
    ts_df: pd.DataFrame,
    value_col: str,
    sig_df: pd.DataFrame,
    fig_dir: Path,
) -> None:
    """
    For each signature, plot three time series (min, median, max) with signature values annotated.
    """
    if sig_df.empty or ts_df.empty:
        return

    fig_dir.mkdir(parents=True, exist_ok=True)
    # Remove duplicates by averaging repeated entries
    sig_df_unique = sig_df.groupby(["id_mp", "signature"], as_index=False)["value"].mean()
    pivot = sig_df_unique.pivot(index="id_mp", columns="signature", values="value")

    for sig in sorted(sig_df["signature"].unique()):
        series_vals = pivot[sig].dropna()
        if series_vals.empty:
            continue
        min_id = series_vals.idxmin()
        max_id = series_vals.idxmax()
        median_target = series_vals.median()
        mean_target = series_vals.mean()
        median_id = (series_vals - median_target).abs().idxmin()
        mean_id = (series_vals - mean_target).abs().idxmin()
        ids = [min_id, median_id, mean_id, max_id]
        labels_row = ["Min", "Median", "Mean", "Max"]

        fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=False)
        for ax, mp_id, title_suffix in zip(axes, ids, labels_row):
            sub = ts_df[ts_df["id_mp"] == mp_id]
            ax.plot(sub["TimeInstant"], sub[value_col], color="#336699", linewidth=0.9)
            ax.set_title(f"{sig} - {title_suffix} ({mp_id})", fontsize=12)
            ax.grid(True, axis="y", linestyle="--", linewidth=0.4, alpha=0.6)
            ax.tick_params(axis='both', labelsize=9)
            # remove top/right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Signature value highlight
            sig_val = pivot.loc[mp_id, sig]
            if sig == "cv_period_mean":
                fmt = ".3f"
            else:
                fmt = ".2f"
            ax.text(0.02, 0.98, f"{sig}: {sig_val:{fmt}}", transform=ax.transAxes, fontsize=11, fontweight="bold",
                    va="top", bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))

    plt.tight_layout()
    plt.savefig(fig_dir / f"ts_{sig}.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_signature_compact(
    ts_df: pd.DataFrame,
    value_col: str,
    sig_df: pd.DataFrame,
    fig_dir: Path,
    signature: str,
    figsize: tuple[float, float] = (5.85, 6.4),
    start_date: str | None = None,
    end_date: str | None = None,
    single_axis: bool = False,
) -> None:
    """
    Create a compact (half A4) figure for a single signature.
    """
    if sig_df.empty or ts_df.empty:
        return

    fig_dir.mkdir(parents=True, exist_ok=True)
    sig_df_unique = sig_df.groupby(["id_mp", "signature"], as_index=False)["value"].mean()
    pivot = sig_df_unique.pivot(index="id_mp", columns="signature", values="value")

    if signature not in pivot.columns:
        return

    series_vals = pivot[signature].dropna()
    if series_vals.empty:
        return

    min_id = series_vals.idxmin()
    max_id = series_vals.idxmax()
    median_target = series_vals.median()
    mean_target = series_vals.mean()
    median_id = (series_vals - median_target).abs().idxmin()
    mean_id = (series_vals - mean_target).abs().idxmin()
    ids = [min_id, median_id, mean_id, max_id]
    labels_row = ["Min", "Median", "Mean", "Max"]

    if single_axis:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        fig.suptitle(signature, fontsize=12, y=0.995)
        colors = {
            "Min": "#1f77b4",
            "Median": "#ff7f0e",
            "Mean": "#2ca02c",
            "Max": "#d62728",
        }
        for mp_id, label in zip(ids, labels_row):
            sub = ts_df[ts_df["id_mp"] == mp_id]
            if start_date is not None:
                sub = sub[sub["TimeInstant"] >= pd.Timestamp(start_date)]
            if end_date is not None:
                sub = sub[sub["TimeInstant"] <= pd.Timestamp(end_date)]
            ax.plot(sub["TimeInstant"], sub[value_col], color=colors[label], linewidth=1.0, label=f"{label} ({mp_id})")

        if start_date is not None and end_date is not None:
            ax.set_xlim(pd.Timestamp(start_date), pd.Timestamp(end_date))
        ax.grid(True, axis="y", linestyle="--", linewidth=0.4, alpha=0.6)
        ax.tick_params(axis="both", labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="upper left", fontsize=8, frameon=True, framealpha=0.8)

        plt.tight_layout()
        fig.savefig(fig_dir / f"ts_{signature}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=False)
    fig.suptitle(signature, fontsize=12, y=0.995)
    start_ts = pd.Timestamp(start_date) if start_date is not None else None
    end_ts = pd.Timestamp(end_date) if end_date is not None else None
    for ax, mp_id, title_suffix in zip(axes, ids, labels_row):
        sub = ts_df[ts_df["id_mp"] == mp_id]
        if start_date is not None:
            sub = sub[sub["TimeInstant"] >= pd.Timestamp(start_date)]
        if end_date is not None:
            sub = sub[sub["TimeInstant"] <= pd.Timestamp(end_date)]
        ax.plot(sub["TimeInstant"], sub[value_col], color="#336699", linewidth=0.9)
        ax.set_title(f"{title_suffix} ({mp_id})", fontsize=9)
        if start_ts is not None and end_ts is not None:
            ax.set_xlim(start_ts, end_ts)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.4, alpha=0.6)
        ax.tick_params(axis="both", labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        sig_val = pivot.loc[mp_id, signature]
        fmt = ".3f" if signature == "cv_period_mean" else ".2f"
        ax.text(
            0.02,
            0.98,
            f"{signature}: {sig_val:{fmt}}",
            transform=ax.transAxes,
            fontsize=9,
            fontweight="bold",
            va="top",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
        )

    plt.tight_layout()
    fig.savefig(fig_dir / f"ts_{signature}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_signature_maps(sig_df: pd.DataFrame, metadata_csv: Path, fig_dir: Path, boundary_shp: Path | None = None) -> None:
    """
    Plot maps for each signature with inset histogram and stats using existing utility.
    """
    if sig_df.empty:
        return

    meta_path = Path(metadata_csv)
    if not meta_path.exists():
        print(f"Metadata not found at {meta_path}, skipping signature maps.")
        return

    meta = pd.read_csv(meta_path)
    meta = meta.drop_duplicates(subset="id_mp")
    gdf_points = gpd.GeoDataFrame(
        meta,
        geometry=gpd.points_from_xy(meta.xutm, meta.yutm),
        crs="EPSG:3035",
    )

    # Merge signature values onto metadata
    sig_wide = sig_df.pivot_table(index="id_mp", columns="signature", values="value", aggfunc="mean")
    gdf_points = gdf_points.merge(sig_wide, on="id_mp", how="left")

    gdfload = None
    if boundary_shp is not None and Path(boundary_shp).exists():
        gdfload = gpd.read_file(Path(boundary_shp))
        gdfload["Country"] = gdfload["Country"].replace("Belgium", "Belgium- Flanders")
    else:
        print(f"Note: boundary shapefile not found at {boundary_shp} - plotting points without country outlines.")

    fig_dir.mkdir(parents=True, exist_ok=True)

    for sig in sorted(sig_df["signature"].unique()):
        if sig not in gdf_points.columns:
            continue
        gsub = gdf_points.dropna(subset=[sig])
        if gsub.empty:
            continue
        values = gsub[sig].to_numpy()
        cap_max = np.nanpercentile(values, 99) if np.isfinite(values).any() else np.nan
        if not np.isfinite(cap_max) or cap_max == 0:
            cap_max = np.nanmax(values) if np.isfinite(values).any() else 1.0
        cap_max = float(cap_max) if np.isfinite(cap_max) else 1.0
        cap_min = None
        # Override caps for specific signatures
        if sig == "cv_date_max":
            cap_max = 4.0
        elif sig == "cv_date_min":
            cap_max = 6.0
        elif sig == "interannual_variation":
            cap_max = 15.0
        elif sig == "magnitude":
            cap_max = 15.0
        elif sig == "duration_curve_slope":
            cap_min = -25.0
        elif sig == "cv_period_mean":
            cap_min = -1.0
        # Use finer formatting for selected continuous signatures
        fine_ticks = {
            "magnitude",
            "mean_annual_maximum",
            "parde_seasonality",
            "colwell_constancy",
            "colwell_contingency",
            "cv_period_mean",
        }
        tick_fmt = (
            ".3f" if sig == "cv_period_mean"
            else ".2f" if sig in fine_ticks or sig == "cv_date_min"
            else None
        )
        stats_dec = (
            3 if sig == "cv_period_mean"
            else 2 if sig in fine_ticks or sig == "cv_date_min"
            else 1
        )
        overflow = sig in {"cv_date_max", "cv_date_min", "interannual_variation", "magnitude"}
        plot_numeric_map_with_histogram(
            gdf_points=gsub.assign(value=gsub[sig]),
            gdf_boundary=gdfload,
            column="value",
            label=sig,
            cap_max=cap_max,
            cap_min=cap_min,
            cmap_name="viridis",
            tick_format=tick_fmt,
            stats_decimals=stats_dec,
            show_overflow_label=overflow,
            stats_uncapped=True,  # 99th-percentile cap is for visualization only here -
                                   # reported stats should match the manuscript text, not
                                   # be silently reduced by the map's colour-scale clipping.
            save_path=fig_dir / f"map_{sig}.png",
        )


def compose_signature_maps_grid(fig_dir: Path) -> None:
    """
    Build a 4x4 grid image from individual signature maps in a fixed order with labels.
    """
    order = [
        "date_min",
        "date_max",
        "cv_date_min",
        "cv_date_max",
        "autocorr_time",
        "colwell_constancy",
        "colwell_contingency",
        "cv_period_mean",
        "parde_seasonality",
        "avg_seasonal_fluctuation",
        "interannual_variation",
        "bimodality_coefficient",
        "mean_annual_maximum",
        "duration_curve_slope",
        "duration_curve_ratio",
        "magnitude",
    ]
    fig_dir.mkdir(parents=True, exist_ok=True)
    imgs = []
    for sig in order:
        img_path = fig_dir / f"map_{sig}.png"
        if img_path.exists():
            imgs.append((sig, mpimg.imread(img_path)))
        else:
            imgs.append((sig, None))

    cols = 4
    rows = 4
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 3.5))
    axes = axes.flatten()
    letters = [chr(ord('a') + i) for i in range(len(order))]

    for ax, (sig, img), letter in zip(axes, imgs, letters):
        ax.axis("off")
        if img is not None:
            ax.imshow(img)
        ax.text(0.02, 0.97, f"{letter}. {sig}", transform=ax.transAxes, fontsize=11, fontweight="bold",
                va="top", ha="left", bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

    plt.tight_layout()
    fig.savefig(fig_dir / "maps_grid.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def compose_signature_maps_grid_reduced(fig_dir: Path) -> None:
    """
    Build a 2x3 grid image from selected signature maps with labels.
    """
    order = [
        "cv_date_max",
        "autocorr_time",
        "colwell_contingency",
        "interannual_variation",
        "mean_annual_maximum",
        "magnitude",
    ]
    fig_dir.mkdir(parents=True, exist_ok=True)
    imgs = []
    for sig in order:
        img_path = fig_dir / f"map_{sig}.png"
        if img_path.exists():
            imgs.append((sig, mpimg.imread(img_path)))
        else:
            imgs.append((sig, None))

    cols = 2
    rows = 3
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6.0, rows * 4.2))
    axes = axes.flatten()
    letters = [chr(ord('a') + i) for i in range(len(order))]

    for ax, (sig, img), letter in zip(axes, imgs, letters):
        ax.axis("off")
        if img is not None:
            ax.imshow(img)
        ax.text(
            0.02,
            0.97,
            f"{letter}. {sig}",
            transform=ax.transAxes,
            fontsize=16,
            fontweight="normal",
            va="top",
            ha="left",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )

    plt.tight_layout()
    fig.savefig(fig_dir / "maps_grid_reduced.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_signatures_pipeline(
    imputed_csv: Path,
    metadata_csv: Path,
    fig_dir: Path,
    sig_out: Path,
    boundary_shp: Path | None = None,
    compute: bool = True,
) -> None:
    """
    Entry point to compute signatures and create the Fig. 6 grid.
    """
    imputed_path = Path(imputed_csv)
    ts_df, value_col = _load_imputed_ts(imputed_path)

    all_signatures = ps.stats.signatures.__all__
    # Drop daily/unstable signatures, those we replace with monthly variants, and Pastas magnitude
    signatures = [
        s for s in all_signatures
        if s not in EXCLUDE_FOR_MONTHLY and s not in REPLACE_WITH_MONTHLY and s != "magnitude"
    ]
    skipped = sorted(set(all_signatures) - set(signatures))
    if skipped:
        print("Skipping signatures not suitable for monthly data:", ", ".join(skipped))

    # Optional limit via env var (set by CLI)
    import os
    max_mp = os.getenv("SIGNATURES_MAX_MP")
    max_mp = int(max_mp) if max_mp is not None else None

    sig_out = Path(sig_out)

    if compute or not sig_out.exists():
        sig_df = compute_signatures_dataframe(
            ts_df,
            value_col=value_col,
            signatures=signatures,
            max_points=max_mp,
            show_progress=True,
        )
        sig_df.to_csv(sig_out, index=False)
    else:
        sig_df = pd.read_csv(sig_out)
        print(f"Loaded existing signatures from {sig_out}")

    fig_dir = Path(fig_dir)
    # This copy is trimmed to reproduce only the two grid figures actually
    # used in the manuscript (Figure 6 = maps_grid_reduced.png). The other
    # figure types (violins, example time series, per-signature compact
    # plots) are exploratory outputs not part of the manuscript - kept here,
    # commented out, for reference.
    # plot_signature_violins(sig_df, fig_dir)
    # plot_example_timeseries_by_signature(ts_df, value_col=value_col, sig_df=sig_df, fig_dir=fig_dir)
    plot_signature_maps(sig_df, metadata_csv=metadata_csv, fig_dir=fig_dir, boundary_shp=boundary_shp)  # produces the per-signature maps the two grids below are composed from
    compose_signature_maps_grid(fig_dir)
    compose_signature_maps_grid_reduced(fig_dir)
    # plot_signature_compact(
    #     ts_df,
    #     value_col=value_col,
    #     sig_df=sig_df,
    #     fig_dir=fig_dir,
    #     signature="avg_seasonal_fluctuation",
    #     start_date="2006-01-01",
    #     end_date="2021-12-31",
    #     single_axis=False,
    # )
    # plot_signature_compact(
    #     ts_df,
    #     value_col=value_col,
    #     sig_df=sig_df,
    #     fig_dir=fig_dir,
    #     signature="cv_date_max",
    #     start_date="1990-01-01",
    #     end_date="2021-12-31",
    #     single_axis=False,
    # )
    # plot_signature_compact(
    #     ts_df,
    #     value_col=value_col,
    #     sig_df=sig_df,
    #     fig_dir=fig_dir,
    #     signature="autocorr_time",
    #     start_date="1960-01-01",
    #     end_date="2021-12-31",
    #     single_axis=False,
    # )
    # plot_signature_compact(
    #     ts_df,
    #     value_col=value_col,
    #     sig_df=sig_df,
    #     fig_dir=fig_dir,
    #     signature="colwell_contingency",
    #     start_date="1990-01-01",
    #     end_date="2021-12-31",
    #     single_axis=False,
    # )

    print(f"Computed signatures for {sig_df['id_mp'].nunique()} monitoring points.")
    print(f"Saved signatures table to {sig_out}")
    print(f"Saved signature maps and grids to {fig_dir}")


