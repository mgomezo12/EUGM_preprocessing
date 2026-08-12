#!/usr/bin/env python3
"""
Compute the Standardised Precipitation Index (SPI1/6/12/48) from the
point-extracted ERA5-Land time series (02_extract_era5land_points.py) and
assemble the final dynamic.csv-shaped table (id_mp, TimeInstant, t2m, sp,
rh, tp, spi1, spi6, spi12, spi48).

SPI is computed per monitoring point on a rolling sum of precipitation
(window = 1/6/12/48 months), fitted to a gamma distribution and mapped to
the standard normal distribution (Kumar et al., 2009), with the zero-rainfall
probability handled explicitly (a month can have literally zero rain, which
the gamma distribution alone cannot represent).

The output is restricted to each station's actual (id_mp, TimeInstant)
coverage in data/EUGM_gwl.csv, not ERA5's full 1950-present range - this
matches the published dynamic.csv (a station's dynamic covariates only
extend as far as its own groundwater record) and matters for SPI itself:
the rolling window's lead-in period is computed from the station's own
record start, not from decades of ERA5 data the station never had
groundwater observations for.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gamma, norm

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent

SPI_WINDOWS = {"spi1": 1, "spi6": 6, "spi12": 12, "spi48": 48}


def _compute_spi_from_accumulated(accum: pd.Series) -> pd.Series:
    values = pd.to_numeric(accum, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(values)
    out = np.full(values.shape, np.nan, dtype=float)
    if valid.sum() < 3:
        return pd.Series(out, index=accum.index)

    vals = values[valid]
    zero_mask = vals <= 0
    q_zero = float(zero_mask.mean())
    positive = vals[~zero_mask]

    if positive.size < 3:
        mean, std = float(np.nanmean(vals)), float(np.nanstd(vals, ddof=0))
        out[valid] = (vals - mean) / std if std > 0 else 0.0
        return pd.Series(out, index=accum.index)

    try:
        shape, loc, scale = gamma.fit(positive, floc=0)
        cdf = np.empty_like(vals, dtype=float)
        cdf[zero_mask] = q_zero
        cdf[~zero_mask] = q_zero + (1.0 - q_zero) * gamma.cdf(positive, a=shape, loc=loc, scale=scale)
        cdf = np.clip(cdf, 1e-12, 1 - 1e-12)
        out[valid] = norm.ppf(cdf)
    except Exception:
        mean, std = float(np.nanmean(vals)), float(np.nanstd(vals, ddof=0))
        out[valid] = (vals - mean) / std if std > 0 else 0.0

    return pd.Series(out, index=accum.index)


def add_spi_features(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("TimeInstant").copy()
    tp = pd.to_numeric(group["tp"], errors="coerce")
    for spi_name, window in SPI_WINDOWS.items():
        accumulated = tp.rolling(window=window, min_periods=window).sum()
        group[spi_name] = _compute_spi_from_accumulated(accumulated)
    return group


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points-csv", type=Path, default=HERE / "era5land_points.csv",
                         help="Output of 02_extract_era5land_points.py.")
    parser.add_argument("--gwl-csv", type=Path, default=REPO_ROOT / "data" / "EUGM_gwl.csv",
                         help="Restricts output to each station's actual (id_mp, TimeInstant) "
                              "coverage - dynamic.csv doesn't extend beyond a station's own "
                              "groundwater record.")
    parser.add_argument("--output-csv", type=Path, default=REPO_ROOT / "data" / "dynamic.csv",
                         help="Final dynamic.csv-shaped output.")
    args = parser.parse_args()

    df = pd.read_csv(args.points_csv, parse_dates=["time"])
    df = df.rename(columns={"time": "TimeInstant"})
    for col in ["t2m", "sp", "rh", "tp"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    gwl_index = pd.read_csv(args.gwl_csv, usecols=["id_mp", "TimeInstant"], parse_dates=["TimeInstant"])
    df = gwl_index.merge(df, on=["id_mp", "TimeInstant"], how="left")

    out = df.groupby("id_mp", group_keys=False).apply(add_spi_features).reset_index(drop=True)
    out = out[["id_mp", "TimeInstant", "t2m", "sp", "rh", "tp", "spi1", "spi6", "spi12", "spi48"]]
    out = out.sort_values(["id_mp", "TimeInstant"])

    out.to_csv(args.output_csv, index=False)
    print(f"Saved {len(out):,} rows for {out['id_mp'].nunique():,} monitoring points to {args.output_csv}")


if __name__ == "__main__":
    main()
