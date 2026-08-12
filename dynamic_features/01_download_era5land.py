#!/usr/bin/env python3
"""
Download ERA5-Land monthly-mean NetCDF files (2 m temperature, total
precipitation, 2 m dewpoint temperature, surface pressure) covering the
bounding box of all EUGM monitoring points, via the Copernicus Climate Data
Store (CDS) API.

Requires CDS API credentials in ~/.cdsapirc (see
https://cds.climate.copernicus.eu/how-to-api). Downloads are cached by
filename - re-running only fetches months not already present in
--output-dir, so this is safe to re-run to extend coverage over time.
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import cdsapi
import pandas as pd
import requests
from pyproj import Transformer

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent

VARIABLES = ["2m_temperature", "total_precipitation", "2m_dewpoint_temperature", "surface_pressure"]


def compute_bbox(points_csv: Path, source_crs: str, pad_degrees: float) -> list[float]:
    """Bounding box [North, West, South, East] around all monitoring points, in EPSG:4326."""
    points = pd.read_csv(points_csv, usecols=["xutm", "yutm"]).dropna()
    transformer = Transformer.from_crs(source_crs, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(points["xutm"].to_numpy(), points["yutm"].to_numpy())
    return [
        float(lat.max() + pad_degrees),
        float(lon.min() - pad_degrees),
        float(lat.min() - pad_degrees),
        float(lon.max() + pad_degrees),
    ]


def iter_year_months(start_year: int, end_year: int, end_month: int):
    y, m = start_year, 1
    while (y, m) <= (end_year, end_month):
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def is_no_data_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(needle in text for needle in ["mars returned no data", "no data", "multiadaptornodataerror"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points-csv", type=Path, default=REPO_ROOT / "data" / "EUGM_mp.csv",
                         help="Monitoring point metadata with xutm/yutm (EPSG:3035).")
    parser.add_argument("--source-crs", default="EPSG:3035")
    parser.add_argument("--pad-degrees", type=float, default=0.5, help="Padding around the point bbox, in degrees.")
    parser.add_argument("--output-dir", type=Path, default=HERE / "era5land_raw",
                         help="Where to save downloaded monthly NetCDF/zip files.")
    parser.add_argument("--start-year", type=int, default=1950)
    parser.add_argument("--end-year", type=int, default=None, help="Defaults to the current year.")
    parser.add_argument("--end-month", type=int, default=None, help="Defaults to last month.")
    args = parser.parse_args()

    today = date.today()
    if args.end_year is None or args.end_month is None:
        last_day_prev_month = date(today.year, today.month, 1) - timedelta(days=1)
        args.end_year = args.end_year or last_day_prev_month.year
        args.end_month = args.end_month or last_day_prev_month.month

    area = compute_bbox(args.points_csv, args.source_crs, args.pad_degrees)
    print(f"ERA5-Land area [N,W,S,E]: {area}")
    print(f"Download window: {args.start_year}-01 to {args.end_year}-{args.end_month:02d}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()

    for year, month in iter_year_months(args.start_year, args.end_year, args.end_month):
        target = args.output_dir / f"era5land_monthly_means_{year}_{month:02d}.nc"
        if target.exists():
            continue
        request = {
            "product_type": "monthly_averaged_reanalysis",
            "data_format": "netcdf",
            "variable": VARIABLES,
            "year": f"{year}",
            "month": [f"{month:02d}"],
            "time": ["00:00"],
            "area": area,
        }
        print(f"[download] {target.name}")
        try:
            client.retrieve("reanalysis-era5-land-monthly-means", request, str(target))
        except requests.HTTPError as exc:
            if is_no_data_error(exc):
                print(f"[skip-no-data] {target.name}: not published by Copernicus yet")
                continue
            raise

    print(f"Done. Files in {args.output_dir}")


if __name__ == "__main__":
    main()
