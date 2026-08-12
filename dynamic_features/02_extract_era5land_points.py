#!/usr/bin/env python3
"""
Extract point-wise ERA5-Land time series at every EUGM monitoring point from
the monthly NetCDF files downloaded by 01_download_era5land.py.

For each point, uses the nearest ERA5-Land grid cell that has data (falling
back to a widening search radius for points near the coast/grid edge, where
the nearest literal cell may be ocean/no-data). Also derives relative
humidity from 2 m temperature and 2 m dewpoint temperature.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent

CDS_TO_SHORT = {
    "t2m": "t2m",
    "tp": "tp",
    "d2m": "d2m",
    "sp": "sp",
}


def _normalize_downloaded_files(raw_dir: Path) -> list[Path]:
    """CDS sometimes delivers a zip containing the .nc file instead of a raw .nc file."""
    normalized_dir = raw_dir / "_normalized"
    normalized_dir.mkdir(exist_ok=True)
    valid_files = []
    for src in sorted(raw_dir.glob("era5land_monthly_means_*.nc")):
        with src.open("rb") as f:
            sig = f.read(4)
        if sig.startswith(b"CDF") or sig.startswith(b"\x89HDF"):
            valid_files.append(src)
        elif sig == b"PK\x03\x04":
            target = normalized_dir / f"{src.stem}__inner.nc"
            if not target.exists():
                with zipfile.ZipFile(src) as zf:
                    nc_members = [n for n in zf.namelist() if n.lower().endswith(".nc")]
                    with zf.open(nc_members[0]) as member, target.open("wb") as out:
                        out.write(member.read())
            valid_files.append(target)
    return valid_files


def find_nearest_valid_cell(lat, lon, lat_vals, lon_vals, valid_mask, max_radius_cells=25):
    i0 = int(np.abs(lat_vals - lat).argmin())
    j0 = int(np.abs(lon_vals - lon).argmin())
    if valid_mask[i0, j0]:
        return i0, j0
    n_lat, n_lon = valid_mask.shape
    for r in range(1, max_radius_cells + 1):
        i_min, i_max = max(0, i0 - r), min(n_lat, i0 + r + 1)
        j_min, j_max = max(0, j0 - r), min(n_lon, j0 + r + 1)
        window = valid_mask[i_min:i_max, j_min:j_max]
        if not window.any():
            continue
        ii_rel, jj_rel = np.where(window)
        ii, jj = ii_rel + i_min, jj_rel + j_min
        d2 = (lat_vals[ii] - lat) ** 2 + (lon_vals[jj] - lon) ** 2
        k = int(np.argmin(d2))
        return int(ii[k]), int(jj[k])
    ii, jj = np.where(valid_mask)
    if len(ii) == 0:
        return None
    d2 = (lat_vals[ii] - lat) ** 2 + (lon_vals[jj] - lon) ** 2
    k = int(np.argmin(d2))
    return int(ii[k]), int(jj[k])


def compute_relative_humidity_percent(t2m_k: pd.Series, d2m_k: pd.Series) -> pd.Series:
    t_c, td_c = t2m_k - 273.15, d2m_k - 273.15
    rh = 100.0 * np.exp((17.625 * td_c) / (243.04 + td_c) - (17.625 * t_c) / (243.04 + t_c))
    return rh.clip(lower=0.0, upper=100.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=HERE / "era5land_raw",
                         help="Directory of monthly NetCDF files from 01_download_era5land.py.")
    parser.add_argument("--points-csv", type=Path, default=REPO_ROOT / "data" / "EUGM_mp.csv",
                         help="Monitoring point metadata with id_mp/xutm/yutm (EPSG:3035).")
    parser.add_argument("--source-crs", default="EPSG:3035")
    parser.add_argument("--output-csv", type=Path, default=HERE / "era5land_points.csv",
                         help="Output: one row per (id_mp, time) with t2m/sp/d2m/tp/rh.")
    args = parser.parse_args()

    print("Normalizing and opening downloaded NetCDF files...")
    files = _normalize_downloaded_files(args.raw_dir)
    if not files:
        raise FileNotFoundError(f"No ERA5-Land NetCDF files found in {args.raw_dir}. Run 01_download_era5land.py first.")
    datasets = [xr.open_dataset(f, engine="netcdf4") for f in files]
    ds = xr.combine_by_coords(datasets, combine_attrs="override")
    if "valid_time" in ds.coords and "time" not in ds.coords:
        ds = ds.rename({"valid_time": "time"})

    points = pd.read_csv(args.points_csv, usecols=["id_mp", "xutm", "yutm"]).dropna(subset=["xutm", "yutm"])
    transformer = Transformer.from_crs(args.source_crs, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(points["xutm"].to_numpy(), points["yutm"].to_numpy())
    points = points.assign(lon=lon, lat=lat)

    lat_vals, lon_vals = ds["latitude"].values, ds["longitude"].values
    source_vars = [v for v in CDS_TO_SHORT if v in ds.data_vars]
    valid_mask = None
    for var in source_vars:
        da = ds[var].notnull()
        for d in [dim for dim in da.dims if dim not in ("latitude", "longitude")]:
            da = da.any(dim=d)
        valid_mask = da.values if valid_mask is None else (valid_mask | da.values)

    print(f"Extracting {len(points)} points...")
    frames = []
    for idx, row in points.iterrows():
        pick = find_nearest_valid_cell(row["lat"], row["lon"], lat_vals, lon_vals, valid_mask)
        if pick is None:
            continue
        i, j = pick
        point_df = ds.isel(latitude=i, longitude=j).to_dataframe().reset_index()
        keep = ["time"] + [v for v in source_vars]
        point_df = point_df[keep].rename(columns=CDS_TO_SHORT)
        point_df["id_mp"] = row["id_mp"]
        frames.append(point_df)
        if (idx + 1) % 2000 == 0:
            print(f"  {idx + 1}/{len(points)}")

    extracted = pd.concat(frames, ignore_index=True)
    if "t2m" in extracted.columns and "d2m" in extracted.columns:
        extracted["rh"] = compute_relative_humidity_percent(extracted["t2m"], extracted["d2m"])

    cols = ["time", "id_mp"] + [c for c in ["t2m", "sp", "rh", "tp", "d2m"] if c in extracted.columns]
    extracted = extracted[cols]
    extracted.to_csv(args.output_csv, index=False)
    print(f"Saved {len(extracted):,} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
