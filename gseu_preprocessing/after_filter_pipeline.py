from pathlib import Path
import pandas as pd
import geopandas as gpd
import numpy as np
import pyarrow.parquet as pq

from .utils_geo import (
    compute_spatial_density,
    plot_density_histograms,
    plot_depth_map_continuous,
    plot_numeric_map_with_histogram,
    plot_years_map_with_histogram,
    plot_start_date_map_with_histogram,
    plot_categorical_map_and_histogram_per_column,
    verify_country_prefix_alignment,
    plot_multipoint_map_cdf,
)
from .utils_plot import (
    plot_elevation_histogram,
    plot_well_depth_histogram,
    plot_screen_length_histogram,
    plot_num_years_histogram,
    plot_monthly_station_counts,
)


def _load_filtered_metadata(imputed_csv: Path, metadata_csv: Path) -> gpd.GeoDataFrame:
    """
    Load metadata and filter to monitoring points present in the imputed time series file.
    """
    if not imputed_csv.exists():
        raise FileNotFoundError(f"Imputed file not found: {imputed_csv}")
    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_csv}")

    # Read only id_mp column to keep memory small
    if imputed_csv.suffix.lower() == ".parquet":
        id_mp_series = pd.read_parquet(imputed_csv, columns=["id_mp"])
    else:
        id_mp_series = pd.read_csv(imputed_csv, usecols=["id_mp"])
    id_mp_series = id_mp_series["id_mp"].dropna().astype(str)
    id_mp_set = set(id_mp_series.unique())

    metadata = pd.read_csv(metadata_csv)
    metadata = metadata[metadata['id_mp'].astype(str).isin(id_mp_set)].copy()
    metadata = metadata.drop_duplicates(subset=['id_mp'])

    # Clean elevation, depth, screen length
    metadata['elevation'] = pd.to_numeric(metadata['elevation'], errors='coerce')
    metadata.loc[metadata['elevation'] <= -7, 'elevation'] = np.nan

    metadata['depth'] = pd.to_numeric(metadata['depth'], errors='coerce')
    metadata.loc[metadata['depth'] <= 0, 'depth'] = np.nan
    metadata = metadata[~metadata['depth'].isin([np.inf, -np.inf])]

    metadata['ScreenLength'] = pd.to_numeric(metadata['ScreenLength'], errors='coerce')
    metadata.loc[metadata['ScreenLength'] <= 0, 'ScreenLength'] = np.nan

    metadata['ScreenTop'] = pd.to_numeric(metadata.get('ScreenTop'), errors='coerce')
    metadata.loc[metadata['ScreenTop'] <= -9000, 'ScreenTop'] = np.nan
    metadata = metadata[~metadata['ScreenTop'].isin([np.inf, -np.inf])]

    metadata['ScreenBottom'] = pd.to_numeric(metadata.get('ScreenBottom'), errors='coerce')
    metadata.loc[metadata['ScreenBottom'] <= -9000, 'ScreenBottom'] = np.nan
    metadata = metadata[~metadata['ScreenBottom'].isin([np.inf, -np.inf])]

    gdf_points = gpd.GeoDataFrame(
        metadata,
        geometry=gpd.points_from_xy(metadata.xutm, metadata.yutm),
        crs='EPSG:3035'
    )

    return gdf_points


def _detect_value_column(columns: list[str]) -> str:
    # Value_comp (Value_org where observed, else Imputed_value) is the
    # complete gap-filled series and must be preferred whenever present -
    # this is what data/EUGM_gwl.csv provides. Imputed_value alone is only
    # populated where Value_org was missing (~3% of rows), so picking it
    # here would silently starve every downstream calculation of data.
    for cand in ['Value_comp', 'Value_imp', 'Imputed_value', 'Value', 'Value_org']:
        if cand in columns:
            return cand
    raise ValueError("No value column found in imputed file.")


def _load_imputed_ts(imputed_csv: Path) -> tuple[pd.DataFrame, str]:
    """
    Load imputed time series with minimal columns and return dataframe plus chosen value column.
    """
    if imputed_csv.suffix.lower() == ".parquet":
        cols = pq.read_schema(imputed_csv).names
    else:
        cols = pd.read_csv(imputed_csv, nrows=0).columns.tolist()
    value_col = _detect_value_column(cols)
    usecols = ["id_mp", "TimeInstant", value_col]
    if imputed_csv.suffix.lower() == ".parquet":
        df = pd.read_parquet(imputed_csv, columns=usecols)
        df["TimeInstant"] = pd.to_datetime(df["TimeInstant"], errors="coerce")
    else:
        df = pd.read_csv(imputed_csv, usecols=usecols, parse_dates=["TimeInstant"])
    df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
    df = df.dropna(subset=[value_col])
    df = df.dropna(subset=['TimeInstant'])
    df['TimeInstant'] = df['TimeInstant'].dt.to_period("M").dt.to_timestamp(how="start")
    return df, value_col


def _compute_mp_duration(imputed_csv: Path, df_ts: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Compute start, end, and number of years of observations per monitoring point using the imputed file.
    """
    if df_ts is None:
        df_ts, _ = _load_imputed_ts(imputed_csv)

    stats = df_ts.groupby('id_mp').agg(
        start_date=('TimeInstant', 'min'),
        end_date=('TimeInstant', 'max')
    ).reset_index()
    stats['num_years'] = (stats['end_date'] - stats['start_date']).dt.days / 365.25
    return stats


def run_after_filter_plots(
    imputed_csv: Path,
    metadata_csv: Path,
    fig_dir: Path,
    boundary_shp: Path | None = None,
) -> None:
    """
    Generate plots for the filtered/imputed time series set (Fig. 2, 3, A1):
    - Fig. 2a/2b: start-date and record-length maps
    - Fig. 3a-3d: AquiferType/AquiferMediaType/depth/ScreenLength maps
    - Fig. A1: multi-point location map with CDF inset
    """
    imputed_path = Path(imputed_csv)
    metadata_path = Path(metadata_csv)

    gdfload = None
    if boundary_shp is not None and Path(boundary_shp).exists():
        gdfload = gpd.read_file(Path(boundary_shp))
        gdfload['Country'] = gdfload['Country'].replace('Belgium', 'Belgium- Flanders')
    else:
        print(f"Note: boundary shapefile not found at {boundary_shp} - plotting points without country outlines.")

    ts_df, value_col = _load_imputed_ts(imputed_path)

    gdf_points = _load_filtered_metadata(imputed_path, metadata_path)
    if gdf_points.empty:
        raise ValueError("No monitoring points matched between metadata and imputed file.")

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # This copy is trimmed to reproduce only the figures actually used in
    # the manuscript. The other plots below (density, elevation/depth/
    # screen-length histograms, ScreenTop/ScreenBottom maps, the plain
    # years-of-data histogram, the Pumping category map, monthly station
    # counts, country-prefix QA) are exploratory/diagnostic outputs not
    # part of the manuscript - kept here, commented out, for reference.

    # density_df, _ = compute_spatial_density(gdf_points, gdfload)
    # plot_density_histograms(density_df, fig_dir / "d")
    # plot_elevation_histogram(gdf_points, fig_dir / "elevation_histv3.png")
    # plot_well_depth_histogram(gdf_points, fig_dir / "depthv3.png")

    # Depth map with histogram inset -> Figure 3c
    plot_depth_map_continuous(
        gdf_points=gdf_points,
        gdf_boundary=gdfload,
        depth_column='depth',
        save_path=fig_dir / "depth_map_hist.png",
    )

    # plot_screen_length_histogram(gdf_points, fig_dir / "screenlength_hist.png")

    # Screen length map with inset histogram -> Figure 3d
    plot_numeric_map_with_histogram(
        gdf_points=gdf_points,
        gdf_boundary=gdfload,
        column='ScreenLength',
        label='Screen length (m)',
        cap_max=100.0,
        cmap_name='viridis',
        save_path=fig_dir / "screenlength_map_hist.png",
    )
    # plot_numeric_map_with_histogram(
    #     gdf_points=gdf_points,
    #     gdf_boundary=gdfload,
    #     column='ScreenTop',
    #     label='Screen top (m)',
    #     cap_max=300.0,
    #     cmap_name='viridis',
    #     save_path=fig_dir / "screentop_map_hist.png",
    # )
    # plot_numeric_map_with_histogram(
    #     gdf_points=gdf_points,
    #     gdf_boundary=gdfload,
    #     column='ScreenBottom',
    #     label='Screen bottom (m)',
    #     cap_max=400.0,
    #     cmap_name='viridis',
    #     save_path=fig_dir / "screenbottom_map_hist.png",
    # )

    # Years of data map with histogram inset -> Figure 2b
    ts_duration_df = _compute_mp_duration(imputed_path, df_ts=ts_df)
    plot_years_map_with_histogram(
        gdf_points=gdf_points,
        gdf_boundary=gdfload,
        ts_duration_df=ts_duration_df,
        save_path=fig_dir / "years_mp_after_filtering.png",
    )
    # plot_num_years_histogram(
    #     ts_duration_df,
    #     save_path=fig_dir / "hist_years_mp_after_filtering.png",
    # )

    # Start-year map with histogram inset -> Figure 2a
    plot_start_date_map_with_histogram(
        gdf_points=gdf_points,
        gdf_boundary=gdfload,
        ts_duration_df=ts_duration_df,
        save_path=fig_dir / "start_dates_map_hist.png",
    )

    # Prepare categorical metadata fields for after-filter maps.
    gdf_points['AquiferMediaType'] = gdf_points['AquiferMediaType'].replace('NaN', 'unknown')
    gdf_points['AquiferType'] = gdf_points['AquiferType'].fillna('unknown')

    # -> Figure 3b
    plot_categorical_map_and_histogram_per_column(
        gdf_points=gdf_points,
        gdf=gdfload,
        column='AquiferMediaType',
        save_path=fig_dir / "AquiferMediaType",
    )
    # -> Figure 3a
    plot_categorical_map_and_histogram_per_column(
        gdf_points=gdf_points,
        gdf=gdfload,
        column='AquiferType',
        save_path=fig_dir / "AquiferType",
    )
    # plot_categorical_map_and_histogram_per_column(
    #     gdf_points=gdf_points,
    #     gdf=gdfload,
    #     column='Pumping',
    #     save_path=fig_dir / "pumping",
    # )
    # plot_monthly_station_counts(
    #     ts_df,
    #     time_col='TimeInstant',
    #     id_col='id_mp',
    #     save_path=fig_dir / "obs_per_month.png",
    # )
    # verify_country_prefix_alignment(
    #     gdf_points=gdf_points,
    #     gdf_boundary=gdfload,
    #     save_path=fig_dir / "country_prefix_check.png",
    # )

    # Multi-point locations map with CDF inset -> Figure A1
    plot_multipoint_map_cdf(
        gdf_points=gdf_points,
        gdf_boundary=gdfload,
        cutoff_m=12.0,
        save_path=fig_dir / "multipoint_map_cdf.png",
    )
