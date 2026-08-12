from pathlib import Path
from os.path import join
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt

from .io import Paths
from .timeseries import parse_and_resample, count_ts_per_mp
from .audit import write_preprocessing_audit
from .utils_plot import plot_elevation_histogram, plot_well_depth_histogram
from .utils_geo import (compute_spatial_density, plot_density_histograms, 
                       plot_categorical_map_and_histogram_per_column, 
                       plot_map_booleancolumns, plot_num_ts_map)


def run_pipeline(paths: Paths) -> None:
    # Load shapefile
    gdfload = gpd.read_file(Path(paths.shp_gseu) / 'Europe_map_GSEU_Partners.shp')
    gdfload['Country'] = gdfload['Country'].replace('Belgium', 'Belgium- Flanders')

    # Load CSVs
    ts_csv = Path(paths.eugm_icgc) / f'EUGM_Timeseries_{paths.snapshot}.csv'
    mp_csv = Path(paths.eugm_icgc) / f'EUGM_MP_{paths.snapshot}.csv'
    dfts_conv_updated = pd.read_csv(ts_csv, sep=';')
    dfts_meta_upd = pd.read_csv(mp_csv, sep=';')

    dfts_conv_updated['id_mp'] = dfts_conv_updated['id_ts'].str.replace(r'_[A-Z]$', '', regex=True)
    write_preprocessing_audit(
        dfts_conv_updated,
        paths.project_root / 'outputs' / 'audit',
        source_file=str(ts_csv),
    )

    print(f"Number of TS: {dfts_conv_updated.id_ts.nunique()}")
    print(f"Number of MP: {dfts_meta_upd.id_mp.nunique()}")

    dfts_mp = dfts_conv_updated[dfts_conv_updated['id_mp'].isin(dfts_meta_upd['id_mp'].unique())]
    dfts_meta_mp = dfts_meta_upd[dfts_meta_upd['id_mp'].isin(dfts_conv_updated['id_mp'].unique())]
    dfts_meta_mp = dfts_meta_mp.drop_duplicates(subset=['id_mp'])
    print(f"Number of TS with linked MP: {dfts_mp['id_ts'].nunique()}")
    print(f"Number of MP linked to one or more TS: {dfts_meta_mp['id_mp'].nunique()}")

    # Counts and cleaning elevation/depth
    dfts_meta_mp = count_ts_per_mp(dfts_conv_updated, dfts_meta_mp)
    dfts_meta_mp = dfts_meta_mp.drop_duplicates(subset=['id_mp'], keep='first')
    dfts_meta_mp['elevation'] = dfts_meta_mp['elevation'].astype(str).str.replace(',', '.').astype(float)
    dfts_meta_mp.loc[dfts_meta_mp['elevation'] <= -7, 'elevation'] = np.nan

    dfts_meta_mp['depth'] = pd.to_numeric(dfts_meta_mp['depth'], errors='coerce')
    dfts_meta_mp.loc[dfts_meta_mp['depth'] <= 0, 'depth'] = np.nan
    dfts_meta_mp = dfts_meta_mp[~dfts_meta_mp['depth'].isin([np.inf, -np.inf])]
    gdf_points = gpd.GeoDataFrame(dfts_meta_mp, geometry=gpd.points_from_xy(dfts_meta_mp.xutm, dfts_meta_mp.yutm), crs='EPSG:3035')

    # This copy is trimmed to reproduce only the manuscript's figures. This
    # stage's plots (elevation/depth histograms, spatial density, raw
    # aquifer-category maps, NRT map, observation-count map) are all
    # pre-filtering diagnostics superseded by the final after-filter maps
    # in after_filter_pipeline.py - kept here, commented out, for reference.
    # plot_elevation_histogram(dfts_meta_mp, Path(paths.figures) / 'elevation_histv3.png')
    # plot_well_depth_histogram(dfts_meta_mp, Path(paths.figures) / 'depthv3.png')
    # density_df, total_density = compute_spatial_density(gdf_points, gdfload)
    # plot_density_histograms(density_df, Path(paths.figures) / 'd')
    # gdf_points['AquiferMediaType'] = gdf_points['AquiferMediaType'].replace('NaN', 'unknown')
    # gdf_points['AquiferType'] = gdf_points['AquiferType'].fillna('unknown')
    # plot_categorical_map_and_histogram_per_column(
    #     gdf_points,
    #     gdfload,
    #     column='AquiferMediaType',
    #     save_path=Path(paths.figures) / 'aquifermediatypev3'
    # )
    # plot_categorical_map_and_histogram_per_column(
    #     gdf_points,
    #     gdfload,
    #     column='AquiferType',
    #     save_path=Path(paths.figures) / 'aquifertypev3'
    # )
    # plot_map_booleancolumns(gdf_points, gdfload, column='NRT', save_path=Path(paths.figures) / 'NRT_updatev3')
    # plot_num_ts_map(gdf_points, gdfload, dfts_conv_updated, columnnan='nonnan_count',
    #                save_path=Path(paths.figures) / 'gdf_original.png')

    # Parse/resample TS and save parquet
    resampled_df_monthly = parse_and_resample(dfts_conv_updated)
    valid_id_mp = set(resampled_df_monthly['id_mp'].dropna().astype(str).unique())
    dfts_meta_mp = dfts_meta_mp[dfts_meta_mp['id_mp'].astype(str).isin(valid_id_mp)].copy()
    dfts_meta_mp.to_csv(Path(paths.data_processed) / 'MonitoringPointsv3.csv', index=False)
    resampled_path = Path(paths.data_processed) / 'TS_mresamplev3.parquet'
    resampled_df_monthly.to_parquet(resampled_path, engine='pyarrow', compression='snappy')

    print("Failed datetime parse (rows):", dfts_conv_updated['TimeInstant'].isna().sum() if 'TimeInstant' in dfts_conv_updated else 'N/A')
