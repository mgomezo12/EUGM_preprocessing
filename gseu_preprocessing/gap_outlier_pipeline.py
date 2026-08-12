"""
Gap analysis and outlier detection pipeline for EUGM time series data.

This module provides the main pipeline for gap analysis and outlier detection,
integrating with the preprocessing workflow.
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path
from typing import Optional

from .io import Paths
from .gap_analysis import (
    detect_stagnant_periods_with_nan,
    calculate_max_gap_length,
    truncate_time_series,
    truncate_time_series_after_gap,
    truncate_trailing_nan,
    calculate_missing_values_percentage,
    compute_last_valid_entry,
    compute_ts_duration,
    ensure_monthly_continuity,
    consolidate_timeseries_by_mp
)
from .outlier_detection import (
    detect_outliers_moving_average,
    detect_outliers_rate_of_change,
    mark_outliers_in_ts,
    plot_outliers_summary,
    plot_outliers_by_type
)
from .utils_geo import (
    plot_num_ts_map,
    plot_num_ts_map_range,
    plot_depth_map_continuous,
    plot_categorical_map_and_histogram_per_column,
)
from .utils_plot import plot_num_years_histogram, plot_stations_by_last_valid_date


def load_exclusion_ids(path: Path) -> set:
    """
    Load the manually-curated id_mp exclusion list (see data/EUGM_processed/
    Exclusion_ids/readme_id_mp_not_in_analysis.md). A missing file just means no
    exclusions are configured - not an error, since this list is optional.
    """
    if not path.exists():
        return set()
    return set(pd.read_csv(path, sep=';')['id_mp'].astype(str))


def export_ppf1(paths: Paths, max_gap_threshold: float = 18) -> None:
    """
    Export EUGM_MP_ppf1 / EUGM_TS_ppf1 from the already-computed outlier-flagged
    intermediates (gdf_mp_*_outliersv3.gpkg, ts_*_outliersv3.parquet), applying the
    current id_mp_not_in_analysis exclusion list.

    Outlier detection (rate-of-change + moving-average over the full time series)
    is the expensive part of this pipeline and is invariant to the exclusion list -
    it doesn't need to be redone just because a station was added to or removed
    from the exclusion list. This function only re-reads the already-flagged data
    and re-applies the current exclusion list, so it takes seconds rather than the
    ~1h+ that full outlier detection takes. Called both at the end of a full
    `run_outlier_detection_pipeline` run and standalone (--stage export) whenever
    only the exclusion list changed.
    """
    print("Exporting EUGM_MP_ppf1 / EUGM_TS_ppf1 from existing outlier-flagged data...")

    gdf_points2 = gpd.read_file(paths.data_processed / f'gdf_mp_{max_gap_threshold}g6y_outliersv3.gpkg')
    ts_f1 = pd.read_parquet(paths.data_processed / f'ts_{max_gap_threshold}g6y_outliersv3.parquet', engine='pyarrow')

    excluded_ids = load_exclusion_ids(paths.id_mp_not_in_analysis)
    if excluded_ids:
        print(f"Excluding {len(excluded_ids)} station(s) listed in {paths.id_mp_not_in_analysis.name}")

    mp_ppf1 = gdf_points2.drop(columns=['id_ts_count', 'nonnan_count'], errors='ignore').copy()
    mp_ppf1 = mp_ppf1[~mp_ppf1['id_mp'].astype(str).isin(excluded_ids)]
    mp_ppf1.to_csv(paths.data_processed / f'EUGM_MP_ppf1_{paths.snapshot}.csv', index=False)

    ts_ppf1 = ts_f1[['id_ts', 'id_mp', 'TimeInstant', 'Value', 'Value_cl']].copy()
    ts_ppf1 = ts_ppf1.rename(columns={'Value': 'Value_org', 'Value_cl': 'Value'})
    ts_ppf1 = ts_ppf1[~ts_ppf1['id_mp'].astype(str).isin(excluded_ids)]

    # Safety net: never hand imputation a series that ends in NaN (e.g. a
    # stagnant-period run that reaches all the way to the series' end) - it
    # would have no real anchor to extrapolate towards. See
    # `truncate_trailing_nan`'s docstring for the real cases this caught.
    n_before = len(ts_ppf1)
    ts_ppf1 = truncate_trailing_nan(ts_ppf1, id_col='id_ts', date_col='TimeInstant', column='Value_org')
    n_trimmed = n_before - len(ts_ppf1)
    if n_trimmed:
        print(f"Trimmed {n_trimmed} trailing-NaN row(s) before export (open-ended stagnant/missing tails)")

    ts_ppf1.to_parquet(paths.data_processed / f'EUGM_TS_ppf1_{paths.snapshot}.parquet', engine='pyarrow', compression='snappy')
    ts_ppf1.to_csv(paths.data_processed / f'EUGM_TS_ppf1_{paths.snapshot}.csv', index=False)

    print(f"Saved EUGM_MP_ppf1 ({len(mp_ppf1)} stations) and EUGM_TS_ppf1 ({len(ts_ppf1)} rows)")


def run_gap_analysis_pipeline(paths: Paths,
                            max_gap_threshold: float = 18,
                            min_obs_threshold: int = 60,  # 5 years * 12 months
                            roc_threshold: int = 10,
                            ma_window: int = 12,
                            ma_sigma: float = 3.1) -> None:
    """
    Run the complete gap analysis and outlier detection pipeline.

    Parameters:
    -----------
    paths : Paths
        Configuration paths
    max_gap_threshold : float
        Maximum gap length threshold in months
    min_obs_threshold : int
        Minimum number of observations required
    roc_threshold : int
        Rate of change threshold for outlier detection
    ma_window : int
        Moving average window size
    ma_sigma : float
        Moving average sigma threshold
    """
    print("Starting gap analysis and outlier detection pipeline...")
    
    # Load processed data from previous pipeline
    monitoring_points_path = paths.data_processed / 'MonitoringPointsv3.csv'
    timeseries_path = paths.data_processed / 'TS_mresamplev3.parquet'
    shapefile_path = paths.shp_gseu / 'Europe_map_GSEU_Partners.shp'
    
    # Load data
    print("Loading processed data...")
    dfts_meta_mp = pd.read_csv(monitoring_points_path, index_col=False)
    dfts_meta_mp = dfts_meta_mp.drop(columns=['id_ts'], errors='ignore').copy()
    resampled_df_monthly = pd.read_parquet(timeseries_path, engine='pyarrow')
    gdfload = gpd.read_file(shapefile_path)
    
    print(f"Loaded {len(resampled_df_monthly)} time series records")
    print(f"Loaded {len(dfts_meta_mp)} monitoring points")
    
    # Step 1: Consolidate time series by monitoring point
    print("Consolidating time series by monitoring point...")
    dfts_joint = consolidate_timeseries_by_mp(resampled_df_monthly)
    
    # Step 2: Ensure monthly continuity
    print("Ensuring monthly continuity...")
    dfts_joints = ensure_monthly_continuity(
        dfts_joint,
        group_cols=("id_mp", "id_ts"),
        date_col="TimeInstant",
        value_col="Value"
    )
    
    # Step 3: Update metadata with time series information
    df_idts = dfts_joints[['id_mp', 'id_ts']].drop_duplicates()
    dfts_meta_mp = dfts_meta_mp.merge(df_idts, on='id_mp', how='inner')
    dfts_meta_mp = dfts_meta_mp.drop_duplicates(subset='id_mp', keep='first')
    dfts_meta_mp = dfts_meta_mp.loc[:, ~dfts_meta_mp.columns.duplicated()]
    
    # Step 4: Detect stagnant periods
    print("Detecting stagnant periods...")
    df_ts = detect_stagnant_periods_with_nan(
        dfts_joints, 
        column='Value', 
        threshold=0.0001, 
        min_duration=366
    )
    
    # Step 5: Calculate gap lengths
    print("Calculating gap lengths...")
    gap_lengths = calculate_max_gap_length(df_ts)
    gap_lengths_filtered = gap_lengths[gap_lengths.max_gap_length > 17]
    
    # Step 6: Truncate time series based on gaps
    print("Truncating time series based on gaps...")
    truncated_df = truncate_time_series(df_ts, gap_lengths, cutoff_date='2010-01-01', max_gap_threshold=18)
    gap_lengths2 = calculate_max_gap_length(truncated_df)
    truncated_df2 = truncate_time_series(truncated_df, gap_lengths2, cutoff_date='2010-01-01', max_gap_threshold=18)
    truncated_df3 = truncate_time_series_after_gap(truncated_df2, gap_lengths2, cutoff_date='2020-01-01', max_gap_threshold=18)
    
    # Step 7: Filter based on missing values and gap lengths
    print("Filtering based on missing values and gap lengths...")
    missing_values = calculate_missing_values_percentage(truncated_df3)
    st_missing_values = missing_values[missing_values['missing_percentage'] > 25]['id_ts']
    st_gap_lengths = gap_lengths2[gap_lengths2['max_gap_length'] >= 3.4]['id_ts']
    filtered_stations = set(st_missing_values).intersection(st_gap_lengths)
    
    gapf1_month_df = truncated_df2[~truncated_df2['id_ts'].isin(filtered_stations)]
    gapf1_month_df['id_mp'] = gapf1_month_df['id_ts'].str.split('_').str[:-1].str.join('_')
    non_nan_counts = gapf1_month_df.groupby('id_ts')['Value'].apply(lambda x: x.notna().sum()).reset_index(name='nonnan_count2')
    gapf1_month_df = gapf1_month_df[gapf1_month_df['id_ts'].isin(non_nan_counts[non_nan_counts['nonnan_count2'] > min_obs_threshold]['id_ts'])]
    
    # Step 8: Filter by last valid entry date
    print("Filtering by last valid entry date...")
    last_valid_dates = compute_last_valid_entry(gapf1_month_df)
    # Diagnostic plot, not used in the manuscript.
    # plot_stations_by_last_valid_date(last_valid_dates, save_path=paths.figures / 'stations_by_last_valid_date.png')
    stations_after_2015 = last_valid_dates[last_valid_dates['last_valid_date'] >= '2000-01-01']
    gapf1_month_df = gapf1_month_df[gapf1_month_df.id_ts.isin(stations_after_2015.id_ts)]
    
    # Step 9: Final gap filtering
    print("Applying final gap filtering...")
    max_gap_len = gap_lengths[gap_lengths['max_gap_length'] <= max_gap_threshold + 0.1]['id_ts']
    gapf2_month_df = gapf1_month_df[gapf1_month_df['id_ts'].isin(max_gap_len)]
    
    # Step 10: Update metadata with final time series
    dfts_meta_mp = dfts_meta_mp.merge(non_nan_counts, on='id_ts', how='left')
    dfts_meta_mp['nonnan_count2'] = dfts_meta_mp['nonnan_count2'].fillna(0).astype(int)
    dftsf2_meta_mp = dfts_meta_mp[dfts_meta_mp.id_ts.isin(gapf2_month_df.id_ts.unique())]
    gdf_points2 = gpd.GeoDataFrame(dftsf2_meta_mp, geometry=gpd.points_from_xy(dftsf2_meta_mp.xutm, dftsf2_meta_mp.yutm), crs="EPSG:3035")
    print(gdf_points2.columns)
    # Step 11: observation count map - intermediate-stage diagnostic, not
    # used in the manuscript (the final Fig 2/3 maps come from the
    # ppf1-filtered dataset via after_filter_pipeline.py, not this stage).
    # print("Creating observation count map...")
    # plot_num_ts_map(gdf_points2, gdfload, gapf2_month_df, columnnan='nonnan_count2',
    #                save_path=paths.figures / f'gdf_mp_{max_gap_threshold}g6y_modv3.png')

    # Step 12: Save intermediate results
    print("Saving intermediate results...")
    #gdf_points2.to_file(paths.data_processed / f'gdf_mp_{max_gap_threshold}g6y_modv3.shp')
    gdf_points2.to_file(paths.data_processed / f'gdf_mp_{max_gap_threshold}g6y_modv3.gpkg', driver="GPKG", engine="fiona")
    gdf_points2.drop(columns=['Unnamed: 0', 'nonnan_count2', 'capped_values', 'geometry'], errors='ignore').to_csv(
        paths.data_processed / f"gdf_mp_{max_gap_threshold}g6y_modv3.csv", index=False)
    gapf2_month_df.to_parquet(paths.data_processed / f'ts_mp_{max_gap_threshold}g6y_modv3.parquet', engine='pyarrow', compression='snappy')
    
    # Step 13: Time series duration analysis - intermediate-stage duplicate
    # of Figure 2b (the manuscript figure comes from the ppf1-filtered
    # dataset via after_filter_pipeline.py, not this stage).
    print("Analyzing time series duration...")
    gapf2_month_df['TimeInstant'] = pd.to_datetime(gapf2_month_df['TimeInstant'], format="%Y/%m/%d", errors='coerce')
    ts_duration_df = compute_ts_duration(gapf2_month_df)
    # plot_num_ts_map_range(gdf_points2, gdfload, ts_duration_df,
    #                      save_path=paths.figures / f'years_mp_{max_gap_threshold}g6yv3.png')
    # plot_num_years_histogram(ts_duration_df,
    #                        save_path=paths.figures / f'hist_years_mp_{max_gap_threshold}g6yv3.png')

    # Step 14: NRT map - diagnostic, not used in the manuscript.
    # print("Creating NRT map...")
    # gdf_points2['NRT'] = gdf_points2['NRT'].astype(bool)
    # from .utils_geo import plot_map_booleancolumns
    # plot_map_booleancolumns(gdf_points2, gdfload, column='NRT',
    #                        save_path=paths.figures / f'gdf_aftergap_NRTv3')

    print("Gap analysis pipeline completed successfully!")


def run_outlier_detection_pipeline(paths: Paths,
                                 roc_threshold: int = 10,
                                 ma_window: int = 12,
                                 ma_sigma: float = 3.1,
                                 max_gap_threshold: float = 18) -> None:
    """
    Run the outlier detection pipeline on gap-filtered data.

    Parameters:
    -----------
    paths : Paths
        Configuration paths
    roc_threshold : int
        Rate of change threshold for outlier detection
    ma_window : int
        Moving average window size
    ma_sigma : float
        Moving average sigma threshold
    max_gap_threshold : float
        Maximum gap threshold used in gap analysis
    """
    print("Starting outlier detection pipeline...")
    
    # Load gap-filtered data
    ts_f1_path = paths.data_processed / f'ts_mp_{max_gap_threshold}g6y_modv3.parquet'
    csv_points_path = paths.data_processed / f'gdf_mp_{max_gap_threshold}g6y_modv3.csv'
    shapefile_path = paths.shp_gseu / 'Europe_map_GSEU_Partners.shp'
    
    print("Loading gap-filtered data...")
    ts_f1 = pd.read_parquet(ts_f1_path, engine='pyarrow')
    csv_points = pd.read_csv(csv_points_path)
    gdf_points = gpd.GeoDataFrame(csv_points, geometry=gpd.points_from_xy(csv_points.xutm, csv_points.yutm), crs="EPSG:3035")
    gdfload = gpd.read_file(shapefile_path)
    
    print(f"Loaded {len(ts_f1)} time series records")
    print(f"Loaded {len(gdf_points)} monitoring points")
    
    # Filter monitoring points to match time series
    gdf_points2 = gdf_points[gdf_points['id_ts'].isin(list(ts_f1['id_ts'].unique()))].copy()
    
    # Step 1: Detect outliers using both methods
    print("Detecting outliers using rate of change method...")
    outliers_roc = detect_outliers_rate_of_change(ts_f1, column='Value', threshold=roc_threshold)
    
    print("Detecting outliers using moving average method...")
    outliers_ma = detect_outliers_moving_average(ts_f1, column='Value', window=ma_window, sig=ma_sigma)
    
    # Step 2: Calculate outlier counts per station
    print("Calculating outlier counts per station...")
    outliers_ma_count = outliers_ma.groupby('id_ts')['ma_outlier'].sum().to_frame(name='outliers_ma_count').reset_index()
    outliers_roc_count = outliers_roc.groupby('id_ts')['roc_outlier'].sum().to_frame(name='outliers_roc_count').reset_index()
    
    # Step 3: Update monitoring points with outlier counts
    gdf_points2 = gdf_points2.merge(outliers_ma_count, on='id_ts', how='left').fillna({'outliers_ma_count': 0})
    gdf_points2 = gdf_points2.merge(outliers_roc_count, on='id_ts', how='left').fillna({'outliers_roc_count': 0})
    gdf_points2['outliers_ma_count'] = gdf_points2['outliers_ma_count'].astype(int)
    gdf_points2['outliers_roc_count'] = gdf_points2['outliers_roc_count'].astype(int)
    
    # Step 4: Save monitoring points with outlier information
    print("Saving monitoring points with outlier information...")
    gdf_points2.to_file(paths.data_processed / f'gdf_mp_{max_gap_threshold}g6y_outliersv3.gpkg', driver="GPKG", engine="fiona")

    # Step 5: Mark outliers in time series and create cleaned version
    print("Marking outliers in time series...")
    ts_f1 = mark_outliers_in_ts(ts_f1, outliers_ma, outliers_roc)

    # Create a cleaned value column: set to NaN where either outlier is True
    ts_f1['Value_cl'] = ts_f1['Value']
    ts_f1.loc[ts_f1['ma_outlier'] | ts_f1['roc_outlier'], 'Value_cl'] = np.nan

    # Step 6: Save time series with outlier information
    print("Saving time series with outlier information...")
    ts_f1.to_parquet(paths.data_processed / f'ts_{max_gap_threshold}g6y_outliersv3.parquet', engine='pyarrow', compression='snappy')

    # Step 7: Export EUGM_MP_ppf1 / EUGM_TS_ppf1 (applies the exclusion list; see
    # export_ppf1's docstring for why this reads back from disk instead of using
    # the dataframes above directly - it's the same code path used by the fast
    # --stage export re-run).
    export_ppf1(paths, max_gap_threshold=max_gap_threshold)

    # Step 8: outlier visualizations - diagnostic outputs at this
    # intermediate stage, not used in the manuscript (Fig 3's aquifer/depth
    # maps and their reported stats come from the final ppf1-filtered
    # dataset via after_filter_pipeline.py, not here). Kept, commented out,
    # for QA reference.
    # print("Creating outlier visualizations...")
    # outliers_fig_path = paths.figures / 'outliers'
    # outliers_fig_path.mkdir(exist_ok=True)
    # plot_outliers_summary(outliers_ma=outliers_ma, outliers_roc=outliers_roc,
    #                      dfts=ts_f1, path_fig=paths.figures, w=ma_window)
    # plot_outliers_by_type(gdf_points2, outliers_ma, outliers_roc, gdfload,
    #                      save_path=paths.figures / 'map_outlier.png')
    # plot_depth_map_continuous(gdf_points2, gdfload, depth_column='depth', save_path=paths.figures / 'map_depth.png')
    # try:
    #     plot_categorical_map_and_histogram_per_column(
    #         gdf_points2, gdfload, column='AquiferMediaType',
    #         save_path=paths.figures / 'aquifer_filtered'
    #     )
    #     plot_categorical_map_and_histogram_per_column(
    #         gdf_points2, gdfload, column='AquiferType',
    #         save_path=paths.figures / 'aquifer_filtered'
    #     )
    # except Exception as e:
    #     print(f"Warning: Could not generate aquifer maps for filtered stations: {e}")

    print("Outlier detection pipeline completed successfully!")
    print(f"Outlier detection results saved to: {paths.data_processed}")
    print(f"Outlier visualizations saved to: {paths.figures}")

def run_outlier_plots_only(paths: Paths,
                           ma_window: int = 12,
                           max_gap_threshold: float = 18) -> None:
    """
    Regenerate outlier visualizations without recomputing outliers.

    Expects previously saved time series with 'ma_outlier' and 'roc_outlier' flags
    and corresponding monitoring points CSV from the gap analysis stage.

    Parameters:
    -----------
    paths : Paths
        Configuration paths
    ma_window : int
        Window size used for labeling in summary filenames
    max_gap_threshold : float
        Maximum gap threshold used when naming saved artifacts
    """
    print("Starting plots-only stage (no outlier recomputation)...")

    # Resolve inputs saved by prior stages
    ts_path = paths.data_processed / f'ts_{max_gap_threshold}g6y_outliersv3.parquet'
    csv_points_path = paths.data_processed / f'gdf_mp_{max_gap_threshold}g6y_modv3.csv'
    shapefile_path = paths.shp_gseu / 'Europe_map_GSEU_Partners.shp'

    # Load saved datasets
    ts = pd.read_parquet(ts_path, engine='pyarrow')
    csv_points = pd.read_csv(csv_points_path)
    gdf_points = gpd.GeoDataFrame(csv_points, geometry=gpd.points_from_xy(csv_points.xutm, csv_points.yutm), crs="EPSG:3035")
    gdf_boundary = gpd.read_file(shapefile_path)

    # Align points to available time series
    gdf_points2 = gdf_points[gdf_points['id_ts'].isin(ts['id_ts'].unique())].copy()

    # Build minimal outlier DataFrames from flags
    outliers_ma = ts[['id_ts', 'TimeInstant', 'ma_outlier']].copy()
    outliers_roc = ts[['id_ts', 'TimeInstant', 'roc_outlier']].copy()

    # Generate plots
    plot_outliers_summary(outliers_ma=outliers_ma, outliers_roc=outliers_roc,
                         dfts=ts, path_fig=paths.figures, w=ma_window)

    plot_outliers_by_type(gdf_points2, outliers_ma, outliers_roc, gdf_boundary,
                         save_path=paths.figures / 'map_outlier.png')

    # Depth map on filtered monthly resampled stations
    from .utils_geo import plot_depth_map_continuous
    plot_depth_map_continuous(gdf_points2, gdf_boundary, depth_column='depth', save_path=paths.figures / 'map_depth.png')

    # Additional maps: Aquifer Media Type and Aquifer Type for filtered stations (plots-only stage)
    try:
        plot_categorical_map_and_histogram_per_column(
            gdf_points2, gdf_boundary, column='AquiferMediaType',
            save_path=paths.figures / 'aquifer_filtered'
        )
        plot_categorical_map_and_histogram_per_column(
            gdf_points2, gdf_boundary, column='AquiferType',
            save_path=paths.figures / 'aquifer_filtered'
        )
    except Exception as e:
        print(f"Warning: Could not generate aquifer maps (plots-only stage): {e}")

    print("Plots-only stage completed successfully!")
    print(f"Outlier visualizations saved to: {paths.figures}")


def run_complete_gap_outlier_pipeline(paths: Paths, **kwargs) -> None:
    """
    Run the complete gap analysis and outlier detection pipeline.

    Parameters:
    -----------
    paths : Paths
        Configuration paths
    **kwargs
        Additional parameters for gap analysis and outlier detection
    """
    print("=" * 60)
    print("RUNNING COMPLETE GAP ANALYSIS AND OUTLIER DETECTION PIPELINE")
    print("=" * 60)
    
    # Run gap analysis pipeline
    run_gap_analysis_pipeline(paths, **kwargs)
    
    # 2) Outlier stage gets only what it understands
    allowed = {k: kwargs[k] for k in ("roc_threshold", "ma_window", "ma_sigma", "max_gap_threshold") if k in kwargs}
    run_outlier_detection_pipeline(paths, **allowed)

    print("=" * 60)
    print("COMPLETE PIPELINE FINISHED SUCCESSFULLY")
    print("=" * 60)
