"""
Outlier detection module for EUGM time series data.

This module provides functions for detecting outliers in time series data
using various methods including moving average and rate of change analysis.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Tuple
from matplotlib.colors import ListedColormap


def detect_outliers_moving_average(df: pd.DataFrame, column: str = 'Value', 
                                  window: int = 12, sig: float = 5) -> pd.DataFrame:
    """
    Detect outliers based on a moving average and moving standard deviation.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with time series data, including 'id_ts' and 'TimeInstant'
    column : str
        Name of the column to analyze (default is 'Value')
    window : int
        Rolling window size for moving average and standard deviation
    sig : float
        Number of standard deviations to set the bounds

    Returns:
    --------
    pd.DataFrame
        DataFrame with outlier information including 'ma_outlier' boolean column
    """
    outliers_ma = pd.DataFrame()  # To store results for all stations

    for ts_id, group in df.groupby('id_ts'):
        group = group.set_index('TimeInstant').sort_index()  # Sort by time for correct rolling calculations

        # Calculate moving average and moving standard deviation
        group['moving_avg'] = group[column].rolling(window=window, min_periods=1, center=True).mean()
        group['moving_std'] = group[column].rolling(window=window, min_periods=1, center=True).std()

        # Define upper and lower bounds for outlier detection
        group['upper_bound'] = group['moving_avg'] + sig * group['moving_std']
        group['lower_bound'] = group['moving_avg'] - sig * group['moving_std']

        # Identify outliers
        group['ma_outlier'] = ((group[column] > group['upper_bound']) | 
                               (group[column] < group['lower_bound']))

        # Append results for this station to the main DataFrame
        outliers_ma = pd.concat([outliers_ma, group[['id_ts', column, 'moving_avg', 'upper_bound', 'lower_bound', 'ma_outlier']]])

    return outliers_ma.reset_index()


def detect_outliers_rate_of_change(df: pd.DataFrame, column: str = 'Value', 
                                  threshold: float = 3) -> pd.DataFrame:
    """
    Detect outliers based on the rate of change with stability conditions.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'id_ts' and the specified value column
    column : str
        Name of the column on which to base the rate of change analysis
    threshold : float
        Multiplier of the mean change rate to classify outliers
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with outlier information including 'roc_outlier' boolean column
    """
    outliers_roc = pd.DataFrame()

    for id_ts, group in df.groupby('id_ts'):
        group = group.sort_values(by='TimeInstant').copy()
        group['TimeInstant'] = pd.to_datetime(group['TimeInstant'])

        # Calculate the mean absolute change rate
        mean_change = group[column].diff().abs().mean()

        # Calculate forward and backward change rates for ±1 step
        forward_change_1 = group[column].diff().abs()
        backward_change_1 = group[column].diff(-1).abs()

        # Identify potential outliers based on rate of change threshold
        potential_outliers = (
            (forward_change_1 > threshold * mean_change) & 
            (backward_change_1 > threshold * mean_change)
        )

        # Stability conditions:
        # Check that t-1 and t+1 values are stable with respect to each other
        stable_neighbors_1 = (
            (group[column].shift(1) - group[column].shift(-1)).abs() < mean_change
        )
        
        # Check that t-2 and t+2 values are stable with respect to each other
        stable_neighbors_2 = (
            (group[column].shift(2) - group[column].shift(-2)).abs() < mean_change
        )

        # Mark outliers that meet both potential outlier and stability conditions for both pairs
        group['roc_outlier'] = potential_outliers & stable_neighbors_1 & stable_neighbors_2
        
        # Combine max change from ±1 steps for reference
        group['change_rate'] = forward_change_1.combine(backward_change_1, max)

        # Collect the outlier information
        outliers_roc = pd.concat([outliers_roc, group[['id_ts', 'TimeInstant', column, 'change_rate', 'roc_outlier']]])

    # Reset index for the output DataFrame
    return outliers_roc.reset_index(drop=True)


def mark_outliers_in_ts(ts_df: pd.DataFrame, outliers_ma: pd.DataFrame, 
                       outliers_roc: pd.DataFrame) -> pd.DataFrame:
    """
    Add boolean columns to ts_df indicating if each observation is an outlier.

    Parameters:
    -----------
    ts_df : pd.DataFrame
        DataFrame with columns ['id_ts', 'TimeInstant', 'Value']
    outliers_ma : pd.DataFrame
        DataFrame with columns ['id_ts', 'TimeInstant', 'ma_outlier']
    outliers_roc : pd.DataFrame
        DataFrame with columns ['id_ts', 'TimeInstant', 'roc_outlier']

    Returns:
    --------
    pd.DataFrame
        Updated DataFrame with 'ma_outlier' and 'roc_outlier' columns added
    """
    # Ensure 'TimeInstant' is in datetime format for consistency
    ts_df['TimeInstant'] = pd.to_datetime(ts_df['TimeInstant'])
    outliers_ma['TimeInstant'] = pd.to_datetime(outliers_ma['TimeInstant'])
    outliers_roc['TimeInstant'] = pd.to_datetime(outliers_roc['TimeInstant'])

    # Add 'ma_outlier' column
    ts_df = ts_df.merge(
        outliers_ma[['id_ts', 'TimeInstant', 'ma_outlier']],
        on=['id_ts', 'TimeInstant'],
        how='left'
    )
    ts_df['ma_outlier'] = ts_df['ma_outlier'].fillna(False)  # Default to False for non-outliers

    # Add 'roc_outlier' column
    ts_df = ts_df.merge(
        outliers_roc[['id_ts', 'TimeInstant', 'roc_outlier']],
        on=['id_ts', 'TimeInstant'],
        how='left'
    )
    ts_df['roc_outlier'] = ts_df['roc_outlier'].fillna(False)  # Default to False for non-outliers

    return ts_df


def plot_outliers_summary(outliers_ma: Optional[pd.DataFrame] = None, 
                         outliers_roc: Optional[pd.DataFrame] = None, 
                         dfts: Optional[pd.DataFrame] = None, 
                         path_fig: Optional[Path] = None, 
                         w: Optional[int] = None) -> None:
    """
    Plot a summary of the number of outliers and unique stations detected.

    Parameters:
    -----------
    outliers_ma : pd.DataFrame, optional
        DataFrame with Moving Average outliers
    outliers_roc : pd.DataFrame, optional
        DataFrame with Rate of Change outliers
    dfts : pd.DataFrame, optional
        Original time series DataFrame
    path_fig : Path, optional
        Path to save the plot
    w : int, optional
        Window size parameter for filename
    """
    methods = []
    counts = []
    station_counts = []
    bar_colors = []

    # Moving Average Outliers
    if outliers_ma is not None:
        n_outliers_ma = outliers_ma['ma_outlier'].sum()
        stations_outliers_ma = outliers_ma[outliers_ma['ma_outlier']]['id_ts'].nunique()
        methods.append('Moving Avg Outliers')
        counts.append(n_outliers_ma)
        station_counts.append(stations_outliers_ma)
        bar_colors.append('#2ca02c')

    # ROC Outliers
    if outliers_roc is not None:
        n_outliers_roc = outliers_roc['roc_outlier'].sum()
        stations_outliers_roc = outliers_roc[outliers_roc['roc_outlier']]['id_ts'].nunique()
        methods.append('ROC Outliers')
        counts.append(n_outliers_roc)
        station_counts.append(stations_outliers_roc)
        bar_colors.append('#ff7f0e')  # Orange for ROC outliers

    # Overlap of Moving Average and ROC Outliers
    if outliers_roc is not None and outliers_ma is not None:
        stations_roc_ma_overlap = pd.merge(outliers_roc[outliers_roc['roc_outlier']],
                                           outliers_ma[outliers_ma['ma_outlier']],
                                           on=['TimeInstant', 'id_ts'])['id_ts'].nunique()
        methods.append('ROC & Moving Avg Overlap')
        counts.append(stations_roc_ma_overlap)
        station_counts.append(stations_roc_ma_overlap)
        bar_colors.append('#ff9896')  # Light orange for overlap

    # Plot if there's data
    if methods:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.barh(methods, counts, color=bar_colors)

        # Add count labels to each bar
        for i, (count, station_count) in enumerate(zip(counts, station_counts)):
            ax.text(count + 0.5, i, f'{count} outliers\n{station_count} stations', va='center', fontsize=10)

        ax.set_xlabel('Number of Outliers and Stations')
        ax.set_title('Outliers and Stations Identified by Each Method and Overlaps')
        plt.grid(axis='x', color='grey', linestyle='--', linewidth=0.5, alpha=0.4)
        plt.tight_layout()
        
        # Save plot if path provided
        if path_fig:
            plt.savefig(f"{path_fig}/TS_outliers_{dfts.id_ts.nunique()}_{w}_sample.png", dpi=150)
        plt.close()
    else:
        print("No outlier data provided to plot.")


def plot_outliers_by_type(gdf_points: pd.DataFrame, outliers_ma: pd.DataFrame, 
                         outliers_roc: pd.DataFrame, gdf_boundary: pd.DataFrame, 
                         save_path: Optional[Path] = None) -> None:
    """
    Plot a map showing outlier presence per monitoring point.

    Parameters:
    -----------
    gdf_points : pd.DataFrame
        GeoDataFrame containing monitoring points with 'id_ts' and 'id_mp'
    outliers_ma : pd.DataFrame
        DataFrame with Moving Average outliers
    outliers_roc : pd.DataFrame
        DataFrame with Rate of Change outliers
    gdf_boundary : pd.DataFrame
        GeoDataFrame representing the European boundary
    save_path : Path, optional
        Path to save the plot
    """
    # Calculate outlier counts by method per station
    ma_outlier_counts = outliers_ma[outliers_ma['ma_outlier']].groupby('id_ts').size()
    roc_outlier_counts = outliers_roc[outliers_roc['roc_outlier']].groupby('id_ts').size()
    
    # Merge with `gdf_points`, filling missing values with zero (no outliers)
    gdf_points = gdf_points.copy()  # Avoid SettingWithCopyWarning
    gdf_points['ma_outliers'] = gdf_points['id_ts'].map(ma_outlier_counts).fillna(0)
    gdf_points['roc_outliers'] = gdf_points['id_ts'].map(roc_outlier_counts).fillna(0)
    
    # Define bins and labels for highlighting only points with outliers
    ma_bins = [0, 1, 2, 3, 4, 5, np.inf]
    roc_bins = [0, 1, 2, 3, 4, 5, np.inf]
    ma_labels = ['No MA Outliers', '1 MA Outlier', '2 MA Outliers', '3 MA Outliers', '4 MA Outliers', '>5 MA Outliers']
    roc_labels = ['No ROC Outliers', '1 ROC Outlier', '2 ROC Outliers', '3 ROC Outliers', '4 ROC Outliers', '>5 ROC Outliers']

    # Apply binning to categorize only points with outliers
    gdf_points['ma_outlier_category'] = pd.cut(gdf_points['ma_outliers'], bins=ma_bins, labels=ma_labels, right=False, include_lowest=True)
    gdf_points['roc_outlier_category'] = pd.cut(gdf_points['roc_outliers'], bins=roc_bins, labels=roc_labels, right=False, include_lowest=True)

    # Separate points with and without outliers
    gdf_no_outliers = gdf_points[(gdf_points['ma_outliers'] == 0) & (gdf_points['roc_outliers'] == 0)]
    gdf_with_outliers = gdf_points[(gdf_points['ma_outliers'] > 0) | (gdf_points['roc_outliers'] > 0)]

    # Define improved color maps with higher contrast
    ma_cmap = ListedColormap(['#9ecae1', '#3182bd', '#08519c', '#08306b', '#001f3f', '#000a1a'])  # Blue gradient for MA
    roc_cmap = ListedColormap(['#a1d99b', '#31a354', '#006d2c', '#00441b', '#003314', '#001f0f'])  # Green gradient for ROC

    # Plot setup
    fig, ax = plt.subplots(figsize=(12, 9))
    gdf_boundary.boundary.plot(ax=ax, color='grey', linewidth=0.5)

    # Plot monitoring points without outliers in gray, alpha=0.5, without a legend label
    gdf_no_outliers.plot(
        ax=ax,
        marker='o',
        color='grey',
        markersize=2,
        alpha=0.3
    )

    # Plot MA outliers with different colors and larger marker size
    gdf_with_outliers[gdf_with_outliers['ma_outliers'] > 0].plot(
        ax=ax,
        marker='o',
        column='ma_outlier_category',
        markersize=8,
        cmap=ma_cmap,
        legend=False,
        alpha=0.8
    )

    # Plot ROC outliers with different colors and larger marker size
    gdf_with_outliers[gdf_with_outliers['roc_outliers'] > 0].plot(
        ax=ax,
        marker='o',
        column='roc_outlier_category',
        markersize=8,
        cmap=roc_cmap,
        legend=False,
        alpha=0.6
    )

    # Set map boundaries for Europe in EPSG:3035
    ax.set_xlim([2500000, 7500000])
    ax.set_ylim([1300000, 5500000])
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.spines[['top', 'right', 'bottom', 'left']].set_visible(False)

    # Create legends, positioned inside the plot near the right (close to Russia)
    ma_legend_elements = [
        plt.Line2D(
            [0], [0], marker='o', color='w',
            markerfacecolor=ma_cmap.colors[i], markersize=12,
            label=ma_labels[i + 1]
        ) for i in range(len(ma_labels) - 1)
    ]
    roc_legend_elements = [
        plt.Line2D(
            [0], [0], marker='o', color='w',
            markerfacecolor=roc_cmap.colors[i], markersize=12,
            label=roc_labels[i + 1]
        ) for i in range(len(roc_labels) - 1)
    ]

    ma_legend = ax.legend(
        handles=ma_legend_elements,
        title="MA Outliers",
        loc="upper right",
        bbox_to_anchor=(0.85, 0.95),
        frameon=True,
        fontsize=11,
        title_fontsize=12,
    )
    roc_legend = ax.legend(
        handles=roc_legend_elements,
        title="ROC Outliers",
        loc="lower right",
        bbox_to_anchor=(0.85, 0.15),
        frameon=True,
        fontsize=11,
        title_fontsize=12,
    )
    ax.add_artist(ma_legend)

    # Save or show the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
