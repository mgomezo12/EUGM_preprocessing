"""
Gap analysis module for EUGM time series data.

This module provides functions for analyzing gaps in time series data,
detecting stagnant periods, and performing data quality assessments.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Tuple


def detect_stagnant_periods_with_nan(df: pd.DataFrame, column: str = 'Value', 
                                    threshold: float = 0.0001, 
                                    min_duration: int = 366) -> pd.DataFrame:
    """
    Detect periods where consecutive values change less than threshold
    for duration exceeding min_duration and set those periods to NaN.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'id_ts', 'TimeInstant', and value column
    column : str
        Name of the column to analyze for stagnant periods
    threshold : float
        Maximum allowed difference between consecutive values
    min_duration : int
        Minimum duration (in days) to consider a stagnant period

    Returns:
    --------
    pd.DataFrame
        Modified DataFrame where stagnant periods are replaced with NaN
    """
    df = df.copy()
    for id_ts, group in df.groupby('id_ts'):
        group = group.sort_values(by='TimeInstant').copy()
        group['delta'] = group[column].diff().abs()

        # Identify stagnant periods
        group['stagnant'] = group['delta'] <= threshold

        # Group consecutive stagnant points
        group['stagnant_group'] = (group['stagnant'] != group['stagnant'].shift()).cumsum()

        # Calculate the duration of each stagnant group
        stagnant_groups = group[group['stagnant']].groupby('stagnant_group').agg(
            start_time=('TimeInstant', 'min'),
            end_time=('TimeInstant', 'max'),
            count=('TimeInstant', 'size')
        )
        stagnant_groups['duration_days'] = (stagnant_groups['end_time'] - stagnant_groups['start_time']).dt.days

        # Mark periods exceeding the minimum duration as stagnant
        valid_stagnant_groups = stagnant_groups[stagnant_groups['duration_days'] >= min_duration].index
        group.loc[group['stagnant_group'].isin(valid_stagnant_groups), column] = np.nan

        # Update the original DataFrame
        df.loc[group.index, column] = group[column]

    return df


def calculate_max_gap_length(df: pd.DataFrame, freq: str = 'MS') -> pd.DataFrame:
    """
    Calculate the maximum gap length in months for each station.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'id_ts', 'TimeInstant', and 'Value' columns
    freq : str
        Frequency string for expected periods (default is 'MS' for monthly)

    Returns:
    --------
    pd.DataFrame
        DataFrame with 'id_ts', 'max_gap_length' in months, 'gap_start', and 'gap_end'
    """
    gap_data = []

    for id_ts, group in df.groupby('id_ts'):
        # Sort by TimeInstant, filter out NaNs, and reset index
        group = group.sort_values(by='TimeInstant').copy()
        group = group[~group['Value'].isna()].reset_index(drop=True)
        group['TimeInstant'] = pd.to_datetime(group['TimeInstant'])

        if len(group) < 2:  # No gap if fewer than 2 valid entries
            gap_data.append({
                'id_ts': id_ts,
                'max_gap_length': 0,
                'gap_start': None,
                'gap_end': None
            })
            continue

        # Calculate time differences between consecutive valid entries
        group['TimeDiff'] = group['TimeInstant'].diff()
        max_gap = group['TimeDiff'].max()

        # Convert the maximum gap to months if it's valid, otherwise set to 0
        max_gap_months = max_gap / pd.Timedelta(days=30) if pd.notna(max_gap) else 0

        # Find start and end of the max gap
        if pd.notna(max_gap):
            max_gap_idx = group['TimeDiff'].idxmax()
            gap_start = group.loc[max_gap_idx - 1, 'TimeInstant'] if max_gap_idx - 1 >= 0 else None
            gap_end = group.loc[max_gap_idx, 'TimeInstant']
        else:
            gap_start = None
            gap_end = None

        gap_data.append({
            'id_ts': id_ts,
            'max_gap_length': max_gap_months,
            'gap_start': gap_start,
            'gap_end': gap_end
        })

    return pd.DataFrame(gap_data)


def truncate_time_series(df: pd.DataFrame, gap_lengths: pd.DataFrame, 
                        cutoff_date: str = '1990-01-01', 
                        max_gap_threshold: float = 6) -> pd.DataFrame:
    """
    Truncate time series if the maximum gap length exceeds threshold and ends before cutoff date.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'id_ts', 'TimeInstant', and 'Value' columns
    gap_lengths : pd.DataFrame
        DataFrame with 'id_ts', 'max_gap_length', 'gap_end' from calculate_max_gap_length
    cutoff_date : str
        The cutoff date as a threshold for truncation
    max_gap_threshold : float
        Maximum gap length in months for truncation condition

    Returns:
    --------
    pd.DataFrame
        Truncated DataFrame
    """
    # Ensure cutoff_date is in datetime format
    cutoff_date = pd.to_datetime(cutoff_date)

    # Filter gap_lengths for series that meet truncation criteria
    truncation_criteria = (gap_lengths['max_gap_length'] > max_gap_threshold) & (gap_lengths['gap_end'] < cutoff_date)
    gaps_to_truncate = gap_lengths.loc[truncation_criteria, ['id_ts', 'gap_end']]

    # Merge gap_end information with the original time series
    df = df.merge(gaps_to_truncate, on='id_ts', how='left')

    # Apply truncation condition: keep records after the gap end date for truncated series
    truncated_df = df[(df['gap_end'].isna()) | (df['TimeInstant'] >= df['gap_end'])]

    # Drop the helper 'gap_end' column and reset index for the result
    truncated_df = truncated_df.drop(columns=['gap_end']).reset_index(drop=True)

    return truncated_df


def truncate_time_series_after_gap(df: pd.DataFrame, gap_lengths: pd.DataFrame, 
                                  cutoff_date: str = '2020-01-01', 
                                  max_gap_threshold: float = 6) -> pd.DataFrame:
    """
    Truncate time series if the maximum gap length exceeds threshold and starts after cutoff date.
    Keeps data before the large gap.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'id_ts', 'TimeInstant', and 'Value' columns
    gap_lengths : pd.DataFrame
        DataFrame with 'id_ts', 'max_gap_length', 'gap_start' from calculate_max_gap_length
    cutoff_date : str
        The cutoff date as a threshold for considering gaps
    max_gap_threshold : float
        Maximum gap length in months for truncation condition

    Returns:
    --------
    pd.DataFrame
        Truncated DataFrame with only data before the large gaps
    """
    # Ensure cutoff_date is in datetime format
    cutoff_date = pd.to_datetime(cutoff_date)

    # Filter gap_lengths for series that meet truncation criteria
    truncation_criteria = (gap_lengths['max_gap_length'] > max_gap_threshold) & (gap_lengths['gap_start'] >= cutoff_date)
    gaps_to_truncate = gap_lengths.loc[truncation_criteria, ['id_ts', 'gap_start']]

    # Merge gap_start information with the original time series
    df = df.merge(gaps_to_truncate, on='id_ts', how='left')

    # Apply truncation condition: keep records before the gap start date for truncated series
    truncated_df = df[(df['gap_start'].isna()) | (df['TimeInstant'] < df['gap_start'])]

    # Drop the helper 'gap_start' column and reset index for the result
    truncated_df = truncated_df.drop(columns=['gap_start']).reset_index(drop=True)

    return truncated_df


def truncate_trailing_nan(df: pd.DataFrame, id_col: str = 'id_ts',
                         date_col: str = 'TimeInstant',
                         column: str = 'Value') -> pd.DataFrame:
    """
    Drop any rows after each station's last non-NaN value.

    `truncate_time_series_after_gap` only trims a trailing gap when it starts
    on/after a fixed cutoff date and exceeds a fixed length threshold - it can
    miss a station whose gap doesn't happen to satisfy both conditions (e.g.
    a stagnant-period run that gets NaN'd starting a few years before the
    cutoff, or one shorter than the length threshold). Left unhandled, such a
    trailing NaN run is exactly what gets handed to imputation next, and the
    model then has no real anchor point to extrapolate towards - it can drift
    arbitrarily far from the last known reading (confirmed on two real EUGM
    stations, BE_1-1111aF2 and FR_BSS001AQHE, where the imputed tail diverged
    by several metres from what the raw sensor was still actually reporting,
    frozen, at the same time). This function is an unconditional safety net:
    regardless of why a station ends in NaN, cut it back to its last real
    observation before imputation ever sees it.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with id_col, date_col, and the value column
    id_col : str
        Column identifying each station's time series
    date_col : str
        Timestamp column
    column : str
        Value column whose trailing NaN run should be trimmed

    Returns:
    --------
    pd.DataFrame
        DataFrame with each station's trailing all-NaN tail removed
    """
    df = df.copy()
    last_valid = (
        df[df[column].notna()]
        .groupby(id_col)[date_col]
        .max()
        .rename('last_valid_date')
    )
    df = df.merge(last_valid, on=id_col, how='left')
    # Stations with no valid observation at all keep every (all-NaN) row -
    # they're handled by the min-observations filter elsewhere, not here.
    keep = df['last_valid_date'].isna() | (df[date_col] <= df['last_valid_date'])
    return df.loc[keep].drop(columns=['last_valid_date']).reset_index(drop=True)


def calculate_missing_values_percentage(df: pd.DataFrame, freq: str = 'MS') -> pd.DataFrame:
    """
    Calculate the percentage of missing values for each time series.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'id_ts', 'TimeInstant', and 'Value' columns
    freq : str
        Frequency string for expected periods

    Returns:
    --------
    pd.DataFrame
        DataFrame with missing value percentages per time series
    """
    missing_data = []
    
    for id_ts, group in df.groupby('id_ts'):
        start_date = group['TimeInstant'].min()
        end_date = group['TimeInstant'].max()
        
        # Calculate total periods based on the frequency
        total_periods = pd.date_range(start=start_date, end=end_date, freq=freq).size
        missing_periods = group['Value'].isna().sum()
        
        missing_percentage = (missing_periods / total_periods) * 100
        
        missing_data.append({
            'id_ts': id_ts,
            'start_date': start_date,
            'end_date': end_date,
            'missing_percentage': missing_percentage
        })
    
    return pd.DataFrame(missing_data)


def compute_last_valid_entry(resampled_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the end date of the last valid entry (non-NaN value) for each time series.

    Parameters:
    -----------
    resampled_df : pd.DataFrame
        DataFrame containing the resampled time series data

    Returns:
    --------
    pd.DataFrame
        DataFrame with 'id_ts' and 'last_valid_date' columns
    """
    # Ensure the TimeInstant column is datetime
    resampled_df['TimeInstant'] = pd.to_datetime(resampled_df['TimeInstant'])

    # Filter non-NaN values and compute the last valid date for each id_ts
    last_valid_entries = (
        resampled_df.dropna(subset=['Value'])  # Drop rows where 'Value' is NaN
        .groupby('id_ts')['TimeInstant']
        .max()  # Get the last date for non-NaN values
        .reset_index()
        .rename(columns={'TimeInstant': 'last_valid_date'})  # Rename the column for clarity
    )

    return last_valid_entries


def compute_ts_duration(truncated_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the beginning, end, and number of years for each time series.

    Parameters:
    -----------
    truncated_df : pd.DataFrame
        DataFrame with columns 'id_ts', 'TimeInstant', 'Value'

    Returns:
    --------
    pd.DataFrame
        DataFrame with columns 'id_ts', 'start_date', 'end_date', and 'num_years'
    """
    # Drop rows where 'Value' is NaN
    truncated_df = truncated_df.dropna(subset=['Value'])

    # Calculate start and end dates, and duration in years for each time series
    ts_stats = truncated_df.groupby('id_ts').agg(
        start_date=('TimeInstant', 'min'),
        end_date=('TimeInstant', 'max')
    ).reset_index()

    # Calculate the number of years of data for each time series
    ts_stats['num_years'] = (ts_stats['end_date'] - ts_stats['start_date']).dt.days / 365.25

    return ts_stats


def ensure_monthly_continuity(df: pd.DataFrame, group_cols: Tuple[str, str] = ("id_mp", "id_ts"),
                            date_col: str = "TimeInstant", value_col: str = "Value") -> pd.DataFrame:
    """
    Make each group strictly monthly (month-start).

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    group_cols : Tuple[str, str]
        Column names to group by
    date_col : str
        Name of the date column
    value_col : str
        Name of the value column

    Returns:
    --------
    pd.DataFrame
        DataFrame with continuous monthly TimeInstant per group
    """
    df = df.copy()

    # Parse dates and normalize to month start (MS)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    # convert to monthly Period, then to timestamp at the *start* of month
    df[date_col] = df[date_col].dt.to_period("M").dt.to_timestamp(how="start")

    # Standardize values to numeric NaN
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    # Sort & drop duplicate months within groups
    group_cols = tuple(group_cols) if isinstance(group_cols, (list, tuple)) else (group_cols,)
    df = df.sort_values(list(group_cols) + [date_col])
    df = df.drop_duplicates(subset=list(group_cols) + [date_col], keep="first")

    # Reindex each group to full monthly range
    parts = []
    for keys, g in df.groupby(list(group_cols), dropna=False, sort=False):
        start, end = g[date_col].min(), g[date_col].max()
        full_idx = pd.date_range(start=start, end=end, freq="MS")
        original_dates = set(g[date_col].astype('int64'))  # for gap flagging

        g2 = (
            g.set_index(date_col)
             .reindex(full_idx)  # adds rows for missing months
             .reset_index()
             .rename(columns={"index": date_col})
        )

        # put back group key columns
        if not isinstance(keys, tuple):
            keys = (keys,)
        for col, val in zip(group_cols, keys):
            g2[col] = val

        # Optional: mark newly inserted rows in NullReason if present
        if "NullReason" in df.columns:
            if "NullReason" not in g2.columns:
                g2["NullReason"] = pd.NA
            is_inserted = ~g2[date_col].astype('int64').isin(original_dates)
            g2.loc[is_inserted & g2[value_col].isna(), "NullReason"] = g2.loc[
                is_inserted & g2[value_col].isna(), "NullReason"
            ].fillna("Added gap (auto)")

        parts.append(g2)

    out = pd.concat(parts, ignore_index=True)

    # Order columns nicely and sort
    other_cols = [c for c in out.columns if c not in (*group_cols, date_col)]
    out = out[list(group_cols) + [date_col] + other_cols]
    out = out.sort_values(list(group_cols) + [date_col]).reset_index(drop=True)

    return out


def consolidate_timeseries_by_mp(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each id_mp, merges all id_ts. In overlapping dates, keeps only values
    from the id_ts with the latest end date for that id_mp.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame with columns: 'id_ts', 'id_mp', 'TimeInstant', 'Value'

    Returns:
    --------
    pd.DataFrame
        A consolidated DataFrame with one time series per id_mp
    """
    df_copy = df.copy()
    df_copy['TimeInstant'] = pd.to_datetime(df_copy['TimeInstant'], errors='coerce')
    df_copy = df_copy.dropna(subset=['TimeInstant'])

    # Step 1: Calculate end date per id_ts
    ts_end_dates = df_copy.groupby('id_ts')['TimeInstant'].max().rename('end_date')
    df_with_end = df_copy.merge(ts_end_dates, on='id_ts')

    # Step 2: Determine the latest-ending id_ts per id_mp
    latest_ts_per_mp = (
        df_with_end.groupby('id_mp', as_index=False)
        .apply(lambda g: g.sort_values('end_date', ascending=False).iloc[0][['id_mp', 'id_ts']])
        .reset_index(drop=True)
    )
    latest_ts_map = latest_ts_per_mp.set_index('id_mp')['id_ts']

    # Step 3: Build final dataset per id_mp
    result = []
    for mp, group in df_with_end.groupby('id_mp'):
        latest_ts = latest_ts_map.loc[mp]
        df_latest = group[group['id_ts'] == latest_ts]
        df_other = group[group['id_ts'] != latest_ts]

        # In overlapping dates, keep only from latest_ts
        overlap_dates = set(df_latest['TimeInstant']).intersection(df_other['TimeInstant'])
        df_other_clean = df_other[~df_other['TimeInstant'].isin(overlap_dates)]

        combined = pd.concat([df_latest, df_other_clean], ignore_index=True)
        combined['id_ts'] = latest_ts  # unify id_ts assignment
        result.append(combined)

    final_df = pd.concat(result, ignore_index=True)
    return final_df[['id_mp', 'id_ts', 'TimeInstant', 'Value']].sort_values(by=['id_mp', 'TimeInstant'])
