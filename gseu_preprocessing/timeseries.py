import pandas as pd
import numpy as np


def remove_invalid_timeinstant(df: pd.DataFrame) -> pd.DataFrame:
    invalid_timeinstant = df[df["TimeInstant"].isna()]
    print(f"Number of rows with invalid 'TimeInstant': {len(invalid_timeinstant)}")
    return df.drop(invalid_timeinstant.index)


def count_ts_per_mp(dfts_conv: pd.DataFrame, dfts_meta_mp: pd.DataFrame) -> pd.DataFrame:
    id_ts_nonnan_counts = dfts_conv.groupby(["id_mp", "id_ts"])['Value'].count().reset_index()
    id_ts_nonnan_counts.columns = ['id_mp', 'id_ts', 'nonnan_count']
    id_mp_nonnan_counts = id_ts_nonnan_counts.groupby('id_mp')['nonnan_count'].sum().reset_index()
    id_mp_nonnan_counts.columns = ['id_mp', 'id_ts_count']
    dfts_meta_mp = dfts_meta_mp.merge(id_mp_nonnan_counts, on='id_mp', how='left')
    result_df = dfts_meta_mp.merge(id_ts_nonnan_counts, on='id_mp', how='left')
    return result_df


def assign_id_ts_to_meta(dfts_meta_mp: pd.DataFrame, resampled_df_monthly: pd.DataFrame) -> pd.DataFrame:
    df_idts = resampled_df_monthly[["id_mp", "id_ts"]].dropna().drop_duplicates()
    df_idts['sort_priority'] = df_idts['id_ts'].str.endswith('_A').astype(int)
    df_idts = df_idts.sort_values(['id_mp', 'sort_priority'], ascending=[True, False])
    idmp_to_idts = df_idts.groupby('id_mp')['id_ts'].apply(list)

    dfts_meta_mp = dfts_meta_mp.copy()
    dfts_meta_mp['id_ts_list'] = dfts_meta_mp['id_mp'].map(idmp_to_idts)
    dfts_meta_mp = dfts_meta_mp[~dfts_meta_mp['id_ts_list'].isna()].copy()
    dfts_meta_mp = dfts_meta_mp.explode('id_ts_list').rename(columns={'id_ts_list': 'id_ts'}).reset_index(drop=True)
    return dfts_meta_mp


def consolidate_timeseries_by_mp(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    df_copy['TimeInstant'] = pd.to_datetime(df_copy['TimeInstant'], errors='coerce')
    df_copy = df_copy.dropna(subset=['TimeInstant'])

    ts_end_dates = df_copy.groupby('id_ts')['TimeInstant'].max().rename('end_date')
    df_with_end = df_copy.merge(ts_end_dates, on='id_ts')

    latest_ts_per_mp = (
        df_with_end.groupby('id_mp', as_index=False)
        .apply(lambda g: g.sort_values('end_date', ascending=False).iloc[0][['id_mp', 'id_ts']])
        .reset_index(drop=True)
    )
    latest_ts_map = latest_ts_per_mp.set_index('id_mp')['id_ts']

    result = []
    for mp, group in df_with_end.groupby('id_mp'):
        latest_ts = latest_ts_map.loc[mp]
        df_latest = group[group['id_ts'] == latest_ts]
        df_other = group[group['id_ts'] != latest_ts]
        overlap_dates = set(df_latest['TimeInstant']).intersection(df_other['TimeInstant'])
        df_other_clean = df_other[~df_other['TimeInstant'].isin(overlap_dates)]
        combined = pd.concat([df_latest, df_other_clean], ignore_index=True)
        combined['id_ts'] = latest_ts
        result.append(combined)

    final_df = pd.concat(result, ignore_index=True)
    return final_df[['id_mp', 'id_ts', 'TimeInstant', 'Value']].sort_values(by=['id_mp', 'TimeInstant'])


def parse_and_resample(dfts_conv_updated: pd.DataFrame) -> pd.DataFrame:
    dfts_conv_updated = dfts_conv_updated.copy()
    dfts_conv_updated['id_mp'] = dfts_conv_updated['id_ts'].str.replace(r'_[A-Z]$', '', regex=True)

    dfts_conv_updated['TimeInstant_fixed'] = dfts_conv_updated['TimeInstant'].astype(str).str.replace(r' (\d):', r' 0\1:', regex=True)
    formats = [
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%d.%m.%Y",
    ]

    dates = dfts_conv_updated['TimeInstant_fixed'].astype(str)
    parsed = pd.to_datetime(pd.Series([pd.NaT] * len(dfts_conv_updated)), errors='coerce')
    for fmt in formats:
        mask = parsed.isna()
        if not mask.any():
            break
        parsed.loc[mask] = pd.to_datetime(dates.loc[mask], format=fmt, errors='coerce')
    mask = parsed.isna()
    if mask.any():
        parsed.loc[mask] = pd.to_datetime(dates.loc[mask], errors='coerce')

    dfts_conv_updated['TimeInstant_dt'] = parsed
    dfts_conv_updated['TimeInstant_std'] = parsed.dt.strftime("%Y/%m/%d %H:%M:%S")

    dfts_conv_updated['Value'] = pd.to_numeric(dfts_conv_updated['Value'].astype(str).str.replace(',', ''), errors='coerce')
    dfts = dfts_conv_updated[(dfts_conv_updated['Value'].between(-200, 2500)) | (pd.isna(dfts_conv_updated['Value']))]
    dfts = dfts.groupby('id_ts').filter(lambda x: len(x) >= 1)
    dfts = dfts.dropna(subset=["TimeInstant_dt"])  # drop rows without parsed datetime

    resampled_df_monthly = (
        dfts.set_index('TimeInstant_dt')
        .groupby('id_ts')
        .resample('MS')
        .median(numeric_only=True)
        .reset_index()
    )
    resampled_df_monthly = resampled_df_monthly.rename(columns={'TimeInstant_dt': 'TimeInstant'})
    resampled_df_monthly = resampled_df_monthly.merge(
        dfts[["id_ts", "id_mp", "NullReason"]].drop_duplicates('id_ts'),
        on='id_ts',
        how='left',
    )
    resampled_df_monthly['TimeInstant'] = resampled_df_monthly['TimeInstant'].dt.strftime("%Y/%m/%d")
    resampled_df_monthly = resampled_df_monthly[['id_mp', 'id_ts', 'TimeInstant', 'Value', 'NullReason']]
    return resampled_df_monthly


