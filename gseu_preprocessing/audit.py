from pathlib import Path

import pandas as pd


def _parse_timeinstant_with_normalization(timeinstant: pd.Series):
    timeinstant_raw = timeinstant.astype(str)
    timeinstant_fixed = timeinstant_raw.str.replace(r' (\d):', r' 0\1:', regex=True)
    normalized_mask = timeinstant_fixed.ne(timeinstant_raw)

    formats = [
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%d.%m.%Y",
    ]
    parsed = pd.to_datetime(pd.Series([pd.NaT] * len(timeinstant_raw)), errors='coerce')
    for fmt in formats:
        mask = parsed.isna()
        if not mask.any():
            break
        parsed.loc[mask] = pd.to_datetime(timeinstant_fixed.loc[mask], format=fmt, errors='coerce')

    mask = parsed.isna()
    if mask.any():
        parsed.loc[mask] = pd.to_datetime(timeinstant_fixed.loc[mask], errors='coerce')

    return parsed, normalized_mask


def write_preprocessing_audit(dfts_conv_updated: pd.DataFrame, output_dir: Path, source_file: str = "") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    df = dfts_conv_updated.copy()
    df['id_mp'] = df['id_ts'].astype(str).str.replace(r'_[A-Z]$', '', regex=True)

    parsed, normalized_mask = _parse_timeinstant_with_normalization(df['TimeInstant'])
    invalid_time_mask = parsed.isna()

    values = pd.to_numeric(df['Value'].astype(str).str.replace(',', '', regex=False), errors='coerce')
    outside_value_mask = values.notna() & ~values.between(-200, 2500)

    stations_invalid = sorted(df.loc[invalid_time_mask, 'id_mp'].dropna().astype(str).unique().tolist())
    stations_outside = sorted(df.loc[outside_value_mask, 'id_mp'].dropna().astype(str).unique().tolist())
    stations_normalized = sorted(df.loc[normalized_mask, 'id_mp'].dropna().astype(str).unique().tolist())

    pd.DataFrame({'id_mp': stations_invalid}).to_csv(output_dir / 'stations_invalid_timeinstant.csv', index=False)
    pd.DataFrame({'id_mp': stations_outside}).to_csv(output_dir / 'stations_value_outside_range.csv', index=False)
    pd.DataFrame({'id_mp': stations_normalized}).to_csv(output_dir / 'stations_datetime_normalized.csv', index=False)

    report_lines = [
        "# Preprocessing Audit",
        "",
        f"- Source file: `{source_file}`" if source_file else "- Source file: `unknown`",
        f"- Total rows scanned: **{len(df):,}**",
        "",
        "## 1) Invalid TimeInstant dropped",
        f"- Rows with invalid/unparseable `TimeInstant`: **{int(invalid_time_mask.sum()):,}**",
        f"- Stations (`id_mp`) affected: **{len(stations_invalid):,}**",
        "- Station IDs file: `outputs/audit/stations_invalid_timeinstant.csv`",
        "",
        "## 2) Value range filter (-200 <= Value <= 2500)",
        f"- Rows outside range (non-null numeric values): **{int(outside_value_mask.sum()):,}**",
        f"- Stations (`id_mp`) affected: **{len(stations_outside):,}**",
        "- Station IDs file: `outputs/audit/stations_value_outside_range.csv`",
        "",
        "## 3) Datetime normalization applied",
        "- Rule checked: regex replacement `r' (\\d):' -> ' 0\\1:'` on `TimeInstant`",
        f"- Rows changed by normalization: **{int(normalized_mask.sum()):,}**",
        f"- Stations (`id_mp`) affected: **{len(stations_normalized):,}**",
        "- Station IDs file: `outputs/audit/stations_datetime_normalized.csv`",
        "",
    ]
    (output_dir / 'preprocessing_audit_report.md').write_text("\n".join(report_lines), encoding='utf-8')
