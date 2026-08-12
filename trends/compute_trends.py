"""
Trend analysis (Sect. 6, Fig. 7) - Mann-Kendall test with Theil-Sen /
seasonal Sen's slope, adapted from W.J. Zaadnoordijk's (TNO-GDN) original
methodology code (mks_trend.py in this same folder, transcribed unmodified)
to run directly against this release's published data/EUGM_gwl.csv instead
of the raw per-station CSV export his original scripts expected.

Reproduces the tac_* trend columns of data/trends_cluster.csv. It does NOT
reproduce cluster_number: the SGI-based k-means clustering (Sect. 6) is a
separate step computed by BGS/TNO and there is no code for it in this
release - trends_cluster.csv remains the file to use for cluster_number.

Usage:
    python compute_trends.py
    python compute_trends.py --start 2002-01-01 --end 2021-12-01 --out my_trends.csv
"""

import argparse
from pathlib import Path

import pandas as pd

from mks_trend import mannkendallsen

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent


def compute_trends(gwl_csv: Path, value_col: str, d_start: pd.Timestamp, d_end: pd.Timestamp) -> pd.DataFrame:
    gwl = pd.read_csv(gwl_csv, usecols=["id_mp", "TimeInstant", value_col], parse_dates=["TimeInstant"])
    gwl = gwl.sort_values(["id_mp", "TimeInstant"])

    records = []
    for id_mp, group in gwl.groupby("id_mp", sort=False):
        sub = group.set_index("TimeInstant")
        # only stations whose record fully spans the trend period are eligible,
        # matching the manuscript's "timeseries that contain values for all
        # months in the period 2002-2021" (Sect. 6)
        if sub.index[0] > d_start or sub.index[-1] < d_end:
            continue
        dft = sub[(sub.index >= d_start) & (sub.index <= d_end)]
        try:
            mks = mannkendallsen(dft[value_col], 12)
        except Exception as e:
            print(f"Skipping {id_mp}: {e}")
            continue
        records.append({
            "id_mp": id_mp,
            "tac_testP_mon_RP": mks.p,
            "tac_testTau_mon_RP": mks.tau,
            "tac_slope_mon_RP": mks.slope,
            "tac_slope_sig_mon_RP": mks.slope if mks.significance else pd.NA,
            "tac_slope_intercept_mon_RP": mks.intercept,
            "tac_slopeCI_low_mon_RP": mks.lower_bound,
            "tac_slopeCI_up_mon_RP": mks.upper_bound,
        })
    return pd.DataFrame.from_records(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gwl-csv", type=Path, default=REPO_ROOT / "data" / "EUGM_gwl.csv",
                         help="Published groundwater level time series.")
    parser.add_argument("--value-col", type=str, default="Value_comp",
                         help="Column to compute trends on (gap-filled series, matching the original methodology).")
    parser.add_argument("--start", type=str, default="2002-01-01", help="Trend period start (yyyy-mm-dd).")
    parser.add_argument("--end", type=str, default="2021-12-01", help="Trend period end (yyyy-mm-dd).")
    parser.add_argument("--out", type=Path, default=HERE / "trends_reproduced.csv",
                         help="Output CSV (tac_* columns only - no cluster_number, see module docstring).")
    args = parser.parse_args()

    df = compute_trends(args.gwl_csv, args.value_col, pd.Timestamp(args.start), pd.Timestamp(args.end))
    df.to_csv(args.out, index=False)
    print(f"{len(df)} stations with a trend computed for {args.start} to {args.end}.")
    print(f"Written to {args.out}")
