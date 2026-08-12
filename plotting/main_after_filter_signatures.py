#!/usr/bin/env python3
"""
Compute groundwater signatures (Table 4) and generate Fig. 6 directly from
the published dataset (data/EUGM_gwl.csv + data/EUGM_mp.csv) - no raw
partner data needed.
"""

import argparse
from pathlib import Path
import sys

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(HERE.parent))  # scripts/, so gseu_preprocessing is importable

from gseu_preprocessing.signatures_pipeline import run_signatures_pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gwl-csv", type=Path, default=REPO_ROOT / "data" / "EUGM_gwl.csv",
                         help="Published groundwater level time series.")
    parser.add_argument("--mp-csv", type=Path, default=REPO_ROOT / "data" / "EUGM_mp.csv",
                         help="Published monitoring point metadata.")
    parser.add_argument("--boundary-shp", type=Path,
                         default=REPO_ROOT / "data" / "_aux" / "SHP" / "Europe_map_GSEU_Partners.shp",
                         help="Optional country-boundary shapefile for map context. Not bundled with "
                              "this release - if not found at the default path, the maps are drawn "
                              "without country outlines rather than failing.")
    parser.add_argument("--fig-dir", type=Path, default=HERE,
                         help="Output directory for the figures.")
    parser.add_argument("--sig-out", type=Path, default=REPO_ROOT / "data" / "gw_signatures.csv",
                         help="Output path for the computed signatures table.")
    parser.add_argument("--max-mp", type=int, default=None,
                         help="Limit number of monitoring points for a quick trial run (e.g., 100).")
    parser.add_argument("--skip-compute", action="store_true",
                         help="Do not recompute signatures; reuse the file at --sig-out if present.")
    args = parser.parse_args()

    if args.max_mp is not None:
        import os
        os.environ["SIGNATURES_MAX_MP"] = str(args.max_mp)

    run_signatures_pipeline(
        imputed_csv=args.gwl_csv,
        metadata_csv=args.mp_csv,
        fig_dir=args.fig_dir,
        sig_out=args.sig_out,
        boundary_shp=args.boundary_shp,
        compute=not args.skip_compute,
    )


if __name__ == "__main__":
    main()
