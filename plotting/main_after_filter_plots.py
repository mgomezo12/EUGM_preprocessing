#!/usr/bin/env python3
"""
Generate Fig. 2a/2b, 3a-3d, and A1 directly from the published, filtered
dataset (data/EUGM_gwl.csv + data/EUGM_mp.csv) - no raw partner data needed.
"""

import argparse
from pathlib import Path
import sys

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(HERE.parent))  # scripts/, so gseu_preprocessing is importable

from gseu_preprocessing.after_filter_pipeline import run_after_filter_plots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gwl-csv", type=Path, default=REPO_ROOT / "data" / "EUGM_gwl.csv",
                         help="Published groundwater level time series.")
    parser.add_argument("--mp-csv", type=Path, default=REPO_ROOT / "data" / "EUGM_mp.csv",
                         help="Published monitoring point metadata.")
    parser.add_argument("--boundary-shp", type=Path,
                         default=REPO_ROOT / "data" / "_aux" / "SHP" / "Europe_map_GSEU_Partners.shp",
                         help="Optional country-boundary shapefile for map context (needs a 'Country' "
                              "column; any source works, e.g. Natural Earth's admin-0 countries layer). "
                              "Not bundled with this release - if not found at the default path, the "
                              "maps are drawn without country outlines rather than failing.")
    parser.add_argument("--fig-dir", type=Path, default=HERE,
                         help="Output directory for the figures.")
    args = parser.parse_args()

    run_after_filter_plots(
        imputed_csv=args.gwl_csv,
        metadata_csv=args.mp_csv,
        fig_dir=args.fig_dir,
        boundary_shp=args.boundary_shp,
    )


if __name__ == "__main__":
    main()
