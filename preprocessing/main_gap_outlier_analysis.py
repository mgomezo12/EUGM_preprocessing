#!/usr/bin/env python3
"""
Main script for gap analysis and outlier detection pipeline.

This script runs the complete gap analysis and outlier detection workflow
on the processed time series data from the main preprocessing pipeline.
"""

import argparse
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from gseu_preprocessing.io import load_paths
from gseu_preprocessing.gap_outlier_pipeline import (
    run_complete_gap_outlier_pipeline,
    run_outlier_detection_pipeline,
    run_outlier_plots_only,
    export_ppf1,
)


def main():
    """Main function for gap analysis and outlier detection."""
    parser = argparse.ArgumentParser(description="Run GSEU gap analysis and outlier detection pipeline")
    parser.add_argument("--config", type=str, default=str(Path("configs/preprocessing.ini")), 
                       help="Path to preprocessing.ini")
    parser.add_argument("--max-gap", type=float, default=18.0, 
                       help="Maximum gap threshold in months (default: 18)")
    parser.add_argument("--min-obs", type=int, default=60, 
                       help="Minimum number of observations required (default: 60)")
    parser.add_argument("--roc-threshold", type=int, default=10, 
                       help="Rate of change threshold for outlier detection (default: 10)")
    parser.add_argument("--ma-window", type=int, default=12, 
                       help="Moving average window size (default: 12)")
    parser.add_argument("--ma-sigma", type=float, default=3.1, 
                       help="Moving average sigma threshold (default: 3.1)")
    parser.add_argument(
        "--stage",
        choices=["complete", "outliers", "plots", "export"],
        default="complete",
        help=(
            "Stage to run: 'complete' (gap + outliers), "
            "'outliers' (outlier detection + plots), "
            "'plots' (only regenerate plots using existing outlier flags), or "
            "'export' (only re-export EUGM_MP_ppf1/EUGM_TS_ppf1 from the existing "
            "outlier-flagged data, applying the current id_mp_not_in_analysis "
            "exclusion list - use this when only that list changed, no new raw "
            "data or config; takes seconds instead of redoing outlier detection)."
        ),
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("GSEU GAP ANALYSIS AND OUTLIER DETECTION PIPELINE")
    print("=" * 60)
    print(f"Configuration: {args.config}")
    print(f"Max gap threshold: {args.max_gap} months")
    print(f"Min observations: {args.min_obs}")
    print(f"ROC threshold: {args.roc_threshold}")
    print(f"MA window: {args.ma_window}")
    print(f"MA sigma: {args.ma_sigma}")
    print("=" * 60)
    
    # Load configuration
    try:
        paths = load_paths(Path(args.config))
        print(f"OK: Configuration loaded successfully")
        print(f"  Data processed: {paths.data_processed}")
        print(f"  Figures: {paths.figures}")
    except Exception as e:
        print(f"ERROR: Failed to load configuration: {e}")
        return 1
    
    # Optional short-circuit: run only outlier detection or only plots
    if args.stage == "outliers":
        try:
            run_outlier_detection_pipeline(
                paths,
                roc_threshold=args.roc_threshold,
                ma_window=args.ma_window,
                ma_sigma=args.ma_sigma,
                max_gap_threshold=args.max_gap,
            )
            print("Outlier detection stage completed successfully!")
            return 0
        except Exception as e:
            print(f"Error running outlier detection stage: {e}")
            import traceback
            traceback.print_exc()
            return 1
    elif args.stage == "plots":
        try:
            run_outlier_plots_only(
                paths,
                ma_window=args.ma_window,
                max_gap_threshold=args.max_gap,
            )
            print("Plotting-only stage completed successfully!")
            return 0
        except Exception as e:
            print(f"Error running plotting-only stage: {e}")
            import traceback
            traceback.print_exc()
            return 1
    elif args.stage == "export":
        try:
            export_ppf1(paths, max_gap_threshold=args.max_gap)
            print("Export stage completed successfully!")
            return 0
        except Exception as e:
            print(f"Error running export stage: {e}")
            import traceback
            traceback.print_exc()
            return 1

    # Run the complete pipeline
    try:
        run_complete_gap_outlier_pipeline(
            paths,
            max_gap_threshold=args.max_gap,
            min_obs_threshold=args.min_obs,
            roc_threshold=args.roc_threshold,
            ma_window=args.ma_window,
            ma_sigma=args.ma_sigma
        )
        print("OK: Pipeline completed successfully!")
        return 0
    except Exception as e:
        print(f"ERROR: Error running pipeline: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
