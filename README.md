# EUGM data paper — code

Code behind *Analysis-ready European Groundwater Monitoring (EUGM) database*
(Gomez et al.).

The dataset itself is not in this repository, it's archived separately on
Zenodo: https://doi.org/10.5281/zenodo.21891416. To run any script here that
reads `data/...`, download the dataset from that DOI and place the CSV files
in a `data/` folder at the root of this repo; it's `.gitignore`d here since
it doesn't belong in a code repository.

## Contents

These scripts fall into two categories:

- **Scripts that run directly against the published `data/*.csv` files** —
  `plotting/` and the map/table-generating parts of `gseu_preprocessing/`.
  These are fully reproducible once you've downloaded `data/` from Zenodo.
- **Scripts that describe how the raw groundwater level time series became the
  published dataset** — `preprocessing/` and the QC/filtering/imputation
  parts of `gseu_preprocessing/`. The raw time series themselves are **not**
  published anywhere,  so these scripts cannot be re-run end-to-end. They're
  published so the exact method is inspectable and any question about it
  can be answered from the code.

- `preprocessing/` — quality control, monthly resampling, gap and outlier
  filtering that turns raw partner submissions into the filtered dataset
  (paper §2 and §4). `shift_detection/detect_level_shifts_ppf1.py` is the
  method described in §4.2 (rolling pre/post-median comparison) — the basis
  for `shift_detected` in `EUGM_gwl.csv` and the station exclusions/
  corrections it led to. By default it only writes the score tables (one
  row per station); pass `--max-plots N` for per-station diagnostic figures
  (there can be hundreds of flagged stations, so this isn't on by default).
- `plotting/` — reproduces Fig. 2, 3, 5, 6, 7, and A1 directly from `data/`,
  no raw partner data needed for any of it. `main_after_filter_plots.py`
  (Fig. 2a/2b, 3a-3d, A1) and `main_after_filter_signatures.py` (Fig. 6,
  Table 4) are CLI entry points for the pipeline logic in
  `gseu_preprocessing/after_filter_pipeline.py`/`signatures_pipeline.py`;
  `plot_static_features.py` (Fig. 5) and `plot_trend_map.py` (Fig. 7) run
  standalone. The clustering appendix figure (Fig. A2/A3) uses plotting
  code included here, but the underlying cluster analysis isn't reproducible
  here — that script consumes `data/trends_cluster.csv`'s `cluster_number`
  column, already published. 
- `trends/` — Mann-Kendall trend test with Theil-Sen/seasonal Sen's slope
  (§6, Fig. 7), reproducing `trends_cluster.csv`'s `tac_*` columns from
  `data/EUGM_gwl.csv` alone. `mks_trend.py` is transcribed unmodified from
  the original methodology code (W.J. Zaadnoordijk, TNO-GDN);
  `compute_trends.py` is the only adapted part, pointed at the published
  schema instead of the original per-station CSV export. It does **not**
  reproduce `cluster_number` — the SGI-based k-means clustering is a
  separate BGS/TNO step with no code in this release, so `trends_cluster.csv`
  remains the file to use for cluster assignments. `pymannkendall.py` is a
  vendored copy of a third-party package (v1.4.2, Hussain et al., 2019,
  MIT-licensed) pinned deliberately: newer pip releases drop an attribute
  (`slope_ci`) this code depends on, which silently breaks trend computation
  for most stations.
- `gseu_preprocessing/` — shared library used by both of the above; includes
  a safety net (`truncate_trailing_nan` in `gap_analysis.py`) that trims any
  station's trailing all-NaN run before imputation, plus two manually
  identified station corrections not caught by any automated rule — see the
  data repository's `README.md` for detail.
- `dynamic_features/` — ERA5-Land download, point-extraction, and SPI
  scripts (Table 2): downloads ERA5-Land monthly-mean NetCDF from the
  Copernicus Climate Data Store for the area covered by the monitoring
  points, extracts values at each point, and computes relative humidity and
  the Standardised Precipitation Index (SPI1/6/12/48). Unlike
  `preprocessing/`, this one *is* runnable end-to-end
  (`data/EUGM_mp.csv` provides the point locations) — it just needs your
  own CDS API credentials and will re-download the full 1950-present
  archive (a genuine, if slow, dependency on an external public data
  source, not on unpublished raw data).

## License

CC BY 4.0 — see `LICENSE`.

## Citation

Please cite the accompanying data paper when using this code. The dataset
it operates on is archived at Zenodo:
https://doi.org/10.5281/zenodo.21891416.

## Note
The authors used LLM to assist with the development and testing of the data-processing and analysis code in this repository.
