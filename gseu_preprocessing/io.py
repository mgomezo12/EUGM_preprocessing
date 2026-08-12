import configparser
import re
from pathlib import Path
from dataclasses import dataclass


@dataclass
class Paths:
    project_root: Path
    data_root: Path
    eugm_icgc: Path
    data_processed: Path
    figures: Path
    shp_gseu: Path
    snapshot: str
    id_mp_not_in_analysis: Path


def _extract_snapshot(eugm_icgc: Path) -> str:
    """
    Derive the raw-input snapshot tag (e.g. '20260707') from the EUGM_ICGC folder
    name, so pipeline outputs land in a snapshot-specific folder instead of being
    silently overwritten by the next run against a different snapshot.
    """
    match = re.search(r"(\d{8})$", eugm_icgc.name)
    if not match:
        raise ValueError(
            f"Could not derive a snapshot tag from EUGM_ICGC folder name "
            f"'{eugm_icgc.name}'. Expected a trailing 8-digit date, e.g. "
            f"'EUGM_csv_20260707'."
        )
    return match.group(1)


def load_paths(config_path: Path) -> Paths:
    config = configparser.ConfigParser()
    config.read(config_path)

    # Allow relative paths from repo root
    cfg_paths = config["PATHS"]
    project_root = Path(config_path).resolve().parents[1]

    def resolve(p: str) -> Path:
        pp = Path(p)
        return (project_root / pp).resolve() if not pp.is_absolute() else pp

    data_root = resolve(cfg_paths.get("data_root", "data"))
    eugm_icgc = resolve(cfg_paths.get("EUGM_ICGC", "data/EUGM_ICGC/EUGM_csv_20250903"))
    snapshot = _extract_snapshot(eugm_icgc)
    data_processed_base = resolve(cfg_paths.get("data_processed", "data/EUGM_processed"))
    # Outputs for a given input snapshot live in their own subfolder, nested inside
    # data_processed (e.g. data/EUGM_processed/20260707/) so different runs never
    # collide and the data/ root stays clean.
    data_processed = data_processed_base / snapshot
    figures = resolve(cfg_paths.get("figures_folder", "outputs/figures"))
    shp_gseu = resolve(cfg_paths.get("shp_gseu_folder", "data/_aux/SHP"))
    # Manually-curated exclusion list (see data/EUGM_processed/Exclusion_ids/
    # readme_id_mp_not_in_analysis.md). Lives outside the per-snapshot folder on
    # purpose: the same exclusions apply regardless of which raw snapshot is active.
    id_mp_not_in_analysis = resolve(cfg_paths.get(
        "id_mp_not_in_analysis",
        "data/EUGM_processed/Exclusion_ids/id_mp_not_in_analysis.csv",
    ))

    data_processed.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    return Paths(
        project_root=project_root,
        data_root=data_root,
        eugm_icgc=eugm_icgc,
        data_processed=data_processed,
        figures=figures,
        shp_gseu=shp_gseu,
        snapshot=snapshot,
        id_mp_not_in_analysis=id_mp_not_in_analysis,
    )


