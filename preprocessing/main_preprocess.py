import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from gseu_preprocessing.io import load_paths
from gseu_preprocessing.preprocess import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Run GSEU preprocessing pipeline")
    parser.add_argument("--config", type=str, default=str(Path("configs/preprocessing.ini")), help="Path to preprocessing.ini")
    args = parser.parse_args()

    paths = load_paths(Path(args.config))
    run_pipeline(paths)


if __name__ == "__main__":
    main()


