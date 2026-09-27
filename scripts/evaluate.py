"""python scripts/evaluate.py --help"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from anywhere without installing

from esr.cli import main  # noqa: E402

if __name__ == "__main__":
    main(["evaluate", *sys.argv[1:]])
