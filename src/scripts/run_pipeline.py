#!/usr/bin/env python3
"""Repository entry point; keeps ``dubbing_pipeline`` importable from src/."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dubbing_pipeline.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
