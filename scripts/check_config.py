#!/usr/bin/env python3
"""Validate YAML configuration without loading models or datasets."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vlm_benchmark.config import validate_all


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=ROOT / "configs" / "models.yaml")
    parser.add_argument("--datasets", default=ROOT / "configs" / "datasets.yaml")
    parser.add_argument("--experiment", default=ROOT / "configs" / "experiment.yaml")
    parser.add_argument("--mode", choices=("config", "official"), default="config")
    args = parser.parse_args()

    try:
        errors = validate_all(
            args.models,
            args.datasets,
            args.experiment,
            official_stage=args.mode == "official",
        )
    except Exception as exc:
        print(f"CONFIG_ERROR: {exc}")
        return 2

    if errors:
        print("Configuration validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("Configuration validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
