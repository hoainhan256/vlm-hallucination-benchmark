#!/usr/bin/env python3
"""Configuration-only benchmark entry point."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vlm_benchmark.config import validate_all


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("check-config", "official"), required=True)
    parser.add_argument("--models", default=ROOT / "configs" / "models.yaml")
    parser.add_argument("--datasets", default=ROOT / "configs" / "datasets.yaml")
    parser.add_argument("--experiment", default=ROOT / "configs" / "experiment.yaml")
    args = parser.parse_args()

    official = args.mode == "official"
    errors = validate_all(args.models, args.datasets, args.experiment, official_stage=official)
    if errors:
        print("Benchmark execution blocked:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Configuration approved for the requested mode.")
    print("No model, dataset, inference or scoring work was executed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
