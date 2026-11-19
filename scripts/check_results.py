#!/usr/bin/env python3
"""Validate a JSONL prediction file and JSON result artifacts."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vlm_benchmark.results import read_jsonl, validate_predictions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    args = parser.parse_args()

    try:
        records = read_jsonl(args.predictions)
        errors = validate_predictions(records)
    except (OSError, ValueError) as exc:
        print(f"RESULT_ERROR: {exc}")
        return 2

    if errors:
        print("Prediction result validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"Prediction result validation passed ({len(records)} records).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
