#!/usr/bin/env python3
"""Collect local environment metadata; never downloads model or dataset data."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vlm_benchmark.environment import write_environment


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=ROOT / "results" / "local_environment.json",
    )
    args = parser.parse_args()
    write_environment(args.output, ROOT)
    print(f"Environment metadata written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
