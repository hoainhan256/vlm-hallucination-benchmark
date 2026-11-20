#!/usr/bin/env python3
"""Check a run manifest and required result artifacts."""

import argparse
import json
from pathlib import Path


REQUIRED_FILES = (
    "predictions.jsonl",
    "metrics.json",
    "run_config.yaml",
    "run_metadata.json",
    "errors.jsonl",
    "run_status.json",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()

    manifest_path = args.run_dir / "manifest.json"
    if not manifest_path.exists():
        print("MANIFEST_ERROR: manifest.json is missing")
        return 2
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"MANIFEST_ERROR: {exc}")
        return 2

    missing = [name for name in REQUIRED_FILES if not (args.run_dir / name).exists()]
    if missing:
        print("Manifest validation failed:")
        for name in missing:
            print(f"- missing: {name}")
        return 1
    if not isinstance(manifest.get("experiment_id"), str):
        print("Manifest validation failed:\n- missing experiment_id")
        return 1
    if not isinstance(manifest.get("run_id"), str):
        print("Manifest validation failed:\n- missing run_id")
        return 1
    print("Manifest and required result files passed basic validation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
