#!/usr/bin/env python3
"""Check required repository directories and documentation."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_DIRS = ("benchmarks", "configs", "docs", "models", "notebooks", "results", "scripts", "tests")
REQUIRED_FILES = (
    "README.md",
    ".gitignore",
    "requirements.txt",
    "configs/models.yaml",
    "configs/datasets.yaml",
    "configs/experiment.yaml",
)


def main() -> int:
    missing = [
        str(path)
        for path in (
            *(ROOT / name for name in REQUIRED_DIRS),
            *(ROOT / name for name in REQUIRED_FILES),
        )
        if not path.exists()
    ]
    if missing:
        print("Repository structure validation failed:")
        for path in missing:
            print(f"- missing: {path}")
        return 1
    print("Repository structure validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
