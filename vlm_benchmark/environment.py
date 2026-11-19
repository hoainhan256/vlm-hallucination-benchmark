"""Collect environment metadata without downloading models or datasets."""

import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def package_version(name: str) -> str | None:
    try:
        module = __import__(name)
        return getattr(module, "__version__", None)
    except Exception:
        return None


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def collect_environment(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root)
    return {
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "git_commit": git_commit(),
        "packages": {
            "torch": package_version("torch"),
            "transformers": package_version("transformers"),
            "accelerate": package_version("accelerate"),
            "yaml": package_version("yaml"),
        },
        "cuda": None,
        "gpu": [],
        "repo_root": str(root.resolve()),
    }


def write_environment(path: str | Path, repo_root: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(collect_environment(repo_root), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
