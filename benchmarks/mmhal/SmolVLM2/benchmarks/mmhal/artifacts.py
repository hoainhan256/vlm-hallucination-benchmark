"""Crash-conscious local artifacts. A run directory has exactly one writer."""

import hashlib
import json
import os
from pathlib import Path

import yaml

from vlm_benchmark.results import read_jsonl, stable_config_hash, validate_predictions


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def append_jsonl(path, record):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


class Run:
    def __init__(self, root, config, identity, metadata, resume=False):
        self.root = Path(root)
        self.config = config
        self.identity = identity
        self.metadata = metadata
        self.hash = stable_config_hash(identity)
        self.resume = resume
        self.records = []

    def __enter__(self):
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = self.root / ".writer.lock"
        self.fd = os.open(self.lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(self.fd, str(os.getpid()).encode())
        try:
            status_path = self.root / "run_status.json"
            if status_path.exists():
                status = json.loads(status_path.read_text(encoding="utf-8"))
                if status["status"] == "COMPLETED":
                    raise ValueError("Refusing to overwrite completed run")
                if not self.resume:
                    raise ValueError("Existing run requires --resume")
                old = json.loads((self.root / "run_metadata.json").read_text(encoding="utf-8"))
                if old["run_config_hash"] != self.hash:
                    raise ValueError("Resume identity/config/environment mismatch")
                self.records = read_jsonl(self.root / "predictions.jsonl")
                errors = validate_predictions(self.records)
                if errors or any(r["run_config_hash"] != self.hash for r in self.records):
                    raise ValueError(f"Invalid resume predictions: {errors}")
                self.metadata = old
            else:
                if any(p.name != ".writer.lock" for p in self.root.iterdir()):
                    raise ValueError("Refusing nonempty unrecognized run directory")
                (self.root / "run_config.yaml").write_text(yaml.safe_dump(self.config, sort_keys=True), encoding="utf-8")
                self.metadata.update(run_config_hash=self.hash, identity=self.identity)
                write_json(self.root / "run_metadata.json", self.metadata)
                for name in ("predictions.jsonl", "errors.jsonl", "progress.jsonl"):
                    (self.root / name).touch(exist_ok=False)
                write_json(self.root / "metrics.json", {"status": "NOT_SCORED"})
            self.set_status("RUNNING")
            return self
        except BaseException:
            os.close(self.fd)
            self.lock.unlink()
            raise

    def set_status(self, status):
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        write_json(self.root / "run_status.json", {"status": status, "updated_at": now})
        self.metadata.update(status=status, updated_at=now)
        write_json(self.root / "run_metadata.json", self.metadata)

    def add(self, record):
        if record["sample_id"] in {r["sample_id"] for r in self.records}:
            raise ValueError("Duplicate sample; attempted samples are immutable")
        record = dict(record, run_config_hash=self.hash)
        append_jsonl(self.root / "predictions.jsonl", record)
        self.records.append(record)
        append_jsonl(self.root / "progress.jsonl", {"sample_id": record["sample_id"], "status": record["status"]})

    def error(self, sample_id, exc, kind):
        # Avoid arbitrary exception strings that could include request headers.
        append_jsonl(self.root / "errors.jsonl", {"sample_id": sample_id, "kind": kind,
                     "exception_type": type(exc).__name__, "run_config_hash": self.hash})

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc is not None:
                self.error(None, exc, "run_error")
                from .model import HardwareBlocked
                self.set_status("BLOCKED" if isinstance(exc, HardwareBlocked) else "FAILED")
            elif any(r["status"] != "COMPLETED" for r in self.records):
                self.set_status("FAILED")
            else:
                self.set_status("COMPLETED")
            manifest = {"experiment_id": self.config["experiment_id"], "run_id": self.root.name,
                        "run_config_hash": self.hash, "normalized_config": self.config,
                        "files": {p.name: checksum(p) for p in self.root.iterdir()
                                  if p.is_file() and p.name not in ("manifest.json", ".writer.lock")}}
            write_json(self.root / "manifest.json", manifest)
        finally:
            os.close(self.fd)
            self.lock.unlink()


def verify_run(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    required = {"predictions.jsonl", "metrics.json", "run_config.yaml", "run_metadata.json",
                "errors.jsonl", "run_status.json"}
    if not required <= set(manifest["files"]):
        raise ValueError("Required artifacts missing from manifest")
    for name, digest in manifest["files"].items():
        if Path(name).name != name or checksum(root / name) != digest:
            raise ValueError(f"Artifact checksum mismatch: {name}")
    records = read_jsonl(root / "predictions.jsonl")
    metadata = json.loads((root / "run_metadata.json").read_text(encoding="utf-8"))
    if stable_config_hash(metadata["identity"]) != manifest["run_config_hash"]:
        raise ValueError("Manifest identity mismatch")
    if validate_predictions(records) or any(r["run_config_hash"] != manifest["run_config_hash"] for r in records):
        raise ValueError("Invalid predictions/config identity")
    return records, manifest
