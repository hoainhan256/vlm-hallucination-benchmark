"""Lightweight validation for result artifacts and run identity."""

import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping


REQUIRED_PREDICTED_FIELDS = {"sample_id", "prediction", "status", "run_config_hash"}


def read_jsonl(path: str | Path) -> list[dict]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: record must be a JSON object")
            records.append(record)
    return records


def validate_predictions(records: Iterable[Mapping]) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    hashes: set[str] = set()
    for record in records:
        missing = REQUIRED_PREDICTED_FIELDS - set(record)
        if missing:
            errors.append(f"Missing fields: {sorted(missing)}")
        sample_id = record.get("sample_id")
        if sample_id is not None:
            if sample_id in seen:
                errors.append(f"Duplicate sample_id: {sample_id}")
            seen.add(str(sample_id))
        run_hash = record.get("run_config_hash")
        if run_hash is not None:
            hashes.add(str(run_hash))
    if len(hashes) > 1:
        errors.append("Predictions contain multiple incompatible run_config_hash values")
    return errors


def merge_result_records(existing: Iterable[Mapping], incoming: Iterable[Mapping]) -> list[dict]:
    existing_list = [dict(record) for record in existing]
    incoming_list = [dict(record) for record in incoming]
    existing_by_id = {str(record.get("sample_id")): record for record in existing_list}
    incoming_by_id = {str(record.get("sample_id")): record for record in incoming_list}

    duplicates = set(existing_by_id).intersection(incoming_by_id)
    conflicts = [
        sample_id
        for sample_id in duplicates
        if existing_by_id[sample_id].get("status") == "COMPLETED"
        and existing_by_id[sample_id] != incoming_by_id[sample_id]
    ]
    if conflicts:
        raise ValueError(f"Refusing to overwrite completed samples: {sorted(conflicts)}")

    merged = dict(existing_by_id)
    for sample_id, record in incoming_by_id.items():
        if sample_id not in merged or record.get("status") == "COMPLETED":
            merged[sample_id] = record
    return list(merged.values())


def stable_config_hash(config: Mapping) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
