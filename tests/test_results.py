import json

import pytest

from vlm_benchmark.results import merge_result_records, read_jsonl, validate_predictions


def test_jsonl_fixture_is_valid():
    records = read_jsonl("tests/fixtures/synthetic_predictions.jsonl")
    assert validate_predictions(records) == []


def test_duplicate_sample_id_is_detected(tmp_path):
    records = [
        {"sample_id": "x", "prediction": "yes", "status": "COMPLETED", "run_config_hash": "h"},
        {"sample_id": "x", "prediction": "no", "status": "COMPLETED", "run_config_hash": "h"},
    ]
    assert any("Duplicate sample_id" in error for error in validate_predictions(records))


def test_missing_result_is_detected():
    records = [{"sample_id": "x", "status": "COMPLETED", "run_config_hash": "h"}]
    assert any("prediction" in error for error in validate_predictions(records))


def test_incompatible_run_configs_are_rejected():
    records = [
        {"sample_id": "x", "prediction": "yes", "status": "COMPLETED", "run_config_hash": "h1"},
        {"sample_id": "y", "prediction": "no", "status": "COMPLETED", "run_config_hash": "h2"},
    ]
    assert any("multiple incompatible" in error for error in validate_predictions(records))


def test_resume_continues_missing_sample():
    existing = [
        {"sample_id": "x", "prediction": "yes", "status": "COMPLETED", "run_config_hash": "h"}
    ]
    incoming = [
        {"sample_id": "y", "prediction": "no", "status": "COMPLETED", "run_config_hash": "h"}
    ]
    assert merge_result_records(existing, incoming) == existing + incoming


def test_resume_does_not_overwrite_completed_sample():
    existing = [
        {"sample_id": "x", "prediction": "yes", "status": "COMPLETED", "run_config_hash": "h"}
    ]
    incoming = [
        {"sample_id": "x", "prediction": "no", "status": "COMPLETED", "run_config_hash": "h"}
    ]
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        merge_result_records(existing, incoming)


def test_incomplete_existing_sample_can_be_updated():
    existing = [
        {"sample_id": "x", "prediction": None, "status": "RUNNING", "run_config_hash": "h"}
    ]
    incoming = [
        {"sample_id": "x", "prediction": "yes", "status": "COMPLETED", "run_config_hash": "h"}
    ]
    assert merge_result_records(existing, incoming)[0]["status"] == "COMPLETED"
