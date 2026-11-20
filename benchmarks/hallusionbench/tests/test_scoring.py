"""CPU-only unit tests for HallusionBench scoring and resume safety."""

import json
import pytest
from benchmarks.hallusionbench.scoring import (
    export_hallusionbench_artifacts,
    parse_yes_no,
    score_hallusionbench,
)


@pytest.fixture
def sample_dataset() -> dict[str, dict]:
    return {
        "s1": {"reference_answer": "yes", "category": "VD", "figure_key": "fig1", "pair_key": "pair1"},
        "s2": {"reference_answer": "no", "category": "VD", "figure_key": "fig1", "pair_key": "pair1"},
        "s3": {"reference_answer": "yes", "category": "VS", "figure_key": "fig2", "pair_key": "pair2"},
        "s4": {"reference_answer": "yes", "category": "VS", "figure_key": "fig2", "pair_key": "pair2"},
    }


def test_parse_yes_no_variants():
    assert parse_yes_no("Yes") == "yes"
    assert parse_yes_no("No, there is no cat.") == "no"
    assert parse_yes_no("Based on the chart, the answer is yes.") == "yes"
    assert parse_yes_no("Yes and no, it depends.") is None
    assert parse_yes_no("") is None


def test_partial_pair_failure_affects_facc_and_qacc(sample_dataset):
    # s1 đúng, s2 sai -> aAcc = 3/4 (0.75), nhưng fig1 và pair1 bị tính sai -> fAcc = 0.5, qAcc = 0.5
    preds = [
        {"sample_id": "s1", "prediction": "Yes", "status": "COMPLETED", "run_config_hash": "h1"},
        {"sample_id": "s2", "prediction": "Yes", "status": "COMPLETED", "run_config_hash": "h1"},
        {"sample_id": "s3", "prediction": "Yes", "status": "COMPLETED", "run_config_hash": "h1"},
        {"sample_id": "s4", "prediction": "Yes", "status": "COMPLETED", "run_config_hash": "h1"},
    ]
    metrics, errors = score_hallusionbench(preds, sample_dataset)
    assert metrics["metrics"]["aAcc"] == 0.75
    assert metrics["metrics"]["fAcc"] == 0.5
    assert metrics["metrics"]["qAcc"] == 0.5
    assert len(errors) == 0


def test_technical_error_excluded_from_denominator(sample_dataset):
    # s4 bị lỗi CUDA OOM -> chỉ chấm trên 3 mẫu COMPLETED, ghi s4 vào errors
    preds = [
        {"sample_id": "s1", "prediction": "Yes", "status": "COMPLETED", "run_config_hash": "h1"},
        {"sample_id": "s2", "prediction": "No", "status": "COMPLETED", "run_config_hash": "h1"},
        {"sample_id": "s3", "prediction": "Yes", "status": "COMPLETED", "run_config_hash": "h1"},
        {"sample_id": "s4", "prediction": None, "status": "FAILED", "run_config_hash": "h1"},
    ]
    metrics, errors = score_hallusionbench(preds, sample_dataset)
    assert metrics["counts"]["scored_questions"] == 3
    assert metrics["metrics"]["aAcc"] == 1.0
    assert metrics["counts"]["incomplete_pairs_skipped"] == 1
    assert len(errors) == 1
    assert errors[0]["sample_id"] == "s4"


def test_export_resume_refuses_overwrite_completed(tmp_path, sample_dataset):
    batch1 = [{"sample_id": "s1", "prediction": "Yes", "status": "COMPLETED", "run_config_hash": "h1"}]
    export_hallusionbench_artifacts(batch1, sample_dataset, tmp_path)

    # Cố tình ghi đè s1 đã COMPLETED với câu trả lời khác -> phải báo lỗi
    conflict_batch = [{"sample_id": "s1", "prediction": "No", "status": "COMPLETED", "run_config_hash": "h1"}]
    with pytest.raises(ValueError, match="Refusing to overwrite completed samples"):
        export_hallusionbench_artifacts(conflict_batch, sample_dataset, tmp_path)