"""CPU-only unit tests for HallusionBench scoring, subgroup breakdown, and resume safety."""

import pytest
from benchmarks.hallusionbench.scoring import (
    export_hallusionbench_artifacts,
    parse_yes_no,
    score_hallusionbench,
)


@pytest.fixture
def sample_dataset() -> dict[str, dict]:
    return {
        "s1": {
            "reference_answer": "yes",
            "category": "VD",
            "difficulty": "easy",
            "figure_key": "fig1",
            "pair_key": "pair1",
        },
        "s2": {
            "reference_answer": "no",
            "category": "VD",
            "difficulty": "hard",
            "figure_key": "fig1",
            "pair_key": "pair1",
        },
        "s3": {
            "reference_answer": "yes",
            "category": "VS",
            "difficulty": "easy",
            "figure_key": "fig2",
            "pair_key": "pair2",
        },
        "s4": {
            "reference_answer": "yes",
            "category": "VS",
            "difficulty": "hard",
            "figure_key": "fig2",
            "pair_key": "pair2",
        },
    }


def test_parse_yes_no_variants():
    assert parse_yes_no("Yes") == "yes"
    assert parse_yes_no("No, there is no cat.") == "no"
    assert parse_yes_no("Based on the chart, the answer is yes.") == "yes"
    assert parse_yes_no("Yes and no, it depends.") is None
    assert parse_yes_no("Yes/No") is None
    assert parse_yes_no("") is None


def test_partial_pair_failure_and_subgroups(sample_dataset):
    # s1 đúng (VD, easy), s2 sai (VD, hard), s3 & s4 đúng (VS, easy & hard)
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

    # Kiểm tra phân nhóm VD vs VS
    assert metrics["by_category"]["VD"]["aAcc"] == 0.5
    assert metrics["by_category"]["VD"]["qAcc"] == 0.0
    assert metrics["by_category"]["VS"]["aAcc"] == 1.0
    assert metrics["by_category"]["VS"]["qAcc"] == 1.0

    # Kiểm tra phân nhóm easy vs hard
    assert metrics["by_difficulty"]["easy"]["aAcc"] == 1.0
    assert metrics["by_difficulty"]["hard"]["aAcc"] == 0.5


def test_fallback_group_keys_from_raw_fields():
    raw_dataset = {
        "q1": {
            "reference_answer": "yes",
            "category": "VD",
            "subcategory": "illusion",
            "set_id": "10",
            "figure_id": "1",
            "question_id": "1",
        },
        "q2": {
            "reference_answer": "no",
            "category": "VD",
            "subcategory": "illusion",
            "set_id": "10",
            "figure_id": "2",
            "question_id": "1",
        },
    }
    preds = [
        {"sample_id": "q1", "prediction": "Yes", "status": "COMPLETED", "run_config_hash": "h1"},
        {"sample_id": "q2", "prediction": "No", "status": "COMPLETED", "run_config_hash": "h1"},
    ]
    metrics, _ = score_hallusionbench(preds, raw_dataset)
    assert metrics["metrics"]["aAcc"] == 1.0
    assert metrics["metrics"]["fAcc"] == 1.0
    assert metrics["metrics"]["qAcc"] == 1.0


def test_technical_error_excluded_from_denominator(sample_dataset):
    preds = [
        {"sample_id": "s1", "prediction": "Yes", "status": "COMPLETED", "run_config_hash": "h1"},
        {"sample_id": "s2", "prediction": "No", "status": "COMPLETED", "run_config_hash": "h1"},
        {"sample_id": "s3", "prediction": "Yes", "status": "COMPLETED", "run_config_hash": "h1"},
        {"sample_id": "s4", "prediction": None, "status": "FAILED", "run_config_hash": "h1"},
    ]
    metrics, errors = score_hallusionbench(preds, sample_dataset)
    assert metrics["summary_counts"]["scored_questions"] == 3
    assert metrics["metrics"]["aAcc"] == 1.0
    assert metrics["summary_counts"]["incomplete_pairs_skipped"] == 1
    assert len(errors) == 1
    assert errors[0]["sample_id"] == "s4"


def test_duplicate_sample_id_raises_error(sample_dataset):
    preds = [
        {"sample_id": "s1", "prediction": "Yes", "status": "COMPLETED", "run_config_hash": "h1"},
        {"sample_id": "s1", "prediction": "No", "status": "COMPLETED", "run_config_hash": "h1"},
    ]
    with pytest.raises(ValueError, match="Duplicate sample_id"):
        score_hallusionbench(preds, sample_dataset)


def test_export_resume_refuses_overwrite_completed(tmp_path, sample_dataset):
    batch1 = [{"sample_id": "s1", "prediction": "Yes", "status": "COMPLETED", "run_config_hash": "h1"}]
    export_hallusionbench_artifacts(batch1, sample_dataset, tmp_path)

    conflict_batch = [{"sample_id": "s1", "prediction": "No", "status": "COMPLETED", "run_config_hash": "h1"}]
    with pytest.raises(ValueError, match="Refusing to overwrite completed samples"):
        export_hallusionbench_artifacts(conflict_batch, sample_dataset, tmp_path)