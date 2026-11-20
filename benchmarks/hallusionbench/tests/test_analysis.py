"""CPU-only unit tests for HallusionBench error analysis and before/after comparison."""

import pytest
from benchmarks.hallusionbench.analysis import (
    compare_before_after,
    extract_error_cases,
    find_cross_model_disagreements,
    render_error_table_markdown,
)


@pytest.fixture
def sample_dataset() -> dict[str, dict]:
    return {
        "s1": {
            "reference_answer": "yes",
            "category": "VD",
            "visual_input": True,
            "image_path": "images/vd_01.png",
            "question": "Is the left circle larger than the right circle?",
            "figure_key": "fig1",
            "pair_key": "pair1",
        },
        "s2": {
            "reference_answer": "no",
            "category": "VD",
            "visual_input": True,
            "image_path": "images/vd_02.png",
            "question": "Is the left circle larger than the right circle?",
            "figure_key": "fig2",
            "pair_key": "pair1",
        },
        "s3": {
            "reference_answer": "yes",
            "category": "VS",
            "visual_input": True,
            "image_path": "images/vs_01.png",
            "question": "Does the bar chart show growth in 2023?",
            "figure_key": "fig3",
            "pair_key": "pair2",
        },
    }


def test_extract_error_cases_distinguishes_error_types(sample_dataset):
    preds = [
        # s1 đúng -> không nằm trong danh sách lỗi
        {"sample_id": "s1", "prediction": "Yes", "status": "COMPLETED", "run_config_hash": "h1"},
        # s2 trả lời sai -> chỉ gắn nhãn sơ bộ 'incorrect_prediction', không tự kết luận 'hallucination'
        {"sample_id": "s2", "prediction": "Yes, it looks bigger.", "status": "COMPLETED", "run_config_hash": "h1"},
        # s3 lỗi định dạng -> 'format_violation'
        {"sample_id": "s3", "prediction": "Yes and no, hard to tell.", "status": "COMPLETED", "run_config_hash": "h1"},
        # s4 lỗi kỹ thuật -> 'technical_error'
        {"sample_id": "s4", "prediction": None, "status": "FAILED", "error": "CUDA OOM", "run_config_hash": "h1"},
    ]
    cases = extract_error_cases(preds, sample_dataset, model_id="smolvlm2", run_id="run_001")
    assert len(cases) == 3

    by_id = {c["sample_id"]: c for c in cases}
    assert by_id["s2"]["preliminary_error_type"] == "incorrect_prediction"
    assert by_id["s2"]["verified_error_type"] is None
    assert by_id["s3"]["preliminary_error_type"] == "format_violation"
    assert by_id["s4"]["preliminary_error_type"] == "technical_error"


def test_find_cross_model_disagreements(sample_dataset):
    runs = {
        "smolvlm2": [
            {"sample_id": "s1", "prediction": "Yes", "status": "COMPLETED", "run_config_hash": "h1"},
            {"sample_id": "s2", "prediction": "Yes", "status": "COMPLETED", "run_config_hash": "h1"},
        ],
        "qwen3_vl": [
            {"sample_id": "s1", "prediction": "Yes", "status": "COMPLETED", "run_config_hash": "h1"},
            {"sample_id": "s2", "prediction": "No", "status": "COMPLETED", "run_config_hash": "h1"},
        ],
        "phi4_multimodal": [
            {"sample_id": "s1", "prediction": "Yes", "status": "COMPLETED", "run_config_hash": "h1"},
            {"sample_id": "s2", "prediction": "Unclear", "status": "COMPLETED", "run_config_hash": "h1"},
        ],
    }
    disagreements = find_cross_model_disagreements(runs, sample_dataset)
    # s1 cả 3 model đều trả lời đúng 'yes' -> không đưa vào; s2 bất đồng -> đưa vào
    assert len(disagreements) == 1
    assert disagreements[0]["sample_id"] == "s2"
    assert disagreements[0]["model_outcomes"]["qwen3_vl"]["is_correct"] is True
    assert disagreements[0]["model_outcomes"]["smolvlm2"]["is_correct"] is False


def test_compare_before_after_tracks_improvements_and_regressions(sample_dataset):
    baseline = [
        {"sample_id": "s1", "prediction": "Yes", "status": "COMPLETED", "run_config_hash": "h1"},  # Đúng -> Sai (regressed)
        {"sample_id": "s2", "prediction": "Yes", "status": "COMPLETED", "run_config_hash": "h1"},  # Sai -> Đúng (improved)
        {"sample_id": "s3", "prediction": "Yes", "status": "COMPLETED", "run_config_hash": "h1"},  # Đúng -> Đúng
    ]
    improved = [
        {"sample_id": "s1", "prediction": "No", "status": "COMPLETED", "run_config_hash": "h2"},
        {"sample_id": "s2", "prediction": "No", "status": "COMPLETED", "run_config_hash": "h2"},
        {"sample_id": "s3", "prediction": "Yes", "status": "COMPLETED", "run_config_hash": "h2"},
    ]
    diff = compare_before_after(baseline, improved, sample_dataset)
    assert diff["compared_completed_samples"] == 3
    assert diff["improved_sample_ids"] == ["s2"]
    assert diff["regressed_sample_ids"] == ["s1"]
    assert diff["delta_aAcc"] == 0.0


def test_render_error_table_markdown(sample_dataset):
    preds = [
        {"sample_id": "s2", "prediction": "Yes", "status": "COMPLETED", "run_config_hash": "h1"},
    ]
    cases = extract_error_cases(preds, sample_dataset, model_id="smolvlm2", run_id="run_001")
    md = render_error_table_markdown(cases)
    assert "| `run_001` | s2 (pair1) | `smolvlm2` |" in md
    assert "`incorrect_prediction`" in md