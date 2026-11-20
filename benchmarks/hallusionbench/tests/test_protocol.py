"""CPU-only unit tests for HallusionBench Member B protocol & inference runner."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.hallusionbench.inference import (
    HallusionInferenceRunner,
    HallusionRunConfig,
    write_blocked_run,
)
from benchmarks.hallusionbench.protocol import (
    HallusionBenchPromptConfig,
    PreparedHallusionInput,
    load_raw_hallusion_json,
    prepare_hallusion_input,
    select_pilot_records,
)
from vlm_benchmark.results import read_jsonl, validate_predictions


class FakeModel:
    """Deterministic fake VLM that records every input it receives."""

    def __init__(self, fail_on_ids: dict[str, Exception] | None = None) -> None:
        self.fail_on_ids = fail_on_ids or {}
        self.calls: list[PreparedHallusionInput] = []

    def generate(
        self,
        prepared: PreparedHallusionInput,
        generation_config: Mapping[str, Any],
    ) -> str:
        self.calls.append(prepared)
        if prepared.sample_id in self.fail_on_ids:
            raise self.fail_on_ids[prepared.sample_id]
        return "Yes" if prepared.visual_input else "No"

    def get_runtime_info(self) -> dict[str, Any]:
        return {
            "resolved_model_revision": "fake-rev-001",
            "resolved_processor_revision": "fake-proc-001",
            "attn_implementation": "sdpa",
            "device_allocation": "cpu",
        }


def _make_sample_image(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Minimal placeholder bytes for path existence tests (no PIL needed in CPU test).
    path.write_bytes(b"fake-png-bytes")
    return path


def _make_run_config(stage: str = "pilot", dtype: str = "float16", **overrides: Any) -> HallusionRunConfig:
    defaults: dict[str, Any] = {
        "experiment_id": "exp_unit_test",
        "run_id": "run_unit_01",
        "model_key": "smolvlm2",
        "model_id": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        "stage": stage,
        "dtype": dtype,
        "max_new_tokens": 64,
    }
    defaults.update(overrides)
    return HallusionRunConfig(**defaults)


def test_prompt_never_leaks_reference_or_gt_answer(tmp_path: Path) -> None:
    img = _make_sample_image(tmp_path / "hallusion_bench" / "VD" / "illusion" / "0_0.png")
    raw_record = {
        "category": "VD",
        "subcategory": "illusion",
        "visual_input": "1",
        "set_id": "0",
        "figure_id": "0",
        "question_id": "0",
        "filename": "./hallusion_bench/VD/illusion/0_0.png",
        "question": "Is the right circle larger than the left circle?",
        "gt_answer": "0",
        "gt_answer_details": "Both circles have the exact same diameter despite surrounding context.",
        "reference_answer": "no",
        "sample_note": "secret_note_do_not_leak",
    }

    prepared = prepare_hallusion_input(raw_record, data_root=tmp_path, verify_image_exists=True)

    assert prepared.visual_input is True
    assert Path(prepared.image_path) == img.resolve()
    assert "Please answer yes or no." in prepared.prompt
    assert "gt_answer" not in prepared.metadata
    assert "gt_answer_details" not in prepared.metadata
    assert "reference_answer" not in prepared.metadata
    assert "sample_note" not in prepared.metadata
    assert "exact same diameter" not in prepared.prompt
    assert "secret_note_do_not_leak" not in prepared.prompt
    # Visual input must include an image slot before text.
    user_content = prepared.messages[-1]["content"]
    assert user_content[0] == {"type": "image"}
    assert user_content[1]["type"] == "text"


def test_visual_input_false_never_attaches_image(tmp_path: Path) -> None:
    record = {
        "sample_id": "hb_text_only_01",
        "question": "Is Paris the capital of France?",
        "visual_input": "0",
        "filename": "./hallusion_bench/should_be_ignored.png",
        "category": "VS",
        "subcategory": "math",
        "set_id": "1",
        "figure_id": "0",
        "question_id": "0",
    }

    prepared = prepare_hallusion_input(record, data_root=tmp_path, verify_image_exists=True)

    assert prepared.visual_input is False
    assert prepared.image_path is None
    user_content = prepared.messages[-1]["content"]
    assert all(item.get("type") != "image" for item in user_content)


def test_visual_input_2_and_nested_hallusion_bench_path(tmp_path: Path) -> None:
    img = _make_sample_image(tmp_path / "hallusion_bench" / "VS" / "chart" / "3_1.png")
    record = {
        "category": "VS",
        "subcategory": "chart",
        "visual_input": "2",
        "set_id": "3",
        "figure_id": "1",
        "question_id": "0",
        "filename": "./VS/chart/3_1.png",
        "question": "According to the chart, does China have the largest population?",
        "gt_answer": "0",
    }

    prepared = prepare_hallusion_input(record, data_root=tmp_path, verify_image_exists=True)
    assert prepared.visual_input is True
    assert Path(prepared.image_path) == img.resolve()


def test_rayguan_hf_data_subfolder_layout(tmp_path: Path) -> None:
    img = _make_sample_image(tmp_path / "data" / "VD" / "figure" / "0_0.png")
    record = {
        "category": "VD",
        "subcategory": "figure",
        "visual_input": "1",
        "set_id": "0",
        "figure_id": "0",
        "question_id": "0",
        "filename": "./VD/figure/0_0.png",
        "question": "Is the arrow pointing left?",
        "gt_answer": "1",
    }

    prepared = prepare_hallusion_input(record, data_root=tmp_path, verify_image_exists=True)
    assert prepared.visual_input is True
    assert Path(prepared.image_path) == img.resolve()


def test_case_insensitive_png_extension_resolution(tmp_path: Path) -> None:
    img = _make_sample_image(tmp_path / "data" / "VS" / "chart" / "6_1.PNG")
    record = {
        "category": "VS",
        "subcategory": "chart",
        "visual_input": "1",
        "set_id": "6",
        "figure_id": "1",
        "question_id": "0",
        "filename": "./VS/chart/6_1.png",
        "question": "Does the bar chart show an increase?",
        "gt_answer": "1",
    }

    prepared = prepare_hallusion_input(record, data_root=tmp_path, verify_image_exists=True)
    assert prepared.visual_input is True
    assert Path(prepared.image_path).resolve() == img.resolve()


def test_visual_input_true_missing_image_raises_file_not_found(tmp_path: Path) -> None:
    record = {
        "sample_id": "hb_missing_img",
        "question": "Does the chart show growth?",
        "visual_input": True,
        "image_path": str(tmp_path / "does_not_exist.png"),
    }

    with pytest.raises(FileNotFoundError):
        prepare_hallusion_input(record, verify_image_exists=True)


def test_select_pilot_records_preserves_complete_groups() -> None:
    records = [
        {"category": "VD", "subcategory": "illusion", "set_id": "0", "figure_id": "0", "question_id": "0", "visual_input": "1", "question": "Q1?"},
        {"category": "VD", "subcategory": "illusion", "set_id": "0", "figure_id": "1", "question_id": "1", "visual_input": "1", "question": "Q2?"},
        {"category": "VS", "subcategory": "map", "set_id": "1", "figure_id": "0", "question_id": "0", "visual_input": "0", "question": "Q3?"},
        {"category": "VS", "subcategory": "map", "set_id": "1", "figure_id": "0", "question_id": "1", "visual_input": "1", "question": "Q4?"},
        {"category": "VD", "subcategory": "chart", "set_id": "2", "figure_id": "0", "question_id": "0", "visual_input": "1", "question": "Q5?"},
    ]

    pilot = select_pilot_records(records, max_groups=2, include_text_only=True)
    # Must contain the text-only group (set 1: 2 items) + set 0 (2 items) = 4 items, never half a group.
    assert len(pilot) == 4
    set_ids = {r["set_id"] for r in pilot}
    assert set_ids == {"0", "1"}


def test_load_raw_hallusion_json_creates_unique_sample_ids(tmp_path: Path) -> None:
    _make_sample_image(tmp_path / "hallusion_bench" / "a.png")
    raw_json = tmp_path / "HallusionBench.json"
    raw_json.write_text(
        json.dumps(
            [
                {
                    "category": "VD",
                    "subcategory": "illusion",
                    "visual_input": "1",
                    "set_id": "0",
                    "figure_id": "0",
                    "question_id": "0",
                    "filename": "./hallusion_bench/a.png",
                    "question": "Q1?",
                    "gt_answer": "1",
                },
                {
                    "category": "VD",
                    "subcategory": "illusion",
                    "visual_input": "0",
                    "set_id": "0",
                    "figure_id": "0",
                    "question_id": "0",
                    "filename": None,
                    "question": "Q1 without image?",
                    "gt_answer": "1",
                },
            ]
        ),
        encoding="utf-8",
    )

    records = load_raw_hallusion_json(raw_json, data_root=tmp_path, verify_image_exists=True)
    assert len(records) == 2
    assert records[0]["sample_id"] != records[1]["sample_id"]
    assert records[0]["visual_input"] is True
    assert records[1]["visual_input"] is False


def test_runner_exports_all_seven_valid_artifacts(tmp_path: Path) -> None:
    img = _make_sample_image(tmp_path / "img.png")
    records = [
        {
            "sample_id": "s1",
            "question": "Is the object red?",
            "visual_input": True,
            "image_path": str(img),
            "category": "VD",
            "subcategory": "color",
            "set_id": "0",
            "figure_id": "0",
            "question_id": "0",
            "reference_answer": "yes",
        },
        {
            "sample_id": "s2",
            "question": "Is water wet?",
            "visual_input": False,
            "image_path": None,
            "category": "VS",
            "subcategory": "fact",
            "set_id": "1",
            "figure_id": "0",
            "question_id": "0",
            "reference_answer": "yes",
        },
    ]

    run_dir = tmp_path / "results" / "exp_unit_test" / "run_unit_01"
    runner = HallusionInferenceRunner(
        run_config=_make_run_config(),
        output_dir=run_dir,
        verify_image_exists=True,
    )
    fake_model = FakeModel()
    status = runner.run(
        records,
        fake_model,
        hardware_info={"cuda_available": False, "bf16_supported": False},
    )

    assert status["status"] == "COMPLETED"
    for fname in (
        "predictions.jsonl",
        "metrics.json",
        "run_config.yaml",
        "run_metadata.json",
        "errors.jsonl",
        "run_status.json",
        "manifest.json",
    ):
        assert (run_dir / fname).exists(), f"Missing artifact {fname}"

    preds = read_jsonl(run_dir / "predictions.jsonl")
    assert len(preds) == 2
    assert validate_predictions(preds) == []
    assert preds[0]["prediction"] == "Yes"
    assert preds[0]["raw_prediction"] == "Yes"
    assert preds[1]["prediction"] == "No"

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["experiment_id"] == "exp_unit_test"
    assert manifest["run_id"] == "run_unit_01"
    assert "predictions.jsonl" in manifest["files"]

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["status"] == "PENDING_SCORING"
    assert metrics["scores"] is None


def test_runner_resume_skips_completed_and_retries_failed(tmp_path: Path) -> None:
    records = [
        {"sample_id": "s1", "question": "Q1?", "visual_input": False},
        {"sample_id": "s2", "question": "Q2?", "visual_input": False},
    ]
    run_dir = tmp_path / "run_resume"
    cfg = _make_run_config()

    # First run: s2 fails with simulated CUDA OOM.
    runner1 = HallusionInferenceRunner(cfg, run_dir, verify_image_exists=False)
    model1 = FakeModel(fail_on_ids={"s2": RuntimeError("CUDA out of memory")})
    status1 = runner1.run(records, model1, hardware_info={"cuda_available": False, "bf16_supported": False})
    assert status1["status"] == "FAILED"
    assert status1["completed_samples"] == 1
    assert status1["failed_samples"] == 1

    preds_after_1 = read_jsonl(run_dir / "predictions.jsonl")
    assert validate_predictions(preds_after_1) == []
    by_id = {r["sample_id"]: r for r in preds_after_1}
    assert by_id["s2"]["prediction"] is None
    assert by_id["s2"]["status"] == "FAILED"
    assert by_id["s2"]["error_type"] == "OOM_ERROR"
    assert len(read_jsonl(run_dir / "errors.jsonl")) == 1

    # Second run (resume): s1 MUST NOT be re-executed; s2 succeeds.
    runner2 = HallusionInferenceRunner(cfg, run_dir, verify_image_exists=False)
    model2 = FakeModel()
    status2 = runner2.run(records, model2, hardware_info={"cuda_available": False, "bf16_supported": False}, resume=True)
    assert status2["status"] == "COMPLETED"
    assert [c.sample_id for c in model2.calls] == ["s2"]


def test_runner_rejects_resume_with_different_config_hash(tmp_path: Path) -> None:
    records = [{"sample_id": "s1", "question": "Q1?", "visual_input": False}]
    run_dir = tmp_path / "run_conflict"

    runner1 = HallusionInferenceRunner(_make_run_config(max_new_tokens=64), run_dir, verify_image_exists=False)
    runner1.run(records, FakeModel(), hardware_info={"cuda_available": False, "bf16_supported": False})

    runner2 = HallusionInferenceRunner(_make_run_config(max_new_tokens=128), run_dir, verify_image_exists=False)
    with pytest.raises(ValueError, match="incompatible run_config_hash"):
        runner2.run(records, FakeModel(), hardware_info={"cuda_available": False, "bf16_supported": False}, resume=True)


def test_policy_blocks_bf16_on_t4_quantization_and_fallback(tmp_path: Path) -> None:
    t4_hardware = {"cuda_available": True, "bf16_supported": False, "gpu": ["Tesla T4"]}

    cfg_bf16 = _make_run_config(dtype="bfloat16")
    errors_t4 = cfg_bf16.validate_for_stage(hardware_info=t4_hardware)
    assert any("native BF16" in err for err in errors_t4)

    cfg_quant = _make_run_config(quantization="int4")
    assert any("quantization" in err for err in cfg_quant.validate_for_stage())

    cfg_fallback = _make_run_config(allow_dtype_fallback=True)
    assert any("fallback" in err for err in cfg_fallback.validate_for_stage())

    # Running with bfloat16 on T4 hardware writes an honest BLOCKED run without calling model.
    run_dir = tmp_path / "blocked_t4"
    runner = HallusionInferenceRunner(cfg_bf16, run_dir, verify_image_exists=False)
    fake_model = FakeModel()
    status = runner.run(
        [{"sample_id": "s1", "question": "Q1?", "visual_input": False}],
        fake_model,
        hardware_info=t4_hardware,
    )
    assert status["status"] == "BLOCKED"
    assert fake_model.calls == []
    assert (run_dir / "manifest.json").exists()


def test_official_stage_is_blocked_without_locked_revisions(tmp_path: Path) -> None:
    cfg = _make_run_config(stage="official", dtype="bfloat16")
    errors = cfg.validate_for_stage(
        hardware_info={"cuda_available": True, "bf16_supported": True},
        repo_root=Path(__file__).resolve().parents[3],
    )
    assert any("model_revision" in err for err in errors)
    assert any("dataset_revision" in err for err in errors)


def test_write_blocked_run_creates_valid_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "blocked_manual"
    cfg = _make_run_config(model_key="phi4_multimodal", model_id="microsoft/Phi-4-multimodal-instruct")
    payload = write_blocked_run(
        run_dir,
        cfg,
        reasons=["Phi-4 OOM on single T4 16GB in FP16 without quantization."],
        hardware_info={"cuda_available": True, "bf16_supported": False},
    )
    assert payload["status"] == "BLOCKED"
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "BLOCKED"
    assert manifest["experiment_id"] == cfg.experiment_id
    assert manifest["run_id"] == cfg.run_id
