"""End-to-end CPU-only integration tests covering HallusionBench A -> B -> C + BaseBenchmark.

Compatible with both `python -m pytest` and stdlib `python -m unittest`.
"""

from __future__ import annotations

import base64
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.hallusionbench import (  # noqa: E402
    HallusionBenchBenchmark,
    HallusionBenchDataset,
    compare_before_after,
    export_hallusionbench_artifacts,
    extract_error_cases,
    filter_by_pilot_ids,
    find_cross_model_disagreements,
    parse_yes_no,
    render_error_table_markdown,
    score_hallusionbench,
    score_run_directory,
    select_frozen_pilot,
    summarize_error_cases,
)
from benchmarks.hallusionbench.analysis import main as analysis_cli_main  # noqa: E402
from benchmarks.hallusionbench.dataset import main as dataset_cli_main  # noqa: E402
from benchmarks.hallusionbench.inference import (  # noqa: E402
    HallusionInferenceRunner,
    HallusionRunConfig,
    main as inference_cli_main,
)
from benchmarks.hallusionbench.protocol import PreparedHallusionInput  # noqa: E402
from benchmarks.hallusionbench.scoring import main as scoring_cli_main  # noqa: E402
from vlm_benchmark.results import read_jsonl, validate_predictions  # noqa: E402

TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
)


def _write_png(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(TINY_PNG)
    return path


def _build_fixture_dataset(root: Path) -> Path:
    """Create a synthetic dataset with VD (easy=1, hard=2) and VS (easy=0 text-only, hard=1 visual)."""
    _write_png(root / "data" / "VD" / "illusion" / "0_0.png")
    _write_png(root / "data" / "VD" / "illusion" / "0_1.png")
    _write_png(root / "data" / "VS" / "chart" / "3_1.PNG")

    records = [
        # Group 1 (VD): 2 figures, 1 complete question pair (easy visual=1 + hard visual=2)
        {
            "category": "VD",
            "subcategory": "illusion",
            "visual_input": "1",
            "set_id": "0",
            "figure_id": "0",
            "question_id": "0",
            "filename": "./VD/illusion/0_0.png",
            "question": "Is the right circle larger than the left circle?",
            "gt_answer": "0",
        },
        {
            "category": "VD",
            "subcategory": "illusion",
            "visual_input": "2",
            "set_id": "0",
            "figure_id": "1",
            "question_id": "0",
            "filename": "./VD/illusion/0_1.png",
            "question": "Is the right circle larger than the left circle?",
            "gt_answer": "1",
        },
        # Group 2 (VS): text-only control (visual=0, easy, figure_key=None) + visual figure (visual=1, hard)
        {
            "category": "VS",
            "subcategory": "chart",
            "visual_input": "0",
            "set_id": "3",
            "figure_id": "0",
            "question_id": "0",
            "filename": None,
            "question": "Is Tokyo in Japan?",
            "gt_answer": "1",
        },
        {
            "category": "VS",
            "subcategory": "chart",
            "visual_input": "1",
            "set_id": "3",
            "figure_id": "1",
            "question_id": "0",
            "filename": "./VS/chart/3_1.png",
            "question": "According to the chart, does the value increase?",
            "gt_answer": "0",
        },
    ]
    json_path = root / "HallusionBench.json"
    json_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return json_path


class ScriptedModel:
    """Deterministic test model returning scripted responses or raising errors by sample_id."""

    def __init__(
        self,
        responses: dict[str, str],
        errors: dict[str, Exception] | None = None,
    ) -> None:
        self.responses = responses
        self.errors = errors or {}

    def generate(self, prepared: PreparedHallusionInput, generation_config: Mapping[str, Any]) -> str:
        if prepared.sample_id in self.errors:
            raise self.errors[prepared.sample_id]
        return self.responses[prepared.sample_id]

    def get_runtime_info(self) -> dict[str, Any]:
        return {
            "resolved_model_revision": "rev-test-001",
            "resolved_processor_revision": "proc-test-001",
            "attn_implementation": "sdpa",
            "device_allocation": "cpu",
        }


class TestHallusionBenchPipelineE2E(unittest.TestCase):
    def test_member_c_scoring_and_text_only_figure_exclusion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            json_path = _build_fixture_dataset(root)
            dataset = HallusionBenchDataset(json_path, data_root=root)
            index = dataset.load_index(verify_images=True)

            # Text-only sample must have figure_key=None so fAcc only scores the 3 actual figures,
            # while pair_key is preserved so qAcc still scores both complete pairs!
            self.assertIsNone(index["catVS_subchart_set3_fig0_q0_vis0"]["figure_key"])
            self.assertIsNotNone(index["catVS_subchart_set3_fig0_q0_vis0"]["pair_key"])
            self.assertEqual(index["catVD_subillusion_set0_fig0_q0_vis1"]["difficulty"], "easy")
            self.assertEqual(index["catVD_subillusion_set0_fig1_q0_vis2"]["difficulty"], "hard")
            self.assertEqual(index["catVS_subchart_set3_fig0_q0_vis0"]["difficulty"], "easy")
            self.assertEqual(index["catVS_subchart_set3_fig1_q0_vis1"]["difficulty"], "hard")

            # Parse edge cases
            self.assertEqual(parse_yes_no("Yes, it is."), "yes")
            self.assertEqual(parse_yes_no("No, there is no circle."), "no")
            self.assertIsNone(parse_yes_no("Yes and no, unclear."))
            self.assertIsNone(parse_yes_no("Maybe"))

            preds = [
                # VD pair: fig0 correct (no), fig1 wrong (no instead of yes) -> pair fails
                {"sample_id": "catVD_subillusion_set0_fig0_q0_vis1", "prediction": "No", "status": "COMPLETED", "run_config_hash": "h1"},
                {"sample_id": "catVD_subillusion_set0_fig1_q0_vis2", "prediction": "No", "status": "COMPLETED", "run_config_hash": "h1"},
                # VS pair: text-only correct (yes), fig1 correct (no) -> pair passes!
                {"sample_id": "catVS_subchart_set3_fig0_q0_vis0", "prediction": "Yes", "status": "COMPLETED", "run_config_hash": "h1"},
                {"sample_id": "catVS_subchart_set3_fig1_q0_vis1", "prediction": "No, it decreases.", "status": "COMPLETED", "run_config_hash": "h1"},
            ]
            metrics, errors = score_hallusionbench(preds, index)
            self.assertEqual(errors, [])
            self.assertEqual(metrics["summary_counts"]["scored_questions"], 4)
            # Only 3 visual figures (VD fig0, VD fig1, VS fig1) — text-only excluded from fAcc!
            self.assertEqual(metrics["summary_counts"]["scored_figures"], 3)
            self.assertEqual(metrics["summary_counts"]["scored_complete_pairs"], 2)
            self.assertEqual(metrics["metrics"]["aAcc"], 0.75)
            self.assertAlmostEqual(metrics["metrics"]["fAcc"], round(2 / 3, 6))
            self.assertEqual(metrics["metrics"]["qAcc"], 0.5)
            self.assertEqual(metrics["by_category"]["VD"]["aAcc"], 0.5)
            self.assertEqual(metrics["by_category"]["VS"]["aAcc"], 1.0)
            self.assertEqual(metrics["by_difficulty"]["easy"]["aAcc"], 1.0)
            self.assertEqual(metrics["by_difficulty"]["hard"]["aAcc"], 0.5)
            self.assertEqual(metrics["by_visual_input"]["0"]["aAcc"], 1.0)

            # Resume safety: refusing to overwrite COMPLETED sample
            out_dir = root / "export_test"
            export_hallusionbench_artifacts(preds[:1], index, out_dir)
            with self.assertRaises(ValueError):
                export_hallusionbench_artifacts(
                    [{"sample_id": "catVD_subillusion_set0_fig0_q0_vis1", "prediction": "Yes", "status": "COMPLETED", "run_config_hash": "h1"}],
                    index,
                    out_dir,
                )

    def test_full_a_to_b_to_c_pipeline_and_repo_checkers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            json_path = _build_fixture_dataset(root)
            pilot_path = root / "pilot_ids.json"

            # --- STEP A: Audit & export frozen pilot ---
            self.assertEqual(
                dataset_cli_main(["--mode", "audit", "--dataset-json", str(json_path), "--data-root", str(root)]),
                0,
            )
            self.assertEqual(
                dataset_cli_main(
                    [
                        "--mode",
                        "export-pilot",
                        "--dataset-json",
                        str(json_path),
                        "--data-root",
                        str(root),
                        "--target-groups",
                        "2",
                        "--output",
                        str(pilot_path),
                    ]
                ),
                0,
            )
            pilot_manifest = json.loads(pilot_path.read_text(encoding="utf-8"))
            self.assertEqual(pilot_manifest["sample_count"], 4)

            # --- STEP B (CLI dry-run with Member A's --pilot-file) ---
            dryrun_root = root / "results_dryrun"
            self.assertEqual(
                inference_cli_main(
                    [
                        "--mode",
                        "dry-run",
                        "--model-key",
                        "smolvlm2",
                        "--stage",
                        "pilot",
                        "--dtype",
                        "float16",
                        "--dataset-json",
                        str(json_path),
                        "--data-root",
                        str(root),
                        "--pilot-file",
                        str(pilot_path),
                        "--experiment-id",
                        "exp_e2e",
                        "--run-id",
                        "run_dry_01",
                        "--output-root",
                        str(dryrun_root),
                    ]
                ),
                0,
            )

            # --- STEP B (Runner with scripted model: correct, wrong, format_violation, and OOM technical error) ---
            dataset = HallusionBenchDataset(json_path, data_root=root)
            samples = filter_by_pilot_ids(
                dataset.load(verify_images=True),
                pilot_manifest["sample_ids"],
                require_complete_groups=True,
            )
            model_safe_records = [s.to_model_safe_dict() for s in samples]

            run_a_dir = root / "results" / "exp_e2e" / "run_smolvlm2"
            cfg_a = HallusionRunConfig(
                experiment_id="exp_e2e",
                run_id="run_smolvlm2",
                model_key="smolvlm2",
                model_id="HuggingFaceTB/SmolVLM2-2.2B-Instruct",
                stage="pilot",
                dtype="float16",
                max_new_tokens=64,
            )
            runner_a = HallusionInferenceRunner(cfg_a, run_a_dir, data_root=root, verify_image_exists=True)
            model_a = ScriptedModel(
                responses={
                    "catVD_subillusion_set0_fig0_q0_vis1": "No",                     # correct
                    "catVD_subillusion_set0_fig1_q0_vis2": "No, looks identical.",   # incorrect_prediction
                    "catVS_subchart_set3_fig0_q0_vis0": "Yes and no, unsure.",       # format_violation
                },
                errors={
                    "catVS_subchart_set3_fig1_q0_vis1": RuntimeError("CUDA out of memory"),  # technical_error
                },
            )
            status_a = runner_a.run(
                model_safe_records,
                model_a,
                hardware_info={"cuda_available": False, "bf16_supported": False},
            )
            self.assertEqual(status_a["status"], "FAILED")
            self.assertEqual(status_a["completed_samples"], 3)
            self.assertEqual(status_a["failed_samples"], 1)

            # Second model run (Qwen3-VL) for cross-model disagreement & before/after comparison
            run_b_dir = root / "results" / "exp_e2e" / "run_qwen3"
            cfg_b = HallusionRunConfig(
                experiment_id="exp_e2e",
                run_id="run_qwen3",
                model_key="qwen3_vl",
                model_id="Qwen/Qwen3-VL-4B-Instruct",
                stage="pilot",
                dtype="float16",
                max_new_tokens=64,
            )
            runner_b = HallusionInferenceRunner(cfg_b, run_b_dir, data_root=root, verify_image_exists=True)
            model_b = ScriptedModel(
                responses={
                    "catVD_subillusion_set0_fig0_q0_vis1": "No",
                    "catVD_subillusion_set0_fig1_q0_vis2": "Yes",
                    "catVS_subchart_set3_fig0_q0_vis0": "Yes",
                    "catVS_subchart_set3_fig1_q0_vis1": "No",
                }
            )
            status_b = runner_b.run(
                model_safe_records,
                model_b,
                hardware_info={"cuda_available": False, "bf16_supported": False},
            )
            self.assertEqual(status_b["status"], "COMPLETED")

            # --- STEP C: Score both run directories via scoring CLI ---
            self.assertEqual(
                scoring_cli_main(
                    ["--run-dir", str(run_a_dir), "--dataset-json", str(json_path), "--data-root", str(root)]
                ),
                0,
            )
            self.assertEqual(
                scoring_cli_main(
                    ["--run-dir", str(run_b_dir), "--dataset-json", str(json_path), "--data-root", str(root)]
                ),
                0,
            )

            metrics_a = json.loads((run_a_dir / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics_a["summary_counts"]["scored_questions"], 3)
            self.assertEqual(metrics_a["summary_counts"]["error_records"], 2)  # 1 technical_error + 1 format_violation
            self.assertEqual(metrics_a["run_identity"]["run_id"], "run_smolvlm2")

            metrics_b = json.loads((run_b_dir / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics_b["metrics"]["aAcc"], 1.0)
            self.assertEqual(metrics_b["metrics"]["fAcc"], 1.0)
            self.assertEqual(metrics_b["metrics"]["qAcc"], 1.0)

            # Verify shared repo scripts pass on BOTH scored runs!
            for rdir in (run_a_dir, run_b_dir):
                res_check = subprocess.run(
                    [sys.executable, "-B", str(REPO_ROOT / "scripts" / "check_results.py"), str(rdir / "predictions.jsonl")],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(res_check.returncode, 0, res_check.stdout + res_check.stderr)
                man_check = subprocess.run(
                    [sys.executable, "-B", str(REPO_ROOT / "scripts" / "check_manifest.py"), str(rdir)],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(man_check.returncode, 0, man_check.stdout + man_check.stderr)

            # --- STEP C: Error extraction, cross-model disagreement & before/after CLI ---
            analysis_dir = run_a_dir / "analysis"
            self.assertEqual(
                analysis_cli_main(
                    [
                        "--mode",
                        "extract",
                        "--run-dir",
                        str(run_a_dir),
                        "--dataset-json",
                        str(json_path),
                        "--data-root",
                        str(root),
                        "--output-dir",
                        str(analysis_dir),
                    ]
                ),
                0,
            )
            cases = json.loads((analysis_dir / "error_cases.json").read_text(encoding="utf-8"))
            summary = summarize_error_cases(cases)
            self.assertEqual(summary["total_error_cases"], 3)
            self.assertEqual(summary["by_preliminary_error_type"]["incorrect_prediction"], 1)
            self.assertEqual(summary["by_preliminary_error_type"]["format_violation"], 1)
            self.assertEqual(summary["by_preliminary_error_type"]["technical_error"], 1)
            md_table = (analysis_dir / "error_cases.md").read_text(encoding="utf-8")
            self.assertIn("`run_smolvlm2`", md_table)
            self.assertIn("`incorrect_prediction`", md_table)

            disagree_out = root / "disagreements.json"
            self.assertEqual(
                analysis_cli_main(
                    [
                        "--mode",
                        "disagreements",
                        "--run-dirs",
                        f"smolvlm2={run_a_dir}",
                        f"qwen3_vl={run_b_dir}",
                        "--dataset-json",
                        str(json_path),
                        "--data-root",
                        str(root),
                        "--output",
                        str(disagree_out),
                    ]
                ),
                0,
            )
            disagreements = json.loads(disagree_out.read_text(encoding="utf-8"))
            self.assertEqual(len(disagreements), 2)

            diff_out = root / "before_after.json"
            self.assertEqual(
                analysis_cli_main(
                    [
                        "--mode",
                        "before-after",
                        "--baseline-run-dir",
                        str(run_a_dir),
                        "--improved-run-dir",
                        str(run_b_dir),
                        "--dataset-json",
                        str(json_path),
                        "--data-root",
                        str(root),
                        "--output",
                        str(diff_out),
                    ]
                ),
                0,
            )
            diff = json.loads(diff_out.read_text(encoding="utf-8"))
            self.assertEqual(diff["compared_completed_samples"], 3)
            self.assertEqual(diff["improved_count"], 2)
            self.assertEqual(diff["regressed_count"], 0)

    def test_base_benchmark_adapter_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            json_path = _build_fixture_dataset(root)
            bench = HallusionBenchBenchmark(json_path, data_root=root, verify_images=True)

            raw = list(bench.read_dataset())
            self.assertEqual(len(raw), 4)
            model = ScriptedModel(
                responses={
                    "catVD_subillusion_set0_fig0_q0_vis1": "No",
                    "catVD_subillusion_set0_fig1_q0_vis2": "Yes",
                    "catVS_subchart_set3_fig0_q0_vis0": "Yes",
                    "catVS_subchart_set3_fig1_q0_vis1": "No",
                }
            )
            preds = []
            for item in raw:
                norm = bench.normalize_record(item)
                prepared = bench.prepare_inputs(norm)
                out = bench.call_model(model, prepared)
                preds.append(
                    {
                        "sample_id": prepared.sample_id,
                        "prediction": out,
                        "raw_prediction": out,
                        "status": "COMPLETED",
                        "run_config_hash": "bench_hash_01",
                    }
                )
            self.assertEqual(validate_predictions(preds), [])
            scores = bench.score(preds)
            self.assertEqual(scores["metrics"]["aAcc"], 1.0)
            self.assertEqual(scores["metrics"]["fAcc"], 1.0)
            self.assertEqual(scores["metrics"]["qAcc"], 1.0)

            out_dir = root / "bench_export"
            bench.export_results(preds, str(out_dir))
            self.assertTrue((out_dir / "predictions.jsonl").is_file())
            self.assertTrue((out_dir / "metrics.json").is_file())
            self.assertTrue((out_dir / "errors.jsonl").is_file())
            self.assertEqual(len(read_jsonl(out_dir / "predictions.jsonl")), 4)


if __name__ == "__main__":
    unittest.main()
