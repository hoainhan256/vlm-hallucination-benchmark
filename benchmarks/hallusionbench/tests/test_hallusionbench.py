"""Unified CPU-only unit & end-to-end tests for HallusionBench (A + B + C + BaseBenchmark).

Compatible with both stdlib `python -m unittest` and `python -m pytest`.
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
    FORBIDDEN_MODEL_INPUT_FIELDS,
    DuplicateSampleIdError,
    GroupIntegrityError,
    HallusionBenchBenchmark,
    HallusionBenchDataset,
    HallusionInferenceRunner,
    HallusionRunConfig,
    HallusionSchemaError,
    PreparedHallusionInput,
    build_group_mappings,
    check_group_integrity,
    compare_before_after,
    export_hallusionbench_artifacts,
    extract_error_cases,
    filter_by_pilot_ids,
    find_cross_model_disagreements,
    normalize_hallusion_record,
    parse_yes_no,
    prepare_hallusion_input,
    render_error_table_markdown,
    score_hallusionbench,
    select_frozen_pilot,
    summarize_error_cases,
    write_blocked_run,
)
from benchmarks.hallusionbench.dataset import main as dataset_cli_main  # noqa: E402
from benchmarks.hallusionbench.evaluation import main as evaluation_cli_main  # noqa: E402
from benchmarks.hallusionbench.inference import main as inference_cli_main  # noqa: E402
from vlm_benchmark.results import read_jsonl, validate_predictions  # noqa: E402

TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
)


def _write_png(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(TINY_PNG)
    return path


def _build_fixture_dataset(root: Path, all_eight_subcategories: bool = False) -> Path:
    """Create a synthetic HallusionBench dataset covering VD/VS, visual_input 0/1/2, and .PNG case mismatch."""
    _write_png(root / "data" / "VD" / "illusion" / "0_0.png")
    _write_png(root / "data" / "VD" / "illusion" / "0_1.png")
    _write_png(root / "data" / "VS" / "chart" / "3_1.PNG")  # Intentional uppercase .PNG

    records: list[dict[str, Any]] = [
        # VD::illusion (visual=1 easy + visual=2 hard)
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
            "gt_answer_details": "Both circles have the exact same diameter.",
            "sample_note": "secret_note_do_not_leak",
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
        # VS::chart (visual=0 text-only easy + visual=1 hard with .PNG case mismatch)
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

    if all_eight_subcategories:
        extra_subs = [
            ("VD", "figure"),
            ("VD", "math"),
            ("VD", "video"),
            ("VS", "map"),
            ("VS", "ocr"),
            ("VS", "table"),
        ]
        for idx, (cat, sub) in enumerate(extra_subs, start=10):
            img_rel = f"./{cat}/{sub}/{idx}_0.png"
            _write_png(root / "data" / cat / sub / f"{idx}_0.png")
            records.append(
                {
                    "category": cat,
                    "subcategory": sub,
                    "visual_input": "0" if cat == "VS" else "1",
                    "set_id": str(idx),
                    "figure_id": "0",
                    "question_id": "0",
                    "filename": None if cat == "VS" else img_rel,
                    "question": f"Sample question for {sub}?",
                    "gt_answer": "1",
                }
            )
            img2_rel = f"./{cat}/{sub}/{idx}_1.png"
            _write_png(root / "data" / cat / sub / f"{idx}_1.png")
            records.append(
                {
                    "category": cat,
                    "subcategory": sub,
                    "visual_input": "1" if cat == "VS" else "2",
                    "set_id": str(idx),
                    "figure_id": "1",
                    "question_id": "0",
                    "filename": img2_rel,
                    "question": f"Sample question for {sub}?",
                    "gt_answer": "0",
                }
            )

    json_path = root / "HallusionBench.json"
    json_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return json_path


class ScriptedModel:
    def __init__(self, responses: dict[str, str], errors: dict[str, Exception] | None = None) -> None:
        self.responses = responses
        self.errors = errors or {}
        self.calls: list[PreparedHallusionInput] = []

    def generate(self, prepared: PreparedHallusionInput, generation_config: Mapping[str, Any]) -> str:
        self.calls.append(prepared)
        if prepared.sample_id in self.errors:
            raise self.errors[prepared.sample_id]
        return self.responses.get(prepared.sample_id, "Yes")

    def get_runtime_info(self) -> dict[str, Any]:
        return {
            "resolved_model_revision": "rev-test-001",
            "resolved_processor_revision": "proc-test-001",
            "attn_implementation": "sdpa",
            "device_allocation": "cpu",
        }


class TestHallusionBenchSuite(unittest.TestCase):
    # --- Part A: Dataset, schema, audit & 8-subcategory pilot ---

    def test_schema_validation_text_only_and_corrupt_missing_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_png(root / "VD" / "illusion" / "ignore.png")

            # Text-only MUST set image_path=None even when filename is populated
            text_rec = {
                "category": "VS",
                "subcategory": "map",
                "visual_input": "0",
                "set_id": "1",
                "figure_id": "0",
                "question_id": "0",
                "filename": "./VD/illusion/ignore.png",
                "question": "Is France in Europe?",
                "gt_answer": "1",
            }
            sample = normalize_hallusion_record(text_rec, data_root=root, verify_image_exists=True)
            self.assertFalse(sample.visual_input)
            self.assertIsNone(sample.image_path)
            self.assertIsNone(sample.figure_key)
            self.assertEqual(sample.difficulty, "easy")
            self.assertTrue(FORBIDDEN_MODEL_INPUT_FIELDS.isdisjoint(sample.to_model_safe_dict().keys()))

            # Bad records fail schema validation
            with self.assertRaises(HallusionSchemaError):
                normalize_hallusion_record({"category": "BAD", "visual_input": "0"}, verify_image_exists=False)

            # Missing & corrupt images are caught
            vis_rec = dict(text_rec, category="VD", visual_input="1", filename="./VD/illusion/missing.png")
            with self.assertRaises(FileNotFoundError):
                normalize_hallusion_record(vis_rec, data_root=root, verify_image_exists=True)

            corrupt = root / "VD" / "illusion" / "corrupt.png"
            corrupt.write_bytes(b"not-a-png")
            with self.assertRaises(ValueError):
                normalize_hallusion_record(
                    dict(vis_rec, filename="./VD/illusion/corrupt.png"),
                    data_root=root,
                    verify_image_exists=True,
                    verify_image_readable=True,
                )

    def test_duplicate_ids_and_group_integrity_detection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dup = [
                {"category": "VS", "subcategory": "math", "visual_input": "0", "set_id": "1", "figure_id": "0", "question_id": "0", "question": "Q?", "gt_answer": "1"},
                {"category": "VS", "subcategory": "math", "visual_input": "0", "set_id": "1", "figure_id": "0", "question_id": "0", "question": "Q2?", "gt_answer": "0"},
            ]
            p = root / "dup.json"
            p.write_text(json.dumps(dup), encoding="utf-8")
            with self.assertRaises(DuplicateSampleIdError):
                HallusionBenchDataset(p, data_root=root).load(verify_images=False)

            # Mismatched images within same figure_group_key
            img_a = _write_png(root / "VD" / "chart" / "a.png")
            img_b = _write_png(root / "VD" / "chart" / "b.png")
            bad_fig = [
                {"category": "VD", "subcategory": "chart", "visual_input": "1", "set_id": "5", "figure_id": "0", "question_id": "0", "filename": str(img_a), "question": "Q0?", "gt_answer": "1"},
                {"category": "VD", "subcategory": "chart", "visual_input": "1", "set_id": "5", "figure_id": "0", "question_id": "1", "filename": str(img_b), "question": "Q1?", "gt_answer": "0"},
            ]
            p2 = root / "bad_fig.json"
            p2.write_text(json.dumps(bad_fig), encoding="utf-8")
            ds2 = HallusionBenchDataset(p2, data_root=root)
            samples2 = ds2.load(verify_images=True)
            self.assertEqual(build_group_mappings(samples2)["order_index"][samples2[1].sample_id], 1)
            self.assertTrue(any("multiple distinct image paths" in msg for msg in check_group_integrity(samples2)))
            with self.assertRaises(GroupIntegrityError):
                ds2.load(verify_images=True, verify_group_integrity=True)

    def test_balanced_8_subcategory_pilot_and_split_group_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            json_path = _build_fixture_dataset(root, all_eight_subcategories=True)
            ds = HallusionBenchDataset(json_path, data_root=root)
            samples = ds.load(verify_images=True, verify_image_readable=True)
            report = ds.audit(verify_image_readable=True)
            self.assertTrue(report["integrity"]["is_clean"])
            self.assertEqual(report["counts"]["case_mismatch_image_samples"], 1)

            pilot = select_frozen_pilot(samples, target_groups=8, balance_subcategories=True)
            self.assertEqual(pilot["group_count"], 8)
            self.assertEqual(
                set(pilot["breakdown_by_subcategory"].keys()),
                {"chart", "figure", "illusion", "map", "math", "ocr", "table", "video"},
            )
            filtered = filter_by_pilot_ids(samples, pilot["sample_ids"], require_complete_groups=True)
            self.assertEqual(len(filtered), pilot["sample_count"])

            with self.assertRaises(GroupIntegrityError):
                filter_by_pilot_ids(samples, pilot["sample_ids"][:-1], require_complete_groups=True)

    # --- Part B: Prompt anti-leakage, policy gates & resume safety ---

    def test_prompt_no_leakage_policy_gates_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            json_path = _build_fixture_dataset(root)
            samples = HallusionBenchDataset(json_path, data_root=root).load(verify_images=True)

            prepared = prepare_hallusion_input(samples[0].to_dict(), data_root=root, verify_image_exists=True)
            self.assertIn("Please answer yes or no.", prepared.prompt)
            self.assertNotIn("exact same diameter", prepared.prompt)
            self.assertTrue(FORBIDDEN_MODEL_INPUT_FIELDS.isdisjoint(prepared.metadata.keys()))

            # Policy gates: T4 BF16, quantization, fallback, and unverified official stage
            t4_hw = {"cuda_available": True, "bf16_supported": False}
            cfg_bf16 = HallusionRunConfig(
                experiment_id="e", run_id="r", model_key="smolvlm2", model_id="HuggingFaceTB/SmolVLM2-2.2B-Instruct",
                stage="pilot", dtype="bfloat16",
            )
            self.assertTrue(any("native BF16" in e for e in cfg_bf16.validate_for_stage(hardware_info=t4_hw)))
            cfg_quant = HallusionRunConfig(
                experiment_id="e", run_id="r", model_key="smolvlm2", model_id="HuggingFaceTB/SmolVLM2-2.2B-Instruct",
                dtype="float16", quantization="int4",
            )
            self.assertTrue(any("quantization" in e for e in cfg_quant.validate_for_stage()))
            cfg_official = HallusionRunConfig(
                experiment_id="e", run_id="r", model_key="smolvlm2", model_id="HuggingFaceTB/SmolVLM2-2.2B-Instruct",
                stage="official", dtype="bfloat16",
            )
            self.assertTrue(any("model_revision" in e for e in cfg_official.validate_for_stage(repo_root=REPO_ROOT)))

            # Blocked run writes all 7 artifacts cleanly
            blocked_dir = root / "blocked"
            blocked = write_blocked_run(blocked_dir, cfg_bf16, reasons=["T4 lacks native BF16"], hardware_info=t4_hw)
            self.assertEqual(blocked["status"], "BLOCKED")
            self.assertTrue((blocked_dir / "manifest.json").is_file())

            # Resume skips COMPLETED and retries FAILED, and rejects config hash mismatch
            run_dir = root / "resume_run"
            cfg_ok = HallusionRunConfig(
                experiment_id="e", run_id="r1", model_key="smolvlm2", model_id="HuggingFaceTB/SmolVLM2-2.2B-Instruct",
                stage="full_baseline", dtype="float16", max_new_tokens=64,
            )
            recs = [s.to_model_safe_dict() for s in samples[:2]]
            m1 = ScriptedModel({recs[0]["sample_id"]: "No"}, errors={recs[1]["sample_id"]: RuntimeError("CUDA out of memory")})
            st1 = HallusionInferenceRunner(cfg_ok, run_dir, data_root=root).run(recs, m1, hardware_info={"cuda_available": False, "bf16_supported": False})
            self.assertEqual(st1["status"], "FAILED")
            self.assertEqual(st1["completed_samples"], 1)

            m2 = ScriptedModel({recs[1]["sample_id"]: "Yes"})
            st2 = HallusionInferenceRunner(cfg_ok, run_dir, data_root=root).run(recs, m2, hardware_info={"cuda_available": False, "bf16_supported": False}, resume=True)
            self.assertEqual(st2["status"], "COMPLETED")
            self.assertEqual([c.sample_id for c in m2.calls], [recs[1]["sample_id"]])

            cfg_diff = HallusionRunConfig(
                experiment_id="e", run_id="r1", model_key="smolvlm2", model_id="HuggingFaceTB/SmolVLM2-2.2B-Instruct",
                stage="full_baseline", dtype="float16", max_new_tokens=128,
            )
            with self.assertRaises(ValueError):
                HallusionInferenceRunner(cfg_diff, run_dir, data_root=root).run(recs, m2, hardware_info={"cuda_available": False, "bf16_supported": False}, resume=True)

    # --- Part C + Full A -> B -> C E2E + BaseBenchmark + Shared Repo Checkers ---

    def test_scoring_analysis_and_full_e2e_pipeline(self) -> None:
        self.assertEqual(parse_yes_no("Yes"), "yes")
        self.assertEqual(parse_yes_no("No, there is no circle."), "no")
        self.assertEqual(parse_yes_no("Based on the chart, yes."), "yes")
        self.assertIsNone(parse_yes_no("Yes and no, it depends."))
        self.assertIsNone(parse_yes_no("Unclear"))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            json_path = _build_fixture_dataset(root)
            pilot_path = root / "pilot_ids.json"

            # Step A CLI: audit + export-pilot + verify-pilot
            self.assertEqual(dataset_cli_main(["--mode", "audit", "--dataset-json", str(json_path), "--data-root", str(root)]), 0)
            self.assertEqual(
                dataset_cli_main(["--mode", "export-pilot", "--dataset-json", str(json_path), "--data-root", str(root), "--target-groups", "2", "--output", str(pilot_path)]),
                0,
            )
            self.assertEqual(
                dataset_cli_main(["--mode", "verify-pilot", "--dataset-json", str(json_path), "--data-root", str(root), "--pilot-file", str(pilot_path)]),
                0,
            )

            # Step B CLI dry-run with Member A's frozen pilot
            dry_root = root / "dry_results"
            self.assertEqual(
                inference_cli_main(
                    [
                        "--mode", "dry-run", "--model-key", "smolvlm2", "--stage", "pilot", "--dtype", "float16",
                        "--dataset-json", str(json_path), "--data-root", str(root), "--pilot-file", str(pilot_path),
                        "--experiment-id", "exp_e2e", "--run-id", "dry_01", "--output-root", str(dry_root),
                    ]
                ),
                0,
            )

            # Step B runner for 2 models (including technical_error, format_violation, incorrect_prediction)
            ds = HallusionBenchDataset(json_path, data_root=root)
            samples = ds.load(verify_images=True)
            index = ds.load_index(verify_images=True)
            recs = [s.to_model_safe_dict() for s in samples]

            run_a = root / "results" / "exp_e2e" / "run_smolvlm2"
            cfg_a = HallusionRunConfig(experiment_id="exp_e2e", run_id="run_smolvlm2", model_key="smolvlm2", model_id="HuggingFaceTB/SmolVLM2-2.2B-Instruct", stage="pilot", dtype="float16")
            m_a = ScriptedModel(
                {
                    "catVD_subillusion_set0_fig0_q0_vis1": "No",
                    "catVD_subillusion_set0_fig1_q0_vis2": "No, looks same.",
                    "catVS_subchart_set3_fig0_q0_vis0": "Yes and no",
                },
                errors={"catVS_subchart_set3_fig1_q0_vis1": RuntimeError("CUDA out of memory")},
            )
            HallusionInferenceRunner(cfg_a, run_a, data_root=root).run(recs, m_a, hardware_info={"cuda_available": False, "bf16_supported": False})

            run_b = root / "results" / "exp_e2e" / "run_qwen3"
            cfg_b = HallusionRunConfig(experiment_id="exp_e2e", run_id="run_qwen3", model_key="qwen3_vl", model_id="Qwen/Qwen3-VL-4B-Instruct", stage="pilot", dtype="float16")
            m_b = ScriptedModel(
                {
                    "catVD_subillusion_set0_fig0_q0_vis1": "No",
                    "catVD_subillusion_set0_fig1_q0_vis2": "Yes",
                    "catVS_subchart_set3_fig0_q0_vis0": "Yes",
                    "catVS_subchart_set3_fig1_q0_vis1": "No",
                }
            )
            HallusionInferenceRunner(cfg_b, run_b, data_root=root).run(recs, m_b, hardware_info={"cuda_available": False, "bf16_supported": False})

            # Step C CLI: score, extract, disagreements, before-after
            for rdir in (run_a, run_b):
                self.assertEqual(evaluation_cli_main(["--mode", "score", "--run-dir", str(rdir), "--dataset-json", str(json_path), "--data-root", str(root)]), 0)
                for script_args in (
                    [str(REPO_ROOT / "scripts" / "check_results.py"), str(rdir / "predictions.jsonl")],
                    [str(REPO_ROOT / "scripts" / "check_manifest.py"), str(rdir)],
                ):
                    proc = subprocess.run([sys.executable, "-B", *script_args], capture_output=True, text=True, check=False)
                    self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

            metrics_b = json.loads((run_b / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics_b["summary_counts"]["scored_figures"], 3)  # text-only excluded from fAcc
            self.assertEqual(metrics_b["summary_counts"]["scored_complete_pairs"], 2)
            self.assertEqual(metrics_b["metrics"], {"aAcc": 1.0, "fAcc": 1.0, "qAcc": 1.0})
            self.assertIn("illusion", metrics_b["by_subcategory"])
            self.assertIn("chart", metrics_b["by_subcategory"])

            self.assertEqual(evaluation_cli_main(["--mode", "extract", "--run-dir", str(run_a), "--dataset-json", str(json_path), "--data-root", str(root)]), 0)
            cases = json.loads((run_a / "analysis" / "error_cases.json").read_text(encoding="utf-8"))
            summary = summarize_error_cases(cases)
            self.assertEqual(summary["total_error_cases"], 3)
            self.assertIn("`run_smolvlm2`", render_error_table_markdown(cases))

            dis_out = root / "disagreements.json"
            self.assertEqual(
                evaluation_cli_main(["--mode", "disagreements", "--run-dirs", f"smolvlm2={run_a}", f"qwen3_vl={run_b}", "--dataset-json", str(json_path), "--data-root", str(root), "--output", str(dis_out)]),
                0,
            )
            self.assertEqual(len(json.loads(dis_out.read_text(encoding="utf-8"))), 2)

            diff_out = root / "diff.json"
            self.assertEqual(
                evaluation_cli_main(["--mode", "before-after", "--baseline-run-dir", str(run_a), "--improved-run-dir", str(run_b), "--dataset-json", str(json_path), "--data-root", str(root), "--output", str(diff_out)]),
                0,
            )
            diff = json.loads(diff_out.read_text(encoding="utf-8"))
            self.assertEqual(diff["improved_count"], 2)
            self.assertEqual(diff["regressed_count"], 0)

            # BaseBenchmark adapter contract + resume-safe export
            bench = HallusionBenchBenchmark(json_path, data_root=root, verify_images=True)
            preds = []
            for item in bench.read_dataset():
                norm = bench.normalize_record(item)
                prep = bench.prepare_inputs(norm)
                out = bench.call_model(m_b, prep)
                preds.append({"sample_id": prep.sample_id, "prediction": out, "status": "COMPLETED", "run_config_hash": "bh1"})
            self.assertEqual(validate_predictions(preds), [])
            self.assertEqual(bench.score(preds)["metrics"]["aAcc"], 1.0)
            exp_dir = root / "bench_out"
            bench.export_results(preds, str(exp_dir))
            self.assertEqual(len(read_jsonl(exp_dir / "predictions.jsonl")), 4)

            # Refuses overwriting COMPLETED sample on conflict
            with self.assertRaises(ValueError):
                export_hallusionbench_artifacts(
                    [{"sample_id": preds[0]["sample_id"], "prediction": "Yes", "status": "COMPLETED", "run_config_hash": "bh1"}],
                    index,
                    exp_dir,
                )

    def test_dual_format_mmhal_deliverables_backfill_and_offload_hash_stability(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            json_path = _build_fixture_dataset(root)
            ds = HallusionBenchDataset(json_path, data_root=root)
            samples = ds.load(verify_images=True)
            recs = [s.to_model_safe_dict() for s in samples]

            # Hardware offload knobs must NOT break config_hash() compatibility with existing runs
            cfg_plain = HallusionRunConfig(
                experiment_id="exp_dual", run_id="run_phi4", model_key="phi4_multimodal",
                model_id="microsoft/Phi-4-multimodal-instruct", stage="pilot", dtype="float16",
                attn_implementation="eager", trust_remote_code=True,
            )
            cfg_offload = HallusionRunConfig(
                experiment_id="exp_dual", run_id="run_phi4", model_key="phi4_multimodal",
                model_id="microsoft/Phi-4-multimodal-instruct", stage="pilot", dtype="float16",
                attn_implementation="eager", trust_remote_code=True,
                max_gpu_memory="10GiB", max_cpu_memory="12GiB", offload_folder=str(root / "offload"),
            )
            self.assertEqual(cfg_plain.config_hash(), cfg_offload.config_hash())

            # Run 1 (new runner emits both 7 official artifacts and MMHal-style files)
            run_phi4 = root / "runs" / "phi4"
            m_phi4 = ScriptedModel(
                {
                    "catVD_subillusion_set0_fig0_q0_vis1": "No",
                    "catVD_subillusion_set0_fig1_q0_vis2": "Yes",
                    "catVS_subchart_set3_fig0_q0_vis0": "Yes",
                    "catVS_subchart_set3_fig1_q0_vis1": "No",
                }
            )
            HallusionInferenceRunner(cfg_offload, run_phi4, data_root=root).run(
                recs, m_phi4, hardware_info={"cuda_available": False, "bf16_supported": False}
            )
            for fname in (
                "inference_config.json",
                "inference_summary.json",
                "inference_log.txt",
                "phi4_multimodal_hallusionbench_responses.json",
            ):
                self.assertTrue((run_phi4 / fname).is_file(), f"Missing {fname}")
            responses = json.loads((run_phi4 / "phi4_multimodal_hallusionbench_responses.json").read_text(encoding="utf-8"))
            self.assertEqual(len(responses), 4)
            self.assertNotIn("gt_answer", responses[0])  # No label leakage in inference responses

            # Run 2 simulates a legacy run (only 7 official artifacts) -> scoring & export must backfill MMHal files
            run_smol = root / "runs" / "smolvlm2"
            cfg_smol = HallusionRunConfig(
                experiment_id="exp_dual", run_id="run_smol", model_key="smolvlm2",
                model_id="HuggingFaceTB/SmolVLM2-2.2B-Instruct", stage="pilot", dtype="float16",
            )
            m_smol = ScriptedModel(
                {
                    "catVD_subillusion_set0_fig0_q0_vis1": "Yes",
                    "catVD_subillusion_set0_fig1_q0_vis2": "Yes",
                    "catVS_subchart_set3_fig0_q0_vis0": "Yes",
                    "catVS_subchart_set3_fig1_q0_vis1": "No",
                }
            )
            HallusionInferenceRunner(cfg_smol, run_smol, data_root=root).run(
                recs, m_smol, hardware_info={"cuda_available": False, "bf16_supported": False}
            )
            for legacy_remove in (
                "inference_config.json",
                "inference_summary.json",
                "smolvlm2_hallusionbench_responses.json",
            ):
                (run_smol / legacy_remove).unlink()

            deliverables_out = root / "repo_results"
            self.assertEqual(
                evaluation_cli_main(
                    [
                        "--mode", "export-deliverables",
                        "--run-dirs", f"smolvlm2={run_smol}", f"phi4_multimodal={run_phi4}",
                        "--dataset-json", str(json_path), "--data-root", str(root),
                        "--output-dir", str(deliverables_out),
                    ]
                ),
                0,
            )

            # Verify both models pass official repo checkers AND have complete MMHal-style deliverables
            for mk in ("smolvlm2", "phi4_multimodal"):
                mdir = deliverables_out / mk
                for fname in (
                    "predictions.jsonl",
                    "metrics.json",
                    "run_config.yaml",
                    "run_metadata.json",
                    "errors.jsonl",
                    "run_status.json",
                    "manifest.json",
                    "inference_config.json",
                    "inference_summary.json",
                    f"{mk}_hallusionbench_responses.json",
                    f"{mk}_hallusionbench_evaluation.json",
                    f"{mk}_hallusionbench_summary.json",
                    "README.md",
                ):
                    self.assertTrue((mdir / fname).is_file(), f"Missing exported {mk}/{fname}")
                self.assertTrue((mdir / "analysis" / "error_cases.md").is_file())
                for script_args in (
                    [str(REPO_ROOT / "scripts" / "check_results.py"), str(mdir / "predictions.jsonl")],
                    [str(REPO_ROOT / "scripts" / "check_manifest.py"), str(mdir)],
                ):
                    proc = subprocess.run([sys.executable, "-B", *script_args], capture_output=True, text=True, check=False)
                    self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

            master_readme = (deliverables_out / "README.md").read_text(encoding="utf-8")
            self.assertIn("`smolvlm2`", master_readme)
            self.assertIn("`phi4_multimodal`", master_readme)
            self.assertIn("`qwen3_vl`", master_readme)  # Honestly listed as PENDING
            self.assertIn("PENDING", master_readme)
            self.assertTrue((deliverables_out / "cross_model_disagreements.json").is_file())

            # Refuses overwriting COMPLETED export unless --overwrite-export is passed
            with self.assertRaises(FileExistsError):
                evaluation_cli_main(
                    [
                        "--mode", "export-deliverables",
                        "--run-dirs", f"phi4_multimodal={run_phi4}",
                        "--dataset-json", str(json_path), "--data-root", str(root),
                        "--output-dir", str(deliverables_out),
                    ]
                )
            self.assertEqual(
                evaluation_cli_main(
                    [
                        "--mode", "export-deliverables",
                        "--overwrite-export",
                        "--run-dirs", f"phi4_multimodal={run_phi4}",
                        "--dataset-json", str(json_path), "--data-root", str(root),
                        "--output-dir", str(deliverables_out),
                    ]
                ),
                0,
            )

            # Optional matplotlib visualization check (runs when matplotlib is installed)
            try:
                import matplotlib  # noqa: F401
            except ModuleNotFoundError:
                pass
            else:
                self.assertEqual(
                    evaluation_cli_main(
                        [
                            "--mode", "visualize",
                            "--results-dir", str(deliverables_out),
                            "--dpi", "100",
                        ]
                    ),
                    0,
                )
                for fig_name in (
                    "fig1_overall_metrics.png",
                    "fig2_category_and_difficulty.png",
                    "fig3_subcategory_heatmap.png",
                    "fig4_visual_input_condition.png",
                    "fig5_errors_and_latency.png",
                ):
                    fig_path = deliverables_out / "figures" / fig_name
                    self.assertTrue(fig_path.is_file() and fig_path.stat().st_size > 1000, f"Invalid chart {fig_name}")
                rep_text = (deliverables_out / "REPORT.md").read_text(encoding="utf-8")
                self.assertIn("fig1_overall_metrics.png", rep_text)
                self.assertIn("fig4_visual_input_condition.png", rep_text)


if __name__ == "__main__":
    unittest.main()
