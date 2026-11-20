"""CPU-only unit tests for HallusionBench Member A dataset loader, schema & pilot selector.

Compatible with both `python -m pytest` and stdlib `python -m unittest`.
"""

from __future__ import annotations

import base64
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.hallusionbench.dataset import (  # noqa: E402
    HallusionBenchDataset,
    build_group_mappings,
    check_group_integrity,
    filter_by_pilot_ids,
    main as dataset_cli_main,
    normalize_hallusion_record,
    select_frozen_pilot,
)
from benchmarks.hallusionbench.protocol import (  # noqa: E402
    make_stable_sample_id as protocol_make_stable_sample_id,
    prepare_hallusion_input,
)
from benchmarks.hallusionbench.schema import (  # noqa: E402
    FORBIDDEN_MODEL_INPUT_FIELDS,
    DuplicateSampleIdError,
    GroupIntegrityError,
    HallusionSchemaError,
    validate_normalized_samples,
    validate_raw_record,
)

# Valid 1x1 RGB PNG image bytes so Pillow verification passes without external fixtures.
TINY_VALID_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
)


def _write_valid_png(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(TINY_VALID_PNG_BYTES)
    return path


def _make_mini_dataset(root: Path) -> tuple[Path, list[dict]]:
    """Create a synthetic multi-group mini HallusionBench dataset on disk."""
    _write_valid_png(root / "data" / "VD" / "illusion" / "0_0.png")
    _write_valid_png(root / "data" / "VD" / "illusion" / "0_1.png")
    # Intentional uppercase .PNG to test Linux case-insensitive resolution & audit tracking.
    _write_valid_png(root / "data" / "VS" / "chart" / "3_1.PNG")
    _write_valid_png(root / "hallusion_bench" / "VD" / "math" / "2_0.png")

    records = [
        # Group 1: VD::illusion::set_0 (original visual=1 + edited visual=2, 2 figures, 1 question pair)
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
            "gt_answer_details": "Both circles have identical diameter despite surrounding context.",
            "sample_note": "ebbinghaus",
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
            "gt_answer_details": "The right circle was edited to be visibly larger.",
            "sample_note": "ebbinghaus_edited",
        },
        # Group 2: VS::chart::set_3 (text-only visual=0 + visual=1 with .PNG case mismatch)
        {
            "category": "VS",
            "subcategory": "chart",
            "visual_input": "0",
            "set_id": "3",
            "figure_id": "0",
            "question_id": "0",
            "filename": None,
            "question": "According to standard geography, is Tokyo in Japan?",
            "gt_answer": "1",
            "gt_answer_details": "Text-only control question.",
        },
        {
            "category": "VS",
            "subcategory": "chart",
            "visual_input": "1",
            "set_id": "3",
            "figure_id": "1",
            "question_id": "0",
            "filename": "./VS/chart/3_1.png",
            "question": "According to the chart, does the value increase in 2024?",
            "gt_answer": "0",
            "gt_answer_details": "The bar declines in 2024.",
        },
        # Group 3: VD::math::set_2 (uses hallusion_bench/ prefix layout)
        {
            "category": "VD",
            "subcategory": "math",
            "visual_input": "1",
            "set_id": "2",
            "figure_id": "0",
            "question_id": "0",
            "filename": "./hallusion_bench/VD/math/2_0.png",
            "question": "Are the two triangles congruent?",
            "gt_answer": "1",
        },
        {
            "category": "VD",
            "subcategory": "math",
            "visual_input": "1",
            "set_id": "2",
            "figure_id": "0",
            "question_id": "1",
            "filename": "./hallusion_bench/VD/math/2_0.png",
            "question": "Does the triangle contain a right angle?",
            "gt_answer": "0",
        },
    ]

    json_path = root / "HallusionBench.json"
    json_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return json_path, records


class TestHallusionBenchDataset(unittest.TestCase):
    def test_valid_records_normalize_and_align_with_member_b_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            json_path, raw_records = _make_mini_dataset(root)
            dataset = HallusionBenchDataset(json_path, data_root=root)

            samples = dataset.load(verify_images=True, verify_image_readable=True)
            self.assertEqual(len(samples), 6)
            self.assertEqual(validate_normalized_samples(samples), [])

            # Check visual=1, visual=2, and visual=0 semantics
            self.assertTrue(samples[0].visual_input)
            self.assertEqual(samples[0].visual_input_raw, "1")
            self.assertEqual(samples[0].gt_answer, "0")
            self.assertEqual(samples[0].reference_answer, "no")

            self.assertTrue(samples[1].visual_input)
            self.assertEqual(samples[1].visual_input_raw, "2")
            self.assertEqual(samples[1].reference_answer, "yes")

            self.assertFalse(samples[2].visual_input)
            self.assertEqual(samples[2].visual_input_raw, "0")
            self.assertIsNone(samples[2].image_path)

            # Case mismatch on VS/chart/3_1.PNG must be resolved and transparently flagged
            self.assertTrue(samples[3].visual_input)
            self.assertTrue(samples[3].image_case_mismatch)
            self.assertTrue(Path(samples[3].image_path).is_file())

            # Every sample_id MUST match Member B's protocol.make_stable_sample_id
            for raw, sample in zip(raw_records, samples):
                self.assertEqual(sample.sample_id, protocol_make_stable_sample_id(raw))

                # Model-safe dict MUST NOT contain any forbidden ground-truth fields
                safe_dict = sample.to_model_safe_dict()
                self.assertTrue(FORBIDDEN_MODEL_INPUT_FIELDS.isdisjoint(safe_dict.keys()))

                # Member B's prepare_hallusion_input must accept Member A's output seamlessly
                prepared = prepare_hallusion_input(sample.to_dict(), data_root=root, verify_image_exists=True)
                self.assertEqual(prepared.sample_id, sample.sample_id)
                self.assertEqual(prepared.visual_input, sample.visual_input)
                if sample.visual_input:
                    self.assertIsNotNone(prepared.image_path)
                else:
                    self.assertIsNone(prepared.image_path)

    def test_text_only_never_attaches_image_even_if_filename_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_valid_png(root / "VD" / "illusion" / "should_ignore.png")
            raw = {
                "category": "VS",
                "subcategory": "map",
                "visual_input": "0",
                "set_id": "9",
                "figure_id": "0",
                "question_id": "0",
                "filename": "./VD/illusion/should_ignore.png",
                "question": "Is France in Europe?",
                "gt_answer": "1",
            }
            sample = normalize_hallusion_record(raw, data_root=root, verify_image_exists=True)
            self.assertFalse(sample.visual_input)
            self.assertEqual(sample.visual_input_raw, "0")
            self.assertIsNone(sample.image_path)
            self.assertEqual(sample.filename, "./VD/illusion/should_ignore.png")

    def test_missing_or_invalid_fields_raise_schema_error(self) -> None:
        bad_records = [
            # Missing question
            {
                "category": "VD",
                "subcategory": "illusion",
                "visual_input": "0",
                "set_id": "0",
                "figure_id": "0",
                "question_id": "0",
                "gt_answer": "1",
            },
            # Invalid category
            {
                "category": "UNKNOWN",
                "subcategory": "illusion",
                "visual_input": "0",
                "set_id": "0",
                "figure_id": "0",
                "question_id": "0",
                "question": "Test?",
                "gt_answer": "1",
            },
            # Invalid visual_input
            {
                "category": "VD",
                "subcategory": "illusion",
                "visual_input": "9",
                "set_id": "0",
                "figure_id": "0",
                "question_id": "0",
                "question": "Test?",
                "gt_answer": "1",
            },
            # Invalid gt_answer
            {
                "category": "VD",
                "subcategory": "illusion",
                "visual_input": "0",
                "set_id": "0",
                "figure_id": "0",
                "question_id": "0",
                "question": "Test?",
                "gt_answer": "maybe",
            },
            # visual_input=1 without filename
            {
                "category": "VD",
                "subcategory": "illusion",
                "visual_input": "1",
                "set_id": "0",
                "figure_id": "0",
                "question_id": "0",
                "question": "Test?",
                "gt_answer": "1",
                "filename": None,
            },
        ]
        for idx, rec in enumerate(bad_records):
            errors = validate_raw_record(rec, index=idx)
            self.assertTrue(len(errors) > 0, f"Expected validation error for bad_records[{idx}]")
            with self.assertRaises(HallusionSchemaError):
                normalize_hallusion_record(rec, verify_image_exists=False)

    def test_missing_image_and_corrupt_image_are_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing_rec = {
                "category": "VD",
                "subcategory": "illusion",
                "visual_input": "1",
                "set_id": "0",
                "figure_id": "0",
                "question_id": "0",
                "filename": "./VD/illusion/nonexistent.png",
                "question": "Does this image exist?",
                "gt_answer": "0",
            }
            with self.assertRaises(FileNotFoundError):
                normalize_hallusion_record(missing_rec, data_root=root, verify_image_exists=True)

            corrupt_path = root / "VD" / "illusion" / "corrupt.png"
            corrupt_path.parent.mkdir(parents=True, exist_ok=True)
            corrupt_path.write_bytes(b"not-a-valid-png-file")
            corrupt_rec = dict(missing_rec, filename="./VD/illusion/corrupt.png")
            with self.assertRaises(ValueError):
                normalize_hallusion_record(
                    corrupt_rec,
                    data_root=root,
                    verify_image_exists=True,
                    verify_image_readable=True,
                )

    def test_duplicate_sample_id_raises_error_in_strict_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dup_records = [
                {
                    "category": "VS",
                    "subcategory": "math",
                    "visual_input": "0",
                    "set_id": "1",
                    "figure_id": "0",
                    "question_id": "0",
                    "question": "Is 2 + 2 = 4?",
                    "gt_answer": "1",
                },
                {
                    "category": "VS",
                    "subcategory": "math",
                    "visual_input": "0",
                    "set_id": "1",
                    "figure_id": "0",
                    "question_id": "0",
                    "question": "Duplicate key question?",
                    "gt_answer": "0",
                },
            ]
            json_path = root / "HallusionBench.json"
            json_path.write_text(json.dumps(dup_records), encoding="utf-8")

            dataset = HallusionBenchDataset(json_path, data_root=root)
            with self.assertRaises(DuplicateSampleIdError):
                dataset.load(verify_images=False)

    def test_group_pair_integrity_and_order_preservation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            img_a = _write_valid_png(root / "VD" / "chart" / "a.png")
            img_b = _write_valid_png(root / "VD" / "chart" / "b.png")
            # Two questions claiming the SAME figure_id=0 in the same set, but pointing to DIFFERENT images!
            records = [
                {
                    "category": "VD",
                    "subcategory": "chart",
                    "visual_input": "1",
                    "set_id": "5",
                    "figure_id": "0",
                    "question_id": "0",
                    "filename": str(img_a),
                    "question": "Q0?",
                    "gt_answer": "1",
                },
                {
                    "category": "VD",
                    "subcategory": "chart",
                    "visual_input": "1",
                    "set_id": "5",
                    "figure_id": "0",
                    "question_id": "1",
                    "filename": str(img_b),
                    "question": "Q1?",
                    "gt_answer": "0",
                },
            ]
            json_path = root / "HallusionBench.json"
            json_path.write_text(json.dumps(records), encoding="utf-8")

            dataset = HallusionBenchDataset(json_path, data_root=root)
            samples = dataset.load(verify_images=True)
            mappings = build_group_mappings(samples)
            self.assertEqual(mappings["order_index"][samples[0].sample_id], 0)
            self.assertEqual(mappings["order_index"][samples[1].sample_id], 1)

            issues = check_group_integrity(samples, flag_singleton_question_pairs=True)
            self.assertTrue(any("multiple distinct image paths" in msg for msg in issues))
            self.assertTrue(any("has only 1 sample(s)" in msg for msg in issues))

            with self.assertRaises(GroupIntegrityError):
                dataset.load(verify_images=True, verify_group_integrity=True)

    def test_frozen_pilot_preserves_complete_groups_and_detects_split_groups(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            json_path, _ = _make_mini_dataset(root)
            samples = HallusionBenchDataset(json_path, data_root=root).load(verify_images=True)

            pilot = select_frozen_pilot(samples, target_groups=2)
            # Must pick Group 2 (VS + text-only, 2 items) and Group 1 (VD + edited visual=2, 2 items) = 4 items
            self.assertEqual(pilot["group_count"], 2)
            self.assertEqual(pilot["sample_count"], 4)
            self.assertIn("VD", pilot["breakdown_by_category"])
            self.assertIn("VS", pilot["breakdown_by_category"])
            self.assertIn("0", pilot["breakdown_by_visual_input_raw"])
            self.assertIn("2", pilot["breakdown_by_visual_input_raw"])

            # Deterministic repeat call produces identical checksum
            pilot_repeat = select_frozen_pilot(samples, target_groups=2)
            self.assertEqual(pilot["pilot_ids_sha256"], pilot_repeat["pilot_ids_sha256"])

            # Filtering with full pilot succeeds
            filtered = filter_by_pilot_ids(samples, pilot["sample_ids"], require_complete_groups=True)
            self.assertEqual([s.sample_id for s in filtered], pilot["sample_ids"])

            # Dropping one sample from a group MUST raise GroupIntegrityError
            broken_pilot_ids = pilot["sample_ids"][:-1]
            with self.assertRaises(GroupIntegrityError):
                filter_by_pilot_ids(samples, broken_pilot_ids, require_complete_groups=True)

    def test_audit_report_counts_and_cli_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            json_path, _ = _make_mini_dataset(root)
            dataset = HallusionBenchDataset(
                json_path,
                data_root=root,
                dataset_source="rayguan/HallusionBench",
                dataset_revision="test-rev-123",
            )

            report = dataset.audit(verify_image_readable=True)
            self.assertTrue(report["integrity"]["is_clean"])
            self.assertEqual(report["counts"]["total_raw_records"], 6)
            self.assertEqual(report["counts"]["valid_records"], 6)
            self.assertEqual(report["counts"]["visual_samples"], 5)
            self.assertEqual(report["counts"]["text_only_samples"], 1)
            self.assertEqual(report["counts"]["missing_image_samples"], 0)
            self.assertEqual(report["counts"]["corrupt_image_samples"], 0)
            self.assertEqual(report["counts"]["case_mismatch_image_samples"], 1)
            self.assertEqual(report["counts"]["set_group_count"], 3)
            self.assertEqual(report["distributions"]["category"], {"VD": 4, "VS": 2})
            self.assertEqual(report["distributions"]["visual_input_raw"], {"0": 1, "1": 4, "2": 1})
            self.assertEqual(len(report["json_sha256"]), 64)

            # CLI roundtrip: audit -> export-pilot -> verify-pilot
            audit_out = root / "audit.json"
            pilot_out = root / "pilot_ids.json"
            rc_audit = dataset_cli_main(
                [
                    "--mode",
                    "audit",
                    "--dataset-json",
                    str(json_path),
                    "--data-root",
                    str(root),
                    "--output",
                    str(audit_out),
                ]
            )
            self.assertEqual(rc_audit, 0)
            self.assertTrue(audit_out.is_file())

            rc_export = dataset_cli_main(
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
                    str(pilot_out),
                ]
            )
            self.assertEqual(rc_export, 0)
            self.assertTrue(pilot_out.is_file())

            rc_verify = dataset_cli_main(
                [
                    "--mode",
                    "verify-pilot",
                    "--dataset-json",
                    str(json_path),
                    "--data-root",
                    str(root),
                    "--pilot-file",
                    str(pilot_out),
                ]
            )
            self.assertEqual(rc_verify, 0)


if __name__ == "__main__":
    unittest.main()
