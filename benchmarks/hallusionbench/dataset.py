"""HallusionBench dataset loader, integrity auditor, and frozen pilot selector (Member A).

Complies with:
- docs/dataset_policy.md (read-only, no mutation of questions/images/labels, record revisions/checksums)
- benchmarks/base.py (does NOT download datasets automatically on import or load)
- PHAN_CONG_NHOM_2_HALLUSIONBENCH_V2.md Section 3 (Member A deliverables)

This module depends only on the Python standard library (with optional Pillow support
for pixel-level image readability checks).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.hallusionbench.schema import (  # noqa: E402
    KNOWN_RAW_FIELDS,
    OFFICIAL_GITHUB_SOURCE,
    OFFICIAL_LICENSE,
    DuplicateSampleIdError,
    GroupIntegrityError,
    HallusionSample,
    HallusionSchemaError,
    derive_difficulty,
    make_figure_group_key,
    make_group_or_pair_key,
    make_question_pair_key,
    make_stable_sample_id,
    parse_category,
    parse_gt_answer,
    parse_visual_input,
    validate_normalized_samples,
    validate_raw_record,
)


def sha256_file(path: str | Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    """Compute deterministic SHA-256 hex digest of a JSON-serializable object."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _find_existing_case_insensitive(
    path: Path,
    expected_name: str | None = None,
) -> tuple[Path | None, bool]:
    """Return (matched_path, case_mismatch).

    On case-sensitive filesystems (Linux/Kaggle/Colab), 9 official HallusionBench
    VS/chart files use uppercase '.PNG' while HallusionBench.json references '.png'.
    On case-insensitive filesystems (Windows), Path.resolve() canonicalizes case to
    the on-disk spelling, so we compare against the original `expected_name` to
    detect case mismatches consistently across OSes.
    """
    parent = path.parent
    if not parent.is_dir():
        return None, False

    target_name = expected_name if expected_name is not None else path.name
    target_lower = target_name.lower()

    entries = sorted(p for p in parent.iterdir() if p.is_file())
    for entry in entries:
        if entry.name == target_name:
            return entry, False
    for entry in entries:
        if entry.name.lower() == target_lower:
            return entry, True
    return None, False


def check_image_readable(image_path: str | Path) -> tuple[bool, str | None]:
    """Verify an image file is non-empty and readable (using Pillow if installed, else magic bytes)."""
    path = Path(image_path)
    if not path.is_file():
        return False, f"Image file does not exist: {path}"
    if path.stat().st_size == 0:
        return False, f"Image file is empty (0 bytes): {path}"

    try:
        from PIL import Image

        with Image.open(path) as img:
            img.verify()
        return True, None
    except ImportError:
        # Fallback magic-header check when Pillow is unavailable.
        header = path.read_bytes()[:12]
        valid_magic = (
            header.startswith(b"\x89PNG\r\n\x1a\n")
            or header.startswith(b"\xff\xd8\xff")
            or header.startswith(b"GIF87a")
            or header.startswith(b"GIF89a")
            or (header.startswith(b"RIFF") and header[8:12] == b"WEBP")
            or header.startswith(b"BM")
        )
        if not valid_magic:
            return False, f"Unrecognized or corrupt image header (Pillow unavailable): {path}"
        return True, None
    except Exception as exc:
        return False, f"Corrupt or unreadable image ({type(exc).__name__}): {exc}"


def resolve_image_path(
    raw_record: Mapping[str, Any],
    visual_input: bool,
    data_root: str | Path | None = None,
    verify_image_exists: bool = True,
    verify_image_readable: bool = False,
) -> tuple[str | None, bool]:
    """Resolve image path while strictly enforcing visual_input semantics.

    Returns (resolved_image_path, case_mismatch).
    - If visual_input is False (visual_input_raw == "0"): ALWAYS returns (None, False).
    - If visual_input is True (visual_input_raw in {"1", "2"}): searches direct,
      'data/', and 'hallusion_bench/' layouts under data_root.
    """
    if not visual_input:
        return None, False

    raw_path = raw_record.get("image_path")
    if raw_path is None or str(raw_path).strip() == "":
        raw_path = raw_record.get("filename")
    if raw_path is None or str(raw_path).strip() == "":
        raise HallusionSchemaError("Visual sample requires non-empty 'filename' or 'image_path'")

    candidate = Path(str(raw_path).strip())
    expected_name = candidate.name
    case_mismatch = False

    if not candidate.is_absolute() and data_root is not None:
        root = Path(data_root)
        parts = [p for p in candidate.parts if p not in (".", "")]
        candidates = [(root / candidate).resolve()]
        if parts and parts[0].lower() == "hallusion_bench":
            candidates.append((root / Path(*parts[1:])).resolve())
            candidates.append((root / "data" / Path(*parts[1:])).resolve())
        elif parts:
            candidates.append((root / "data" / Path(*parts)).resolve())
            candidates.append((root / "hallusion_bench" / Path(*parts)).resolve())

        matched_path: Path | None = None
        for option in candidates:
            found, mismatch = _find_existing_case_insensitive(option, expected_name=expected_name)
            if found is not None:
                matched_path = found
                case_mismatch = mismatch
                break
        candidate = matched_path if matched_path is not None else candidates[0]
    else:
        found, mismatch = _find_existing_case_insensitive(candidate, expected_name=expected_name)
        if found is not None:
            candidate = found
            case_mismatch = mismatch

    if verify_image_exists and not candidate.is_file():
        raise FileNotFoundError(f"HallusionBench image file not found: {candidate}")

    if verify_image_readable:
        ok, err = check_image_readable(candidate)
        if not ok:
            raise ValueError(err or f"Image file is unreadable: {candidate}")

    return str(candidate), case_mismatch


def normalize_hallusion_record(
    raw_record: Mapping[str, Any],
    data_root: str | Path | None = None,
    index: int | None = None,
    verify_image_exists: bool = True,
    verify_image_readable: bool = False,
) -> HallusionSample:
    """Validate and normalize one raw HallusionBench.json record into HallusionSample."""
    errors = validate_raw_record(raw_record, index=index)
    if errors:
        raise HallusionSchemaError("; ".join(errors))

    category = parse_category(raw_record["category"])
    subcategory = str(raw_record["subcategory"]).strip()
    set_id = str(raw_record["set_id"]).strip()
    figure_id = str(raw_record["figure_id"]).strip()
    question_id = str(raw_record["question_id"]).strip()
    visual_input, visual_input_raw = parse_visual_input(raw_record["visual_input"])
    gt_answer, reference_answer = parse_gt_answer(raw_record["gt_answer"])
    question = str(raw_record["question"]).strip()

    existing_sid = raw_record.get("sample_id")
    if existing_sid is not None and str(existing_sid).strip():
        sample_id = str(existing_sid).strip()
    else:
        sample_id = make_stable_sample_id(
            category=category,
            subcategory=subcategory,
            set_id=set_id,
            figure_id=figure_id,
            question_id=question_id,
            visual_input_raw=visual_input_raw,
        )

    image_path, case_mismatch = resolve_image_path(
        raw_record=raw_record,
        visual_input=visual_input,
        data_root=data_root,
        verify_image_exists=verify_image_exists,
        verify_image_readable=verify_image_readable,
    )

    raw_filename = raw_record.get("filename")
    filename = str(raw_filename).strip() if raw_filename is not None and str(raw_filename).strip() else None

    gt_details_raw = raw_record.get("gt_answer_details")
    gt_answer_details = str(gt_details_raw) if gt_details_raw is not None else None

    note_raw = raw_record.get("sample_note")
    sample_note = str(note_raw) if note_raw is not None else None

    raw_difficulty = raw_record.get("difficulty")
    if isinstance(raw_difficulty, str) and raw_difficulty.strip():
        difficulty: str | None = raw_difficulty.strip().lower()
    else:
        difficulty = derive_difficulty(category, visual_input_raw)

    extra = {
        str(k): v
        for k, v in raw_record.items()
        if str(k) not in KNOWN_RAW_FIELDS and str(k) != "difficulty"
    }

    return HallusionSample(
        sample_id=sample_id,
        question=question,
        image_path=image_path,
        visual_input=visual_input,
        visual_input_raw=visual_input_raw,
        reference_answer=reference_answer,
        gt_answer=gt_answer,
        category=category,
        subcategory=subcategory,
        set_id=set_id,
        figure_id=figure_id,
        question_id=question_id,
        filename=filename,
        group_or_pair_key=make_group_or_pair_key(category, subcategory, set_id),
        figure_group_key=make_figure_group_key(category, subcategory, set_id, figure_id),
        question_pair_key=make_question_pair_key(category, subcategory, set_id, question_id),
        difficulty=difficulty,
        gt_answer_details=gt_answer_details,
        sample_note=sample_note,
        source_index=index,
        image_case_mismatch=case_mismatch,
        extra_fields=extra,
    )


def samples_to_dataset_index(samples: Sequence[HallusionSample]) -> dict[str, dict[str, Any]]:
    """Index normalized HallusionSample records by sample_id for Member C scoring/analysis."""
    return {sample.sample_id: sample.to_dict() for sample in samples}


def build_group_mappings(samples: Sequence[HallusionSample]) -> dict[str, Any]:
    """Build deterministic, order-preserving group/figure/pair mappings for Member C."""
    by_set_group: dict[str, list[str]] = {}
    by_figure_group: dict[str, list[str]] = {}
    by_visual_figure_group: dict[str, list[str]] = {}
    by_question_pair: dict[str, list[str]] = {}
    by_category: dict[str, list[str]] = {}
    by_subcategory: dict[str, list[str]] = {}
    by_visual_input_raw: dict[str, list[str]] = {}
    order_index: dict[str, int] = {}

    for pos, sample in enumerate(samples):
        sid = sample.sample_id
        order_index[sid] = pos
        by_set_group.setdefault(sample.group_or_pair_key, []).append(sid)
        by_figure_group.setdefault(sample.figure_group_key, []).append(sid)
        if sample.visual_input:
            by_visual_figure_group.setdefault(sample.figure_group_key, []).append(sid)
        by_question_pair.setdefault(sample.question_pair_key, []).append(sid)
        by_category.setdefault(sample.category, []).append(sid)
        by_subcategory.setdefault(sample.subcategory, []).append(sid)
        by_visual_input_raw.setdefault(sample.visual_input_raw, []).append(sid)

    return {
        "by_set_group": by_set_group,
        "by_figure_group": by_figure_group,
        "by_visual_figure_group": by_visual_figure_group,
        "by_question_pair": by_question_pair,
        "by_category": by_category,
        "by_subcategory": by_subcategory,
        "by_visual_input_raw": by_visual_input_raw,
        "order_index": order_index,
    }


def check_group_integrity(
    samples: Sequence[HallusionSample],
    *,
    flag_singleton_question_pairs: bool = True,
) -> list[str]:
    """Inspect structural consistency of HallusionBench sets, figures, and question pairs."""
    issues: list[str] = []

    by_fig_visual_paths: dict[str, set[str]] = {}
    by_qpair: dict[str, list[HallusionSample]] = {}
    by_set_visual_fig_ids: dict[str, set[str]] = {}

    for sample in samples:
        by_qpair.setdefault(sample.question_pair_key, []).append(sample)
        if sample.visual_input and sample.image_path:
            norm_path = str(Path(sample.image_path)).lower()
            by_fig_visual_paths.setdefault(sample.figure_group_key, set()).add(norm_path)
            by_set_visual_fig_ids.setdefault(sample.group_or_pair_key, set()).add(sample.figure_id)

    for fig_key, paths in by_fig_visual_paths.items():
        if len(paths) > 1:
            issues.append(
                f"Figure group '{fig_key}' references multiple distinct image paths: {sorted(paths)}"
            )

    for qpair_key, items in by_qpair.items():
        if flag_singleton_question_pairs and len(items) < 2:
            issues.append(
                f"Question pair '{qpair_key}' has only {len(items)} sample(s) "
                f"({[s.sample_id for s in items]}); verify whether control counterpart is missing"
            )
        fig_vis_keys = [(s.figure_id, s.visual_input_raw) for s in items]
        if len(fig_vis_keys) != len(set(fig_vis_keys)):
            issues.append(
                f"Question pair '{qpair_key}' contains duplicate (figure_id, visual_input_raw) entries: {fig_vis_keys}"
            )

    return issues


def select_frozen_pilot(
    samples: Sequence[HallusionSample],
    target_groups: int = 6,
    *,
    include_vd: bool = True,
    include_vs: bool = True,
    include_text_only: bool = True,
    include_edited_visual: bool = True,
) -> dict[str, Any]:
    """Select a deterministic, reproducible pilot manifest preserving COMPLETE set groups.

    Never splits a (category, subcategory, set_id) group so Member C can evaluate
    aAcc, fAcc, and qAcc accurately on pilot runs.
    """
    if target_groups <= 0:
        raise ValueError("target_groups must be a positive integer")
    if not samples:
        raise ValueError("Cannot select pilot from an empty sample list")

    groups: dict[str, list[HallusionSample]] = {}
    for sample in samples:
        groups.setdefault(sample.group_or_pair_key, []).append(sample)

    ordered_keys = list(groups.keys())
    selected_keys: list[str] = []

    def _pick_first_matching(predicate: Any) -> None:
        if len(selected_keys) >= target_groups:
            return
        for key in ordered_keys:
            if key in selected_keys:
                continue
            if predicate(groups[key]):
                selected_keys.append(key)
                return

    # 1. Prioritize groups that cover every key structural dimension of HallusionBench.
    if include_vs and include_text_only:
        _pick_first_matching(
            lambda items: any(s.category == "VS" for s in items)
            and any(s.visual_input_raw == "0" for s in items)
        )
    if include_vd and include_edited_visual:
        _pick_first_matching(
            lambda items: any(s.category == "VD" for s in items)
            and any(s.visual_input_raw == "2" for s in items)
        )
    if include_vd:
        _pick_first_matching(lambda items: any(s.category == "VD" for s in items))
    if include_vs:
        _pick_first_matching(lambda items: any(s.category == "VS" for s in items))
    if include_text_only:
        _pick_first_matching(lambda items: any(s.visual_input_raw == "0" for s in items))
    if include_edited_visual:
        _pick_first_matching(lambda items: any(s.visual_input_raw == "2" for s in items))

    # 2. Fill remaining slots while balancing subcategories deterministically.
    used_subcategories = {groups[k][0].subcategory for k in selected_keys}
    for key in ordered_keys:
        if len(selected_keys) >= target_groups:
            break
        if key in selected_keys:
            continue
        sub = groups[key][0].subcategory
        if sub not in used_subcategories:
            selected_keys.append(key)
            used_subcategories.add(sub)

    for key in ordered_keys:
        if len(selected_keys) >= target_groups:
            break
        if key not in selected_keys:
            selected_keys.append(key)

    selected_samples: list[HallusionSample] = []
    for key in selected_keys:
        selected_samples.extend(groups[key])

    sample_ids = [s.sample_id for s in selected_samples]
    mappings = build_group_mappings(selected_samples)

    return {
        "manifest_type": "hallusionbench_frozen_pilot",
        "selection_policy": {
            "preserve_complete_set_groups": True,
            "target_groups": target_groups,
            "include_vd": include_vd,
            "include_vs": include_vs,
            "include_text_only": include_text_only,
            "include_edited_visual": include_edited_visual,
        },
        "group_count": len(selected_keys),
        "sample_count": len(sample_ids),
        "figure_group_count": len(mappings["by_figure_group"]),
        "visual_figure_group_count": len(mappings["by_visual_figure_group"]),
        "question_pair_count": len(mappings["by_question_pair"]),
        "breakdown_by_category": dict(Counter(s.category for s in selected_samples)),
        "breakdown_by_subcategory": dict(Counter(s.subcategory for s in selected_samples)),
        "breakdown_by_visual_input_raw": dict(Counter(s.visual_input_raw for s in selected_samples)),
        "selected_group_keys": selected_keys,
        "sample_ids": sample_ids,
        "pilot_ids_sha256": sha256_json(sample_ids),
    }


def filter_by_pilot_ids(
    samples: Sequence[HallusionSample],
    pilot_ids: Sequence[str],
    *,
    require_complete_groups: bool = True,
) -> list[HallusionSample]:
    """Filter samples strictly to a frozen pilot ID list, checking completeness and group integrity."""
    if not pilot_ids:
        raise ValueError("pilot_ids cannot be empty")
    if len(pilot_ids) != len(set(pilot_ids)):
        raise ValueError("pilot_ids contains duplicate sample IDs")

    by_id = {s.sample_id: s for s in samples}
    missing = [sid for sid in pilot_ids if sid not in by_id]
    if missing:
        raise ValueError(f"Frozen pilot IDs missing from dataset: {missing[:10]} (total missing: {len(missing)})")

    selected = [by_id[sid] for sid in pilot_ids]

    if require_complete_groups:
        selected_set = set(pilot_ids)
        full_groups = build_group_mappings(samples)["by_set_group"]
        selected_group_keys = {s.group_or_pair_key for s in selected}
        incomplete: dict[str, list[str]] = {}
        for gkey in sorted(selected_group_keys):
            missing_in_group = [sid for sid in full_groups[gkey] if sid not in selected_set]
            if missing_in_group:
                incomplete[gkey] = missing_in_group
        if incomplete:
            raise GroupIntegrityError(
                f"Pilot selection splits complete set groups (breaks fAcc/qAcc): {incomplete}"
            )

    return selected


class HallusionBenchDataset:
    """Read-only HallusionBench dataset loader preserving original order and group/pair structure."""

    def __init__(
        self,
        json_path: str | Path,
        data_root: str | Path | None = None,
        *,
        dataset_source: str = OFFICIAL_GITHUB_SOURCE,
        dataset_revision: str | None = None,
        license_name: str = OFFICIAL_LICENSE,
    ) -> None:
        self.json_path = Path(json_path)
        self.data_root = Path(data_root) if data_root is not None else self.json_path.parent
        self.dataset_source = dataset_source
        self.dataset_revision = dataset_revision
        self.license_name = license_name

    def read_raw(self) -> list[dict[str, Any]]:
        """Read raw HallusionBench.json without downloading or mutating it."""
        if not self.json_path.is_file():
            raise FileNotFoundError(f"Dataset JSON file not found: {self.json_path}")
        raw_items = json.loads(self.json_path.read_text(encoding="utf-8"))
        if not isinstance(raw_items, list):
            raise HallusionSchemaError(f"Expected a JSON list in {self.json_path}, got {type(raw_items).__name__}")
        return raw_items

    def load(
        self,
        *,
        verify_images: bool = True,
        verify_image_readable: bool = False,
        verify_group_integrity: bool = False,
    ) -> list[HallusionSample]:
        """Load, validate, and normalize all HallusionBench records in strict mode."""
        raw_items = self.read_raw()
        samples: list[HallusionSample] = []
        seen_ids: dict[str, int] = {}

        for idx, raw_item in enumerate(raw_items):
            sample = normalize_hallusion_record(
                raw_item,
                data_root=self.data_root,
                index=idx,
                verify_image_exists=verify_images,
                verify_image_readable=verify_image_readable,
            )
            if sample.sample_id in seen_ids:
                raise DuplicateSampleIdError(
                    f"Duplicate sample_id '{sample.sample_id}' at raw indices {seen_ids[sample.sample_id]} and {idx}"
                )
            seen_ids[sample.sample_id] = idx
            samples.append(sample)

        norm_errors = validate_normalized_samples(samples)
        if norm_errors:
            raise HallusionSchemaError("; ".join(norm_errors))

        if verify_group_integrity:
            group_issues = check_group_integrity(samples, flag_singleton_question_pairs=False)
            if group_issues:
                raise GroupIntegrityError("; ".join(group_issues))

        return samples

    def load_dicts(
        self,
        *,
        verify_images: bool = True,
        verify_image_readable: bool = False,
        model_safe: bool = False,
    ) -> list[dict[str, Any]]:
        """Load normalized records as dictionaries (directly consumable by Member B / Member C)."""
        samples = self.load(
            verify_images=verify_images,
            verify_image_readable=verify_image_readable,
        )
        if model_safe:
            return [s.to_model_safe_dict() for s in samples]
        return [s.to_dict() for s in samples]

    def load_index(
        self,
        *,
        verify_images: bool = True,
        verify_image_readable: bool = False,
    ) -> dict[str, dict[str, Any]]:
        """Load normalized records keyed by sample_id (directly consumable by Member C)."""
        samples = self.load(
            verify_images=verify_images,
            verify_image_readable=verify_image_readable,
        )
        return samples_to_dataset_index(samples)

    def audit(self, *, verify_image_readable: bool = True) -> dict[str, Any]:
        """Run comprehensive programmatic audit of HallusionBench.json and images on disk."""
        raw_items = self.read_raw()
        json_checksum = sha256_file(self.json_path)

        schema_errors: list[dict[str, Any]] = []
        valid_samples: list[HallusionSample] = []
        duplicate_sample_ids: list[dict[str, Any]] = []
        seen_ids: dict[str, int] = {}

        missing_image_samples: list[dict[str, Any]] = []
        corrupt_image_samples: list[dict[str, Any]] = []
        case_mismatch_samples: list[dict[str, Any]] = []
        text_only_with_filename: list[dict[str, Any]] = []
        existing_image_paths: set[str] = set()
        referenced_image_paths: set[str] = set()

        for idx, item in enumerate(raw_items):
            rec_errors = validate_raw_record(item, index=idx)
            if rec_errors:
                schema_errors.append({"index": idx, "errors": rec_errors})
                continue

            sample = normalize_hallusion_record(
                item,
                data_root=self.data_root,
                index=idx,
                verify_image_exists=False,
                verify_image_readable=False,
            )

            if sample.sample_id in seen_ids:
                duplicate_sample_ids.append(
                    {
                        "sample_id": sample.sample_id,
                        "first_index": seen_ids[sample.sample_id],
                        "duplicate_index": idx,
                    }
                )
            else:
                seen_ids[sample.sample_id] = idx

            if not sample.visual_input and sample.filename is not None:
                text_only_with_filename.append(
                    {"sample_id": sample.sample_id, "index": idx, "raw_filename": sample.filename}
                )

            if sample.visual_input and sample.image_path is not None:
                referenced_image_paths.add(sample.image_path)
                img_path = Path(sample.image_path)
                if not img_path.is_file():
                    missing_image_samples.append(
                        {
                            "sample_id": sample.sample_id,
                            "index": idx,
                            "filename": sample.filename,
                            "resolved_path": sample.image_path,
                        }
                    )
                else:
                    existing_image_paths.add(str(img_path.resolve()))
                    if sample.image_case_mismatch:
                        case_mismatch_samples.append(
                            {
                                "sample_id": sample.sample_id,
                                "index": idx,
                                "raw_filename": sample.filename,
                                "resolved_path": sample.image_path,
                            }
                        )
                    if verify_image_readable:
                        ok, err = check_image_readable(img_path)
                        if not ok:
                            corrupt_image_samples.append(
                                {
                                    "sample_id": sample.sample_id,
                                    "index": idx,
                                    "resolved_path": sample.image_path,
                                    "error": err,
                                }
                            )

            valid_samples.append(sample)

        mappings = build_group_mappings(valid_samples)
        pair_size_dist = dict(sorted(Counter(len(v) for v in mappings["by_question_pair"].values()).items()))
        fig_size_dist = dict(sorted(Counter(len(v) for v in mappings["by_figure_group"].values()).items()))
        group_warnings = check_group_integrity(valid_samples, flag_singleton_question_pairs=True)

        return {
            "dataset_name": "HallusionBench",
            "dataset_source": self.dataset_source,
            "dataset_revision": self.dataset_revision,
            "license": self.license_name,
            "json_file": self.json_path.name,
            "json_sha256": json_checksum,
            "counts": {
                "total_raw_records": len(raw_items),
                "valid_records": len(valid_samples),
                "schema_error_records": len(schema_errors),
                "duplicate_sample_id_count": len(duplicate_sample_ids),
                "visual_samples": sum(1 for s in valid_samples if s.visual_input),
                "text_only_samples": sum(1 for s in valid_samples if not s.visual_input),
                "unique_referenced_image_paths": len(referenced_image_paths),
                "unique_existing_image_files": len(existing_image_paths),
                "missing_image_samples": len(missing_image_samples),
                "corrupt_image_samples": len(corrupt_image_samples),
                "case_mismatch_image_samples": len(case_mismatch_samples),
                "text_only_with_ignored_filename": len(text_only_with_filename),
                "set_group_count": len(mappings["by_set_group"]),
                "figure_group_count": len(mappings["by_figure_group"]),
                "visual_figure_group_count": len(mappings["by_visual_figure_group"]),
                "question_pair_count": len(mappings["by_question_pair"]),
            },
            "distributions": {
                "category": dict(sorted(Counter(s.category for s in valid_samples).items())),
                "subcategory": dict(sorted(Counter(s.subcategory for s in valid_samples).items())),
                "visual_input_raw": dict(sorted(Counter(s.visual_input_raw for s in valid_samples).items())),
                "gt_answer": dict(sorted(Counter(s.gt_answer for s in valid_samples).items())),
                "reference_answer": dict(sorted(Counter(s.reference_answer for s in valid_samples).items())),
                "question_pair_sizes": {str(k): v for k, v in pair_size_dist.items()},
                "figure_group_sizes": {str(k): v for k, v in fig_size_dist.items()},
            },
            "integrity": {
                "is_clean": (
                    len(schema_errors) == 0
                    and len(duplicate_sample_ids) == 0
                    and len(missing_image_samples) == 0
                    and len(corrupt_image_samples) == 0
                ),
                "schema_errors": schema_errors,
                "duplicate_sample_ids": duplicate_sample_ids,
                "missing_image_samples": missing_image_samples,
                "corrupt_image_samples": corrupt_image_samples,
                "case_mismatch_samples": case_mismatch_samples,
                "text_only_with_ignored_filename": text_only_with_filename,
                "group_integrity_warnings": group_warnings,
            },
        }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="HallusionBench Member A dataset audit & pilot CLI")
    parser.add_argument(
        "--mode",
        choices=("audit", "export-pilot", "verify-pilot"),
        required=True,
        help="audit: inspect dataset & images; export-pilot: generate frozen pilot IDs; verify-pilot: check pilot IDs",
    )
    parser.add_argument("--dataset-json", type=Path, required=True, help="Path to local HallusionBench.json")
    parser.add_argument("--data-root", type=Path, default=None, help="Root directory containing VD/VS images")
    parser.add_argument("--dataset-source", default=OFFICIAL_GITHUB_SOURCE)
    parser.add_argument("--dataset-revision", default=None)
    parser.add_argument("--skip-image-integrity", action="store_true", help="Skip PIL/header readability check in audit")
    parser.add_argument("--target-groups", type=int, default=6, help="Number of complete set groups for pilot export")
    parser.add_argument("--pilot-file", type=Path, default=None, help="Pilot JSON manifest for verify-pilot")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output file path")
    args = parser.parse_args(argv)

    dataset = HallusionBenchDataset(
        json_path=args.dataset_json,
        data_root=args.data_root,
        dataset_source=args.dataset_source,
        dataset_revision=args.dataset_revision,
    )

    if args.mode == "audit":
        report = dataset.audit(verify_image_readable=not args.skip_image_integrity)
        text = json.dumps(report, indent=2, ensure_ascii=False)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + "\n", encoding="utf-8")
        print(text)
        return 0 if report["integrity"]["is_clean"] else 1

    if args.mode == "export-pilot":
        samples = dataset.load(
            verify_images=True,
            verify_image_readable=not args.skip_image_integrity,
        )
        manifest = select_frozen_pilot(samples, target_groups=args.target_groups)
        manifest["dataset_source"] = args.dataset_source
        manifest["dataset_revision"] = args.dataset_revision
        manifest["json_sha256"] = sha256_file(args.dataset_json)
        text = json.dumps(manifest, indent=2, ensure_ascii=False)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + "\n", encoding="utf-8")
        print(text)
        return 0

    if args.mode == "verify-pilot":
        if args.pilot_file is None:
            raise ValueError("--pilot-file is required for --mode verify-pilot")
        pilot_manifest = json.loads(args.pilot_file.read_text(encoding="utf-8"))
        pilot_ids = pilot_manifest.get("sample_ids")
        if not isinstance(pilot_ids, list):
            raise ValueError(f"Invalid pilot manifest {args.pilot_file}: missing list field 'sample_ids'")
        expected_sha = pilot_manifest.get("pilot_ids_sha256")
        actual_sha = sha256_json(pilot_ids)
        if expected_sha is not None and actual_sha != expected_sha:
            raise ValueError(f"Pilot ID checksum mismatch: expected {expected_sha}, got {actual_sha}")

        samples = dataset.load(verify_images=True, verify_image_readable=False)
        filtered = filter_by_pilot_ids(samples, pilot_ids, require_complete_groups=True)
        summary = {
            "status": "VALID",
            "verified_sample_count": len(filtered),
            "verified_group_count": len({s.group_or_pair_key for s in filtered}),
            "pilot_ids_sha256": actual_sha,
        }
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
