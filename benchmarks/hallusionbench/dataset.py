"""HallusionBench Part A — Schema, read-only dataset loader, integrity auditor, and 8-subcategory pilot selector.

Design rules (docs/dataset_policy.md & benchmarks/base.py):
- Never downloads datasets automatically on import or load.
- Never mutates raw questions, images, or ground-truth labels.
- Preserves text-only controls strictly (visual_input="0" -> image_path=None).
- Protects model inputs from ground-truth label leakage via to_model_safe_dict().
- Preserves set, figure, and question-pair links for aAcc / fAcc / qAcc scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Official HallusionBench metadata & schema constants
# ---------------------------------------------------------------------------

DATASET_NAME = "HallusionBench"
OFFICIAL_GITHUB_SOURCE = "tianyi-lab/HallusionBench"
HF_AUTHOR_MIRROR_SOURCE = "rayguan/HallusionBench"
KNOWN_HF_REVISION_CANDIDATE = "2067dfdce5abc7efdbb0927f4721ae282f0b6c51"
OFFICIAL_LICENSE = "BSD-3-Clause"

VALID_CATEGORIES = ("VD", "VS")
VALID_VISUAL_INPUT_CODES = ("0", "1", "2")
GT_ANSWER_TO_YESNO = {"0": "no", "1": "yes"}
YESNO_TO_GT_ANSWER = {"no": "0", "yes": "1"}

REQUIRED_RAW_FIELDS = (
    "category",
    "subcategory",
    "visual_input",
    "set_id",
    "figure_id",
    "question_id",
    "question",
    "gt_answer",
)
OPTIONAL_RAW_FIELDS = ("filename", "gt_answer_details", "sample_note", "same")
KNOWN_RAW_FIELDS = frozenset(
    REQUIRED_RAW_FIELDS + OPTIONAL_RAW_FIELDS + ("sample_id", "image_path", "reference_answer", "difficulty")
)

# Fields that MUST NEVER be passed into prompt construction or model inputs.
FORBIDDEN_MODEL_INPUT_FIELDS = frozenset(
    {
        "gt_answer",
        "gt_answer_details",
        "reference_answer",
        "reference",
        "answer",
        "label",
        "sample_note",
    }
)


class HallusionSchemaError(ValueError):
    """Raised when a raw or normalized HallusionBench record violates schema rules."""


class DuplicateSampleIdError(HallusionSchemaError):
    """Raised when two records resolve to the exact same stable sample_id."""


class GroupIntegrityError(HallusionSchemaError):
    """Raised when HallusionBench group/figure/question-pair links are broken."""


# ---------------------------------------------------------------------------
# Strict field parsers & deterministic ID / grouping helpers
# ---------------------------------------------------------------------------

def _clean_id_token(value: Any, field_name: str) -> str:
    if value is None or isinstance(value, bool):
        raise HallusionSchemaError(f"Field '{field_name}' must be a non-empty string/int, got {value!r}")
    token = str(value).strip()
    if not token:
        raise HallusionSchemaError(f"Field '{field_name}' cannot be empty")
    return token


def parse_visual_input(value: Any) -> tuple[bool, str]:
    """Parse visual_input strictly into (requires_image, raw_code).

    - '0' / 0 / False -> text-only control (no image)
    - '1' / 1 / True  -> original image
    - '2' / 2         -> edited/manipulated image
    """
    if isinstance(value, bool):
        return (True, "1") if value else (False, "0")
    if isinstance(value, int):
        if value == 0:
            return False, "0"
        if value == 1:
            return True, "1"
        if value == 2:
            return True, "2"
    if isinstance(value, str):
        norm = value.strip().lower()
        if norm in {"0", "false", "no"}:
            return False, "0"
        if norm in {"1", "true", "yes"}:
            return True, "1"
        if norm == "2":
            return True, "2"
    raise HallusionSchemaError(f"Unsupported visual_input value: {value!r}; expected 0, 1, or 2")


def parse_gt_answer(value: Any) -> tuple[str, str]:
    """Parse ground-truth label into (raw_gt_answer, reference_answer)."""
    if isinstance(value, bool):
        raise HallusionSchemaError(f"Ambiguous boolean gt_answer {value!r}; expected '0'/'1' or 'no'/'yes'")
    if isinstance(value, int):
        if value == 0:
            return "0", "no"
        if value == 1:
            return "1", "yes"
    if isinstance(value, str):
        norm = value.strip().lower()
        if norm in GT_ANSWER_TO_YESNO:
            return norm, GT_ANSWER_TO_YESNO[norm]
        if norm in YESNO_TO_GT_ANSWER:
            return YESNO_TO_GT_ANSWER[norm], norm
    raise HallusionSchemaError(f"Unsupported gt_answer value: {value!r}; expected '0', '1', 'no', or 'yes'")


def parse_category(value: Any) -> str:
    """Validate HallusionBench top-level category ('VD' or 'VS')."""
    if not isinstance(value, str) or not value.strip():
        raise HallusionSchemaError(f"Missing or invalid category: {value!r}")
    cat = value.strip().upper()
    if cat not in VALID_CATEGORIES:
        raise HallusionSchemaError(f"Unsupported category {value!r}; expected one of {VALID_CATEGORIES}")
    return cat


def derive_difficulty(category: str, visual_input_raw: str) -> str | None:
    """Derive structural easy/hard tag from HallusionBench category and visual_input."""
    cat = category.strip().upper()
    vis = str(visual_input_raw).strip()
    if cat == "VD":
        return "easy" if vis == "1" else ("hard" if vis == "2" else None)
    if cat == "VS":
        return "easy" if vis == "0" else ("hard" if vis in {"1", "2"} else None)
    return None


def make_stable_sample_id(
    category: str,
    subcategory: str,
    set_id: str,
    figure_id: str,
    question_id: str,
    visual_input_raw: str,
) -> str:
    """Build deterministic composite sample_id (question_id alone is NOT unique)."""
    return (
        f"cat{category}_sub{subcategory}_set{set_id}_"
        f"fig{figure_id}_q{question_id}_vis{visual_input_raw}"
    )


def make_group_or_pair_key(category: str, subcategory: str, set_id: str) -> str:
    return f"{category}::{subcategory}::set_{set_id}"


def make_figure_group_key(category: str, subcategory: str, set_id: str, figure_id: str) -> str:
    return f"{category}::{subcategory}::set_{set_id}::fig_{figure_id}"


def make_question_pair_key(category: str, subcategory: str, set_id: str, question_id: str) -> str:
    return f"{category}::{subcategory}::set_{set_id}::q_{question_id}"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


# ---------------------------------------------------------------------------
# Sample dataclass & validation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HallusionSample:
    """Normalized, immutable HallusionBench sample contract shared across A -> B -> C."""

    sample_id: str
    question: str
    image_path: str | None
    visual_input: bool
    visual_input_raw: str
    reference_answer: str
    gt_answer: str
    category: str
    subcategory: str
    set_id: str
    figure_id: str
    question_id: str
    filename: str | None
    group_or_pair_key: str
    figure_group_key: str
    question_pair_key: str
    difficulty: str | None = None
    gt_answer_details: str | None = None
    sample_note: str | None = None
    source_index: int | None = None
    image_case_mismatch: bool = False
    extra_fields: dict[str, Any] = field(default_factory=dict)

    @property
    def figure_key(self) -> str | None:
        """Figure key for fAcc (None for text-only samples with no visual figure)."""
        return self.figure_group_key if self.visual_input else None

    @property
    def pair_key(self) -> str:
        """Question-pair key for qAcc."""
        return self.question_pair_key

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["figure_key"] = self.figure_key
        payload["pair_key"] = self.pair_key
        return payload

    def to_model_safe_dict(self) -> dict[str, Any]:
        """Return a label-free dict excluding all FORBIDDEN_MODEL_INPUT_FIELDS."""
        payload = self.to_dict()
        for forbidden in FORBIDDEN_MODEL_INPUT_FIELDS:
            payload.pop(forbidden, None)
        return payload


def validate_raw_record(raw_record: Any, index: int | None = None) -> list[str]:
    prefix = f"Record[{index}]" if index is not None else "Record"
    if not isinstance(raw_record, Mapping):
        return [f"{prefix}: expected a JSON object/mapping, got {type(raw_record).__name__}"]

    errors: list[str] = []
    for req_field in REQUIRED_RAW_FIELDS:
        if req_field not in raw_record or raw_record[req_field] is None:
            errors.append(f"{prefix}: missing required field '{req_field}'")

    if raw_record.get("category") is not None:
        try:
            parse_category(raw_record["category"])
        except HallusionSchemaError as exc:
            errors.append(f"{prefix}: {exc}")

    for id_field in ("subcategory", "set_id", "figure_id", "question_id"):
        if raw_record.get(id_field) is not None:
            try:
                _clean_id_token(raw_record[id_field], id_field)
            except HallusionSchemaError as exc:
                errors.append(f"{prefix}: {exc}")

    question = raw_record.get("question")
    if question is not None and (not isinstance(question, str) or not question.strip()):
        errors.append(f"{prefix}: field 'question' must be a non-empty string")

    visual_bool: bool | None = None
    if raw_record.get("visual_input") is not None:
        try:
            visual_bool, _ = parse_visual_input(raw_record["visual_input"])
        except HallusionSchemaError as exc:
            errors.append(f"{prefix}: {exc}")

    if raw_record.get("gt_answer") is not None:
        try:
            parse_gt_answer(raw_record["gt_answer"])
        except HallusionSchemaError as exc:
            errors.append(f"{prefix}: {exc}")

    if visual_bool is True:
        raw_filename = raw_record.get("filename")
        raw_image_path = raw_record.get("image_path")
        has_filename = isinstance(raw_filename, str) and bool(raw_filename.strip())
        has_image_path = isinstance(raw_image_path, str) and bool(raw_image_path.strip())
        if not (has_filename or has_image_path):
            errors.append(f"{prefix}: visual_input requires a non-empty 'filename' or 'image_path'")

    return errors


def validate_normalized_samples(samples: Sequence[HallusionSample]) -> list[str]:
    errors: list[str] = []
    seen_ids: dict[str, int] = {}
    for idx, sample in enumerate(samples):
        if sample.sample_id in seen_ids:
            errors.append(f"Duplicate sample_id '{sample.sample_id}' at indices {seen_ids[sample.sample_id]} and {idx}")
        else:
            seen_ids[sample.sample_id] = idx
        if not sample.question.strip():
            errors.append(f"Sample {sample.sample_id}: empty question")
        if sample.category not in VALID_CATEGORIES:
            errors.append(f"Sample {sample.sample_id}: invalid category {sample.category!r}")
        if sample.visual_input_raw not in VALID_VISUAL_INPUT_CODES:
            errors.append(f"Sample {sample.sample_id}: invalid visual_input_raw {sample.visual_input_raw!r}")
        if sample.visual_input is False and sample.image_path is not None:
            errors.append(f"Sample {sample.sample_id}: text-only sample must have image_path=None")
        if sample.visual_input is True and (not sample.image_path or not str(sample.image_path).strip()):
            errors.append(f"Sample {sample.sample_id}: visual sample is missing image_path")
        if sample.gt_answer not in GT_ANSWER_TO_YESNO or GT_ANSWER_TO_YESNO[sample.gt_answer] != sample.reference_answer:
            errors.append(f"Sample {sample.sample_id}: gt_answer/reference_answer mismatch")
    return errors


# ---------------------------------------------------------------------------
# Image resolution & readability checks
# ---------------------------------------------------------------------------

def _find_existing_case_insensitive(
    path: Path,
    expected_name: str | None = None,
) -> tuple[Path | None, bool]:
    """Resolve filename case-insensitively while flagging OS-independent case mismatches."""
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
        header = path.read_bytes()[:12]
        valid_magic = (
            header.startswith(b"\x89PNG\r\n\x1a\n")
            or header.startswith(b"\xff\xd8\xff")
            or header.startswith(b"GIF87a")
            or header.startswith(b"GIF89a")
            or (header.startswith(b"RIFF") and header[8:12] == b"WEBP")
            or header.startswith(b"BM")
        )
        return (True, None) if valid_magic else (False, f"Unrecognized image header: {path}")
    except Exception as exc:
        return False, f"Corrupt or unreadable image ({type(exc).__name__}): {exc}"


def resolve_image_path(
    raw_record: Mapping[str, Any],
    visual_input: bool,
    data_root: str | Path | None = None,
    verify_image_exists: bool = True,
    verify_image_readable: bool = False,
) -> tuple[str | None, bool]:
    """Resolve image path across direct, data/, and hallusion_bench/ layouts."""
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


# ---------------------------------------------------------------------------
# Normalization, grouping, pilot selection, and dataset loader/auditor
# ---------------------------------------------------------------------------

def normalize_hallusion_record(
    raw_record: Mapping[str, Any],
    data_root: str | Path | None = None,
    index: int | None = None,
    verify_image_exists: bool = True,
    verify_image_readable: bool = False,
) -> HallusionSample:
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
    sample_id = (
        str(existing_sid).strip()
        if existing_sid is not None and str(existing_sid).strip()
        else make_stable_sample_id(category, subcategory, set_id, figure_id, question_id, visual_input_raw)
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

    raw_difficulty = raw_record.get("difficulty")
    difficulty = (
        raw_difficulty.strip().lower()
        if isinstance(raw_difficulty, str) and raw_difficulty.strip()
        else derive_difficulty(category, visual_input_raw)
    )

    extra = {str(k): v for k, v in raw_record.items() if str(k) not in KNOWN_RAW_FIELDS}

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
        gt_answer_details=str(raw_record["gt_answer_details"]) if raw_record.get("gt_answer_details") is not None else None,
        sample_note=str(raw_record["sample_note"]) if raw_record.get("sample_note") is not None else None,
        source_index=index,
        image_case_mismatch=case_mismatch,
        extra_fields=extra,
    )


def samples_to_dataset_index(samples: Sequence[HallusionSample]) -> dict[str, dict[str, Any]]:
    return {sample.sample_id: sample.to_dict() for sample in samples}


def build_group_mappings(samples: Sequence[HallusionSample]) -> dict[str, Any]:
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
    issues: list[str] = []
    by_fig_visual_paths: dict[str, set[str]] = {}
    by_qpair: dict[str, list[HallusionSample]] = {}

    for sample in samples:
        by_qpair.setdefault(sample.question_pair_key, []).append(sample)
        if sample.visual_input and sample.image_path:
            by_fig_visual_paths.setdefault(sample.figure_group_key, set()).add(str(Path(sample.image_path)).lower())

    for fig_key, paths in by_fig_visual_paths.items():
        if len(paths) > 1:
            issues.append(f"Figure group '{fig_key}' references multiple distinct image paths: {sorted(paths)}")

    for qpair_key, items in by_qpair.items():
        if flag_singleton_question_pairs and len(items) < 2:
            issues.append(f"Question pair '{qpair_key}' has only {len(items)} sample(s) ({[s.sample_id for s in items]})")
        fig_vis = [(s.figure_id, s.visual_input_raw) for s in items]
        if len(fig_vis) != len(set(fig_vis)):
            issues.append(f"Question pair '{qpair_key}' contains duplicate (figure_id, visual_input_raw) entries")

    return issues


def select_frozen_pilot(
    samples: Sequence[HallusionSample],
    target_groups: int = 8,
    *,
    balance_subcategories: bool = True,
) -> dict[str, Any]:
    """Select a deterministic frozen pilot preserving COMPLETE set groups across ALL subcategories.

    By default (target_groups=8, balance_subcategories=True), picks 1 complete control group
    per HallusionBench subcategory (chart, figure, illusion, map, math, ocr, table, video),
    preferring compact groups with multi-condition pairs (visual_input 0/1/2) so aAcc, fAcc,
    qAcc, VD/VS, and Easy/Hard are all well-represented in ~50-80 samples.
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

    def _group_rank(gkey: str) -> tuple[int, int, int, str]:
        items = groups[gkey]
        vis_types = {s.visual_input_raw for s in items}
        pair_counts = Counter(s.question_pair_key for s in items)
        has_complete_pair = any(cnt >= 2 for cnt in pair_counts.values())
        # Prefer: (1) has complete control pair, (2) has >=2 visual_input conditions, (3) compact size.
        return (0 if has_complete_pair else 1, 0 if len(vis_types) >= 2 else 1, len(items), gkey)

    if balance_subcategories:
        by_sub: dict[str, list[str]] = {}
        for gkey in ordered_keys:
            sub = groups[gkey][0].subcategory
            by_sub.setdefault(sub, []).append(gkey)
        for sub in sorted(by_sub.keys()):
            if len(selected_keys) >= target_groups:
                break
            best_key = min(by_sub[sub], key=_group_rank)
            selected_keys.append(best_key)

    # Ensure both VD & VS and text-only / edited visual conditions are covered if slots remain.
    for predicate in (
        lambda items: any(s.visual_input_raw == "0" for s in items),
        lambda items: any(s.visual_input_raw == "2" for s in items),
        lambda items: any(s.category == "VD" for s in items),
        lambda items: any(s.category == "VS" for s in items),
    ):
        if len(selected_keys) >= target_groups:
            break
        already_covered = any(predicate(groups[k]) for k in selected_keys)
        if not already_covered:
            matching = [k for k in ordered_keys if k not in selected_keys and predicate(groups[k])]
            if matching:
                selected_keys.append(min(matching, key=_group_rank))

    for gkey in ordered_keys:
        if len(selected_keys) >= target_groups:
            break
        if gkey not in selected_keys:
            selected_keys.append(gkey)

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
            "balance_subcategories": balance_subcategories,
        },
        "group_count": len(selected_keys),
        "sample_count": len(sample_ids),
        "figure_group_count": len(mappings["by_figure_group"]),
        "visual_figure_group_count": len(mappings["by_visual_figure_group"]),
        "question_pair_count": len(mappings["by_question_pair"]),
        "breakdown_by_category": dict(sorted(Counter(s.category for s in selected_samples).items())),
        "breakdown_by_subcategory": dict(sorted(Counter(s.subcategory for s in selected_samples).items())),
        "breakdown_by_visual_input_raw": dict(sorted(Counter(s.visual_input_raw for s in selected_samples).items())),
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
        incomplete = {
            gkey: [sid for sid in full_groups[gkey] if sid not in selected_set]
            for gkey in sorted({s.group_or_pair_key for s in selected})
            if any(sid not in selected_set for sid in full_groups[gkey])
        }
        if incomplete:
            raise GroupIntegrityError(f"Pilot selection splits complete set groups: {incomplete}")

    return selected


class HallusionBenchDataset:
    """Read-only HallusionBench dataset loader preserving original order and group/pair structure."""

    def __init__(
        self,
        json_path: str | Path,
        data_root: str | Path | None = None,
        *,
        dataset_source: str = HF_AUTHOR_MIRROR_SOURCE,
        dataset_revision: str | None = None,
        license_name: str = OFFICIAL_LICENSE,
    ) -> None:
        self.json_path = Path(json_path)
        self.data_root = Path(data_root) if data_root is not None else self.json_path.parent
        self.dataset_source = dataset_source
        self.dataset_revision = dataset_revision
        self.license_name = license_name

    def read_raw(self) -> list[dict[str, Any]]:
        if not self.json_path.is_file():
            raise FileNotFoundError(f"Dataset JSON file not found: {self.json_path}")
        raw_items = json.loads(self.json_path.read_text(encoding="utf-8"))
        if not isinstance(raw_items, list):
            raise HallusionSchemaError(f"Expected a JSON list in {self.json_path}")
        return raw_items

    def load(
        self,
        *,
        verify_images: bool = True,
        verify_image_readable: bool = False,
        verify_group_integrity: bool = False,
    ) -> list[HallusionSample]:
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
        samples = self.load(verify_images=verify_images, verify_image_readable=verify_image_readable)
        return [s.to_model_safe_dict() if model_safe else s.to_dict() for s in samples]

    def load_index(
        self,
        *,
        verify_images: bool = True,
        verify_image_readable: bool = False,
    ) -> dict[str, dict[str, Any]]:
        return samples_to_dataset_index(
            self.load(verify_images=verify_images, verify_image_readable=verify_image_readable)
        )

    def audit(self, *, verify_image_readable: bool = True) -> dict[str, Any]:
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
                    {"sample_id": sample.sample_id, "first_index": seen_ids[sample.sample_id], "duplicate_index": idx}
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
                        {"sample_id": sample.sample_id, "index": idx, "filename": sample.filename, "resolved_path": sample.image_path}
                    )
                else:
                    existing_image_paths.add(str(img_path.resolve()))
                    if sample.image_case_mismatch:
                        case_mismatch_samples.append(
                            {"sample_id": sample.sample_id, "index": idx, "raw_filename": sample.filename, "resolved_path": sample.image_path}
                        )
                    if verify_image_readable:
                        ok, err = check_image_readable(img_path)
                        if not ok:
                            corrupt_image_samples.append(
                                {"sample_id": sample.sample_id, "index": idx, "resolved_path": sample.image_path, "error": err}
                            )

            valid_samples.append(sample)

        mappings = build_group_mappings(valid_samples)
        pair_size_dist = dict(sorted(Counter(len(v) for v in mappings["by_question_pair"].values()).items()))
        fig_size_dist = dict(sorted(Counter(len(v) for v in mappings["by_figure_group"].values()).items()))
        group_warnings = check_group_integrity(valid_samples, flag_singleton_question_pairs=True)

        return {
            "dataset_name": DATASET_NAME,
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
    parser = argparse.ArgumentParser(description="HallusionBench Part A dataset audit & pilot CLI")
    parser.add_argument("--mode", choices=("audit", "export-pilot", "verify-pilot"), required=True)
    parser.add_argument("--dataset-json", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--dataset-source", default=HF_AUTHOR_MIRROR_SOURCE)
    parser.add_argument("--dataset-revision", default=None)
    parser.add_argument("--skip-image-integrity", action="store_true")
    parser.add_argument("--target-groups", type=int, default=8, help="Number of complete groups (default 8 = all 8 subcategories)")
    parser.add_argument("--pilot-file", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
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
        samples = dataset.load(verify_images=True, verify_image_readable=not args.skip_image_integrity)
        manifest = select_frozen_pilot(samples, target_groups=args.target_groups, balance_subcategories=True)
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
        print(
            json.dumps(
                {
                    "status": "VALID",
                    "verified_sample_count": len(filtered),
                    "verified_group_count": len({s.group_or_pair_key for s in filtered}),
                    "subcategories": sorted({s.subcategory for s in filtered}),
                    "pilot_ids_sha256": actual_sha,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
