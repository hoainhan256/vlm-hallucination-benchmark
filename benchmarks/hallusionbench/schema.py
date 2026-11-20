"""HallusionBench dataset schema, validation, and stable ID/grouping contracts (Member A).

Design constraints enforced here (per docs/dataset_policy.md & team plan Section 3):
- Never mutate raw questions, images, or ground-truth labels.
- Strictly preserve text-only control samples (visual_input="0"): image_path MUST be None.
- Preserve both raw official fields (gt_answer, visual_input_raw, set_id, figure_id,
  question_id, filename, gt_answer_details, sample_note) and derived structural keys
  needed by Member B (inference) and Member C (scoring/error analysis).
- Maintain exact sample_id and group_or_pair_key compatibility with Member B's protocol.py.
- Never leak ground-truth/reference fields into model-facing dictionaries.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence


DATASET_NAME = "HallusionBench"
OFFICIAL_GITHUB_SOURCE = "tianyi-lab/HallusionBench"
OFFICIAL_GITHUB_URL = "https://github.com/tianyi-lab/HallusionBench"
HF_AUTHOR_MIRROR_SOURCE = "rayguan/HallusionBench"
HF_AUTHOR_MIRROR_URL = "https://huggingface.co/datasets/rayguan/HallusionBench"
# Candidate HF commit referenced in team runbooks; official lock requires lead approval in configs/datasets.yaml.
KNOWN_HF_REVISION_CANDIDATE = "2067dfdce5abc7efdbb0927f4721ae282f0b6c51"
OFFICIAL_LICENSE = "BSD-3-Clause"

VALID_CATEGORIES = ("VD", "VS")
VALID_VISUAL_INPUT_CODES = ("0", "1", "2")
GT_ANSWER_TO_YESNO = {
    "0": "no",
    "1": "yes",
}
YESNO_TO_GT_ANSWER = {
    "no": "0",
    "yes": "1",
}

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

OPTIONAL_RAW_FIELDS = (
    "filename",
    "gt_answer_details",
    "sample_note",
    "same",
)

KNOWN_RAW_FIELDS = frozenset(REQUIRED_RAW_FIELDS + OPTIONAL_RAW_FIELDS + ("sample_id", "image_path", "reference_answer"))

# Aligned with benchmarks/hallusionbench/protocol.py FORBIDDEN_INPUT_FIELDS.
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


def _clean_id_token(value: Any, field_name: str) -> str:
    if value is None or isinstance(value, bool):
        raise HallusionSchemaError(f"Field '{field_name}' must be a non-empty string/int, got {value!r}")
    token = str(value).strip()
    if not token:
        raise HallusionSchemaError(f"Field '{field_name}' cannot be empty")
    return token


def parse_visual_input(value: Any) -> tuple[bool, str]:
    """Parse HallusionBench visual_input strictly into (requires_image, raw_code).

    Official HallusionBench semantics:
    - "0" / 0 / False -> text-only control question (no image)
    - "1" / 1 / True  -> visual question with original image
    - "2" / 2         -> visual question with edited/manipulated image
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
    """Parse HallusionBench ground-truth label into (raw_gt_answer, reference_answer).

    In HallusionBench.json, gt_answer is "1" (yes) or "0" (no).
    """
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


def make_stable_sample_id(
    category: str,
    subcategory: str,
    set_id: str,
    figure_id: str,
    question_id: str,
    visual_input_raw: str,
) -> str:
    """Build deterministic composite sample_id matching Member B's protocol.make_stable_sample_id.

    HallusionBench's question_id alone is NOT globally unique because it repeats across
    categories, subcategories, sets, figures, and visual/text-only variants.
    """
    return (
        f"cat{category}_sub{subcategory}_set{set_id}_"
        f"fig{figure_id}_q{question_id}_vis{visual_input_raw}"
    )


def make_group_or_pair_key(category: str, subcategory: str, set_id: str) -> str:
    """Build set-level control group key aligned with protocol.make_group_or_pair_key."""
    return f"{category}::{subcategory}::set_{set_id}"


def make_figure_group_key(category: str, subcategory: str, set_id: str, figure_id: str) -> str:
    """Build figure-level structural key (candidate key for fAcc; Member C verifies against evaluation.py)."""
    return f"{category}::{subcategory}::set_{set_id}::fig_{figure_id}"


def make_question_pair_key(category: str, subcategory: str, set_id: str, question_id: str) -> str:
    """Build question-pair structural key (candidate key for qAcc; Member C verifies against evaluation.py)."""
    return f"{category}::{subcategory}::set_{set_id}::q_{question_id}"


def derive_difficulty(category: str, visual_input_raw: str) -> str | None:
    """Derive structural easy/hard tag from HallusionBench category and visual_input.

    Per HallusionBench CVPR 2024 taxonomy (subject to Member C's cross-check with
    official evaluation.py/utils.py before official reporting):
    - VD: visual_input="1" (original image) -> "easy"; visual_input="2" (edited image) -> "hard"
    - VS: visual_input="0" (text-only control) -> "easy"; visual_input in {"1", "2"} -> "hard"
    """
    cat = category.strip().upper()
    vis = str(visual_input_raw).strip()
    if cat == "VD":
        if vis == "1":
            return "easy"
        if vis == "2":
            return "hard"
    elif cat == "VS":
        if vis == "0":
            return "easy"
        if vis in {"1", "2"}:
            return "hard"
    return None


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
        """Figure key for Member C fAcc (None for text-only samples with no visual figure)."""
        return self.figure_group_key if self.visual_input else None

    @property
    def pair_key(self) -> str:
        """Question-pair key for Member C qAcc."""
        return self.question_pair_key

    def to_dict(self) -> dict[str, Any]:
        """Serialize full normalized sample (compatible with Member B & Member C)."""
        payload = asdict(self)
        payload["figure_key"] = self.figure_key
        payload["pair_key"] = self.pair_key
        return payload

    def to_model_safe_dict(self) -> dict[str, Any]:
        """Return a label-free dict guaranteed to exclude all FORBIDDEN_MODEL_INPUT_FIELDS."""
        payload = self.to_dict()
        for forbidden in FORBIDDEN_MODEL_INPUT_FIELDS:
            payload.pop(forbidden, None)
        return payload


def validate_raw_record(raw_record: Any, index: int | None = None) -> list[str]:
    """Validate one raw HallusionBench.json object without mutating it.

    Returns a list of error strings (empty list means valid).
    """
    prefix = f"Record[{index}]" if index is not None else "Record"
    if not isinstance(raw_record, Mapping):
        return [f"{prefix}: expected a JSON object/mapping, got {type(raw_record).__name__}"]

    errors: list[str] = []
    for req_field in REQUIRED_RAW_FIELDS:
        if req_field not in raw_record or raw_record[req_field] is None:
            errors.append(f"{prefix}: missing required field '{req_field}'")

    if "category" in raw_record and raw_record["category"] is not None:
        try:
            parse_category(raw_record["category"])
        except HallusionSchemaError as exc:
            errors.append(f"{prefix}: {exc}")

    for id_field in ("subcategory", "set_id", "figure_id", "question_id"):
        if id_field in raw_record and raw_record[id_field] is not None:
            try:
                _clean_id_token(raw_record[id_field], id_field)
            except HallusionSchemaError as exc:
                errors.append(f"{prefix}: {exc}")

    question = raw_record.get("question")
    if question is not None and (not isinstance(question, str) or not question.strip()):
        errors.append(f"{prefix}: field 'question' must be a non-empty string")

    visual_bool: bool | None = None
    if "visual_input" in raw_record and raw_record["visual_input"] is not None:
        try:
            visual_bool, _ = parse_visual_input(raw_record["visual_input"])
        except HallusionSchemaError as exc:
            errors.append(f"{prefix}: {exc}")

    if "gt_answer" in raw_record and raw_record["gt_answer"] is not None:
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
    """Validate a sequence of normalized HallusionSample instances for schema & uniqueness invariants."""
    errors: list[str] = []
    seen_ids: dict[str, int] = {}

    for idx, sample in enumerate(samples):
        if sample.sample_id in seen_ids:
            errors.append(
                f"Duplicate sample_id '{sample.sample_id}' at indices {seen_ids[sample.sample_id]} and {idx}"
            )
        else:
            seen_ids[sample.sample_id] = idx

        if not sample.question.strip():
            errors.append(f"Sample {sample.sample_id}: empty question")
        if sample.category not in VALID_CATEGORIES:
            errors.append(f"Sample {sample.sample_id}: invalid category {sample.category!r}")
        if sample.visual_input_raw not in VALID_VISUAL_INPUT_CODES:
            errors.append(f"Sample {sample.sample_id}: invalid visual_input_raw {sample.visual_input_raw!r}")
        if sample.visual_input is False and sample.image_path is not None:
            errors.append(
                f"Sample {sample.sample_id}: text-only sample (visual_input=False) must have image_path=None"
            )
        if sample.visual_input is True and (not sample.image_path or not str(sample.image_path).strip()):
            errors.append(
                f"Sample {sample.sample_id}: visual sample (visual_input=True) is missing image_path"
            )
        if sample.gt_answer not in GT_ANSWER_TO_YESNO:
            errors.append(f"Sample {sample.sample_id}: invalid gt_answer {sample.gt_answer!r}")
        elif GT_ANSWER_TO_YESNO[sample.gt_answer] != sample.reference_answer:
            errors.append(
                f"Sample {sample.sample_id}: gt_answer {sample.gt_answer!r} does not match "
                f"reference_answer {sample.reference_answer!r}"
            )

        expected_group = make_group_or_pair_key(sample.category, sample.subcategory, sample.set_id)
        if sample.group_or_pair_key != expected_group:
            errors.append(
                f"Sample {sample.sample_id}: group_or_pair_key {sample.group_or_pair_key!r} != {expected_group!r}"
            )
        expected_fig = make_figure_group_key(sample.category, sample.subcategory, sample.set_id, sample.figure_id)
        if sample.figure_group_key != expected_fig:
            errors.append(
                f"Sample {sample.sample_id}: figure_group_key {sample.figure_group_key!r} != {expected_fig!r}"
            )
        expected_qpair = make_question_pair_key(sample.category, sample.subcategory, sample.set_id, sample.question_id)
        if sample.question_pair_key != expected_qpair:
            errors.append(
                f"Sample {sample.sample_id}: question_pair_key {sample.question_pair_key!r} != {expected_qpair!r}"
            )

    return errors
