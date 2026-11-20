"""Input protocol and prompt preparation for HallusionBench (Member B).

Design constraints enforced here:
- Never leak reference/ground-truth answers into model prompts.
- Preserve strict separation between visual (visual_input=1) and text-only
  (visual_input=0) samples; never fabricate or attach images to text-only items.
- Accept both Member A's normalized schema and raw HallusionBench.json records
  (for smoke-test/pilot before Member A's loader is finalized) without mutating
  questions, images, or labels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_PROMPT_VERSION = "hallusionbench_yesno_v1"
DEFAULT_PROMPT_SUFFIX = "Please answer yes or no."

# Fields that MUST NEVER be passed into prompt construction or model inputs.
FORBIDDEN_INPUT_FIELDS = frozenset(
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

# Metadata keys preserved for downstream grouping/scoring by Member C.
GROUP_METADATA_FIELDS = (
    "category",
    "subcategory",
    "visual_input",
    "set_id",
    "figure_id",
    "question_id",
    "group_or_pair_key",
)

# Read-only reference specs aligned with configs/models.yaml.
# Do NOT edit configs/models.yaml from inside HallusionBench without lead approval.
MODEL_SPECS: dict[str, dict[str, Any]] = {
    "smolvlm2": {
        "model_id": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        "loader_family": "smolvlm2",
        "default_trust_remote_code": False,
        "recommended_t4_attn": "sdpa",
        "priority_order": 1,
    },
    "qwen3_vl": {
        "model_id": "Qwen/Qwen3-VL-4B-Instruct",
        "loader_family": "qwen3_vl",
        "default_trust_remote_code": False,
        "recommended_t4_attn": "sdpa",
        "priority_order": 2,
    },
    "phi4_multimodal": {
        "model_id": "microsoft/Phi-4-multimodal-instruct",
        "loader_family": "phi4_multimodal",
        "default_trust_remote_code": True,
        "recommended_t4_attn": "eager",
        "priority_order": 3,
    },
}


@dataclass(frozen=True)
class HallusionBenchPromptConfig:
    """Versioned prompt configuration hashed into run_config_hash."""

    prompt_version: str = DEFAULT_PROMPT_VERSION
    suffix: str = DEFAULT_PROMPT_SUFFIX
    append_suffix: bool = True
    system_prompt: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PreparedHallusionInput:
    """Validated, label-free input ready for a VLM model adapter."""

    sample_id: str
    prompt: str
    visual_input: bool
    image_path: str | None
    messages: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)


def parse_visual_input(value: Any) -> bool:
    """Parse HallusionBench visual_input strictly without silent guessing.

    In HallusionBench.json:
    - "1" / 1 / True (and "2" / 2 for edited-image variants) require an image.
    - "0" / 0 / False denote text-only control questions (no image).
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        if value == 0:
            return False
        if value in (1, 2):
            return True
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"0", "false", "no"}:
            return False
        if normalized in {"1", "2", "true", "yes"}:
            return True
    raise ValueError(f"Unsupported visual_input value: {value!r}")


def make_stable_sample_id(record: Mapping[str, Any], index: int | None = None) -> str:
    """Return an existing sample_id or build a deterministic composite ID.

    Per team plan section 3.1, question_id alone is NOT unique in HallusionBench.
    """
    existing = record.get("sample_id")
    if existing is not None and str(existing).strip():
        return str(existing).strip()

    key_parts = [
        ("cat", record.get("category")),
        ("sub", record.get("subcategory")),
        ("set", record.get("set_id")),
        ("fig", record.get("figure_id")),
        ("q", record.get("question_id")),
        ("vis", record.get("visual_input")),
    ]
    present = [f"{prefix}{val}" for prefix, val in key_parts if val is not None and str(val).strip() != ""]
    if len(present) >= 4:
        return "_".join(present)
    if index is not None:
        return f"hb_idx_{index:05d}"
    raise ValueError("Cannot derive stable sample_id: missing sample_id and composite keys")


def make_group_or_pair_key(record: Mapping[str, Any]) -> str:
    """Build a grouping key preserving HallusionBench set/figure structure."""
    existing = record.get("group_or_pair_key")
    if existing is not None and str(existing).strip():
        return str(existing).strip()
    cat = record.get("category", "UNK")
    sub = record.get("subcategory", "UNK")
    set_id = record.get("set_id", "UNK")
    return f"{cat}::{sub}::set_{set_id}"


def _find_existing_case_insensitive(path: Path) -> Path | None:
    """Return path if it exists, or a deterministic case-insensitive filename match in the same dir.

    Needed on Linux (Colab/Kaggle) because 9 official HallusionBench VS/chart images have uppercase
    '.PNG' filenames while HallusionBench.json references lowercase '.png'.
    """
    if path.exists():
        return path
    parent = path.parent
    if not parent.is_dir():
        return None
    target_lower = path.name.lower()
    matches = sorted(p for p in parent.iterdir() if p.is_file() and p.name.lower() == target_lower)
    return matches[0] if matches else None


def resolve_image_path(
    record: Mapping[str, Any],
    visual_input: bool,
    data_root: str | Path | None = None,
    verify_image_exists: bool = False,
) -> str | None:
    """Resolve image path strictly respecting visual_input.

    - If visual_input is False: always returns None (never attaches an image).
    - If visual_input is True: requires image_path or filename to be present.
    """
    if not visual_input:
        return None

    raw_path = record.get("image_path")
    if raw_path is None or str(raw_path).strip() == "":
        raw_path = record.get("filename")
    if raw_path is None or str(raw_path).strip() == "":
        sample_id = record.get("sample_id", "<unknown>")
        raise ValueError(f"Sample {sample_id}: visual_input is True but image_path/filename is missing")

    candidate = Path(str(raw_path).strip())
    if not candidate.is_absolute() and data_root is not None:
        root = Path(data_root)
        # Try direct path (e.g. "./VS/chart/3_1.png" or "./hallusion_bench/VS/chart/3_1.png"),
        # plus common nested/un-nested "data" or "hallusion_bench" layouts.
        parts = [p for p in candidate.parts if p not in (".", "")]
        candidates = [(root / candidate).resolve()]
        if parts and parts[0].lower() == "hallusion_bench":
            candidates.append((root / Path(*parts[1:])).resolve())
            candidates.append((root / "data" / Path(*parts[1:])).resolve())
        elif parts:
            candidates.append((root / "data" / Path(*parts)).resolve())
            candidates.append((root / "hallusion_bench" / Path(*parts)).resolve())

        matched = next(
            (found for p in candidates if (found := _find_existing_case_insensitive(p)) is not None),
            None,
        )
        candidate = matched if matched is not None else candidates[0]
    else:
        found = _find_existing_case_insensitive(candidate)
        if found is not None:
            candidate = found

    if verify_image_exists and not candidate.exists():
        raise FileNotFoundError(
            f"Sample {record.get('sample_id', '<unknown>')}: image file not found at {candidate}"
        )
    return str(candidate)


def normalize_raw_hallusion_record(
    raw_record: Mapping[str, Any],
    data_root: str | Path | None = None,
    index: int | None = None,
    verify_image_exists: bool = False,
) -> dict[str, Any]:
    """Temporary adapter converting a raw HallusionBench.json entry to the team A/B/C contract.

    Used ONLY so Member B can run smoke-test/pilot before Member A finishes dataset.py.
    Does not mutate question text, images, or ground-truth labels.
    """
    if not isinstance(raw_record, Mapping):
        raise TypeError("HallusionBench record must be a mapping")

    question = raw_record.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"Record at index {index}: missing or empty 'question'")

    visual_input = parse_visual_input(raw_record.get("visual_input"))
    sample_id = make_stable_sample_id(raw_record, index=index)
    image_path = resolve_image_path(
        raw_record,
        visual_input=visual_input,
        data_root=data_root,
        verify_image_exists=verify_image_exists,
    )

    return {
        "sample_id": sample_id,
        "question": question.strip(),
        "image_path": image_path,
        "visual_input": visual_input,
        "visual_input_raw": raw_record.get("visual_input"),
        "reference_answer": raw_record.get("reference_answer", raw_record.get("gt_answer")),
        "category": raw_record.get("category"),
        "subcategory": raw_record.get("subcategory"),
        "set_id": raw_record.get("set_id"),
        "figure_id": raw_record.get("figure_id"),
        "question_id": raw_record.get("question_id"),
        "filename": raw_record.get("filename"),
        "group_or_pair_key": make_group_or_pair_key(raw_record),
    }


def build_prompt(
    question: str,
    prompt_config: HallusionBenchPromptConfig | None = None,
) -> str:
    """Build deterministic HallusionBench prompt text from the question only."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a non-empty string")

    cfg = prompt_config or HallusionBenchPromptConfig()
    clean_question = question.strip()
    if not cfg.append_suffix or not cfg.suffix.strip():
        return clean_question

    suffix = cfg.suffix.strip()
    if clean_question.lower().endswith(suffix.lower()):
        return clean_question
    return f"{clean_question} {suffix}"


def build_chat_messages(
    prompt: str,
    visual_input: bool,
    system_prompt: str | None = None,
) -> list[dict[str, Any]]:
    """Construct standard multimodal/text-only chat messages without image objects."""
    messages: list[dict[str, Any]] = []
    if system_prompt and system_prompt.strip():
        messages.append(
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt.strip()}],
            }
        )

    user_content: list[dict[str, Any]] = []
    if visual_input:
        user_content.append({"type": "image"})
    user_content.append({"type": "text", "text": prompt})
    messages.append({"role": "user", "content": user_content})
    return messages


def assert_no_label_leakage(
    prepared: PreparedHallusionInput,
    source_record: Mapping[str, Any],
) -> None:
    """Fail fast if forbidden label fields were copied into prepared metadata or messages."""
    leaked_keys = FORBIDDEN_INPUT_FIELDS.intersection(prepared.metadata.keys())
    if leaked_keys:
        raise AssertionError(f"Label leakage detected in prepared metadata: {sorted(leaked_keys)}")

    # Ensure long ground-truth explanations (gt_answer_details) are not injected into the prompt.
    gt_details = source_record.get("gt_answer_details")
    if isinstance(gt_details, str):
        detail_clean = gt_details.strip()
        question_clean = str(source_record.get("question", "")).strip()
        if (
            len(detail_clean) >= 12
            and detail_clean.lower() not in question_clean.lower()
            and detail_clean.lower() in prepared.prompt.lower()
        ):
            raise AssertionError("Label leakage detected: gt_answer_details appeared in prompt")


def prepare_hallusion_input(
    record: Mapping[str, Any],
    data_root: str | Path | None = None,
    prompt_config: HallusionBenchPromptConfig | None = None,
    verify_image_exists: bool = False,
) -> PreparedHallusionInput:
    """Prepare one label-free model input from either Member A's schema or raw JSON."""
    if not isinstance(record, Mapping):
        raise TypeError("Record must be a mapping")

    sample_id = make_stable_sample_id(record)
    question = record.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"Sample {sample_id}: missing or empty 'question'")

    visual_input = parse_visual_input(record.get("visual_input"))
    image_path = resolve_image_path(
        record,
        visual_input=visual_input,
        data_root=data_root,
        verify_image_exists=verify_image_exists,
    )

    cfg = prompt_config or HallusionBenchPromptConfig()
    prompt = build_prompt(question, prompt_config=cfg)
    messages = build_chat_messages(
        prompt=prompt,
        visual_input=visual_input,
        system_prompt=cfg.system_prompt,
    )

    metadata: dict[str, Any] = {}
    for field_name in GROUP_METADATA_FIELDS:
        if field_name in record and field_name not in FORBIDDEN_INPUT_FIELDS:
            metadata[field_name] = record[field_name]
    metadata["visual_input"] = visual_input
    metadata["group_or_pair_key"] = make_group_or_pair_key(record)

    prepared = PreparedHallusionInput(
        sample_id=sample_id,
        prompt=prompt,
        visual_input=visual_input,
        image_path=image_path,
        messages=messages,
        metadata=metadata,
    )
    assert_no_label_leakage(prepared, record)
    return prepared


def select_pilot_records(
    records: Sequence[Mapping[str, Any]],
    max_groups: int = 5,
    include_text_only: bool = True,
) -> list[dict[str, Any]]:
    """Select a deterministic pilot subset preserving COMPLETE groups/control pairs.

    Never splits a (category, subcategory, set_id) group so Member C can test
    aAcc, fAcc, and qAcc on pilot predictions.
    """
    if max_groups <= 0:
        raise ValueError("max_groups must be positive")

    grouped: dict[str, list[dict[str, Any]]] = {}
    group_has_text_only: dict[str, bool] = {}
    for rec in records:
        rec_dict = dict(rec)
        key = make_group_or_pair_key(rec_dict)
        grouped.setdefault(key, []).append(rec_dict)
        vis = parse_visual_input(rec_dict.get("visual_input"))
        if not vis:
            group_has_text_only[key] = True

    ordered_keys = list(grouped.keys())
    selected_keys: list[str] = []

    if include_text_only:
        # Ensure at least one group containing text-only controls is included when available.
        for key in ordered_keys:
            if group_has_text_only.get(key) and key not in selected_keys:
                selected_keys.append(key)
                break

    for key in ordered_keys:
        if len(selected_keys) >= max_groups:
            break
        if key not in selected_keys:
            selected_keys.append(key)

    pilot: list[dict[str, Any]] = []
    for key in selected_keys:
        pilot.extend(grouped[key])
    return pilot


def load_raw_hallusion_json(
    json_path: str | Path,
    data_root: str | Path | None = None,
    verify_image_exists: bool = False,
) -> list[dict[str, Any]]:
    """Load and normalize HallusionBench.json while checking sample_id uniqueness."""
    import json

    path = Path(json_path)
    raw_items = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw_items, list):
        raise ValueError(f"Expected a JSON list in {path}")

    root = Path(data_root) if data_root is not None else path.parent
    normalized: list[dict[str, Any]] = []
    seen_ids: dict[str, int] = {}

    for idx, item in enumerate(raw_items):
        rec = normalize_raw_hallusion_record(
            item,
            data_root=root,
            index=idx,
            verify_image_exists=verify_image_exists,
        )
        sid = rec["sample_id"]
        if sid in seen_ids:
            # Disambiguate deterministically without dropping records, and record collision.
            rec["sample_id"] = f"{sid}_dup{idx}"
            rec["sample_id_collision_with"] = sid
        seen_ids[rec["sample_id"]] = idx
        normalized.append(rec)
    return normalized
