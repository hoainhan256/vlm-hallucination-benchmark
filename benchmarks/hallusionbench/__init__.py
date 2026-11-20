"""HallusionBench benchmark package (unified Member A + B + C modules)."""

from .analysis import (
    VALID_ERROR_TYPES,
    compare_before_after,
    export_error_report,
    extract_error_cases,
    find_cross_model_disagreements,
    render_error_table_markdown,
    summarize_error_cases,
)
from .benchmark import HallusionBenchBenchmark
from .dataset import (
    HallusionBenchDataset,
    build_group_mappings,
    check_group_integrity,
    filter_by_pilot_ids,
    normalize_hallusion_record,
    samples_to_dataset_index,
    select_frozen_pilot,
)
from .protocol import (
    DEFAULT_PROMPT_SUFFIX,
    DEFAULT_PROMPT_VERSION,
    FORBIDDEN_INPUT_FIELDS,
    MODEL_SPECS,
    HallusionBenchPromptConfig,
    PreparedHallusionInput,
    assert_no_label_leakage,
    build_prompt,
    make_stable_sample_id,
    normalize_raw_hallusion_record,
    parse_visual_input,
    prepare_hallusion_input,
    select_pilot_records,
)
from .schema import (
    DuplicateSampleIdError,
    GroupIntegrityError,
    HallusionSample,
    HallusionSchemaError,
    derive_difficulty,
)
from .scoring import (
    export_hallusionbench_artifacts,
    parse_yes_no,
    score_hallusionbench,
    score_run_directory,
)

__all__ = [
    # Unified BaseBenchmark adapter
    "HallusionBenchBenchmark",
    # Member A — Dataset & Schema
    "DuplicateSampleIdError",
    "GroupIntegrityError",
    "HallusionBenchDataset",
    "HallusionSample",
    "HallusionSchemaError",
    "build_group_mappings",
    "check_group_integrity",
    "derive_difficulty",
    "filter_by_pilot_ids",
    "normalize_hallusion_record",
    "samples_to_dataset_index",
    "select_frozen_pilot",
    # Member B — Protocol & Inference
    "DEFAULT_PROMPT_SUFFIX",
    "DEFAULT_PROMPT_VERSION",
    "FORBIDDEN_INPUT_FIELDS",
    "MODEL_SPECS",
    "HallusionBenchPromptConfig",
    "PreparedHallusionInput",
    "assert_no_label_leakage",
    "build_prompt",
    "make_stable_sample_id",
    "normalize_raw_hallusion_record",
    "parse_visual_input",
    "prepare_hallusion_input",
    "select_pilot_records",
    # Member C — Scoring & Error Analysis
    "VALID_ERROR_TYPES",
    "compare_before_after",
    "export_error_report",
    "export_hallusionbench_artifacts",
    "extract_error_cases",
    "find_cross_model_disagreements",
    "parse_yes_no",
    "render_error_table_markdown",
    "score_hallusionbench",
    "score_run_directory",
    "summarize_error_cases",
]
