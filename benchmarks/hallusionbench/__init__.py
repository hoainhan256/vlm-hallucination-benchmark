"""HallusionBench benchmark package (clean 3-module layout: dataset + inference + evaluation)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

from benchmarks.base import BaseBenchmark
from .dataset import (
    DATASET_NAME,
    FORBIDDEN_MODEL_INPUT_FIELDS,
    HF_AUTHOR_MIRROR_SOURCE,
    KNOWN_HF_REVISION_CANDIDATE,
    OFFICIAL_GITHUB_SOURCE,
    OFFICIAL_LICENSE,
    DuplicateSampleIdError,
    GroupIntegrityError,
    HallusionBenchDataset,
    HallusionSample,
    HallusionSchemaError,
    build_group_mappings,
    check_group_integrity,
    derive_difficulty,
    filter_by_pilot_ids,
    make_stable_sample_id,
    normalize_hallusion_record,
    parse_visual_input,
    samples_to_dataset_index,
    select_frozen_pilot,
)
from .evaluation import (
    VALID_ERROR_TYPES,
    build_detailed_evaluation_records,
    build_mmhal_style_summary,
    compare_before_after,
    export_deliverables,
    export_error_report,
    export_hallusionbench_artifacts,
    extract_error_cases,
    find_cross_model_disagreements,
    parse_yes_no,
    render_error_table_markdown,
    render_master_results_readme,
    render_model_results_readme,
    score_hallusionbench,
    score_run_directory,
    summarize_error_cases,
)
from .inference import (
    DEFAULT_PROMPT_SUFFIX,
    DEFAULT_PROMPT_VERSION,
    MODEL_SPECS,
    VALID_STAGES,
    HallusionBenchPromptConfig,
    HallusionInferenceRunner,
    HallusionRunConfig,
    PreparedHallusionInput,
    assert_no_label_leakage,
    build_prompt,
    prepare_hallusion_input,
    write_blocked_run,
)


class HallusionBenchBenchmark(BaseBenchmark):
    """Concrete BaseBenchmark adapter uniting dataset (A), inference (B), and evaluation (C)."""

    def __init__(
        self,
        json_path: str | Path,
        data_root: str | Path | None = None,
        *,
        prompt_config: HallusionBenchPromptConfig | None = None,
        generation_config: Mapping[str, Any] | None = None,
        verify_images: bool = True,
        protocol_version: str = "unverified_draft_rule_based",
        evaluation_mode: str = "alternative_evaluation_rule_based",
    ) -> None:
        self.dataset = HallusionBenchDataset(json_path=json_path, data_root=data_root)
        self.data_root = self.dataset.data_root
        self.prompt_config = prompt_config or HallusionBenchPromptConfig()
        self.generation_config = dict(generation_config or {"do_sample": False, "max_new_tokens": 128})
        self.verify_images = verify_images
        self.protocol_version = protocol_version
        self.evaluation_mode = evaluation_mode
        self._index_cache: dict[str, dict[str, Any]] | None = None

    def read_dataset(self) -> Iterable[Mapping[str, Any]]:
        return self.dataset.read_raw()

    def normalize_record(self, record: Mapping[str, Any]) -> Mapping[str, Any]:
        return normalize_hallusion_record(
            record, data_root=self.data_root, verify_image_exists=self.verify_images
        ).to_dict()

    def prepare_inputs(self, record: Mapping[str, Any]) -> PreparedHallusionInput:
        return prepare_hallusion_input(
            record, data_root=self.data_root, prompt_config=self.prompt_config, verify_image_exists=self.verify_images
        )

    def call_model(self, model: Any, prepared_input: Any) -> Any:
        if hasattr(model, "generate"):
            return model.generate(prepared_input, self.generation_config)
        raise TypeError(f"Model {type(model).__name__} must implement generate(prepared_input, generation_config)")

    def _get_index(self) -> dict[str, dict[str, Any]]:
        if self._index_cache is None:
            self._index_cache = self.dataset.load_index(verify_images=self.verify_images, verify_image_readable=False)
        return self._index_cache

    def score(self, predictions: Iterable[Mapping[str, Any]]) -> Mapping[str, Any]:
        metrics, _ = score_hallusionbench(
            predictions,
            self._get_index(),
            protocol_version=self.protocol_version,
            evaluation_mode=self.evaluation_mode,
        )
        return metrics

    def export_results(self, predictions: Iterable[Mapping[str, Any]], output_dir: str) -> None:
        export_hallusionbench_artifacts(
            predictions,
            self._get_index(),
            output_dir=output_dir,
            protocol_version=self.protocol_version,
            evaluation_mode=self.evaluation_mode,
        )


__all__ = [
    "DATASET_NAME",
    "DEFAULT_PROMPT_SUFFIX",
    "DEFAULT_PROMPT_VERSION",
    "FORBIDDEN_MODEL_INPUT_FIELDS",
    "HF_AUTHOR_MIRROR_SOURCE",
    "KNOWN_HF_REVISION_CANDIDATE",
    "MODEL_SPECS",
    "OFFICIAL_GITHUB_SOURCE",
    "OFFICIAL_LICENSE",
    "VALID_ERROR_TYPES",
    "VALID_STAGES",
    "DuplicateSampleIdError",
    "GroupIntegrityError",
    "HallusionBenchBenchmark",
    "HallusionBenchDataset",
    "HallusionBenchPromptConfig",
    "HallusionInferenceRunner",
    "HallusionRunConfig",
    "HallusionSample",
    "HallusionSchemaError",
    "PreparedHallusionInput",
    "assert_no_label_leakage",
    "build_detailed_evaluation_records",
    "build_group_mappings",
    "build_mmhal_style_summary",
    "build_prompt",
    "check_group_integrity",
    "compare_before_after",
    "derive_difficulty",
    "export_deliverables",
    "export_error_report",
    "export_hallusionbench_artifacts",
    "extract_error_cases",
    "filter_by_pilot_ids",
    "find_cross_model_disagreements",
    "make_stable_sample_id",
    "normalize_hallusion_record",
    "parse_visual_input",
    "parse_yes_no",
    "prepare_hallusion_input",
    "render_error_table_markdown",
    "render_master_results_readme",
    "render_model_results_readme",
    "samples_to_dataset_index",
    "score_hallusionbench",
    "score_run_directory",
    "select_frozen_pilot",
    "summarize_error_cases",
    "write_blocked_run",
]
