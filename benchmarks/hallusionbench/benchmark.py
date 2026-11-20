"""Concrete BaseBenchmark adapter uniting HallusionBench Member A, B, and C modules."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

from benchmarks.base import BaseBenchmark
from benchmarks.hallusionbench.dataset import (
    HallusionBenchDataset,
    normalize_hallusion_record,
)
from benchmarks.hallusionbench.protocol import (
    HallusionBenchPromptConfig,
    PreparedHallusionInput,
    prepare_hallusion_input,
)
from benchmarks.hallusionbench.scoring import (
    export_hallusionbench_artifacts,
    score_hallusionbench,
)


class HallusionBenchBenchmark(BaseBenchmark):
    """End-to-end HallusionBench adapter implementing benchmarks/base.py::BaseBenchmark.

    - A (dataset.py / schema.py): read_dataset & normalize_record
    - B (protocol.py / inference.py): prepare_inputs & call_model
    - C (scoring.py / analysis.py): score & export_results
    """

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
        self._dataset_index_cache: dict[str, dict[str, Any]] | None = None

    def read_dataset(self) -> Iterable[Mapping[str, Any]]:
        """Read raw HallusionBench.json without downloading or mutating it."""
        return self.dataset.read_raw()

    def normalize_record(self, record: Mapping[str, Any]) -> Mapping[str, Any]:
        """Normalize one raw record into HallusionSample dict preserving group/pair keys."""
        sample = normalize_hallusion_record(
            record,
            data_root=self.data_root,
            verify_image_exists=self.verify_images,
        )
        return sample.to_dict()

    def prepare_inputs(self, record: Mapping[str, Any]) -> PreparedHallusionInput:
        """Prepare label-free visual/text-only model input via Member B's protocol."""
        return prepare_hallusion_input(
            record,
            data_root=self.data_root,
            prompt_config=self.prompt_config,
            verify_image_exists=self.verify_images,
        )

    def call_model(self, model: Any, prepared_input: Any) -> Any:
        """Call a HallusionModelAdapter (or BaseVLMModel-compatible object) without mutating prompt rules."""
        if hasattr(model, "generate"):
            return model.generate(prepared_input, self.generation_config)
        raise TypeError(f"Model {type(model).__name__} does not implement generate(prepared_input, generation_config)")

    def _get_dataset_index(self) -> dict[str, dict[str, Any]]:
        if self._dataset_index_cache is None:
            self._dataset_index_cache = self.dataset.load_index(
                verify_images=self.verify_images,
                verify_image_readable=False,
            )
        return self._dataset_index_cache

    def score(self, predictions: Iterable[Mapping[str, Any]]) -> Mapping[str, Any]:
        """Compute HallusionBench aAcc, fAcc, qAcc, VD/VS, and Easy/Hard metrics via Member C."""
        metrics, _ = score_hallusionbench(
            predictions,
            self._get_dataset_index(),
            protocol_version=self.protocol_version,
            evaluation_mode=self.evaluation_mode,
        )
        return metrics

    def export_results(self, predictions: Iterable[Mapping[str, Any]], output_dir: str) -> None:
        """Export resume-safe predictions.jsonl, metrics.json, and errors.jsonl."""
        export_hallusionbench_artifacts(
            predictions,
            self._get_dataset_index(),
            output_dir=output_dir,
            protocol_version=self.protocol_version,
            evaluation_mode=self.evaluation_mode,
        )
