"""HallusionBench Part B — Prompt protocol, label-leakage guard, HF 3-model adapter, and resumable runner.

Zero-shot inference only (0 training epochs). Complies with:
- docs/model_policy.md (no quantization, no silent dtype fallback, locked revisions for official)
- docs/result_schema.md (all 7 artifacts, required prediction fields, honest status transitions)
- vlm_benchmark/results.py (stable_config_hash, validate_predictions, merge_result_records)

Importing this module NEVER downloads weights or datasets.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - fallback when PyYAML is not installed locally
    yaml = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.hallusionbench.dataset import (  # noqa: E402
    FORBIDDEN_MODEL_INPUT_FIELDS,
    HF_AUTHOR_MIRROR_SOURCE,
    HallusionBenchDataset,
    filter_by_pilot_ids,
    make_group_or_pair_key,
    make_stable_sample_id,
    parse_visual_input,
    resolve_image_path,
    sha256_file,
    sha256_json,
)
from vlm_benchmark.environment import collect_environment  # noqa: E402
from vlm_benchmark.results import (  # noqa: E402
    merge_result_records,
    read_jsonl,
    stable_config_hash,
    validate_predictions,
)


DEFAULT_PROMPT_VERSION = "hallusionbench_yesno_v1"
DEFAULT_PROMPT_SUFFIX = "Please answer yes or no."
VALID_STAGES = ("smoke_test", "pilot", "full_baseline", "official")
ALLOWED_DTYPES = ("bfloat16", "float16", "float32")

GROUP_METADATA_FIELDS = (
    "category",
    "subcategory",
    "visual_input",
    "visual_input_raw",
    "set_id",
    "figure_id",
    "question_id",
    "group_or_pair_key",
)

MODEL_SPECS: dict[str, dict[str, Any]] = {
    "smolvlm2": {
        "model_id": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        "loader_family": "smolvlm2",
        "default_trust_remote_code": False,
        "recommended_t4_attn": "sdpa",
    },
    "qwen3_vl": {
        "model_id": "Qwen/Qwen3-VL-4B-Instruct",
        "loader_family": "qwen3_vl",
        "default_trust_remote_code": False,
        "recommended_t4_attn": "sdpa",
    },
    "phi4_multimodal": {
        "model_id": "microsoft/Phi-4-multimodal-instruct",
        "loader_family": "phi4_multimodal",
        "default_trust_remote_code": True,
        "recommended_t4_attn": "eager",
    },
}


# ---------------------------------------------------------------------------
# Prompt & label-free input preparation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HallusionBenchPromptConfig:
    prompt_version: str = DEFAULT_PROMPT_VERSION
    suffix: str = DEFAULT_PROMPT_SUFFIX
    append_suffix: bool = True
    system_prompt: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PreparedHallusionInput:
    sample_id: str
    prompt: str
    visual_input: bool
    image_path: str | None
    messages: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)


def build_prompt(question: str, prompt_config: HallusionBenchPromptConfig | None = None) -> str:
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
    messages: list[dict[str, Any]] = []
    if system_prompt and system_prompt.strip():
        messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt.strip()}]})
    user_content: list[dict[str, Any]] = []
    if visual_input:
        user_content.append({"type": "image"})
    user_content.append({"type": "text", "text": prompt})
    messages.append({"role": "user", "content": user_content})
    return messages


def assert_no_label_leakage(prepared: PreparedHallusionInput, source_record: Mapping[str, Any]) -> None:
    leaked_keys = FORBIDDEN_MODEL_INPUT_FIELDS.intersection(prepared.metadata.keys())
    if leaked_keys:
        raise AssertionError(f"Label leakage detected in prepared metadata: {sorted(leaked_keys)}")
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
    """Prepare one label-free model input from a HallusionSample dict or raw JSON record."""
    if not isinstance(record, Mapping):
        raise TypeError("Record must be a mapping")

    question = record.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Missing or empty 'question'")

    visual_input, vis_raw = parse_visual_input(record.get("visual_input"))

    existing_sid = record.get("sample_id")
    if existing_sid is not None and str(existing_sid).strip():
        sample_id = str(existing_sid).strip()
    else:
        sample_id = make_stable_sample_id(
            str(record.get("category", "UNK")),
            str(record.get("subcategory", "UNK")),
            str(record.get("set_id", "UNK")),
            str(record.get("figure_id", "UNK")),
            str(record.get("question_id", "UNK")),
            vis_raw,
        )

    image_path, _ = resolve_image_path(
        raw_record=record,
        visual_input=visual_input,
        data_root=data_root,
        verify_image_exists=verify_image_exists,
        verify_image_readable=False,
    )

    cfg = prompt_config or HallusionBenchPromptConfig()
    prompt = build_prompt(question, prompt_config=cfg)
    messages = build_chat_messages(prompt=prompt, visual_input=visual_input, system_prompt=cfg.system_prompt)

    metadata: dict[str, Any] = {
        k: record[k]
        for k in GROUP_METADATA_FIELDS
        if k in record and k not in FORBIDDEN_MODEL_INPUT_FIELDS
    }
    metadata["visual_input"] = visual_input
    metadata["visual_input_raw"] = str(record.get("visual_input_raw", vis_raw))
    metadata["group_or_pair_key"] = str(
        record.get("group_or_pair_key")
        or make_group_or_pair_key(
            str(record.get("category", "UNK")),
            str(record.get("subcategory", "UNK")),
            str(record.get("set_id", "UNK")),
        )
    )

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


# ---------------------------------------------------------------------------
# Run configuration, environment inspection, and artifact helpers
# ---------------------------------------------------------------------------

class HallusionModelAdapter(Protocol):
    def generate(self, prepared: PreparedHallusionInput, generation_config: Mapping[str, Any]) -> str: ...
    def get_runtime_info(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class HallusionRunConfig:
    experiment_id: str
    run_id: str
    model_key: str
    model_id: str
    stage: str = "pilot"
    model_revision: str | None = None
    processor_revision: str | None = None
    dtype: str = "bfloat16"
    allow_dtype_fallback: bool = False
    quantization: str | None = None
    attn_implementation: str | None = None
    device_map: str | dict[str, Any] | None = "auto"
    trust_remote_code: bool = False
    dataset_source: str = HF_AUTHOR_MIRROR_SOURCE
    dataset_revision: str | None = None
    prompt_config: HallusionBenchPromptConfig = field(default_factory=HallusionBenchPromptConfig)
    do_sample: bool = False
    max_new_tokens: int = 128
    temperature: float | None = None
    top_p: float | None = None
    seed: int = 0
    batch_size: int = 1

    def generation_config_dict(self) -> dict[str, Any]:
        cfg: dict[str, Any] = {"do_sample": self.do_sample, "max_new_tokens": self.max_new_tokens}
        if self.do_sample:
            if self.temperature is not None:
                cfg["temperature"] = self.temperature
            if self.top_p is not None:
                cfg["top_p"] = self.top_p
        return cfg

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["generation_config"] = self.generation_config_dict()
        return data

    def to_hashable_dict(self) -> dict[str, Any]:
        return {
            "benchmark": "hallusionbench",
            "stage": self.stage,
            "model_key": self.model_key,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "processor_revision": self.processor_revision,
            "dtype": self.dtype,
            "quantization": self.quantization,
            "attn_implementation": self.attn_implementation,
            "trust_remote_code": self.trust_remote_code,
            "dataset_source": self.dataset_source,
            "dataset_revision": self.dataset_revision,
            "prompt_config": self.prompt_config.to_dict(),
            "generation_config": self.generation_config_dict(),
            "seed": self.seed,
            "batch_size": self.batch_size,
        }

    def config_hash(self) -> str:
        return stable_config_hash(self.to_hashable_dict())

    def validate_for_stage(
        self,
        hardware_info: Mapping[str, Any] | None = None,
        repo_root: str | Path | None = None,
    ) -> list[str]:
        errors: list[str] = []
        if self.stage not in VALID_STAGES:
            errors.append(f"Invalid stage {self.stage!r}; expected one of {VALID_STAGES}")
        if self.model_key not in MODEL_SPECS:
            errors.append(f"Unknown model_key {self.model_key!r}; expected one of {sorted(MODEL_SPECS)}")
        if self.quantization not in (None, "none"):
            errors.append("Policy violation: quantization is not allowed")
        if self.allow_dtype_fallback is not False:
            errors.append("Policy violation: automatic dtype fallback must be disabled")
        if self.dtype not in ALLOWED_DTYPES:
            errors.append(f"Unsupported dtype {self.dtype!r}; expected one of {ALLOWED_DTYPES}")
        if self.batch_size != 1:
            errors.append("Policy violation: batch_size must be 1")
        if self.max_new_tokens is None or self.max_new_tokens <= 0:
            errors.append("Invalid generation_config: max_new_tokens must be positive")

        if hardware_info is not None:
            if self.dtype == "bfloat16" and hardware_info.get("cuda_available") and not hardware_info.get("bf16_supported"):
                errors.append(
                    "Hardware incompatibility: requested bfloat16 on GPU without native BF16 support "
                    "(e.g., T4). Request lead approval for float16 or use a BF16-capable GPU."
                )

        if self.stage == "official":
            if not self.model_revision:
                errors.append("Official stage requires locked model_revision")
            if not self.processor_revision:
                errors.append("Official stage requires locked processor_revision")
            if not self.dataset_revision:
                errors.append("Official stage requires locked dataset_revision")
            if repo_root is not None:
                root = Path(repo_root)
                try:
                    from vlm_benchmark.config import validate_all

                    for err in validate_all(
                        root / "configs" / "models.yaml",
                        root / "configs" / "datasets.yaml",
                        root / "configs" / "experiment.yaml",
                        official_stage=True,
                    ):
                        errors.append(f"Repository official gate: {err}")
                except ModuleNotFoundError as exc:
                    errors.append(f"Repository official gate unavailable ({exc}): install requirements.txt")

        return errors


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dump_yaml(payload: Mapping[str, Any]) -> str:
    if yaml is not None:
        return yaml.safe_dump(dict(payload), sort_keys=False, allow_unicode=True)
    return json.dumps(dict(payload), indent=2, ensure_ascii=False) + "\n"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(dict(rec), ensure_ascii=False) + "\n")


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(record), ensure_ascii=False) + "\n")
        handle.flush()


def _check_native_bf16(torch_module: Any) -> tuple[bool, bool]:
    if not torch_module.cuda.is_available():
        return False, False
    emulated = bool(torch_module.cuda.is_bf16_supported())
    try:
        native = bool(torch_module.cuda.is_bf16_supported(including_emulation=False))
    except TypeError:
        native = emulated and torch_module.cuda.get_device_properties(0).major >= 8
    return native, emulated


def collect_runtime_environment(repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    env = collect_environment(repo_root)
    env.update(
        {
            "cuda_available": False,
            "gpu_count": 0,
            "bf16_supported": False,
            "bf16_emulated_supported": False,
            "gpus": [],
        }
    )
    for pkg in ("PIL", "huggingface_hub"):
        try:
            env["packages"][pkg] = getattr(__import__(pkg), "__version__", None)
        except Exception:
            env["packages"][pkg] = None

    try:
        import torch

        env["packages"]["torch"] = getattr(torch, "__version__", None)
        env["cuda"] = getattr(torch.version, "cuda", None)
        if torch.cuda.is_available():
            count = int(torch.cuda.device_count())
            bf16_native, bf16_emulated = _check_native_bf16(torch)
            gpus = []
            for idx in range(count):
                props = torch.cuda.get_device_properties(idx)
                gpus.append(
                    {
                        "index": idx,
                        "name": torch.cuda.get_device_name(idx),
                        "total_memory_gib": round(props.total_memory / (2**30), 2),
                        "major": props.major,
                        "minor": props.minor,
                    }
                )
            env.update(
                {
                    "cuda_available": True,
                    "gpu_count": count,
                    "bf16_supported": bf16_native,
                    "bf16_emulated_supported": bf16_emulated,
                    "gpu": [g["name"] for g in gpus],
                    "gpus": gpus,
                }
            )
    except Exception:
        pass
    return env


def write_manifest(run_dir: Path, run_config: HallusionRunConfig, status: str) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    for name in ("predictions.jsonl", "metrics.json", "run_config.yaml", "run_metadata.json", "errors.jsonl", "run_status.json"):
        file_path = run_dir / name
        if file_path.exists():
            files[name] = {"size_bytes": file_path.stat().st_size, "sha256": sha256_file(file_path)}
    manifest = {
        "experiment_id": run_config.experiment_id,
        "run_id": run_config.run_id,
        "benchmark": "hallusionbench",
        "stage": run_config.stage,
        "model_key": run_config.model_key,
        "model_id": run_config.model_id,
        "run_config_hash": run_config.config_hash(),
        "status": status,
        "generated_at_utc": _utc_now_iso(),
        "files": files,
    }
    _write_json(run_dir / "manifest.json", manifest)
    return manifest


def write_blocked_run(
    run_dir: str | Path,
    run_config: HallusionRunConfig,
    reasons: Sequence[str],
    repo_root: str | Path = REPO_ROOT,
    hardware_info: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    out_dir = Path(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(hardware_info) if hardware_info is not None else collect_runtime_environment(repo_root)
    now = _utc_now_iso()

    _write_jsonl(out_dir / "predictions.jsonl", [])
    _write_jsonl(
        out_dir / "errors.jsonl",
        [{"sample_id": None, "error_type": "POLICY_OR_HARDWARE_BLOCKED", "message": r, "timestamp_utc": now} for r in reasons],
    )
    (out_dir / "run_config.yaml").write_text(_dump_yaml(run_config.to_dict()), encoding="utf-8")
    _write_json(out_dir / "metrics.json", {"status": "BLOCKED", "note": "Run blocked before inference.", "scores": None})
    _write_json(
        out_dir / "run_metadata.json",
        {
            "experiment_id": run_config.experiment_id,
            "run_id": run_config.run_id,
            "stage": run_config.stage,
            "model_key": run_config.model_key,
            "model_id": run_config.model_id,
            "model_revision": run_config.model_revision,
            "processor_revision": run_config.processor_revision,
            "dataset_source": run_config.dataset_source,
            "dataset_revision": run_config.dataset_revision,
            "dtype": run_config.dtype,
            "attn_implementation": run_config.attn_implementation,
            "run_config_hash": run_config.config_hash(),
            "status": "BLOCKED",
            "blocked_reasons": list(reasons),
            "device_allocation": None,
            "started_at_utc": now,
            "finished_at_utc": now,
            "environment": env,
        },
    )
    status_payload = {
        "experiment_id": run_config.experiment_id,
        "run_id": run_config.run_id,
        "stage": run_config.stage,
        "status": "BLOCKED",
        "reasons": list(reasons),
        "updated_at_utc": now,
    }
    _write_json(out_dir / "run_status.json", status_payload)
    write_manifest(out_dir, run_config, status="BLOCKED")
    return status_payload


def _classify_exception(exc: Exception) -> str:
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    if "outofmemory" in name or "out of memory" in msg or "cuda oom" in msg:
        return "OOM_ERROR"
    if isinstance(exc, FileNotFoundError):
        return "DATA_ERROR"
    if isinstance(exc, (ValueError, KeyError, TypeError)):
        return "INPUT_OR_SCHEMA_ERROR"
    return "TECHNICAL_ERROR"


# ---------------------------------------------------------------------------
# Resumable inference runner
# ---------------------------------------------------------------------------

class HallusionInferenceRunner:
    def __init__(
        self,
        run_config: HallusionRunConfig,
        output_dir: str | Path,
        repo_root: str | Path = REPO_ROOT,
        data_root: str | Path | None = None,
        verify_image_exists: bool = True,
    ) -> None:
        self.run_config = run_config
        self.output_dir = Path(output_dir)
        self.repo_root = Path(repo_root)
        self.data_root = Path(data_root) if data_root is not None else None
        self.verify_image_exists = verify_image_exists
        self.config_hash = run_config.config_hash()

    def _load_existing_predictions(self, predictions_path: Path) -> list[dict[str, Any]]:
        if not predictions_path.exists():
            return []
        existing = read_jsonl(predictions_path)
        if not existing:
            return []
        val_errors = validate_predictions(existing)
        if val_errors:
            raise ValueError(f"Existing predictions.jsonl is invalid: {val_errors}")
        existing_hashes = {str(r.get("run_config_hash")) for r in existing}
        if existing_hashes != {self.config_hash}:
            raise ValueError(
                f"Refusing to resume: incompatible run_config_hash {sorted(existing_hashes)} != {self.config_hash}"
            )
        return existing

    def run(
        self,
        records: Sequence[Mapping[str, Any]],
        model_adapter: HallusionModelAdapter,
        hardware_info: Mapping[str, Any] | None = None,
        resume: bool = True,
    ) -> dict[str, Any]:
        env = dict(hardware_info) if hardware_info is not None else collect_runtime_environment(self.repo_root)
        policy_errors = self.run_config.validate_for_stage(hardware_info=env, repo_root=self.repo_root)
        if policy_errors:
            return write_blocked_run(self.output_dir, self.run_config, policy_errors, self.repo_root, env)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = self.output_dir / "predictions.jsonl"
        errors_path = self.output_dir / "errors.jsonl"
        run_status_path = self.output_dir / "run_status.json"

        existing_records: list[dict[str, Any]] = []
        if resume:
            existing_records = self._load_existing_predictions(predictions_path)
        elif predictions_path.exists() and predictions_path.stat().st_size > 0:
            raise FileExistsError(f"Refusing to overwrite {predictions_path}; enable resume=True or use a new run_id.")

        completed_ids = {str(r["sample_id"]) for r in existing_records if r.get("status") == "COMPLETED"}

        if not existing_records or not errors_path.exists():
            errors_path.write_text("", encoding="utf-8")

        (self.output_dir / "run_config.yaml").write_text(_dump_yaml(self.run_config.to_dict()), encoding="utf-8")
        started_at = _utc_now_iso()
        _write_json(
            run_status_path,
            {
                "experiment_id": self.run_config.experiment_id,
                "run_id": self.run_config.run_id,
                "stage": self.run_config.stage,
                "status": "RUNNING",
                "total_samples": len(records),
                "completed_samples": len(completed_ids),
                "updated_at_utc": started_at,
            },
        )

        gen_cfg = self.run_config.generation_config_dict()
        merged_records = list(existing_records)
        new_completed = new_failed = skipped_completed = 0
        seen_scope_ids: set[str] = set()

        for idx, raw_record in enumerate(records):
            sample_id_hint = str(raw_record.get("sample_id") or f"idx_{idx}")
            prepared: PreparedHallusionInput | None = None
            t0 = time.perf_counter()
            try:
                prepared = prepare_hallusion_input(
                    raw_record,
                    data_root=self.data_root,
                    prompt_config=self.run_config.prompt_config,
                    verify_image_exists=self.verify_image_exists,
                )
                sample_id = prepared.sample_id
                if sample_id in seen_scope_ids:
                    raise ValueError(f"Duplicate sample_id in input scope: {sample_id}")
                seen_scope_ids.add(sample_id)

                if sample_id in completed_ids:
                    skipped_completed += 1
                    continue

                raw_response = model_adapter.generate(prepared, gen_cfg)
                if not isinstance(raw_response, str):
                    raise TypeError(f"Model adapter returned non-string: {type(raw_response).__name__}")

                elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
                pred_record = {
                    "sample_id": sample_id,
                    "prediction": raw_response,
                    "raw_prediction": raw_response,
                    "status": "COMPLETED",
                    "run_config_hash": self.config_hash,
                    "model_key": self.run_config.model_key,
                    "model_id": self.run_config.model_id,
                    "stage": self.run_config.stage,
                    "visual_input": prepared.visual_input,
                    "prompt": prepared.prompt,
                    "timing_ms": elapsed_ms,
                    "metadata": prepared.metadata,
                }
                merged_records = merge_result_records(merged_records, [pred_record])
                _write_jsonl(predictions_path, merged_records)
                completed_ids.add(sample_id)
                new_completed += 1

            except Exception as exc:
                elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
                sample_id = prepared.sample_id if prepared is not None else sample_id_hint
                err_type = _classify_exception(exc)
                _append_jsonl(
                    errors_path,
                    {
                        "sample_id": sample_id,
                        "error_type": err_type,
                        "exception_class": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                        "timing_ms": elapsed_ms,
                        "timestamp_utc": _utc_now_iso(),
                    },
                )
                if sample_id not in completed_ids:
                    failed_record = {
                        "sample_id": sample_id,
                        "prediction": None,
                        "raw_prediction": None,
                        "status": "FAILED",
                        "error_type": err_type,
                        "error": f"{err_type}: {exc}",
                        "run_config_hash": self.config_hash,
                        "model_key": self.run_config.model_key,
                        "model_id": self.run_config.model_id,
                        "stage": self.run_config.stage,
                        "visual_input": prepared.visual_input if prepared is not None else None,
                        "prompt": prepared.prompt if prepared is not None else None,
                        "timing_ms": elapsed_ms,
                        "metadata": prepared.metadata if prepared is not None else {},
                    }
                    merged_records = merge_result_records(merged_records, [failed_record])
                    _write_jsonl(predictions_path, merged_records)
                new_failed += 1
                if err_type == "OOM_ERROR":
                    try:
                        import torch

                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass

        val_errors = validate_predictions(merged_records)
        if val_errors:
            raise RuntimeError(f"Generated predictions failed validation: {val_errors}")

        total_completed = sum(1 for r in merged_records if r.get("status") == "COMPLETED")
        total_failed = sum(1 for r in merged_records if r.get("status") != "COMPLETED")
        final_status = "COMPLETED" if total_failed == 0 and total_completed == len(records) else "FAILED"
        finished_at = _utc_now_iso()

        runtime_info = model_adapter.get_runtime_info()
        _write_json(
            self.output_dir / "metrics.json",
            {
                "status": "PENDING_SCORING",
                "benchmark": "hallusionbench",
                "stage": self.run_config.stage,
                "note": "Run evaluation.py --mode score to compute aAcc, fAcc, qAcc, VD/VS, and Easy/Hard.",
                "scores": None,
            },
        )
        _write_json(
            self.output_dir / "run_metadata.json",
            {
                "experiment_id": self.run_config.experiment_id,
                "run_id": self.run_config.run_id,
                "benchmark": "hallusionbench",
                "stage": self.run_config.stage,
                "model_key": self.run_config.model_key,
                "model_id": self.run_config.model_id,
                "configured_model_revision": self.run_config.model_revision,
                "configured_processor_revision": self.run_config.processor_revision,
                "resolved_model_revision": runtime_info.get("resolved_model_revision"),
                "resolved_processor_revision": runtime_info.get("resolved_processor_revision"),
                "dataset_source": self.run_config.dataset_source,
                "dataset_revision": self.run_config.dataset_revision,
                "dtype": self.run_config.dtype,
                "attn_implementation": runtime_info.get("attn_implementation", self.run_config.attn_implementation),
                "device_allocation": runtime_info.get("device_allocation"),
                "run_config_hash": self.config_hash,
                "status": final_status,
                "counts": {
                    "scope_total": len(records),
                    "total_completed": total_completed,
                    "total_failed": total_failed,
                    "new_completed_this_invocation": new_completed,
                    "new_failed_this_invocation": new_failed,
                    "skipped_already_completed": skipped_completed,
                },
                "started_at_utc": started_at,
                "finished_at_utc": finished_at,
                "environment": env,
            },
        )
        final_status_payload = {
            "experiment_id": self.run_config.experiment_id,
            "run_id": self.run_config.run_id,
            "stage": self.run_config.stage,
            "status": final_status,
            "total_samples": len(records),
            "completed_samples": total_completed,
            "failed_samples": total_failed,
            "run_config_hash": self.config_hash,
            "updated_at_utc": finished_at,
        }
        _write_json(run_status_path, final_status_payload)
        write_manifest(self.output_dir, self.run_config, status=final_status)
        return final_status_payload


# ---------------------------------------------------------------------------
# Hugging Face VLM adapter (SmolVLM2, Qwen3-VL, Phi-4-multimodal)
# ---------------------------------------------------------------------------

class HuggingFaceHallusionModel:
    def __init__(self, run_config: HallusionRunConfig) -> None:
        self.run_config = run_config
        self.spec = MODEL_SPECS[run_config.model_key]
        self.processor: Any = None
        self.model: Any = None
        self._runtime_info: dict[str, Any] = {
            "resolved_model_revision": None,
            "resolved_processor_revision": None,
            "attn_implementation": run_config.attn_implementation,
            "device_allocation": None,
        }

    def load_model(self) -> None:
        import torch
        from transformers import AutoProcessor

        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        torch_dtype = dtype_map[self.run_config.dtype]

        bf16_native, _ = _check_native_bf16(torch)
        if self.run_config.dtype == "bfloat16" and torch.cuda.is_available() and not bf16_native:
            raise RuntimeError("Refusing silent dtype fallback: bfloat16 requested on GPU without native BF16.")

        proc_kwargs: dict[str, Any] = {"trust_remote_code": self.run_config.trust_remote_code}
        if self.run_config.processor_revision:
            proc_kwargs["revision"] = self.run_config.processor_revision
        self.processor = AutoProcessor.from_pretrained(self.run_config.model_id, **proc_kwargs)

        model_kwargs: dict[str, Any] = {
            "torch_dtype": torch_dtype,
            "device_map": self.run_config.device_map,
            "trust_remote_code": self.run_config.trust_remote_code,
        }
        if self.run_config.model_revision:
            model_kwargs["revision"] = self.run_config.model_revision
        if self.run_config.attn_implementation:
            model_kwargs["attn_implementation"] = self.run_config.attn_implementation

        family = self.spec["loader_family"]
        if family == "smolvlm2":
            from transformers import AutoModelForImageTextToText

            self.model = AutoModelForImageTextToText.from_pretrained(self.run_config.model_id, **model_kwargs)
        elif family == "qwen3_vl":
            from transformers import Qwen3VLForConditionalGeneration

            self.model = Qwen3VLForConditionalGeneration.from_pretrained(self.run_config.model_id, **model_kwargs)
        elif family == "phi4_multimodal":
            from transformers import AutoConfig, AutoModelForCausalLM

            attn = self.run_config.attn_implementation or "eager"
            cfg_kwargs: dict[str, Any] = {"trust_remote_code": self.run_config.trust_remote_code}
            if self.run_config.model_revision:
                cfg_kwargs["revision"] = self.run_config.model_revision
            phi4_cfg = AutoConfig.from_pretrained(self.run_config.model_id, **cfg_kwargs)
            phi4_cfg._attn_implementation = attn
            phi4_cfg._attn_implementation_autoset = False
            if hasattr(phi4_cfg, "attn_implementation"):
                phi4_cfg.attn_implementation = attn
            model_kwargs["config"] = phi4_cfg
            model_kwargs["attn_implementation"] = attn
            self.model = AutoModelForCausalLM.from_pretrained(self.run_config.model_id, **model_kwargs)
        else:
            raise ValueError(f"Unsupported loader family: {family}")

        self.model.eval()
        self._capture_runtime_info()

    def _resolve_hub_commit_sha(self, revision: str | None) -> str | None:
        try:
            from huggingface_hub import HfApi

            sha = getattr(HfApi().model_info(self.run_config.model_id, revision=revision), "sha", None)
            if isinstance(sha, str) and sha.strip():
                return sha.strip()
        except Exception:
            pass
        return revision

    def _capture_runtime_info(self) -> None:
        model_commit = getattr(getattr(self.model, "config", None), "_commit_hash", None)
        proc_commit = getattr(self.processor, "_commit_hash", None) or getattr(
            getattr(self.processor, "tokenizer", None), "_commit_hash", None
        )
        if not model_commit:
            model_commit = self._resolve_hub_commit_sha(self.run_config.model_revision)
        if not proc_commit:
            proc_commit = self._resolve_hub_commit_sha(self.run_config.processor_revision) or model_commit

        hf_device_map = getattr(self.model, "hf_device_map", None)
        allocation: Any = (
            {str(k): str(v) for k, v in hf_device_map.items()}
            if hf_device_map is not None
            else str(self._primary_device())
        )
        self._runtime_info = {
            "resolved_model_revision": model_commit,
            "resolved_processor_revision": proc_commit,
            "attn_implementation": getattr(
                getattr(self.model, "config", None), "_attn_implementation", self.run_config.attn_implementation
            ),
            "device_allocation": allocation,
        }

    def _primary_device(self) -> Any:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return "cpu"

    def get_runtime_info(self) -> dict[str, Any]:
        return dict(self._runtime_info)

    def generate(self, prepared: PreparedHallusionInput, generation_config: Mapping[str, Any]) -> str:
        if self.model is None or self.processor is None:
            raise RuntimeError("Model is not loaded; call load_model() first.")
        import torch
        from PIL import Image

        pil_image = Image.open(prepared.image_path).convert("RGB") if prepared.visual_input and prepared.image_path else None
        if prepared.visual_input and pil_image is None:
            raise ValueError(f"Sample {prepared.sample_id}: visual_input is True but image_path is None")

        if self.spec["loader_family"] == "phi4_multimodal":
            image_tag = "<|image_1|>" if prepared.visual_input else ""
            formatted = f"<|user|>{image_tag}{prepared.prompt}<|end|><|assistant|>"
            inputs = (
                self.processor(text=formatted, images=[pil_image], return_tensors="pt")
                if prepared.visual_input
                else self.processor(text=formatted, return_tensors="pt")
            )
        else:
            chat_text = self.processor.apply_chat_template(
                prepared.messages, tokenize=False, add_generation_prompt=True
            )
            inputs = (
                self.processor(text=[chat_text], images=[pil_image], padding=True, return_tensors="pt")
                if prepared.visual_input
                else self.processor(text=[chat_text], padding=True, return_tensors="pt")
            )

        device = self._primary_device()
        moved = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        gen_kwargs = dict(generation_config)
        if not gen_kwargs.get("do_sample", False):
            gen_kwargs.pop("temperature", None)
            gen_kwargs.pop("top_p", None)

        if self.spec["loader_family"] == "phi4_multimodal":
            import inspect

            # Workaround for Microsoft Phi-4-multimodal remote code on transformers >= 4.49/5.x
            # where generate() may pass num_logits_to_keep=None into forward().
            try:
                if "num_logits_to_keep" in inspect.signature(self.model.forward).parameters:
                    gen_kwargs.setdefault("num_logits_to_keep", 1)
            except (TypeError, ValueError):
                pass

        with torch.inference_mode():
            output_ids = self.model.generate(**moved, **gen_kwargs)

        input_len = int(moved["input_ids"].shape[-1])
        decoded = self.processor.batch_decode(
            output_ids[:, input_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return decoded.strip()

    def release_resources(self) -> None:
        self.model = None
        self.processor = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


class SyntheticDryRunModel:
    def generate(self, prepared: PreparedHallusionInput, generation_config: Mapping[str, Any]) -> str:
        return f"SYNTHETIC_DRY_RUN_ONLY({'visual' if prepared.visual_input else 'text_only'}): yes"

    def get_runtime_info(self) -> dict[str, Any]:
        return {
            "resolved_model_revision": "SYNTHETIC_DRY_RUN",
            "resolved_processor_revision": "SYNTHETIC_DRY_RUN",
            "attn_implementation": "none",
            "device_allocation": "cpu",
        }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="HallusionBench Part B inference CLI (0-epoch zero-shot)")
    parser.add_argument("--mode", choices=("check-env", "dry-run", "run"), required=True)
    parser.add_argument("--model-key", choices=tuple(MODEL_SPECS.keys()), default="smolvlm2")
    parser.add_argument("--stage", choices=VALID_STAGES, default="pilot")
    parser.add_argument("--experiment-id", default="hallusionbench_dev")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--dtype", choices=ALLOWED_DTYPES, default="bfloat16")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--processor-revision", default=None)
    parser.add_argument("--dataset-json", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--dataset-source", default=HF_AUTHOR_MIRROR_SOURCE)
    parser.add_argument("--dataset-revision", default=None)
    parser.add_argument("--pilot-file", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=None, help="Allowed ONLY for --stage smoke_test")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--output-root", type=Path, default=Path(os.environ.get("OUTPUT_DIR", "results")))
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true", default=None)
    parser.add_argument("--allow-model-load", action="store_true")
    args = parser.parse_args(argv)

    env = collect_runtime_environment(REPO_ROOT)
    if args.mode == "check-env":
        print(json.dumps(env, indent=2, ensure_ascii=False))
        return 0

    spec = MODEL_SPECS[args.model_key]
    trust_remote = bool(args.trust_remote_code) if args.trust_remote_code is not None else bool(spec["default_trust_remote_code"])
    run_id = args.run_id or f"{args.stage}_{args.model_key}_{args.dtype}"

    run_config = HallusionRunConfig(
        experiment_id=args.experiment_id,
        run_id=run_id,
        model_key=args.model_key,
        model_id=spec["model_id"],
        stage=args.stage,
        model_revision=args.model_revision,
        processor_revision=args.processor_revision,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        device_map=args.device_map,
        trust_remote_code=trust_remote,
        dataset_source=args.dataset_source,
        dataset_revision=args.dataset_revision,
        max_new_tokens=args.max_new_tokens,
    )

    run_dir = args.output_root / args.experiment_id / run_id
    data_root = args.data_root or (args.dataset_json.parent if args.dataset_json else None)

    if args.dataset_json is not None:
        dataset = HallusionBenchDataset(
            args.dataset_json,
            data_root=data_root,
            dataset_source=args.dataset_source,
            dataset_revision=args.dataset_revision,
        )
        samples = dataset.load(verify_images=False, verify_image_readable=False)
        if args.pilot_file is not None:
            pilot_manifest = json.loads(args.pilot_file.read_text(encoding="utf-8"))
            pilot_ids = pilot_manifest.get("sample_ids")
            if not isinstance(pilot_ids, list) or not pilot_ids:
                raise ValueError(f"Invalid pilot manifest {args.pilot_file}")
            expected_sha = pilot_manifest.get("pilot_ids_sha256")
            if expected_sha is not None and sha256_json(pilot_ids) != expected_sha:
                raise ValueError(f"Pilot manifest checksum mismatch in {args.pilot_file}")
            samples = filter_by_pilot_ids(samples, pilot_ids, require_complete_groups=True)
        records = [s.to_model_safe_dict() for s in samples]
        if args.max_samples is not None:
            if args.stage != "smoke_test":
                raise ValueError("--max-samples is only allowed for --stage smoke_test")
            visual = [r for r in records if r["visual_input"]][: max(1, args.max_samples // 2)]
            text_only = [r for r in records if not r["visual_input"]][: max(1, args.max_samples - len(visual))]
            chosen = (visual + text_only)[: args.max_samples]
            records = chosen if chosen else records[: args.max_samples]
    elif args.mode == "dry-run":
        records = [
            {
                "sample_id": "synthetic_set0_fig0_q0_v0",
                "category": "VS",
                "subcategory": "chart",
                "visual_input": "0",
                "set_id": "0",
                "figure_id": "0",
                "question_id": "0",
                "question": "Is the sky typically blue on a clear sunny day?",
            },
            {
                "sample_id": "synthetic_set0_fig0_q1_v0",
                "category": "VS",
                "subcategory": "chart",
                "visual_input": "0",
                "set_id": "0",
                "figure_id": "0",
                "question_id": "1",
                "question": "Are two circles of different diameters the same size?",
            },
        ]
    else:
        raise ValueError("--dataset-json is required for --mode run")

    runner = HallusionInferenceRunner(
        run_config=run_config,
        output_dir=run_dir,
        repo_root=REPO_ROOT,
        data_root=data_root,
        verify_image_exists=(args.dataset_json is not None),
    )

    if args.mode == "dry-run":
        status = runner.run(
            records=records,
            model_adapter=SyntheticDryRunModel(),
            hardware_info={"cuda_available": False, "bf16_supported": False},
            resume=not args.no_resume,
        )
        print(json.dumps(status, indent=2, ensure_ascii=False))
        return 0 if status["status"] == "COMPLETED" else 1

    if not args.allow_model_load:
        print("Refusing to load model weights without explicit --allow-model-load flag.", file=sys.stderr)
        return 2

    precheck_errors = run_config.validate_for_stage(hardware_info=env, repo_root=REPO_ROOT)
    if precheck_errors:
        status = write_blocked_run(run_dir, run_config, precheck_errors, REPO_ROOT, env)
        print(json.dumps(status, indent=2, ensure_ascii=False))
        return 1

    adapter = HuggingFaceHallusionModel(run_config)
    try:
        adapter.load_model()
        status = runner.run(records=records, model_adapter=adapter, hardware_info=env, resume=not args.no_resume)
        print(json.dumps(status, indent=2, ensure_ascii=False))
        return 0 if status["status"] == "COMPLETED" else 1
    except Exception as exc:
        status = write_blocked_run(
            run_dir, run_config, [f"Model load/runtime failure ({type(exc).__name__}): {exc}"], REPO_ROOT, env
        )
        print(json.dumps(status, indent=2, ensure_ascii=False))
        return 1
    finally:
        adapter.release_resources()


if __name__ == "__main__":
    raise SystemExit(main())
