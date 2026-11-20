"""Inference runner, hardware/policy checks, and artifact export for HallusionBench (Member B).

Complies with:
- docs/model_policy.md (no quantization, no silent dtype fallback, locked revisions for official)
- docs/result_schema.md (all 7 artifacts, required prediction fields, honest status transitions)
- vlm_benchmark/results.py (stable_config_hash, validate_predictions, merge_result_records)

Importing this module NEVER downloads weights or datasets.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal local envs without PyYAML
    yaml = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.hallusionbench.dataset import (  # noqa: E402
    HallusionBenchDataset,
    filter_by_pilot_ids,
    sha256_json,
)
from benchmarks.hallusionbench.protocol import (  # noqa: E402
    MODEL_SPECS,
    HallusionBenchPromptConfig,
    PreparedHallusionInput,
    load_raw_hallusion_json,
    prepare_hallusion_input,
    select_pilot_records,
)
from vlm_benchmark.environment import collect_environment  # noqa: E402
from vlm_benchmark.results import (  # noqa: E402
    merge_result_records,
    read_jsonl,
    stable_config_hash,
    validate_predictions,
)


def _dump_yaml(payload: Mapping[str, Any]) -> str:
    if yaml is not None:
        return yaml.safe_dump(dict(payload), sort_keys=False, allow_unicode=True)
    # JSON is valid YAML 1.2 and keeps dry-runs/tests working even before PyYAML is installed.
    return json.dumps(dict(payload), indent=2, ensure_ascii=False) + "\n"


VALID_STAGES = ("smoke_test", "pilot", "official")
VALID_RUN_STATUSES = ("PENDING", "RUNNING", "COMPLETED", "BLOCKED", "FAILED")
ALLOWED_DTYPES = ("bfloat16", "float16", "float32")


class HallusionModelAdapter(Protocol):
    """Minimal inference interface (decouples runner from Hugging Face for CPU tests)."""

    def generate(
        self,
        prepared: PreparedHallusionInput,
        generation_config: Mapping[str, Any],
    ) -> str: ...

    def get_runtime_info(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class HallusionRunConfig:
    """Explicit, hashable configuration for a single HallusionBench inference run."""

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
    dataset_source: str = "tianyi-lab/HallusionBench"
    dataset_revision: str | None = None
    prompt_config: HallusionBenchPromptConfig = field(default_factory=HallusionBenchPromptConfig)
    do_sample: bool = False
    max_new_tokens: int = 128
    temperature: float | None = None
    top_p: float | None = None
    seed: int = 0
    batch_size: int = 1

    def generation_config_dict(self) -> dict[str, Any]:
        cfg: dict[str, Any] = {
            "do_sample": self.do_sample,
            "max_new_tokens": self.max_new_tokens,
        }
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
        """Fields that define experiment identity for resume compatibility."""
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
        """Validate run config against project policy and detected hardware."""
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
            errors.append("Policy violation: batch_size must be 1 per configs/experiment.yaml")
        if self.max_new_tokens is None or self.max_new_tokens <= 0:
            errors.append("Invalid generation_config: max_new_tokens must be a positive integer")

        if hardware_info is not None:
            cuda_available = bool(hardware_info.get("cuda_available"))
            bf16_supported = bool(hardware_info.get("bf16_supported"))
            if self.dtype == "bfloat16" and cuda_available and not bf16_supported:
                errors.append(
                    "Hardware incompatibility: requested bfloat16 on GPU without native BF16 support "
                    "(e.g., Kaggle T4). Request lead approval for float16 or use BF16-capable GPU; "
                    "automatic fallback is forbidden."
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

                    repo_errors = validate_all(
                        root / "configs" / "models.yaml",
                        root / "configs" / "datasets.yaml",
                        root / "configs" / "experiment.yaml",
                        official_stage=True,
                    )
                    for err in repo_errors:
                        errors.append(f"Repository official gate: {err}")
                except ModuleNotFoundError as exc:
                    errors.append(
                        f"Repository official gate unavailable ({exc}): install requirements.txt (PyYAML) "
                        "before running official stage"
                    )

        return errors


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_native_bf16(torch_module: Any) -> tuple[bool, bool]:
    """Return (bf16_native_supported, bf16_emulated_supported).

    Newer PyTorch versions default is_bf16_supported() to including_emulation=True,
    which reports True on Turing GPUs like Tesla T4 (sm_75) that lack native BF16.
    """
    if not torch_module.cuda.is_available():
        return False, False
    emulated = bool(torch_module.cuda.is_bf16_supported())
    try:
        native = bool(torch_module.cuda.is_bf16_supported(including_emulation=False))
    except TypeError:
        props = torch_module.cuda.get_device_properties(0)
        native = emulated and props.major >= 8
    return native, emulated


def collect_runtime_environment(repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    """Collect full runtime environment metadata including GPU/CUDA/BF16 if torch exists."""
    env = collect_environment(repo_root)
    env["cuda_available"] = False
    env["gpu_count"] = 0
    env["bf16_supported"] = False
    env["bf16_emulated_supported"] = False
    env["gpus"] = []
    env["packages"]["PIL"] = None
    env["packages"]["huggingface_hub"] = None

    for pkg in ("PIL", "huggingface_hub"):
        try:
            mod = __import__(pkg)
            env["packages"][pkg] = getattr(mod, "__version__", None)
        except Exception:
            env["packages"][pkg] = None

    try:
        import torch

        env["packages"]["torch"] = getattr(torch, "__version__", None)
        env["cuda"] = getattr(torch.version, "cuda", None)
        cuda_avail = bool(torch.cuda.is_available())
        env["cuda_available"] = cuda_avail
        if cuda_avail:
            count = int(torch.cuda.device_count())
            env["gpu_count"] = count
            bf16_native, bf16_emulated = _check_native_bf16(torch)
            env["bf16_supported"] = bf16_native
            env["bf16_emulated_supported"] = bf16_emulated
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
            env["gpu"] = [g["name"] for g in gpus]
            env["gpus"] = gpus
    except Exception:
        pass

    return env


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), ensure_ascii=False) + "\n")


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(record), ensure_ascii=False) + "\n")
        handle.flush()


def write_manifest(
    run_dir: Path,
    run_config: HallusionRunConfig,
    status: str,
) -> dict[str, Any]:
    """Write manifest.json with SHA-256 checksums for all required run artifacts."""
    artifact_names = (
        "predictions.jsonl",
        "metrics.json",
        "run_config.yaml",
        "run_metadata.json",
        "errors.jsonl",
        "run_status.json",
    )
    files: dict[str, dict[str, Any]] = {}
    for name in artifact_names:
        file_path = run_dir / name
        if file_path.exists():
            files[name] = {
                "size_bytes": file_path.stat().st_size,
                "sha256": sha256_file(file_path),
            }

    manifest = {
        "experiment_id": run_config.experiment_id,
        "run_id": run_config.run_id,
        "benchmark": "hallusionbench",
        "stage": run_config.stage,
        "model_key": run_config.model_key,
        "model_id": run_config.model_id,
        "run_config_hash": run_config.config_hash(),
        "status": status,
        "generated_at_utc": utc_now_iso(),
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
    """Export an honest BLOCKED run with all 7 required artifacts and evidence."""
    out_dir = Path(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = dict(hardware_info) if hardware_info is not None else collect_runtime_environment(repo_root)
    now = utc_now_iso()

    _write_jsonl(out_dir / "predictions.jsonl", [])
    _write_jsonl(
        out_dir / "errors.jsonl",
        [
            {
                "sample_id": None,
                "error_type": "POLICY_OR_HARDWARE_BLOCKED",
                "message": reason,
                "timestamp_utc": now,
            }
            for reason in reasons
        ],
    )
    (out_dir / "run_config.yaml").write_text(
        _dump_yaml(run_config.to_dict()),
        encoding="utf-8",
    )
    _write_json(
        out_dir / "metrics.json",
        {
            "status": "BLOCKED",
            "note": "Run was blocked before inference; no metrics computed.",
            "scores": None,
        },
    )
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
    name = type(exc).__name__
    msg = str(exc).lower()
    if "outofmemory" in name.lower() or "out of memory" in msg or "cuda oom" in msg:
        return "OOM_ERROR"
    if isinstance(exc, FileNotFoundError):
        return "DATA_ERROR"
    if isinstance(exc, (ValueError, KeyError, TypeError)):
        return "INPUT_OR_SCHEMA_ERROR"
    return "TECHNICAL_ERROR"


class HallusionInferenceRunner:
    """Resumable, policy-compliant HallusionBench inference runner."""

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
        validation_errors = validate_predictions(existing)
        if validation_errors:
            raise ValueError(f"Existing predictions.jsonl is invalid: {validation_errors}")
        existing_hashes = {str(r.get("run_config_hash")) for r in existing}
        if existing_hashes != {self.config_hash}:
            raise ValueError(
                "Refusing to resume: existing predictions.jsonl has incompatible run_config_hash "
                f"{sorted(existing_hashes)} != {self.config_hash}. Create a new run_id instead."
            )
        return existing

    def run(
        self,
        records: Sequence[Mapping[str, Any]],
        model_adapter: HallusionModelAdapter,
        hardware_info: Mapping[str, Any] | None = None,
        resume: bool = True,
    ) -> dict[str, Any]:
        """Execute inference over records and export all 7 required artifacts."""
        env = dict(hardware_info) if hardware_info is not None else collect_runtime_environment(self.repo_root)
        policy_errors = self.run_config.validate_for_stage(
            hardware_info=env,
            repo_root=self.repo_root,
        )
        if policy_errors:
            return write_blocked_run(
                self.output_dir,
                self.run_config,
                reasons=policy_errors,
                repo_root=self.repo_root,
                hardware_info=env,
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = self.output_dir / "predictions.jsonl"
        errors_path = self.output_dir / "errors.jsonl"
        run_status_path = self.output_dir / "run_status.json"

        existing_records: list[dict[str, Any]] = []
        if resume:
            existing_records = self._load_existing_predictions(predictions_path)
        elif predictions_path.exists() and predictions_path.stat().st_size > 0:
            raise FileExistsError(
                f"Refusing to overwrite existing predictions at {predictions_path}; "
                "enable resume=True or choose a new run_id."
            )

        completed_ids = {
            str(r["sample_id"])
            for r in existing_records
            if r.get("status") == "COMPLETED"
        }

        if not existing_records or not errors_path.exists():
            # Start with a clean errors.jsonl on fresh runs (clears stale pre-inference BLOCKED entries
            # when retrying a previously blocked run_id with 0 predictions).
            errors_path.write_text("", encoding="utf-8")

        (self.output_dir / "run_config.yaml").write_text(
            _dump_yaml(self.run_config.to_dict()),
            encoding="utf-8",
        )

        started_at = utc_now_iso()
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
        new_completed = 0
        new_failed = 0
        skipped_completed = 0

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
                    raise TypeError(f"Model adapter returned non-string response: {type(raw_response).__name__}")

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
                        "timestamp_utc": utc_now_iso(),
                    },
                )

                # Never turn technical/data errors into fake predictions.
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

        # Validate predictions.jsonl strictly with shared repository validator.
        val_errors = validate_predictions(merged_records)
        if val_errors:
            raise RuntimeError(f"Generated predictions failed vlm_benchmark.results validation: {val_errors}")

        total_completed = sum(1 for r in merged_records if r.get("status") == "COMPLETED")
        total_failed = sum(1 for r in merged_records if r.get("status") != "COMPLETED")
        final_status = "COMPLETED" if total_failed == 0 and total_completed == len(records) else "FAILED"
        finished_at = utc_now_iso()

        runtime_info = model_adapter.get_runtime_info()
        _write_json(
            self.output_dir / "metrics.json",
            {
                "status": "PENDING_SCORING",
                "benchmark": "hallusionbench",
                "stage": self.run_config.stage,
                "note": (
                    "Raw predictions generated by Member B. Metrics (aAcc, fAcc, qAcc, VD/VS, easy/hard) "
                    "must be computed by Member C using the verified HallusionBench scoring protocol."
                ),
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


class HuggingFaceHallusionModel:
    """Real Hugging Face VLM adapter for SmolVLM2, Qwen3-VL, and Phi-4-multimodal.

    Lazy-imports torch/transformers/PIL only inside load_model() and generate().
    Strictly obeys dtype (no silent fallback) and strips prompt tokens on decode.
    """

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

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map[self.run_config.dtype]

        bf16_native, _ = _check_native_bf16(torch)
        if self.run_config.dtype == "bfloat16" and torch.cuda.is_available() and not bf16_native:
            raise RuntimeError(
                "Refusing silent dtype fallback: bfloat16 requested on GPU without native BF16 support."
            )

        proc_kwargs: dict[str, Any] = {
            "trust_remote_code": self.run_config.trust_remote_code,
        }
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

            self.model = AutoModelForImageTextToText.from_pretrained(
                self.run_config.model_id,
                **model_kwargs,
            )
        elif family == "qwen3_vl":
            from transformers import Qwen3VLForConditionalGeneration

            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.run_config.model_id,
                **model_kwargs,
            )
        elif family == "phi4_multimodal":
            from transformers import AutoConfig, AutoModelForCausalLM

            attn = self.run_config.attn_implementation or "eager"
            cfg_kwargs: dict[str, Any] = {
                "trust_remote_code": self.run_config.trust_remote_code,
            }
            if self.run_config.model_revision:
                cfg_kwargs["revision"] = self.run_config.model_revision
            phi4_cfg = AutoConfig.from_pretrained(self.run_config.model_id, **cfg_kwargs)
            # Phi-4-multimodal's remote config.json hardcodes _attn_implementation="flash_attention_2";
            # override it explicitly on the config object before from_pretrained for T4/non-FA2 GPUs.
            phi4_cfg._attn_implementation = attn
            phi4_cfg._attn_implementation_autoset = False
            if hasattr(phi4_cfg, "attn_implementation"):
                phi4_cfg.attn_implementation = attn
            model_kwargs["config"] = phi4_cfg
            model_kwargs["attn_implementation"] = attn
            model_kwargs["_attn_implementation"] = attn
            self.model = AutoModelForCausalLM.from_pretrained(
                self.run_config.model_id,
                **model_kwargs,
            )
        else:
            raise ValueError(f"Unsupported loader family: {family}")

        self.model.eval()
        self._capture_runtime_info()

    def _resolve_hub_commit_sha(self, revision: str | None) -> str | None:
        """Resolve exact 40-char Hugging Face commit SHA when transformers _commit_hash is None."""
        try:
            from huggingface_hub import HfApi

            info = HfApi().model_info(self.run_config.model_id, revision=revision)
            sha = getattr(info, "sha", None)
            if isinstance(sha, str) and sha.strip():
                return sha.strip()
        except Exception:
            pass
        return revision

    def _capture_runtime_info(self) -> None:
        model_commit = getattr(getattr(self.model, "config", None), "_commit_hash", None)
        proc_commit = getattr(self.processor, "_commit_hash", None)
        if proc_commit is None:
            tok = getattr(self.processor, "tokenizer", None)
            proc_commit = getattr(tok, "_commit_hash", None)

        if not model_commit:
            model_commit = self._resolve_hub_commit_sha(self.run_config.model_revision)
        if not proc_commit:
            proc_commit = self._resolve_hub_commit_sha(self.run_config.processor_revision) or model_commit

        hf_device_map = getattr(self.model, "hf_device_map", None)
        if hf_device_map is not None:
            allocation: Any = {str(k): str(v) for k, v in hf_device_map.items()}
        else:
            allocation = str(self._primary_device())

        self._runtime_info = {
            "resolved_model_revision": model_commit,
            "resolved_processor_revision": proc_commit,
            "attn_implementation": getattr(
                getattr(self.model, "config", None),
                "_attn_implementation",
                self.run_config.attn_implementation,
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

    def _move_inputs_to_device(self, inputs: Any, device: Any) -> Any:
        import torch

        moved: dict[str, Any] = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                moved[k] = v.to(device)
            else:
                moved[k] = v
        return moved

    def generate(
        self,
        prepared: PreparedHallusionInput,
        generation_config: Mapping[str, Any],
    ) -> str:
        if self.model is None or self.processor is None:
            raise RuntimeError("Model is not loaded; call load_model() first.")

        import torch
        from PIL import Image

        pil_image = None
        if prepared.visual_input:
            if not prepared.image_path:
                raise ValueError(f"Sample {prepared.sample_id}: visual_input is True but image_path is None")
            pil_image = Image.open(prepared.image_path).convert("RGB")

        family = self.spec["loader_family"]
        if family == "phi4_multimodal":
            image_tag = "<|image_1|>" if prepared.visual_input else ""
            formatted_prompt = f"<|user|>{image_tag}{prepared.prompt}<|end|><|assistant|>"
            if prepared.visual_input:
                inputs = self.processor(
                    text=formatted_prompt,
                    images=[pil_image],
                    return_tensors="pt",
                )
            else:
                inputs = self.processor(
                    text=formatted_prompt,
                    return_tensors="pt",
                )
        else:
            # Standard HF chat template flow for SmolVLM2 and Qwen3-VL.
            chat_text = self.processor.apply_chat_template(
                prepared.messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            if prepared.visual_input:
                inputs = self.processor(
                    text=[chat_text],
                    images=[pil_image],
                    padding=True,
                    return_tensors="pt",
                )
            else:
                inputs = self.processor(
                    text=[chat_text],
                    padding=True,
                    return_tensors="pt",
                )

        device = self._primary_device()
        inputs = self._move_inputs_to_device(inputs, device)

        gen_kwargs = dict(generation_config)
        if not gen_kwargs.get("do_sample", False):
            gen_kwargs.pop("temperature", None)
            gen_kwargs.pop("top_p", None)

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        input_len = int(inputs["input_ids"].shape[-1])
        generated_only = output_ids[:, input_len:]
        decoded = self.processor.batch_decode(
            generated_only,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
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
    """Deterministic CPU-only model for pipeline dry-runs (NEVER used for real results)."""

    def generate(
        self,
        prepared: PreparedHallusionInput,
        generation_config: Mapping[str, Any],
    ) -> str:
        mode = "visual" if prepared.visual_input else "text_only"
        return f"SYNTHETIC_DRY_RUN_ONLY({mode}): yes"

    def get_runtime_info(self) -> dict[str, Any]:
        return {
            "resolved_model_revision": "SYNTHETIC_DRY_RUN",
            "resolved_processor_revision": "SYNTHETIC_DRY_RUN",
            "attn_implementation": "none",
            "device_allocation": "cpu",
        }


def _build_synthetic_fixture_records() -> list[dict[str, Any]]:
    return [
        {
            "sample_id": "synthetic_set0_fig0_q0_v0",
            "category": "VD",
            "subcategory": "illusion",
            "visual_input": "0",
            "set_id": "0",
            "figure_id": "0",
            "question_id": "0",
            "question": "Is the sky typically blue on a clear sunny day?",
            "gt_answer": "1",
        },
        {
            "sample_id": "synthetic_set0_fig0_q1_v0",
            "category": "VD",
            "subcategory": "illusion",
            "visual_input": "0",
            "set_id": "0",
            "figure_id": "0",
            "question_id": "1",
            "question": "Are two circles of different diameters the same size?",
            "gt_answer": "0",
        },
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="HallusionBench Member B inference CLI")
    parser.add_argument(
        "--mode",
        choices=("check-env", "dry-run", "run"),
        required=True,
        help="check-env: print GPU/CUDA/BF16 info; dry-run: synthetic CPU test; run: real HF inference",
    )
    parser.add_argument("--model-key", choices=tuple(MODEL_SPECS.keys()), default="smolvlm2")
    parser.add_argument("--stage", choices=VALID_STAGES, default="smoke_test")
    parser.add_argument("--experiment-id", default="hallusionbench_b_dev")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--dtype", choices=ALLOWED_DTYPES, default="bfloat16")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--processor-revision", default=None)
    parser.add_argument("--dataset-json", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--dataset-source", default="tianyi-lab/HallusionBench")
    parser.add_argument("--dataset-revision", default=None)
    parser.add_argument(
        "--pilot-file",
        type=Path,
        default=None,
        help="Optional Member A frozen pilot_ids.json manifest (enforces complete groups & checksum).",
    )
    parser.add_argument("--max-groups", type=int, default=None, help="Limit to N complete groups for pilot/smoke_test")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional hard cap ONLY for 2-sample smoke_test")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--output-root", type=Path, default=Path(os.environ.get("OUTPUT_DIR", "results")))
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true", default=None)
    parser.add_argument(
        "--allow-model-load",
        action="store_true",
        help="Explicit opt-in required for --mode run to download/load real Hugging Face weights.",
    )
    args = parser.parse_args(argv)

    env = collect_runtime_environment(REPO_ROOT)
    if args.mode == "check-env":
        print(json.dumps(env, indent=2, ensure_ascii=False))
        return 0

    spec = MODEL_SPECS[args.model_key]
    trust_remote = (
        bool(args.trust_remote_code)
        if args.trust_remote_code is not None
        else bool(spec["default_trust_remote_code"])
    )
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
        if args.pilot_file is not None:
            pilot_manifest = json.loads(args.pilot_file.read_text(encoding="utf-8"))
            pilot_ids = pilot_manifest.get("sample_ids")
            if not isinstance(pilot_ids, list) or not pilot_ids:
                raise ValueError(f"Invalid pilot manifest {args.pilot_file}: missing non-empty 'sample_ids'")
            expected_sha = pilot_manifest.get("pilot_ids_sha256")
            actual_sha = sha256_json(pilot_ids)
            if expected_sha is not None and actual_sha != expected_sha:
                raise ValueError(
                    f"Pilot manifest checksum mismatch in {args.pilot_file}: "
                    f"expected {expected_sha}, got {actual_sha}"
                )
            dataset = HallusionBenchDataset(
                args.dataset_json,
                data_root=data_root,
                dataset_revision=args.dataset_revision,
            )
            samples = dataset.load(verify_images=False, verify_image_readable=False)
            selected_samples = filter_by_pilot_ids(samples, pilot_ids, require_complete_groups=True)
            records = [s.to_model_safe_dict() for s in selected_samples]
        else:
            try:
                dataset = HallusionBenchDataset(
                    args.dataset_json,
                    data_root=data_root,
                    dataset_revision=args.dataset_revision,
                )
                records = dataset.load_dicts(
                    verify_images=False,
                    verify_image_readable=False,
                    model_safe=True,
                )
            except ValueError:
                # Backward-compatible fallback for minimal raw fixtures lacking strict Member A fields.
                records = load_raw_hallusion_json(
                    args.dataset_json,
                    data_root=data_root,
                    verify_image_exists=False,
                )
        if args.max_groups is not None:
            records = select_pilot_records(records, max_groups=args.max_groups, include_text_only=True)
        if args.max_samples is not None:
            if args.stage != "smoke_test":
                raise ValueError("--max-samples is only allowed for --stage smoke_test; use --max-groups for pilot.")
            # Pick balanced visual + text-only samples for quick smoke test when possible.
            visual = [r for r in records if r["visual_input"]][: max(1, args.max_samples // 2)]
            text_only = [r for r in records if not r["visual_input"]][: max(1, args.max_samples - len(visual))]
            chosen = (visual + text_only)[: args.max_samples]
            records = chosen if chosen else records[: args.max_samples]
    elif args.mode == "dry-run":
        records = _build_synthetic_fixture_records()
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

    # Pre-check policy/hardware BEFORE downloading heavy weights.
    precheck_errors = run_config.validate_for_stage(hardware_info=env, repo_root=REPO_ROOT)
    if precheck_errors:
        status = write_blocked_run(run_dir, run_config, reasons=precheck_errors, repo_root=REPO_ROOT, hardware_info=env)
        print(json.dumps(status, indent=2, ensure_ascii=False))
        return 1

    adapter = HuggingFaceHallusionModel(run_config)
    try:
        adapter.load_model()
        status = runner.run(
            records=records,
            model_adapter=adapter,
            hardware_info=env,
            resume=not args.no_resume,
        )
        print(json.dumps(status, indent=2, ensure_ascii=False))
        return 0 if status["status"] == "COMPLETED" else 1
    except Exception as exc:
        status = write_blocked_run(
            run_dir,
            run_config,
            reasons=[f"Model load/runtime failure ({type(exc).__name__}): {exc}"],
            repo_root=REPO_ROOT,
            hardware_info=env,
        )
        print(json.dumps(status, indent=2, ensure_ascii=False))
        return 1
    finally:
        adapter.release_resources()


if __name__ == "__main__":
    raise SystemExit(main())
