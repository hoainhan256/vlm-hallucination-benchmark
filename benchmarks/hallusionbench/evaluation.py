"""HallusionBench Part C — Scoring (aAcc/fAcc/qAcc, VD/VS, Easy/Hard), error taxonomy & cross-model analysis.

Evaluation mode is explicitly labeled 'alternative_evaluation_rule_based' because it uses
deterministic Yes/No parsing rather than a paid GPT-4 judge.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.hallusionbench.dataset import HallusionBenchDataset, sha256_file  # noqa: E402
from vlm_benchmark.results import merge_result_records, read_jsonl, validate_predictions  # noqa: E402


# 6 standardized error labels per docs/evaluation_policy.md
VALID_ERROR_TYPES = {
    "incorrect_prediction",
    "hallucination",
    "format_violation",
    "technical_error",
    "label_data_error",
    "undetermined",
}


# ---------------------------------------------------------------------------
# Deterministic Yes/No parser & metadata resolvers
# ---------------------------------------------------------------------------

def parse_yes_no(raw_text: Any) -> str | None:
    """Normalize a VLM response into 'yes', 'no', or None (format_violation)."""
    if raw_text is None:
        return None
    text = str(raw_text).strip().lower()
    if not text:
        return None
    if re.search(r"\b(yes\s*(and|or|/)\s*no|no\s*(and|or|/)\s*yes)\b", text):
        return None
    match = re.match(r"^(yes|no)\b", text)
    if match:
        return match.group(1)
    has_yes = bool(re.search(r"\byes\b", text))
    has_no = bool(re.search(r"\bno\b", text))
    if has_yes and not has_no:
        return "yes"
    if has_no and not has_yes:
        return "no"
    return None


def _resolve_reference_answer(meta: Mapping[str, Any]) -> str:
    raw = meta.get("reference_answer")
    if raw is None:
        raw = meta.get("gt_answer")
    if raw is None:
        raise ValueError("Dataset metadata missing 'reference_answer' / 'gt_answer'")
    norm = str(raw).strip().lower()
    if norm in {"1", "yes"}:
        return "yes"
    if norm in {"0", "no"}:
        return "no"
    return norm


def _resolve_figure_key(meta: Mapping[str, Any]) -> str | None:
    if meta.get("figure_key") is not None:
        return str(meta["figure_key"])
    if meta.get("visual_input") is False or str(meta.get("visual_input_raw", "")).strip() == "0":
        return None
    if meta.get("figure_group_key") is not None:
        return str(meta["figure_group_key"])
    if meta.get("set_id") is not None and meta.get("figure_id") is not None:
        return f"{meta.get('category', '')}:{meta.get('subcategory', '')}:{meta['set_id']}:{meta['figure_id']}"
    return None


def _resolve_pair_key(meta: Mapping[str, Any]) -> str | None:
    if meta.get("pair_key") is not None:
        return str(meta["pair_key"])
    if meta.get("question_pair_key") is not None:
        return str(meta["question_pair_key"])
    if meta.get("set_id") is not None and meta.get("question_id") is not None:
        return f"{meta.get('category', '')}:{meta.get('subcategory', '')}:{meta['set_id']}:{meta['question_id']}"
    return None


# ---------------------------------------------------------------------------
# Metric computation (aAcc, fAcc, qAcc, VD/VS, Easy/Hard, subcategory, visual_input)
# ---------------------------------------------------------------------------

def _compute_subset_metrics(items: list[dict[str, Any]]) -> dict[str, Any]:
    total_q = len(items)
    correct_q = sum(1 for r in items if r["is_correct"])
    a_acc = (correct_q / total_q) if total_q > 0 else 0.0

    figure_groups: dict[str, list[bool]] = defaultdict(list)
    for r in items:
        if r["figure_key"] is not None:
            figure_groups[r["figure_key"]].append(r["is_correct"])
    total_f = len(figure_groups)
    correct_f = sum(1 for outcomes in figure_groups.values() if all(outcomes))
    f_acc = (correct_f / total_f) if total_f > 0 else 0.0

    pair_groups: dict[str, list[bool]] = defaultdict(list)
    for r in items:
        if r["pair_key"] is not None:
            pair_groups[r["pair_key"]].append(r["is_correct"])
    complete_pairs = {k: v for k, v in pair_groups.items() if len(v) >= 2}
    total_p = len(complete_pairs)
    correct_p = sum(1 for outcomes in complete_pairs.values() if all(outcomes))
    q_acc = (correct_p / total_p) if total_p > 0 else 0.0

    return {
        "aAcc": round(a_acc, 6),
        "fAcc": round(f_acc, 6),
        "qAcc": round(q_acc, 6),
        "counts": {
            "scored_questions": total_q,
            "scored_figures": total_f,
            "scored_complete_pairs": total_p,
            "incomplete_pairs_skipped": len(pair_groups) - total_p,
        },
    }


def score_hallusionbench(
    predictions: Iterable[Mapping[str, Any]],
    dataset_records: Mapping[str, Mapping[str, Any]],
    protocol_version: str = "unverified_draft_rule_based",
    evaluation_mode: str = "alternative_evaluation_rule_based",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compute overall + subgroup HallusionBench metrics while isolating technical & format errors."""
    pred_list = [dict(p) for p in predictions]
    schema_errors = validate_predictions(pred_list)
    if schema_errors:
        raise ValueError(f"Invalid predictions schema: {schema_errors}")

    error_records: list[dict[str, Any]] = []
    valid_results: list[dict[str, Any]] = []

    for pred in pred_list:
        sample_id = str(pred["sample_id"])
        status = pred.get("status")

        if status != "COMPLETED":
            error_records.append(
                {
                    "sample_id": sample_id,
                    "error_type": "technical_error",
                    "status": status,
                    "detail": pred.get("error") or pred.get("error_type") or "Non-completed inference status",
                }
            )
            continue

        meta = dataset_records.get(sample_id)
        if meta is None:
            error_records.append(
                {
                    "sample_id": sample_id,
                    "error_type": "label_data_error",
                    "detail": "sample_id not found in locked dataset metadata",
                }
            )
            continue

        parsed_pred = parse_yes_no(pred.get("prediction"))
        gt_answer = _resolve_reference_answer(meta)

        if parsed_pred is None:
            error_records.append(
                {
                    "sample_id": sample_id,
                    "error_type": "format_violation",
                    "raw_prediction": pred.get("raw_prediction", pred.get("prediction")),
                    "detail": "Could not extract deterministic Yes/No answer",
                }
            )
            is_correct = False
        else:
            is_correct = parsed_pred == gt_answer

        valid_results.append(
            {
                "sample_id": sample_id,
                "is_correct": is_correct,
                "parsed_pred": parsed_pred,
                "gt_answer": gt_answer,
                "category": str(meta["category"]).strip().upper() if meta.get("category") else None,
                "subcategory": str(meta["subcategory"]).strip() if meta.get("subcategory") else None,
                "difficulty": str(meta["difficulty"]).strip().lower() if meta.get("difficulty") else None,
                "visual_input_raw": str(meta["visual_input_raw"]).strip() if meta.get("visual_input_raw") is not None else None,
                "figure_key": _resolve_figure_key(meta),
                "pair_key": _resolve_pair_key(meta),
            }
        )

    overall = _compute_subset_metrics(valid_results)

    def _group_breakdown(field_name: str) -> dict[str, Any]:
        keys = sorted({r[field_name] for r in valid_results if r[field_name] is not None})
        return {k: _compute_subset_metrics([r for r in valid_results if r[field_name] == k]) for k in keys}

    metrics = {
        "benchmark": "hallusionbench",
        "protocol_version": protocol_version,
        "evaluation_mode": evaluation_mode,
        "summary_counts": {
            "total_predictions_received": len(pred_list),
            "scored_questions": overall["counts"]["scored_questions"],
            "scored_figures": overall["counts"]["scored_figures"],
            "scored_complete_pairs": overall["counts"]["scored_complete_pairs"],
            "incomplete_pairs_skipped": overall["counts"]["incomplete_pairs_skipped"],
            "error_records": len(error_records),
        },
        "metrics": {
            "aAcc": overall["aAcc"],
            "fAcc": overall["fAcc"],
            "qAcc": overall["qAcc"],
        },
        "by_category": _group_breakdown("category"),
        "by_subcategory": _group_breakdown("subcategory"),
        "by_difficulty": _group_breakdown("difficulty"),
        "by_visual_input": _group_breakdown("visual_input_raw"),
    }
    return metrics, error_records


def _refresh_manifest_checksums(run_dir: Path, model_key: str | None = None) -> None:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files") if isinstance(manifest.get("files"), dict) else {}
    resolved_model_key = model_key or manifest.get("model_key")
    tracked = [
        "predictions.jsonl",
        "metrics.json",
        "run_config.yaml",
        "run_metadata.json",
        "errors.jsonl",
        "run_status.json",
        "inference_config.json",
        "inference_summary.json",
        "inference_log.txt",
    ]
    if isinstance(resolved_model_key, str) and resolved_model_key.strip():
        mk = resolved_model_key.strip()
        tracked.extend(
            [
                f"{mk}_hallusionbench_responses.json",
                f"{mk}_hallusionbench_evaluation.json",
                f"{mk}_hallusionbench_summary.json",
            ]
        )
    for name in tracked:
        file_path = run_dir / name
        if file_path.is_file():
            files[name] = {"size_bytes": file_path.stat().st_size, "sha256": sha256_file(file_path)}
    manifest["files"] = files
    manifest["scored_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _infer_model_key(run_dir: Path, run_meta: Mapping[str, Any], preds: Sequence[Mapping[str, Any]]) -> str:
    for candidate in (
        run_meta.get("model_key"),
        preds[0].get("model_key") if preds else None,
    ):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    name = run_dir.name.lower()
    for known in ("smolvlm2", "qwen3_vl", "phi4_multimodal"):
        if known in name:
            return known
    return "model"


def build_detailed_evaluation_records(
    predictions: Sequence[Mapping[str, Any]],
    dataset_records: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build per-sample evaluated records (MMHal-style) aligned with HallusionBench scoring."""
    rows: list[dict[str, Any]] = []
    for idx, pred in enumerate(predictions):
        sid = str(pred.get("sample_id"))
        status = str(pred.get("status", "UNKNOWN"))
        raw_answer = pred.get("raw_prediction", pred.get("prediction"))
        meta = dataset_records.get(sid)
        gt_answer = _resolve_reference_answer(meta) if meta is not None else None

        if status != "COMPLETED":
            parsed_pred = None
            is_correct: bool | None = None
            error_type = "technical_error"
            inference_status = status
        elif meta is None:
            parsed_pred = None
            is_correct = None
            error_type = "label_data_error"
            inference_status = "SUCCESS"
        else:
            parsed_pred = parse_yes_no(pred.get("prediction"))
            inference_status = "SUCCESS"
            if parsed_pred is None:
                is_correct = False
                error_type = "format_violation"
            elif parsed_pred == gt_answer:
                is_correct = True
                error_type = "None"
            else:
                is_correct = False
                error_type = "incorrect_prediction"

        rows.append(
            {
                "index": idx,
                "sample_id": sid,
                "category": meta.get("category") if meta else None,
                "subcategory": meta.get("subcategory") if meta else None,
                "difficulty": meta.get("difficulty") if meta else None,
                "visual_input": meta.get("visual_input") if meta else pred.get("visual_input"),
                "visual_input_raw": meta.get("visual_input_raw") if meta else None,
                "set_id": meta.get("set_id") if meta else None,
                "figure_id": meta.get("figure_id") if meta else None,
                "question_id": meta.get("question_id") if meta else None,
                "filename": meta.get("filename") if meta else None,
                "figure_key": _resolve_figure_key(meta) if meta else None,
                "pair_key": _resolve_pair_key(meta) if meta else None,
                "question": meta.get("question") if meta else None,
                "gt_answer": gt_answer,
                "gt_answer_details": meta.get("gt_answer_details") if meta else None,
                "model_answer": raw_answer if raw_answer is not None else "",
                "parsed_prediction": parsed_pred,
                "is_correct": is_correct,
                "inference_status": inference_status,
                "error_type": error_type,
                "timing_ms": pred.get("timing_ms"),
            }
        )
    return rows


def build_mmhal_style_summary(
    metrics: Mapping[str, Any],
    run_meta: Mapping[str, Any],
    predictions: Sequence[Mapping[str, Any]],
    model_key: str,
) -> dict[str, Any]:
    """Build clean benchmark summary JSON combining MMHal readability with HallusionBench metadata."""
    env = run_meta.get("environment") if isinstance(run_meta.get("environment"), Mapping) else {}
    gpus = env.get("gpu") if isinstance(env.get("gpu"), list) else []
    device_label = ", ".join(str(g) for g in gpus) if gpus else str(run_meta.get("device_allocation") or "unknown")
    success_count = sum(1 for r in predictions if r.get("status") == "COMPLETED")
    failed_count = len(predictions) - success_count
    counts = metrics.get("summary_counts") or {}
    scores = metrics.get("metrics") or {}

    return {
        "benchmark": "HallusionBench",
        "model_key": model_key,
        "model": run_meta.get("model_id") or model_key,
        "model_revision": run_meta.get("resolved_model_revision") or run_meta.get("configured_model_revision"),
        "processor_revision": run_meta.get("resolved_processor_revision") or run_meta.get("configured_processor_revision"),
        "dataset": run_meta.get("dataset_source"),
        "dataset_revision": run_meta.get("dataset_revision"),
        "experiment_id": run_meta.get("experiment_id"),
        "run_id": run_meta.get("run_id"),
        "stage": run_meta.get("stage"),
        "status": run_meta.get("status"),
        "device": device_label,
        "dtype": run_meta.get("dtype"),
        "attn_implementation": run_meta.get("attn_implementation"),
        "run_config_hash": run_meta.get("run_config_hash"),
        "inference_started_at_utc": run_meta.get("started_at_utc"),
        "inference_finished_at_utc": run_meta.get("finished_at_utc"),
        "total_samples": len(predictions),
        "inference_success": success_count,
        "inference_failed": failed_count,
        "evaluated_samples": counts.get("scored_questions", 0),
        "scored_figures": counts.get("scored_figures", 0),
        "scored_complete_pairs": counts.get("scored_complete_pairs", 0),
        "incomplete_pairs_skipped": counts.get("incomplete_pairs_skipped", 0),
        "error_records": counts.get("error_records", 0),
        "metrics": {
            "aAcc": scores.get("aAcc"),
            "fAcc": scores.get("fAcc"),
            "qAcc": scores.get("qAcc"),
        },
        "by_category": metrics.get("by_category", {}),
        "by_subcategory": metrics.get("by_subcategory", {}),
        "by_difficulty": metrics.get("by_difficulty", {}),
        "by_visual_input": metrics.get("by_visual_input", {}),
        "protocol_version": metrics.get("protocol_version"),
        "evaluation_mode": metrics.get("evaluation_mode"),
        "evaluation_method": (
            "Chấm tự động Yes/No tất định (alternative_evaluation_rule_based) cho aAcc, fAcc, qAcc; "
            "không gọi GPT-4 API. Technical errors được loại khỏi mẫu số điểm."
        ),
    }


def _backfill_inference_deliverables(
    out_path: Path,
    model_key: str,
    preds: Sequence[Mapping[str, Any]],
    dataset_records: Mapping[str, Mapping[str, Any]],
    run_meta: Mapping[str, Any],
) -> None:
    """Backfill MMHal-style inference files for runs created before these files were added."""
    resp_path = out_path / f"{model_key}_hallusionbench_responses.json"
    if not resp_path.is_file():
        responses: list[dict[str, Any]] = []
        for rec in preds:
            sid = str(rec.get("sample_id"))
            meta_rec = rec.get("metadata") if isinstance(rec.get("metadata"), Mapping) else {}
            ds_meta = dataset_records.get(sid) or {}
            status = rec.get("status")
            responses.append(
                {
                    "sample_id": sid,
                    "category": meta_rec.get("category", ds_meta.get("category")),
                    "subcategory": meta_rec.get("subcategory", ds_meta.get("subcategory")),
                    "visual_input": rec.get("visual_input", ds_meta.get("visual_input")),
                    "visual_input_raw": meta_rec.get("visual_input_raw", ds_meta.get("visual_input_raw")),
                    "set_id": meta_rec.get("set_id", ds_meta.get("set_id")),
                    "figure_id": meta_rec.get("figure_id", ds_meta.get("figure_id")),
                    "question_id": meta_rec.get("question_id", ds_meta.get("question_id")),
                    "filename": ds_meta.get("filename"),
                    "question": ds_meta.get("question"),
                    "prompt": rec.get("prompt"),
                    "model_answer": rec.get("raw_prediction", rec.get("prediction")) or "",
                    "inference_status": "SUCCESS" if status == "COMPLETED" else str(status or "FAILED"),
                    "timing_ms": rec.get("timing_ms"),
                    "error": rec.get("error"),
                }
            )
        resp_path.write_text(json.dumps(responses, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    cfg_path = out_path / "inference_config.json"
    if not cfg_path.is_file():
        cfg_payload = {
            "experiment_id": run_meta.get("experiment_id"),
            "run_id": run_meta.get("run_id"),
            "benchmark": "hallusionbench",
            "stage": run_meta.get("stage"),
            "model_key": model_key,
            "model_id": run_meta.get("model_id"),
            "model_revision": run_meta.get("resolved_model_revision") or run_meta.get("configured_model_revision"),
            "processor_revision": run_meta.get("resolved_processor_revision") or run_meta.get("configured_processor_revision"),
            "dataset": run_meta.get("dataset_source"),
            "dataset_revision": run_meta.get("dataset_revision"),
            "total_samples": len(preds),
            "target_dtype": run_meta.get("dtype"),
            "attn_implementation": run_meta.get("attn_implementation"),
            "run_config_hash": run_meta.get("run_config_hash"),
            "timestamp": run_meta.get("started_at_utc"),
        }
        cfg_path.write_text(json.dumps(cfg_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    sum_path = out_path / "inference_summary.json"
    if not sum_path.is_file():
        env = run_meta.get("environment") if isinstance(run_meta.get("environment"), Mapping) else {}
        gpus = env.get("gpu") if isinstance(env.get("gpu"), list) else []
        device_label = ", ".join(str(g) for g in gpus) if gpus else str(run_meta.get("device_allocation") or "unknown")
        success_count = sum(1 for r in preds if r.get("status") == "COMPLETED")
        timings = [float(r["timing_ms"]) for r in preds if isinstance(r.get("timing_ms"), (int, float))]
        sum_payload = {
            "experiment_id": run_meta.get("experiment_id"),
            "run_id": run_meta.get("run_id"),
            "benchmark": "hallusionbench",
            "model_key": model_key,
            "model_id": run_meta.get("model_id"),
            "stage": run_meta.get("stage"),
            "status": run_meta.get("status"),
            "total_samples": len(preds),
            "success_count": success_count,
            "failed_count": len(preds) - success_count,
            "duration_seconds": round(sum(timings) / 1000.0, 2) if timings else None,
            "avg_latency_ms": round(sum(timings) / len(timings), 2) if timings else None,
            "device": device_label,
            "dtype": run_meta.get("dtype"),
            "run_config_hash": run_meta.get("run_config_hash"),
            "started_at_utc": run_meta.get("started_at_utc"),
            "finished_at_utc": run_meta.get("finished_at_utc"),
        }
        sum_path.write_text(json.dumps(sum_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def score_run_directory(
    run_dir: str | Path,
    dataset_records: Mapping[str, Mapping[str, Any]],
    *,
    protocol_version: str = "unverified_draft_rule_based",
    evaluation_mode: str = "alternative_evaluation_rule_based",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Score an existing run directory in place without mutating predictions.jsonl."""
    out_path = Path(run_dir)
    pred_file = out_path / "predictions.jsonl"
    if not pred_file.is_file():
        raise FileNotFoundError(f"predictions.jsonl not found in run directory: {out_path}")

    preds = read_jsonl(pred_file)
    metrics, scoring_errors = score_hallusionbench(
        preds,
        dataset_records,
        protocol_version=protocol_version,
        evaluation_mode=evaluation_mode,
    )

    meta_path = out_path / "run_metadata.json"
    run_meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    if run_meta:
        metrics["run_identity"] = {
            k: run_meta.get(k)
            for k in ("experiment_id", "run_id", "stage", "model_key", "model_id", "run_config_hash")
        }

    (out_path / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    errors_file = out_path / "errors.jsonl"
    existing_errors = read_jsonl(errors_file) if errors_file.is_file() else []
    seen_keys = {
        (str(e.get("sample_id")), str(e.get("error_type")), str(e.get("detail") or e.get("message") or ""))
        for e in existing_errors
    }
    merged_errors = list(existing_errors)
    for err in scoring_errors:
        key = (str(err.get("sample_id")), str(err.get("error_type")), str(err.get("detail") or ""))
        if key not in seen_keys:
            merged_errors.append(err)
            seen_keys.add(key)

    with errors_file.open("w", encoding="utf-8") as handle:
        for err in merged_errors:
            handle.write(json.dumps(err, ensure_ascii=False) + "\n")

    model_key = _infer_model_key(out_path, run_meta, preds)
    _backfill_inference_deliverables(out_path, model_key, preds, dataset_records, run_meta)

    detailed_eval = build_detailed_evaluation_records(preds, dataset_records)
    summary_payload = build_mmhal_style_summary(metrics, run_meta, preds, model_key)
    (out_path / f"{model_key}_hallusionbench_evaluation.json").write_text(
        json.dumps(detailed_eval, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out_path / f"{model_key}_hallusionbench_summary.json").write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    _refresh_manifest_checksums(out_path, model_key=model_key)
    return metrics, scoring_errors


def export_hallusionbench_artifacts(
    predictions: Iterable[Mapping[str, Any]],
    dataset_records: Mapping[str, Mapping[str, Any]],
    output_dir: str | Path,
    protocol_version: str = "unverified_draft_rule_based",
    evaluation_mode: str = "alternative_evaluation_rule_based",
) -> None:
    """Export resume-safe predictions.jsonl, metrics.json, and errors.jsonl."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    pred_file = out_path / "predictions.jsonl"
    existing_preds = read_jsonl(pred_file) if pred_file.exists() else []
    merged_preds = merge_result_records(existing_preds, predictions)

    metrics, error_records = score_hallusionbench(
        merged_preds,
        dataset_records,
        protocol_version=protocol_version,
        evaluation_mode=evaluation_mode,
    )

    with pred_file.open("w", encoding="utf-8") as f:
        for rec in merged_preds:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    (out_path / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (out_path / "errors.jsonl").open("w", encoding="utf-8") as f:
        for err in error_records:
            f.write(json.dumps(err, ensure_ascii=False) + "\n")
    _refresh_manifest_checksums(out_path)


# ---------------------------------------------------------------------------
# Error extraction, cross-model disagreement & before/after comparison
# ---------------------------------------------------------------------------

def extract_error_cases(
    predictions: Iterable[Mapping[str, Any]],
    dataset_records: Mapping[str, Mapping[str, Any]],
    model_id: str,
    run_id: str,
) -> list[dict[str, Any]]:
    """Extract all failed/incorrect/format-violation cases for visual evidence review."""
    cases: list[dict[str, Any]] = []
    for pred in predictions:
        sample_id = str(pred.get("sample_id"))
        status = str(pred.get("status", "UNKNOWN"))
        raw_resp = pred.get("raw_prediction", pred.get("prediction"))
        meta = dataset_records.get(sample_id)

        base = {
            "run_id": run_id,
            "model_id": model_id,
            "sample_id": sample_id,
            "figure_key": _resolve_figure_key(meta) if meta else None,
            "pair_key": _resolve_pair_key(meta) if meta else None,
            "category": meta.get("category") if meta else None,
            "subcategory": meta.get("subcategory") if meta else None,
            "difficulty": meta.get("difficulty") if meta else None,
            "visual_input": meta.get("visual_input") if meta else None,
            "visual_input_raw": meta.get("visual_input_raw") if meta else None,
            "image_path": meta.get("image_path") if meta else None,
            "question": meta.get("question") if meta else None,
            "gt_answer": _resolve_reference_answer(meta) if meta else None,
            "raw_response": raw_resp,
            "is_correct": False,
        }

        if status != "COMPLETED":
            cases.append(
                {
                    **base,
                    "parsed_prediction": None,
                    "preliminary_error_type": "technical_error",
                    "verified_error_type": "technical_error",
                    "evidence": pred.get("error") or pred.get("error_type") or f"Inference status is {status}",
                }
            )
            continue

        if meta is None:
            cases.append(
                {
                    **base,
                    "parsed_prediction": None,
                    "preliminary_error_type": "label_data_error",
                    "verified_error_type": "label_data_error",
                    "evidence": "sample_id not found in locked dataset metadata",
                }
            )
            continue

        parsed_pred = parse_yes_no(pred.get("prediction"))
        gt_answer = _resolve_reference_answer(meta)

        if parsed_pred is None:
            cases.append(
                {
                    **base,
                    "parsed_prediction": None,
                    "preliminary_error_type": "format_violation",
                    "verified_error_type": "format_violation",
                    "evidence": "Cannot extract deterministic Yes/No from model response",
                }
            )
        elif parsed_pred != gt_answer:
            cases.append(
                {
                    **base,
                    "parsed_prediction": parsed_pred,
                    "preliminary_error_type": "incorrect_prediction",
                    "verified_error_type": None,
                    "evidence": "Pending visual evidence review (do not assume hallucination by default)",
                }
            )
    return cases


def find_cross_model_disagreements(
    model_predictions: Mapping[str, Iterable[Mapping[str, Any]]],
    dataset_records: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    indexed: dict[str, dict[str, dict[str, Any]]] = {}
    for model_id, preds in model_predictions.items():
        for pred in preds:
            if pred.get("status") != "COMPLETED":
                continue
            sid = str(pred.get("sample_id"))
            meta = dataset_records.get(sid)
            if meta is None:
                continue
            parsed = parse_yes_no(pred.get("prediction"))
            gt = _resolve_reference_answer(meta)
            indexed.setdefault(sid, {})[model_id] = {
                "parsed_prediction": parsed,
                "is_correct": parsed == gt if parsed is not None else False,
                "raw_response": pred.get("raw_prediction", pred.get("prediction")),
            }

    disagreements: list[dict[str, Any]] = []
    for sid, outcomes in sorted(indexed.items()):
        if len(outcomes) < 2:
            continue
        if len({v["parsed_prediction"] for v in outcomes.values()}) > 1 or len({v["is_correct"] for v in outcomes.values()}) > 1:
            meta = dataset_records[sid]
            disagreements.append(
                {
                    "sample_id": sid,
                    "category": meta.get("category"),
                    "subcategory": meta.get("subcategory"),
                    "difficulty": meta.get("difficulty"),
                    "figure_key": _resolve_figure_key(meta),
                    "pair_key": _resolve_pair_key(meta),
                    "image_path": meta.get("image_path"),
                    "question": meta.get("question"),
                    "gt_answer": _resolve_reference_answer(meta),
                    "model_outcomes": outcomes,
                }
            )
    return disagreements


def compare_before_after(
    baseline_predictions: Iterable[Mapping[str, Any]],
    improved_predictions: Iterable[Mapping[str, Any]],
    dataset_records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    base_map = {str(p["sample_id"]): p for p in baseline_predictions if p.get("status") == "COMPLETED"}
    imp_map = {str(p["sample_id"]): p for p in improved_predictions if p.get("status") == "COMPLETED"}
    common_ids = sorted(set(base_map).intersection(imp_map).intersection(dataset_records))

    improved_ids: list[str] = []
    regressed_ids: list[str] = []
    unchanged_correct = unchanged_wrong = 0

    for sid in common_ids:
        gt = _resolve_reference_answer(dataset_records[sid])
        base_ok = parse_yes_no(base_map[sid].get("prediction")) == gt
        imp_ok = parse_yes_no(imp_map[sid].get("prediction")) == gt
        if not base_ok and imp_ok:
            improved_ids.append(sid)
        elif base_ok and not imp_ok:
            regressed_ids.append(sid)
        elif base_ok and imp_ok:
            unchanged_correct += 1
        else:
            unchanged_wrong += 1

    total = len(common_ids)
    base_acc = ((unchanged_correct + len(regressed_ids)) / total) if total > 0 else 0.0
    imp_acc = ((unchanged_correct + len(improved_ids)) / total) if total > 0 else 0.0
    return {
        "compared_completed_samples": total,
        "baseline_aAcc": round(base_acc, 6),
        "improved_aAcc": round(imp_acc, 6),
        "delta_aAcc": round(imp_acc - base_acc, 6),
        "improved_count": len(improved_ids),
        "regressed_count": len(regressed_ids),
        "unchanged_correct_count": unchanged_correct,
        "unchanged_wrong_count": unchanged_wrong,
        "improved_sample_ids": improved_ids,
        "regressed_sample_ids": regressed_ids,
    }


def summarize_error_cases(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "total_error_cases": len(cases),
        "by_preliminary_error_type": dict(sorted(Counter(str(c.get("preliminary_error_type")) for c in cases).items())),
        "by_verified_error_type": dict(sorted(Counter(str(c.get("verified_error_type") or "pending_review") for c in cases).items())),
        "by_category": dict(sorted(Counter(str(c.get("category") or "UNKNOWN") for c in cases).items())),
        "by_subcategory": dict(sorted(Counter(str(c.get("subcategory") or "UNKNOWN") for c in cases).items())),
        "by_difficulty": dict(sorted(Counter(str(c.get("difficulty") or "UNKNOWN") for c in cases).items())),
    }


def render_error_table_markdown(cases: Iterable[Mapping[str, Any]]) -> str:
    header = (
        "| `run_id` | `sample_id` / group | Model | Ảnh & câu hỏi | "
        "Ground truth | Raw response | Chấm đúng/sai | Loại lỗi & chứng cứ |\n"
        "|---|---|---|---|---|---|---|---|"
    )
    rows = [header]
    for c in cases:
        group_info = f"{c.get('sample_id')} ({c.get('pair_key') or c.get('figure_key') or 'N/A'})"
        img_q = f"Img: `{c.get('image_path')}` — Q: {c.get('question')}"
        raw_clean = str(c.get("raw_response", "")).replace("\n", " ").replace("|", "\\|")
        err_label = c.get("verified_error_type") or c.get("preliminary_error_type")
        evidence = str(c.get("evidence", "")).replace("\n", " ").replace("|", "\\|")
        rows.append(
            f"| `{c.get('run_id')}` | {group_info} | `{c.get('model_id')}` | "
            f"{img_q} | `{c.get('gt_answer')}` | {raw_clean} | Sai | `{err_label}`: {evidence} |"
        )
    return "\n".join(rows)


def export_error_report(cases: Sequence[Mapping[str, Any]], output_dir: str | Path) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cases_path = out / "error_cases.json"
    summary_path = out / "error_summary.json"
    md_path = out / "error_cases.md"
    cases_path.write_text(json.dumps(list(cases), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(summarize_error_cases(cases), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(render_error_table_markdown(cases) + "\n", encoding="utf-8")
    return {"error_cases_json": str(cases_path), "error_summary_json": str(summary_path), "error_cases_md": str(md_path)}


def _fmt_pct(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value * 100:.2f}%"
    return "N/A"


def render_model_results_readme(summary: Mapping[str, Any]) -> str:
    """Render per-model README.md (MMHal-style) with HallusionBench metrics and file inventory."""
    mk = str(summary.get("model_key", "model"))
    model_id = str(summary.get("model", mk))
    m = summary.get("metrics") or {}
    lines = [
        f"# Kết quả đánh giá `{model_id}` trên HallusionBench",
        "",
        "## 1. Thông tin thực nghiệm",
        "",
        "| Hạng mục | Giá trị |",
        "|---|---|",
        f"| **Model** | `{model_id}` (`{mk}`) |",
        f"| **Model revision** | `{summary.get('model_revision')}` |",
        f"| **Dataset** | `{summary.get('dataset')}` @ `{summary.get('dataset_revision')}` |",
        f"| **Stage / Status** | `{summary.get('stage')}` / `{summary.get('status')}` |",
        f"| **Thiết bị / Dtype** | `{summary.get('device')}` / `{summary.get('dtype')}` (`{summary.get('attn_implementation')}`) |",
        f"| **Run ID / Config Hash** | `{summary.get('run_id')}` / `{summary.get('run_config_hash')}` |",
        f"| **Mẫu hoàn thành / Tổng** | **{summary.get('inference_success')} / {summary.get('total_samples')}** (lỗi: {summary.get('inference_failed')}) |",
        f"| **Chế độ chấm điểm** | `{summary.get('evaluation_mode')}` (`{summary.get('protocol_version')}`) |",
        "",
        "## 2. Chỉ số tổng quan",
        "",
        "| Chỉ số | Giá trị | Số lượng chấm |",
        "|---|---:|---:|",
        f"| **aAcc** (Question Accuracy) | **{_fmt_pct(m.get('aAcc'))}** ({m.get('aAcc')}) | {summary.get('evaluated_samples')} câu |",
        f"| **fAcc** (Figure Accuracy) | **{_fmt_pct(m.get('fAcc'))}** ({m.get('fAcc')}) | {summary.get('scored_figures')} ảnh |",
        f"| **qAcc** (Question-Pair Accuracy) | **{_fmt_pct(m.get('qAcc'))}** ({m.get('qAcc')}) | {summary.get('scored_complete_pairs')} cặp |",
        "",
        "## 3. Phân tích theo nhóm (`VD/VS`, `Easy/Hard`, 8 `subcategory`)",
        "",
        "| Nhóm | Số câu | aAcc | fAcc | qAcc |",
        "|---|---:|---:|---:|---:|",
    ]
    for section_label, section_key in (
        ("Category", "by_category"),
        ("Difficulty", "by_difficulty"),
        ("Subcategory", "by_subcategory"),
    ):
        group_map = summary.get(section_key) or {}
        for key in sorted(group_map.keys()):
            val = group_map[key]
            q_cnt = (val.get("counts") or {}).get("scored_questions", 0)
            lines.append(
                f"| {section_label}: `{key}` | {q_cnt} | {_fmt_pct(val.get('aAcc'))} | "
                f"{_fmt_pct(val.get('fAcc'))} | {_fmt_pct(val.get('qAcc'))} |"
            )

    lines.extend(
        [
            "",
            "## 4. Danh mục tệp trong thư mục",
            "",
            "- **7 artifact chuẩn repo (`docs/result_schema.md`):** `predictions.jsonl`, `metrics.json`, `run_config.yaml`, `run_metadata.json`, `errors.jsonl`, `run_status.json`, `manifest.json`.",
            f"- **Gói dễ đọc tương đương nhóm MMHal:** `{mk}_hallusionbench_responses.json`, `{mk}_hallusionbench_evaluation.json`, `{mk}_hallusionbench_summary.json`, `inference_config.json`, `inference_summary.json`, `inference_log.txt`.",
            "- **Phân tích lỗi:** `analysis/error_cases.json`, `analysis/error_summary.json`, `analysis/error_cases.md`.",
            "",
        ]
    )
    return "\n".join(lines)


def render_master_results_readme(
    summaries_by_model: Mapping[str, Mapping[str, Any]],
    *,
    expected_models: Sequence[str] = ("smolvlm2", "qwen3_vl", "phi4_multimodal"),
    disagreement_count: int | None = None,
) -> str:
    """Render master HallusionBench results README.md comparing all available models honestly."""
    lines = [
        "# Kết quả thực nghiệm HallusionBench (Nhóm 2)",
        "",
        "Thư mục này lưu đồng thời **7 artifact chuẩn repo (`docs/result_schema.md`)** và **bộ tệp JSON/log dễ đọc (tương đương nhánh MMHal)** cho từng mô hình đã chạy inference thật (0-epoch, không tạo dữ liệu giả).",
        "",
        "## Bảng tổng hợp 3 mô hình",
        "",
        "| Model Key | Model ID | Stage | Status | Dtype | Hoàn thành | aAcc | fAcc | qAcc | Thư mục |",
        "|---|---|---|---|---|---:|---:|---:|---:|---|",
    ]
    model_order = list(expected_models) + [k for k in sorted(summaries_by_model.keys()) if k not in expected_models]
    for mk in model_order:
        s = summaries_by_model.get(mk)
        if s is None:
            lines.append(f"| `{mk}` | *(chưa có run)* | — | `PENDING` | — | 0 | — | — | — | — |")
            continue
        m = s.get("metrics") or {}
        lines.append(
            f"| `{mk}` | `{s.get('model')}` | `{s.get('stage')}` | `{s.get('status')}` | `{s.get('dtype')}` | "
            f"{s.get('inference_success')}/{s.get('total_samples')} | **{_fmt_pct(m.get('aAcc'))}** | "
            f"**{_fmt_pct(m.get('fAcc'))}** | **{_fmt_pct(m.get('qAcc'))}** | [`{mk}/`]({mk}/README.md) |"
        )

    if disagreement_count is not None:
        lines.extend(
            [
                "",
                f"- **Ca bất đồng giữa các mô hình:** **{disagreement_count}** ca — xem [`cross_model_disagreements.json`](cross_model_disagreements.json).",
            ]
        )

    lines.extend(
        [
            "",
            "## Lệnh kiểm chứng lại artifact",
            "",
            "```bash",
        ]
    )
    for mk in sorted(summaries_by_model.keys()):
        lines.append(f"python scripts/check_results.py benchmarks/hallusionbench/results/{mk}/predictions.jsonl")
        lines.append(f"python scripts/check_manifest.py benchmarks/hallusionbench/results/{mk}")
    lines.extend(["```", ""])
    return "\n".join(lines)


def export_deliverables(
    run_dirs_by_model: Mapping[str, str | Path],
    dataset_records: Mapping[str, Mapping[str, Any]],
    output_dir: str | Path,
    *,
    protocol_version: str = "unverified_draft_rule_based",
    evaluation_mode: str = "alternative_evaluation_rule_based",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Package real scored runs into per-model folders (7 official artifacts + MMHal-style files)."""
    if not run_dirs_by_model:
        raise ValueError("run_dirs_by_model cannot be empty")

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    summaries_by_model: dict[str, dict[str, Any]] = {}
    preds_by_model: dict[str, list[dict[str, Any]]] = {}
    exported_models: dict[str, str] = {}

    for model_key, src_dir_raw in sorted(run_dirs_by_model.items()):
        src_dir = Path(src_dir_raw)
        if not (src_dir / "predictions.jsonl").is_file():
            raise FileNotFoundError(f"Missing predictions.jsonl for {model_key}: {src_dir}")

        # Score source run in place so both 7 official artifacts and MMHal-style files are fresh.
        score_run_directory(
            src_dir,
            dataset_records,
            protocol_version=protocol_version,
            evaluation_mode=evaluation_mode,
        )

        dst_dir = out_root / model_key
        dst_status = dst_dir / "run_status.json"
        if dst_status.is_file() and not overwrite:
            existing_st = json.loads(dst_status.read_text(encoding="utf-8")).get("status")
            if existing_st == "COMPLETED":
                raise FileExistsError(
                    f"Refusing to overwrite COMPLETED exported deliverables at {dst_dir}; pass overwrite=True to refresh."
                )

        dst_dir.mkdir(parents=True, exist_ok=True)
        files_to_copy = [
            "predictions.jsonl",
            "metrics.json",
            "run_config.yaml",
            "run_metadata.json",
            "errors.jsonl",
            "run_status.json",
            "manifest.json",
            "inference_config.json",
            "inference_summary.json",
            "inference_log.txt",
            f"{model_key}_hallusionbench_responses.json",
            f"{model_key}_hallusionbench_evaluation.json",
            f"{model_key}_hallusionbench_summary.json",
        ]
        for fname in files_to_copy:
            src_file = src_dir / fname
            if src_file.is_file():
                shutil.copy2(src_file, dst_dir / fname)

        preds = read_jsonl(dst_dir / "predictions.jsonl")
        preds_by_model[model_key] = preds

        meta_path = dst_dir / "run_metadata.json"
        run_meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
        model_id = str(run_meta.get("model_id") or model_key)
        run_id = str(run_meta.get("run_id") or dst_dir.name)

        cases = extract_error_cases(preds, dataset_records, model_id=model_id, run_id=run_id)
        export_error_report(cases, dst_dir / "analysis")

        summary_file = dst_dir / f"{model_key}_hallusionbench_summary.json"
        summary_data = json.loads(summary_file.read_text(encoding="utf-8"))
        summaries_by_model[model_key] = summary_data
        (dst_dir / "README.md").write_text(render_model_results_readme(summary_data), encoding="utf-8")
        _refresh_manifest_checksums(dst_dir, model_key=model_key)
        exported_models[model_key] = str(dst_dir)

    disagreement_count: int | None = None
    disagreements_path: str | None = None
    if len(preds_by_model) >= 2:
        disagreements = find_cross_model_disagreements(preds_by_model, dataset_records)
        disagreement_count = len(disagreements)
        dis_file = out_root / "cross_model_disagreements.json"
        dis_file.write_text(json.dumps(disagreements, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        disagreements_path = str(dis_file)

    readme_path = out_root / "README.md"
    readme_path.write_text(
        render_master_results_readme(summaries_by_model, disagreement_count=disagreement_count),
        encoding="utf-8",
    )

    return {
        "output_dir": str(out_root),
        "exported_models": exported_models,
        "master_readme": str(readme_path),
        "disagreements_file": disagreements_path,
        "disagreement_count": disagreement_count,
    }


# ---------------------------------------------------------------------------
# Unified CLI for scoring + error analysis
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="HallusionBench Part C scoring & error analysis CLI")
    parser.add_argument(
        "--mode",
        choices=("score", "extract", "disagreements", "before-after", "export-deliverables"),
        required=True,
    )
    parser.add_argument("--dataset-json", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--protocol-version", default="unverified_draft_rule_based")
    parser.add_argument("--evaluation-mode", default="alternative_evaluation_rule_based")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-dirs", nargs="+", default=None, help="model=run_dir pairs for disagreements / export-deliverables")
    parser.add_argument("--baseline-run-dir", type=Path, default=None)
    parser.add_argument("--improved-run-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--overwrite-export", action="store_true", help="Allow refreshing already-exported deliverables")
    args = parser.parse_args(argv)

    dataset_index = HallusionBenchDataset(args.dataset_json, data_root=args.data_root).load_index(
        verify_images=False, verify_image_readable=False
    )

    if args.mode == "score":
        if args.run_dir is None:
            raise ValueError("--run-dir is required for --mode score")
        metrics, _ = score_run_directory(
            args.run_dir,
            dataset_index,
            protocol_version=args.protocol_version,
            evaluation_mode=args.evaluation_mode,
        )
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
        return 0

    if args.mode == "extract":
        if args.run_dir is None:
            raise ValueError("--run-dir is required for --mode extract")
        preds = read_jsonl(args.run_dir / "predictions.jsonl")
        meta_path = args.run_dir / "run_metadata.json"
        run_meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
        model_id = str(run_meta.get("model_id") or run_meta.get("model_key") or "unknown_model")
        run_id = str(run_meta.get("run_id") or args.run_dir.name)
        cases = extract_error_cases(preds, dataset_index, model_id=model_id, run_id=run_id)
        out_dir = args.output_dir or (args.run_dir / "analysis")
        paths = export_error_report(cases, out_dir)
        print(json.dumps({"summary": summarize_error_cases(cases), "files": paths}, indent=2, ensure_ascii=False))
        return 0

    if args.mode == "disagreements":
        if not args.run_dirs or len(args.run_dirs) < 2:
            raise ValueError("--run-dirs requires at least 2 model=dir entries")
        model_preds: dict[str, list[dict[str, Any]]] = {}
        for entry in args.run_dirs:
            mkey, rdir = entry.split("=", 1)
            model_preds[mkey.strip()] = read_jsonl(Path(rdir.strip()) / "predictions.jsonl")
        disagreements = find_cross_model_disagreements(model_preds, dataset_index)
        text = json.dumps(disagreements, indent=2, ensure_ascii=False)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + "\n", encoding="utf-8")
        print(text)
        return 0

    if args.mode == "before-after":
        if args.baseline_run_dir is None or args.improved_run_dir is None:
            raise ValueError("--baseline-run-dir and --improved-run-dir are required")
        diff = compare_before_after(
            read_jsonl(args.baseline_run_dir / "predictions.jsonl"),
            read_jsonl(args.improved_run_dir / "predictions.jsonl"),
            dataset_index,
        )
        text = json.dumps(diff, indent=2, ensure_ascii=False)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + "\n", encoding="utf-8")
        print(text)
        return 0

    if args.mode == "export-deliverables":
        if not args.run_dirs:
            raise ValueError("--run-dirs requires at least 1 model=dir entry")
        run_map: dict[str, Path] = {}
        for entry in args.run_dirs:
            mkey, rdir = entry.split("=", 1)
            run_map[mkey.strip()] = Path(rdir.strip())
        out_dir = args.output_dir or (REPO_ROOT / "benchmarks" / "hallusionbench" / "results")
        report = export_deliverables(
            run_map,
            dataset_index,
            output_dir=out_dir,
            protocol_version=args.protocol_version,
            evaluation_mode=args.evaluation_mode,
            overwrite=args.overwrite_export,
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
