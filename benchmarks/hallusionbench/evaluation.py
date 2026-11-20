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


def _refresh_manifest_checksums(run_dir: Path) -> None:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files") if isinstance(manifest.get("files"), dict) else {}
    for name in ("predictions.jsonl", "metrics.json", "run_config.yaml", "run_metadata.json", "errors.jsonl", "run_status.json"):
        file_path = run_dir / name
        if file_path.is_file():
            files[name] = {"size_bytes": file_path.stat().st_size, "sha256": sha256_file(file_path)}
    manifest["files"] = files
    manifest["scored_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


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

    metrics, scoring_errors = score_hallusionbench(
        read_jsonl(pred_file),
        dataset_records,
        protocol_version=protocol_version,
        evaluation_mode=evaluation_mode,
    )

    meta_path = out_path / "run_metadata.json"
    if meta_path.is_file():
        run_meta = json.loads(meta_path.read_text(encoding="utf-8"))
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

    _refresh_manifest_checksums(out_path)
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


# ---------------------------------------------------------------------------
# Unified CLI for scoring + error analysis
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="HallusionBench Part C scoring & error analysis CLI")
    parser.add_argument("--mode", choices=("score", "extract", "disagreements", "before-after"), required=True)
    parser.add_argument("--dataset-json", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--protocol-version", default="unverified_draft_rule_based")
    parser.add_argument("--evaluation-mode", default="alternative_evaluation_rule_based")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-dirs", nargs="+", default=None, help="model=run_dir pairs for --mode disagreements")
    parser.add_argument("--baseline-run-dir", type=Path, default=None)
    parser.add_argument("--improved-run-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
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

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
