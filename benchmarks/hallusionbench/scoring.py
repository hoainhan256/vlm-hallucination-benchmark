"""HallusionBench scoring, run-directory evaluation, and artifact export module (Member C)."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from vlm_benchmark.results import merge_result_records, read_jsonl, validate_predictions
except ModuleNotFoundError:
    # Dự phòng tạm thời khi nhánh benchmark/hallusionbench chưa đồng bộ thư mục vlm_benchmark từ main
    REQUIRED_PREDICTED_FIELDS = {"sample_id", "prediction", "status", "run_config_hash"}

    def read_jsonl(path: str | Path) -> list[dict]:
        records = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
        return records

    def validate_predictions(records: Iterable[Mapping]) -> list[str]:
        errors, seen, hashes = [], set(), set()
        for r in records:
            missing = REQUIRED_PREDICTED_FIELDS - set(r)
            if missing:
                errors.append(f"Missing fields: {sorted(missing)}")
            sid = r.get("sample_id")
            if sid is not None:
                if sid in seen:
                    errors.append(f"Duplicate sample_id: {sid}")
                seen.add(str(sid))
            if r.get("run_config_hash") is not None:
                hashes.add(str(r["run_config_hash"]))
        if len(hashes) > 1:
            errors.append("Predictions contain multiple incompatible run_config_hash values")
        return errors

    def merge_result_records(existing: Iterable[Mapping], incoming: Iterable[Mapping]) -> list[dict]:
        ex_map = {str(r.get("sample_id")): dict(r) for r in existing}
        in_map = {str(r.get("sample_id")): dict(r) for r in incoming}
        conflicts = [
            sid for sid in set(ex_map).intersection(in_map)
            if ex_map[sid].get("status") == "COMPLETED" and ex_map[sid] != in_map[sid]
        ]
        if conflicts:
            raise ValueError(f"Refusing to overwrite completed samples: {sorted(conflicts)}")
        merged = dict(ex_map)
        for sid, rec in in_map.items():
            if sid not in merged or rec.get("status") == "COMPLETED":
                merged[sid] = rec
        return list(merged.values())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_reference_answer(meta: Mapping[str, Any]) -> str:
    """Resolve ground-truth answer to 'yes' or 'no' from Member A schema or raw HallusionBench.json."""
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


def parse_yes_no(raw_text: Any) -> str | None:
    """Chuẩn hóa câu trả lời của VLM về 'yes', 'no' hoặc None (format violation).

    Quy tắc rule-based; cần đối chiếu với evaluation.py gốc của HallusionBench
    trước khi chạy official.
    """
    if raw_text is None:
        return None
    text = str(raw_text).strip().lower()
    if not text:
        return None

    # 1. Bắt các mẫu trả lời nước đôi trực tiếp (ví dụ: "Yes and no...", "Yes/No")
    if re.search(r"\b(yes\s*(and|or|/)\s*no|no\s*(and|or|/)\s*yes)\b", text):
        return None

    # 2. Khớp trực tiếp ở đầu câu (ví dụ: "Yes, the image shows..." hoặc "Yes, there is no cat.")
    match = re.match(r"^(yes|no)\b", text)
    if match:
        return match.group(1)

    # 3. Nếu giải thích dài dòng mới chốt ở cuối, chỉ nhận khi không chứa cả 2 từ trái ngược
    has_yes = bool(re.search(r"\byes\b", text))
    has_no = bool(re.search(r"\bno\b", text))
    if has_yes and not has_no:
        return "yes"
    if has_no and not has_yes:
        return "no"

    return None


def _resolve_figure_key(meta: Mapping[str, Any]) -> str | None:
    """Lấy khóa gộp cấp hình ảnh (fAcc) từ schema chuẩn hoặc ghép từ trường gốc.

    Mẫu text-only (visual_input=False / visual_input_raw='0') không có ảnh nên trả về None
    khi không được gán figure_key tường minh, tránh làm nhiễu mẫu số fAcc.
    """
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
    """Lấy khóa gộp cấp cặp câu hỏi (qAcc) từ schema chuẩn hoặc ghép từ trường gốc."""
    if meta.get("pair_key") is not None:
        return str(meta["pair_key"])
    if meta.get("question_pair_key") is not None:
        return str(meta["question_pair_key"])
    if meta.get("set_id") is not None and meta.get("question_id") is not None:
        return f"{meta.get('category', '')}:{meta.get('subcategory', '')}:{meta['set_id']}:{meta['question_id']}"
    return None


def _compute_subset_metrics(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Tính aAcc, fAcc, qAcc và ghi rõ mẫu số cho một tập mẫu hợp lệ."""
    total_q = len(items)
    correct_q = sum(1 for r in items if r["is_correct"])
    a_acc = (correct_q / total_q) if total_q > 0 else 0.0

    # fAcc: Tất cả câu hỏi thuộc cùng 1 figure phải trả lời đúng
    figure_groups: dict[str, list[bool]] = defaultdict(list)
    for r in items:
        if r["figure_key"] is not None:
            figure_groups[r["figure_key"]].append(r["is_correct"])
    total_f = len(figure_groups)
    correct_f = sum(1 for outcomes in figure_groups.values() if all(outcomes))
    f_acc = (correct_f / total_f) if total_f > 0 else 0.0

    # qAcc: Cả cặp câu hỏi kiểm soát (>= 2 biến thể) phải trả lời đúng
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
    """Tính metric tổng và phân nhóm (VD/VS, Easy/Hard), đồng thời tách lỗi kỹ thuật/định dạng."""
    pred_list = [dict(p) for p in predictions]
    schema_errors = validate_predictions(pred_list)
    if schema_errors:
        raise ValueError(f"Invalid predictions schema: {schema_errors}")

    error_records: list[dict[str, Any]] = []
    valid_results: list[dict[str, Any]] = []

    for pred in pred_list:
        sample_id = str(pred["sample_id"])
        status = pred.get("status")

        # 1. Tách lỗi kỹ thuật từ bước inference (tuyệt đối không tính vào mẫu số điểm)
        if status != "COMPLETED":
            error_records.append({
                "sample_id": sample_id,
                "error_type": "technical_error",
                "status": status,
                "detail": pred.get("error") or pred.get("error_type") or "Non-completed inference status",
            })
            continue

        # 2. Kiểm tra khớp metadata từ bộ dữ liệu đã khóa của Thành viên A
        meta = dataset_records.get(sample_id)
        if meta is None:
            error_records.append({
                "sample_id": sample_id,
                "error_type": "label_data_error",
                "detail": "sample_id not found in locked dataset metadata",
            })
            continue

        # 3. Chuẩn hóa câu trả lời Yes/No
        parsed_pred = parse_yes_no(pred.get("prediction"))
        gt_answer = _resolve_reference_answer(meta)

        if parsed_pred is None:
            error_records.append({
                "sample_id": sample_id,
                "error_type": "format_violation",
                "raw_prediction": pred.get("raw_prediction", pred.get("prediction")),
                "detail": "Could not extract deterministic Yes/No answer",
            })
            is_correct = False
        else:
            is_correct = parsed_pred == gt_answer

        category = str(meta["category"]).strip().upper() if meta.get("category") else None
        difficulty = str(meta["difficulty"]).strip().lower() if meta.get("difficulty") else None
        visual_raw = (
            str(meta["visual_input_raw"]).strip()
            if meta.get("visual_input_raw") is not None
            else None
        )

        valid_results.append({
            "sample_id": sample_id,
            "is_correct": is_correct,
            "parsed_pred": parsed_pred,
            "gt_answer": gt_answer,
            "category": category,
            "subcategory": meta.get("subcategory"),
            "difficulty": difficulty,
            "visual_input_raw": visual_raw,
            "figure_key": _resolve_figure_key(meta),
            "pair_key": _resolve_pair_key(meta),
        })

    overall = _compute_subset_metrics(valid_results)

    # Phân tách theo VD (Visual Dependent) và VS (Visual Supplement)
    by_category: dict[str, Any] = {}
    for cat in sorted({r["category"] for r in valid_results if r["category"]}):
        cat_items = [r for r in valid_results if r["category"] == cat]
        by_category[cat] = _compute_subset_metrics(cat_items)

    # Phân tách theo Easy / Hard (nếu dataset có trường difficulty)
    by_difficulty: dict[str, Any] = {}
    for diff in sorted({r["difficulty"] for r in valid_results if r["difficulty"]}):
        diff_items = [r for r in valid_results if r["difficulty"] == diff]
        by_difficulty[diff] = _compute_subset_metrics(diff_items)

    # Phân tách theo loại đầu vào thị giác (0=text-only, 1=ảnh gốc, 2=ảnh chỉnh sửa)
    by_visual_input: dict[str, Any] = {}
    for vis_code in sorted({r["visual_input_raw"] for r in valid_results if r["visual_input_raw"] is not None}):
        vis_items = [r for r in valid_results if r["visual_input_raw"] == vis_code]
        by_visual_input[vis_code] = _compute_subset_metrics(vis_items)

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
        "by_category": by_category,
        "by_difficulty": by_difficulty,
        "by_visual_input": by_visual_input,
    }
    return metrics, error_records


def _refresh_manifest_checksums(run_dir: Path) -> None:
    """Update SHA-256 checksums inside manifest.json after scoring updates metrics.json/errors.jsonl."""
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files")
    if not isinstance(files, dict):
        files = {}
    for name in (
        "predictions.jsonl",
        "metrics.json",
        "run_config.yaml",
        "run_metadata.json",
        "errors.jsonl",
        "run_status.json",
    ):
        file_path = run_dir / name
        if file_path.is_file():
            files[name] = {
                "size_bytes": file_path.stat().st_size,
                "sha256": _sha256_file(file_path),
            }
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
    """Score an existing Member B run directory without mutating predictions.jsonl.

    Updates metrics.json, merges scoring errors into errors.jsonl (preserving runtime
    inference errors), and refreshes manifest.json checksums so scripts/check_manifest.py passes.
    """
    out_path = Path(run_dir)
    pred_file = out_path / "predictions.jsonl"
    if not pred_file.is_file():
        raise FileNotFoundError(f"predictions.jsonl not found in run directory: {out_path}")

    predictions = read_jsonl(pred_file)
    metrics, scoring_errors = score_hallusionbench(
        predictions,
        dataset_records,
        protocol_version=protocol_version,
        evaluation_mode=evaluation_mode,
    )

    meta_path = out_path / "run_metadata.json"
    if meta_path.is_file():
        run_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        metrics["run_identity"] = {
            "experiment_id": run_meta.get("experiment_id"),
            "run_id": run_meta.get("run_id"),
            "stage": run_meta.get("stage"),
            "model_key": run_meta.get("model_key"),
            "model_id": run_meta.get("model_id"),
            "run_config_hash": run_meta.get("run_config_hash"),
        }

    (out_path / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # Preserve existing runtime errors from Member B and append non-duplicate scoring errors.
    errors_file = out_path / "errors.jsonl"
    existing_errors = read_jsonl(errors_file) if errors_file.is_file() else []
    existing_keys = {
        (str(e.get("sample_id")), str(e.get("error_type")), str(e.get("detail") or e.get("message") or ""))
        for e in existing_errors
    }
    merged_errors = list(existing_errors)
    for err in scoring_errors:
        key = (str(err.get("sample_id")), str(err.get("error_type")), str(err.get("detail") or ""))
        if key not in existing_keys:
            merged_errors.append(err)
            existing_keys.add(key)

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
    """Xuất predictions.jsonl (hỗ trợ resume an toàn), metrics.json và errors.jsonl."""
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

    with (out_path / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    with (out_path / "errors.jsonl").open("w", encoding="utf-8") as f:
        for err in error_records:
            f.write(json.dumps(err, ensure_ascii=False) + "\n")

    _refresh_manifest_checksums(out_path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="HallusionBench Member C scoring CLI")
    parser.add_argument("--run-dir", type=Path, required=True, help="Member B run directory containing predictions.jsonl")
    parser.add_argument("--dataset-json", type=Path, required=True, help="Path to locked HallusionBench.json")
    parser.add_argument("--data-root", type=Path, default=None, help="Dataset root directory")
    parser.add_argument("--protocol-version", default="unverified_draft_rule_based")
    parser.add_argument("--evaluation-mode", default="alternative_evaluation_rule_based")
    args = parser.parse_args(argv)

    from benchmarks.hallusionbench.dataset import HallusionBenchDataset

    dataset = HallusionBenchDataset(args.dataset_json, data_root=args.data_root)
    dataset_index = dataset.load_index(verify_images=False, verify_image_readable=False)
    metrics, _ = score_run_directory(
        args.run_dir,
        dataset_index,
        protocol_version=args.protocol_version,
        evaluation_mode=args.evaluation_mode,
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())