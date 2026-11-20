"""HallusionBench scoring and artifact export module."""

from collections import defaultdict
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

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
    """Lấy khóa gộp cấp hình ảnh (fAcc) từ schema chuẩn hoặc ghép từ trường gốc."""
    if meta.get("figure_key") is not None:
        return str(meta["figure_key"])
    if meta.get("set_id") is not None and meta.get("figure_id") is not None:
        return f"{meta.get('category', '')}:{meta.get('subcategory', '')}:{meta['set_id']}:{meta['figure_id']}"
    return None


def _resolve_pair_key(meta: Mapping[str, Any]) -> str | None:
    """Lấy khóa gộp cấp cặp câu hỏi (qAcc) từ schema chuẩn hoặc ghép từ trường gốc."""
    if meta.get("pair_key") is not None:
        return str(meta["pair_key"])
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
                "detail": pred.get("error", "Non-completed inference status"),
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
        gt_answer = str(meta["reference_answer"]).strip().lower()

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

        valid_results.append({
            "sample_id": sample_id,
            "is_correct": is_correct,
            "parsed_pred": parsed_pred,
            "gt_answer": gt_answer,
            "category": category,
            "subcategory": meta.get("subcategory"),
            "difficulty": difficulty,
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
    }
    return metrics, error_records


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