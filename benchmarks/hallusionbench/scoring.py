"""HallusionBench scoring and artifact export module."""

from collections import defaultdict
import hashlib
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

    Cần đối chiếu và cập nhật thêm các rule biên từ evaluation.py gốc của HallusionBench.
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


def score_hallusionbench(
    predictions: Iterable[Mapping[str, Any]],
    dataset_records: Mapping[str, Mapping[str, Any]],
    protocol_version: str = "unverified_draft",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Tính aAcc, fAcc, qAcc và tách danh sách lỗi kỹ thuật / định dạng."""
    pred_list = [dict(p) for p in predictions]
    schema_errors = validate_predictions(pred_list)
    if schema_errors:
        raise ValueError(f"Invalid predictions schema: {schema_errors}")

    error_records: list[dict[str, Any]] = []
    valid_results: dict[str, dict[str, Any]] = {}

    for pred in pred_list:
        sample_id = str(pred["sample_id"])
        status = pred.get("status")

        # 1. Tách lỗi kỹ thuật từ bước inference (không tính vào mẫu số điểm)
        if status != "COMPLETED":
            error_records.append({
                "sample_id": sample_id,
                "error_type": "technical_error",
                "status": status,
                "detail": pred.get("error", "Non-completed inference status"),
            })
            continue

        # 2. Kiểm tra khớp metadata từ bộ dữ liệu của A
        meta = dataset_records.get(sample_id)
        if meta is None:
            error_records.append({
                "sample_id": sample_id,
                "error_type": "label_data_error",
                "detail": "sample_id not found in locked dataset metadata",
            })
            continue

        # 3. Parse câu trả lời Yes/No
        parsed_pred = parse_yes_no(pred.get("prediction"))
        gt_answer = str(meta["reference_answer"]).strip().lower()

        if parsed_pred is None:
            error_records.append({
                "sample_id": sample_id,
                "error_type": "format_violation",
                "raw_prediction": pred.get("raw_prediction", pred.get("prediction")),
                "detail": "Could not extract deterministic Yes/No answer",
            })
            # Lưu ý: Cần đối chiếu evaluation.py gốc xem format violation bị tính 0 điểm
            # hay loại khỏi tập hợp lệ. Mặc định tạm tính is_correct = False trong valid_results:
            is_correct = False
        else:
            is_correct = parsed_pred == gt_answer

        valid_results[sample_id] = {
            "is_correct": is_correct,
            "parsed_pred": parsed_pred,
            "gt_answer": gt_answer,
            "category": meta.get("category"),         # VD hoặc VS
            "subcategory": meta.get("subcategory"),
            "difficulty": meta.get("difficulty"),     # Easy hoặc Hard (nếu có)
            "figure_key": meta.get("figure_key"),     # Khóa gộp cấp hình ảnh (fAcc)
            "pair_key": meta.get("pair_key"),         # Khóa gộp cấp cặp câu hỏi (qAcc)
        }

    # Tính aAcc (Question-level accuracy)
    total_q = len(valid_results)
    correct_q = sum(1 for r in valid_results.values() if r["is_correct"])
    a_acc = (correct_q / total_q) if total_q > 0 else 0.0

    # Tính fAcc (Figure-level accuracy: tất cả câu hỏi thuộc cùng 1 figure phải đúng)
    figure_groups: dict[str, list[bool]] = defaultdict(list)
    for r in valid_results.values():
        if r["figure_key"] is not None:
            figure_groups[str(r["figure_key"])].append(r["is_correct"])
    total_f = len(figure_groups)
    correct_f = sum(1 for outcomes in figure_groups.values() if all(outcomes))
    f_acc = (correct_f / total_f) if total_f > 0 else 0.0

    # Tính qAcc (Question-pair accuracy: cả cặp kiểm soát phải đúng)
    # Lưu ý: Cần kiểm tra thêm điều kiện cặp đủ 2 câu (len == 2) khi chạy pilot/incomplete run
    pair_groups: dict[str, list[bool]] = defaultdict(list)
    for r in valid_results.values():
        if r["pair_key"] is not None:
            pair_groups[str(r["pair_key"])].append(r["is_correct"])
    complete_pairs = {k: v for k, v in pair_groups.items() if len(v) >= 2}
    total_p = len(complete_pairs)
    correct_p = sum(1 for outcomes in complete_pairs.values() if all(outcomes))
    q_acc = (correct_p / total_p) if total_p > 0 else 0.0

    metrics = {
        "benchmark": "hallusionbench",
        "protocol_version": protocol_version,
        "counts": {
            "total_predictions_received": len(pred_list),
            "scored_questions": total_q,
            "scored_figures": total_f,
            "scored_complete_pairs": total_p,
            "incomplete_pairs_skipped": len(pair_groups) - total_p,
            "error_records": len(error_records),
        },
        "metrics": {
            "aAcc": round(a_acc, 6),
            "fAcc": round(f_acc, 6),
            "qAcc": round(q_acc, 6),
        },
    }
    return metrics, error_records


def export_hallusionbench_artifacts(
    predictions: Iterable[Mapping[str, Any]],
    dataset_records: Mapping[str, Mapping[str, Any]],
    output_dir: str | Path,
    protocol_version: str = "unverified_draft",
) -> None:
    """Xuất predictions.jsonl (hỗ trợ resume), metrics.json và errors.jsonl."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    pred_file = out_path / "predictions.jsonl"
    existing_preds = read_jsonl(pred_file) if pred_file.exists() else []
    merged_preds = merge_result_records(existing_preds, predictions)

    metrics, error_records = score_hallusionbench(
        merged_preds, dataset_records, protocol_version=protocol_version
    )

    with pred_file.open("w", encoding="utf-8") as f:
        for rec in merged_preds:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with (out_path / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    with (out_path / "errors.jsonl").open("w", encoding="utf-8") as f:
        for err in error_records:
            f.write(json.dumps(err, ensure_ascii=False) + "\n")