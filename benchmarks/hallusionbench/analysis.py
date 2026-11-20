"""HallusionBench error extraction, cross-model disagreement, and before/after analysis."""

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from benchmarks.hallusionbench.scoring import (
    _resolve_figure_key,
    _resolve_pair_key,
    parse_yes_no,
)

try:
    from vlm_benchmark.results import read_jsonl
except ModuleNotFoundError:
    def read_jsonl(path: str | Path) -> list[dict]:
        records = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
        return records


# 6 nhãn phân loại lỗi chuẩn theo docs/evaluation_policy.md
VALID_ERROR_TYPES = {
    "incorrect_prediction",
    "hallucination",
    "format_violation",
    "technical_error",
    "label_data_error",
    "undetermined",
}


def extract_error_cases(
    predictions: Iterable[Mapping[str, Any]],
    dataset_records: Mapping[str, Mapping[str, Any]],
    model_id: str,
    run_id: str,
) -> list[dict[str, Any]]:
    """Trích xuất danh sách các ca lỗi từ một run thật kèm ngữ cảnh ảnh/câu hỏi.

    Lưu ý: Các câu trả lời sai nhãn được gắn nhãn sơ bộ là 'incorrect_prediction'
    và yêu cầu đối chiếu bằng chứng thị giác trước khi kết luận 'hallucination'.
    """
    cases: list[dict[str, Any]] = []

    for pred in predictions:
        sample_id = str(pred.get("sample_id"))
        status = str(pred.get("status", "UNKNOWN"))
        raw_resp = pred.get("raw_prediction", pred.get("prediction"))
        meta = dataset_records.get(sample_id)

        # 1. Lỗi kỹ thuật từ quá trình inference
        if status != "COMPLETED":
            cases.append({
                "run_id": run_id,
                "model_id": model_id,
                "sample_id": sample_id,
                "figure_key": _resolve_figure_key(meta) if meta else None,
                "pair_key": _resolve_pair_key(meta) if meta else None,
                "category": meta.get("category") if meta else None,
                "visual_input": meta.get("visual_input") if meta else None,
                "image_path": meta.get("image_path") if meta else None,
                "question": meta.get("question") if meta else None,
                "gt_answer": meta.get("reference_answer") if meta else None,
                "raw_response": raw_resp,
                "parsed_prediction": None,
                "is_correct": False,
                "preliminary_error_type": "technical_error",
                "verified_error_type": "technical_error",
                "evidence": pred.get("error", f"Inference status is {status}"),
            })
            continue

        # 2. Lỗi thiếu hoặc lệch metadata dữ liệu
        if meta is None:
            cases.append({
                "run_id": run_id,
                "model_id": model_id,
                "sample_id": sample_id,
                "figure_key": None,
                "pair_key": None,
                "category": None,
                "visual_input": None,
                "image_path": None,
                "question": None,
                "gt_answer": None,
                "raw_response": raw_resp,
                "parsed_prediction": None,
                "is_correct": False,
                "preliminary_error_type": "label_data_error",
                "verified_error_type": "label_data_error",
                "evidence": "sample_id not found in locked dataset metadata",
            })
            continue

        parsed_pred = parse_yes_no(pred.get("prediction"))
        gt_answer = str(meta["reference_answer"]).strip().lower()

        # 3. Vi phạm định dạng (không chốt được Yes/No)
        if parsed_pred is None:
            cases.append({
                "run_id": run_id,
                "model_id": model_id,
                "sample_id": sample_id,
                "figure_key": _resolve_figure_key(meta),
                "pair_key": _resolve_pair_key(meta),
                "category": meta.get("category"),
                "visual_input": meta.get("visual_input"),
                "image_path": meta.get("image_path"),
                "question": meta.get("question"),
                "gt_answer": gt_answer,
                "raw_response": raw_resp,
                "parsed_prediction": None,
                "is_correct": False,
                "preliminary_error_type": "format_violation",
                "verified_error_type": "format_violation",
                "evidence": "Cannot extract deterministic Yes/No from model response",
            })
            continue

        # 4. Sai dự đoán (chờ đối chiếu ảnh để phân loại hallucination vs. undetermined)
        if parsed_pred != gt_answer:
            cases.append({
                "run_id": run_id,
                "model_id": model_id,
                "sample_id": sample_id,
                "figure_key": _resolve_figure_key(meta),
                "pair_key": _resolve_pair_key(meta),
                "category": meta.get("category"),
                "visual_input": meta.get("visual_input"),
                "image_path": meta.get("image_path"),
                "question": meta.get("question"),
                "gt_answer": gt_answer,
                "raw_response": raw_resp,
                "parsed_prediction": parsed_pred,
                "is_correct": False,
                "preliminary_error_type": "incorrect_prediction",
                "verified_error_type": None,
                "evidence": "Pending visual evidence review (do not assume hallucination by default)",
            })

    return cases


def find_cross_model_disagreements(
    model_predictions: Mapping[str, Iterable[Mapping[str, Any]]],
    dataset_records: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Tìm các mẫu mà các mô hình (SmolVLM2, Qwen3-VL, Phi-4) cho kết quả khác nhau."""
    indexed_by_sample: dict[str, dict[str, dict[str, Any]]] = {}

    for model_id, preds in model_predictions.items():
        for pred in preds:
            if pred.get("status") != "COMPLETED":
                continue
            sample_id = str(pred.get("sample_id"))
            meta = dataset_records.get(sample_id)
            if meta is None:
                continue

            parsed = parse_yes_no(pred.get("prediction"))
            gt = str(meta["reference_answer"]).strip().lower()
            indexed_by_sample.setdefault(sample_id, {})[model_id] = {
                "parsed_prediction": parsed,
                "is_correct": parsed == gt if parsed is not None else False,
                "raw_response": pred.get("raw_prediction", pred.get("prediction")),
            }

    disagreements: list[dict[str, Any]] = []
    for sample_id, outcomes in sorted(indexed_by_sample.items()):
        if len(outcomes) < 2:
            continue

        parsed_set = {info["parsed_prediction"] for info in outcomes.values()}
        correct_set = {info["is_correct"] for info in outcomes.values()}

        if len(parsed_set) > 1 or len(correct_set) > 1:
            meta = dataset_records[sample_id]
            disagreements.append({
                "sample_id": sample_id,
                "category": meta.get("category"),
                "figure_key": _resolve_figure_key(meta),
                "pair_key": _resolve_pair_key(meta),
                "image_path": meta.get("image_path"),
                "question": meta.get("question"),
                "gt_answer": str(meta["reference_answer"]).strip().lower(),
                "model_outcomes": outcomes,
            })

    return disagreements


def compare_before_after(
    baseline_predictions: Iterable[Mapping[str, Any]],
    improved_predictions: Iterable[Mapping[str, Any]],
    dataset_records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Đối chứng Before vs. After (Gate G6) trên cùng tập mẫu hợp lệ."""
    base_map = {
        str(p["sample_id"]): p
        for p in baseline_predictions
        if p.get("status") == "COMPLETED"
    }
    imp_map = {
        str(p["sample_id"]): p
        for p in improved_predictions
        if p.get("status") == "COMPLETED"
    }

    common_ids = sorted(set(base_map).intersection(imp_map).intersection(dataset_records))
    improved_ids: list[str] = []
    regressed_ids: list[str] = []
    unchanged_correct = 0
    unchanged_wrong = 0

    for sid in common_ids:
        gt = str(dataset_records[sid]["reference_answer"]).strip().lower()
        base_parsed = parse_yes_no(base_map[sid].get("prediction"))
        imp_parsed = parse_yes_no(imp_map[sid].get("prediction"))

        base_ok = base_parsed == gt if base_parsed is not None else False
        imp_ok = imp_parsed == gt if imp_parsed is not None else False

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


def render_error_table_markdown(cases: Iterable[Mapping[str, Any]]) -> str:
    """Xuất bảng Markdown theo đúng mẫu Mục 5.4 của tài liệu phân công nhóm."""
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