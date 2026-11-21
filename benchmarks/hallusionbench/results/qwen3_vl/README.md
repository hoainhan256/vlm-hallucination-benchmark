# Kết quả đánh giá `Qwen/Qwen3-VL-4B-Instruct` trên HallusionBench

## 1. Thông tin thực nghiệm

| Hạng mục | Giá trị |
|---|---|
| **Model** | `Qwen/Qwen3-VL-4B-Instruct` (`qwen3_vl`) |
| **Model revision** | `ebb281ec70b05090aa6165b016eac8ec08e71b17` |
| **Dataset** | `rayguan/HallusionBench` @ `2067dfdce5abc7efdbb0927f4721ae282f0b6c51` |
| **Stage / Status** | `pilot` / `COMPLETED` |
| **Thiết bị / Dtype** | `Tesla T4` / `float16` (`sdpa`) |
| **Run ID / Config Hash** | `pilot_qwen3_vl_float16` / `f26ce735c59ce2a3b876c631638c7586946883ef2c07adb5b3533ce9dc501dae` |
| **Mẫu hoàn thành / Tổng** | **34 / 34** (lỗi: 0) |
| **Chế độ chấm điểm** | `alternative_evaluation_rule_based` (`unverified_draft_rule_based`) |

## 2. Chỉ số tổng quan

| Chỉ số | Giá trị | Số lượng chấm |
|---|---:|---:|
| **aAcc** (Question Accuracy) | **67.65%** (0.676471) | 34 câu |
| **fAcc** (Figure Accuracy) | **60.00%** (0.6) | 15 ảnh |
| **qAcc** (Question-Pair Accuracy) | **21.43%** (0.214286) | 14 cặp |

## 3. Phân tích theo nhóm (`VD/VS`, `Easy/Hard`, 8 `subcategory`)

| Nhóm | Số câu | aAcc | fAcc | qAcc |
|---|---:|---:|---:|---:|
| Category: `VD` | 12 | 50.00% | 37.50% | 0.00% |
| Category: `VS` | 22 | 77.27% | 85.71% | 37.50% |
| Difficulty: `easy` | 14 | 64.29% | 75.00% | 0.00% |
| Difficulty: `hard` | 20 | 70.00% | 54.55% | 66.67% |
| Subcategory: `chart` | 4 | 50.00% | 100.00% | 0.00% |
| Subcategory: `figure` | 2 | 50.00% | 50.00% | 0.00% |
| Subcategory: `illusion` | 2 | 50.00% | 50.00% | 0.00% |
| Subcategory: `map` | 6 | 66.67% | 50.00% | 0.00% |
| Subcategory: `math` | 6 | 50.00% | 0.00% | 0.00% |
| Subcategory: `ocr` | 3 | 100.00% | 100.00% | 100.00% |
| Subcategory: `table` | 9 | 88.89% | 100.00% | 66.67% |
| Subcategory: `video` | 2 | 50.00% | 50.00% | 0.00% |

## 4. Danh mục tệp trong thư mục

- **7 artifact chuẩn repo (`docs/result_schema.md`):** `predictions.jsonl`, `metrics.json`, `run_config.yaml`, `run_metadata.json`, `errors.jsonl`, `run_status.json`, `manifest.json`.
- **Gói dễ đọc tương đương nhóm MMHal:** `qwen3_vl_hallusionbench_responses.json`, `qwen3_vl_hallusionbench_evaluation.json`, `qwen3_vl_hallusionbench_summary.json`, `inference_config.json`, `inference_summary.json`, `inference_log.txt`.
- **Phân tích lỗi:** `analysis/error_cases.json`, `analysis/error_summary.json`, `analysis/error_cases.md`.
