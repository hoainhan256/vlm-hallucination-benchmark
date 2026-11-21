# Kết quả đánh giá `microsoft/Phi-4-multimodal-instruct` trên HallusionBench

## 1. Thông tin thực nghiệm

| Hạng mục | Giá trị |
|---|---|
| **Model** | `microsoft/Phi-4-multimodal-instruct` (`phi4_multimodal`) |
| **Model revision** | `93f923e1a7727d1c4f446756212d9d3e8fcc5d81` |
| **Dataset** | `rayguan/HallusionBench` @ `2067dfdce5abc7efdbb0927f4721ae282f0b6c51` |
| **Stage / Status** | `pilot` / `COMPLETED` |
| **Thiết bị / Dtype** | `Tesla T4` / `float16` (`eager`) |
| **Run ID / Config Hash** | `pilot_phi4_multimodal_float16` / `f5b6ff22449f36fa338e0be60e6d163c5c5bd277878f56cc74376691982fe756` |
| **Mẫu hoàn thành / Tổng** | **34 / 34** (lỗi: 0) |
| **Chế độ chấm điểm** | `alternative_evaluation_rule_based` (`unverified_draft_rule_based`) |

## 2. Chỉ số tổng quan

| Chỉ số | Giá trị | Số lượng chấm |
|---|---:|---:|
| **aAcc** (Question Accuracy) | **61.76%** (0.617647) | 34 câu |
| **fAcc** (Figure Accuracy) | **40.00%** (0.4) | 15 ảnh |
| **qAcc** (Question-Pair Accuracy) | **21.43%** (0.214286) | 14 cặp |

## 3. Phân tích theo nhóm (`VD/VS`, `Easy/Hard`, 8 `subcategory`)

| Nhóm | Số câu | aAcc | fAcc | qAcc |
|---|---:|---:|---:|---:|
| Category: `VD` | 12 | 58.33% | 50.00% | 16.67% |
| Category: `VS` | 22 | 63.64% | 28.57% | 25.00% |
| Difficulty: `easy` | 14 | 78.57% | 75.00% | 0.00% |
| Difficulty: `hard` | 20 | 50.00% | 27.27% | 33.33% |
| Subcategory: `chart` | 4 | 25.00% | 0.00% | 0.00% |
| Subcategory: `figure` | 2 | 50.00% | 50.00% | 0.00% |
| Subcategory: `illusion` | 2 | 50.00% | 50.00% | 0.00% |
| Subcategory: `map` | 6 | 66.67% | 0.00% | 0.00% |
| Subcategory: `math` | 6 | 66.67% | 50.00% | 33.33% |
| Subcategory: `ocr` | 3 | 66.67% | 50.00% | 0.00% |
| Subcategory: `table` | 9 | 77.78% | 50.00% | 66.67% |
| Subcategory: `video` | 2 | 50.00% | 50.00% | 0.00% |

## 4. Danh mục tệp trong thư mục

- **7 artifact chuẩn repo (`docs/result_schema.md`):** `predictions.jsonl`, `metrics.json`, `run_config.yaml`, `run_metadata.json`, `errors.jsonl`, `run_status.json`, `manifest.json`.
- **Gói dễ đọc tương đương nhóm MMHal:** `phi4_multimodal_hallusionbench_responses.json`, `phi4_multimodal_hallusionbench_evaluation.json`, `phi4_multimodal_hallusionbench_summary.json`, `inference_config.json`, `inference_summary.json`, `inference_log.txt`.
- **Phân tích lỗi:** `analysis/error_cases.json`, `analysis/error_summary.json`, `analysis/error_cases.md`.
