# Kết quả đánh giá `HuggingFaceTB/SmolVLM2-2.2B-Instruct` trên HallusionBench

## 1. Thông tin thực nghiệm

| Hạng mục | Giá trị |
|---|---|
| **Model** | `HuggingFaceTB/SmolVLM2-2.2B-Instruct` (`smolvlm2`) |
| **Model revision** | `be92141c8d0ceb98596a923c308ac2bbb4a7c466` |
| **Dataset** | `rayguan/HallusionBench` @ `2067dfdce5abc7efdbb0927f4721ae282f0b6c51` |
| **Stage / Status** | `pilot` / `COMPLETED` |
| **Thiết bị / Dtype** | `Tesla T4` / `float16` (`sdpa`) |
| **Run ID / Config Hash** | `pilot_smolvlm2_float16` / `69970a0702d84a36cab936be8c2dfa74d5b029029a7f183c10d6929ce9213edf` |
| **Mẫu hoàn thành / Tổng** | **34 / 34** (lỗi: 0) |
| **Chế độ chấm điểm** | `alternative_evaluation_rule_based` (`unverified_draft_rule_based`) |

## 2. Chỉ số tổng quan

| Chỉ số | Giá trị | Số lượng chấm |
|---|---:|---:|
| **aAcc** (Question Accuracy) | **58.82%** (0.588235) | 34 câu |
| **fAcc** (Figure Accuracy) | **40.00%** (0.4) | 15 ảnh |
| **qAcc** (Question-Pair Accuracy) | **7.14%** (0.071429) | 14 cặp |

## 3. Phân tích theo nhóm (`VD/VS`, `Easy/Hard`, 8 `subcategory`)

| Nhóm | Số câu | aAcc | fAcc | qAcc |
|---|---:|---:|---:|---:|
| Category: `VD` | 12 | 58.33% | 50.00% | 16.67% |
| Category: `VS` | 22 | 59.09% | 28.57% | 0.00% |
| Difficulty: `easy` | 14 | 71.43% | 50.00% | 0.00% |
| Difficulty: `hard` | 20 | 50.00% | 36.36% | 0.00% |
| Subcategory: `chart` | 4 | 50.00% | 0.00% | 0.00% |
| Subcategory: `figure` | 2 | 100.00% | 100.00% | 100.00% |
| Subcategory: `illusion` | 2 | 50.00% | 50.00% | 0.00% |
| Subcategory: `map` | 6 | 66.67% | 50.00% | 0.00% |
| Subcategory: `math` | 6 | 50.00% | 0.00% | 0.00% |
| Subcategory: `ocr` | 3 | 66.67% | 50.00% | 0.00% |
| Subcategory: `table` | 9 | 55.56% | 0.00% | 0.00% |
| Subcategory: `video` | 2 | 50.00% | 50.00% | 0.00% |

## 4. Danh mục tệp trong thư mục

- **7 artifact chuẩn repo (`docs/result_schema.md`):** `predictions.jsonl`, `metrics.json`, `run_config.yaml`, `run_metadata.json`, `errors.jsonl`, `run_status.json`, `manifest.json`.
- **Gói dễ đọc tương đương nhóm MMHal:** `smolvlm2_hallusionbench_responses.json`, `smolvlm2_hallusionbench_evaluation.json`, `smolvlm2_hallusionbench_summary.json`, `inference_config.json`, `inference_summary.json`, `inference_log.txt`.
- **Phân tích lỗi:** `analysis/error_cases.json`, `analysis/error_summary.json`, `analysis/error_cases.md`.
