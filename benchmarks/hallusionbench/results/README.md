# Kết quả thực nghiệm HallusionBench (Nhóm 2)

Thư mục này lưu đồng thời **7 artifact chuẩn repo (`docs/result_schema.md`)**, **bộ tệp JSON/log dễ đọc (tương đương nhánh `benchmark/mmhal`)**, và **bộ 5 biểu đồ PNG 300 DPI + báo cáo trực quan** cho **đủ 3/3 mô hình đã chạy inference thật** (0-epoch, **không có dữ liệu giả**).

## Nguồn gốc & xác thực kết quả

- **Môi trường chạy thật:** Google Colab GPU **Tesla T4** (`14.56 GiB`, `float16`, `do_sample=False`, `max_new_tokens=128`), ngày **21/11/2026**.
- **Dataset thật:** `rayguan/HallusionBench` @ `2067dfdce5abc7efdbb0927f4721ae282f0b6c51` (`HallusionBench.json` SHA-256 `ca6e0fb6...`).
- **Phạm vi:** tập **`pilot` cố định (34 câu hỏi, 15 hình ảnh, 14 cặp kiểm soát phủ đủ 8/8 dạng bài)** — hoàn thành **34/34 câu** với **0 lỗi kỹ thuật / 0 lỗi định dạng** ở cả 3 mô hình.
- **Đường dẫn song song theo schema chung:** `results/hb_colab_t4_pilot/` (chứa đủ 3 run `pilot_smolvlm2_float16`, `pilot_qwen3_vl_float16`, `pilot_phi4_multimodal_float16`).

## Bảng tổng hợp 3 mô hình

| Model Key | Model ID | Revision (HF SHA) | Stage | Status | Dtype | Hoàn thành | aAcc | fAcc | qAcc | Thư mục |
|---|---|---|---|---|---|---:|---:|---:|---:|---|
| `smolvlm2` | `HuggingFaceTB/SmolVLM2-2.2B-Instruct` | `be92141c8d` | `pilot` | `COMPLETED` | `float16` | 34/34 | **58.82%** | **40.00%** | **7.14%** | [`smolvlm2/`](smolvlm2/README.md) |
| `qwen3_vl` | `Qwen/Qwen3-VL-4B-Instruct` | `ebb281ec70` | `pilot` | `COMPLETED` | `float16` | 34/34 | **67.65%** | **60.00%** | **21.43%** | [`qwen3_vl/`](qwen3_vl/README.md) |
| `phi4_multimodal` | `microsoft/Phi-4-multimodal-instruct` | `93f923e1a7` | `pilot` | `COMPLETED` | `float16` *(offload)* | 34/34 | **61.76%** | **40.00%** | **21.43%** | [`phi4_multimodal/`](phi4_multimodal/README.md) |

## Báo cáo trực quan & Phân tích bất đồng

- **Báo cáo trực quan đầy đủ (nhúng sẵn 5 biểu đồ):** [`REPORT.md`](REPORT.md)
- **5 biểu đồ PNG (300 DPI) trong [`figures/`](figures/):**
  1. [`figures/fig1_overall_metrics.png`](figures/fig1_overall_metrics.png) — Tổng quan `aAcc / fAcc / qAcc`
  2. [`figures/fig2_category_and_difficulty.png`](figures/fig2_category_and_difficulty.png) — So sánh `VD vs VS` và `Easy vs Hard`
  3. [`figures/fig3_subcategory_heatmap.png`](figures/fig3_subcategory_heatmap.png) — Heatmap 3 model × 8 dạng bài
  4. [`figures/fig4_visual_input_condition.png`](figures/fig4_visual_input_condition.png) — Tác động của ảnh bẫy chỉnh sửa (`0 / 1 / 2`)
  5. [`figures/fig5_errors_and_latency.png`](figures/fig5_errors_and_latency.png) — Hồ sơ lỗi theo độ khó & Độ chính xác vs Tốc độ
- **Ca bất đồng giữa 3 mô hình:** **18** ca — xem [`cross_model_disagreements.json`](cross_model_disagreements.json).

## Lệnh kiểm chứng lại artifact & tái sinh biểu đồ

```bash
python scripts/check_results.py benchmarks/hallusionbench/results/phi4_multimodal/predictions.jsonl
python scripts/check_manifest.py benchmarks/hallusionbench/results/phi4_multimodal
python scripts/check_results.py benchmarks/hallusionbench/results/qwen3_vl/predictions.jsonl
python scripts/check_manifest.py benchmarks/hallusionbench/results/qwen3_vl
python scripts/check_results.py benchmarks/hallusionbench/results/smolvlm2/predictions.jsonl
python scripts/check_manifest.py benchmarks/hallusionbench/results/smolvlm2
python -m benchmarks.hallusionbench.evaluation --mode visualize --results-dir benchmarks/hallusionbench/results
```
