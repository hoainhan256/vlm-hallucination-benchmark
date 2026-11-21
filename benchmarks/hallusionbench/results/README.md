# Kết quả thực nghiệm HallusionBench (Nhóm 2)

Thư mục này lưu đồng thời **7 artifact chuẩn repo (`docs/result_schema.md`)** và **bộ tệp JSON/log dễ đọc (tương đương nhánh MMHal)** cho từng mô hình đã chạy inference thật (0-epoch, không tạo dữ liệu giả).

## Bảng tổng hợp 3 mô hình

| Model Key | Model ID | Stage | Status | Dtype | Hoàn thành | aAcc | fAcc | qAcc | Thư mục |
|---|---|---|---|---|---:|---:|---:|---:|---|
| `smolvlm2` | `HuggingFaceTB/SmolVLM2-2.2B-Instruct` | `pilot` | `COMPLETED` | `float16` | 34/34 | **58.82%** | **40.00%** | **7.14%** | [`smolvlm2/`](smolvlm2/README.md) |
| `qwen3_vl` | `Qwen/Qwen3-VL-4B-Instruct` | `pilot` | `COMPLETED` | `float16` | 34/34 | **67.65%** | **60.00%** | **21.43%** | [`qwen3_vl/`](qwen3_vl/README.md) |
| `phi4_multimodal` | `microsoft/Phi-4-multimodal-instruct` | `pilot` | `COMPLETED` | `float16` | 34/34 | **61.76%** | **40.00%** | **21.43%** | [`phi4_multimodal/`](phi4_multimodal/README.md) |

- **Ca bất đồng giữa các mô hình:** **18** ca — xem [`cross_model_disagreements.json`](cross_model_disagreements.json).

## Lệnh kiểm chứng lại artifact

```bash
python scripts/check_results.py benchmarks/hallusionbench/results/phi4_multimodal/predictions.jsonl
python scripts/check_manifest.py benchmarks/hallusionbench/results/phi4_multimodal
python scripts/check_results.py benchmarks/hallusionbench/results/qwen3_vl/predictions.jsonl
python scripts/check_manifest.py benchmarks/hallusionbench/results/qwen3_vl
python scripts/check_results.py benchmarks/hallusionbench/results/smolvlm2/predictions.jsonl
python scripts/check_manifest.py benchmarks/hallusionbench/results/smolvlm2
```
