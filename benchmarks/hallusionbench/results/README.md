# Kết quả thực nghiệm HallusionBench (Nhóm 2)

Thư mục này là nơi lưu trữ kết quả inference & đánh giá **chạy thật** (0-epoch, không tạo dữ liệu giả theo `docs/result_schema.md`) cho 3 mô hình trên HallusionBench, kết hợp đồng thời:
1. **7 artifact chuẩn repo (`docs/result_schema.md`):** `predictions.jsonl`, `metrics.json`, `run_config.yaml`, `run_metadata.json`, `errors.jsonl`, `run_status.json`, `manifest.json` (kiểm chứng bằng `scripts/check_results.py` & `scripts/check_manifest.py`).
2. **Bộ tệp JSON / log / README dễ đọc tương đương nhánh `benchmark/mmhal`:** `<model>_hallusionbench_responses.json`, `<model>_hallusionbench_evaluation.json`, `<model>_hallusionbench_summary.json`, `inference_config.json`, `inference_summary.json`, `inference_log.txt`, `analysis/error_cases.md`, và `README.md` bảng điểm từng mô hình.

## Trạng thái commit kết quả hiện tại

| Model Key | Model ID | Trạng thái trên Git |
|---|---|---|
| `smolvlm2` | `HuggingFaceTB/SmolVLM2-2.2B-Instruct` | `PENDING` (chờ xuất gói từ Colab/Drive) |
| `qwen3_vl` | `Qwen/Qwen3-VL-4B-Instruct` | `PENDING` (chờ xuất gói từ Colab/Drive) |
| `phi4_multimodal` | `microsoft/Phi-4-multimodal-instruct` | `PENDING` (chờ chạy riêng & xuất gói từ Colab/Drive) |

> **Cách cập nhật thư mục này sau khi chạy Colab/Kaggle:**
> Chạy Cell 6 → Cell 8 trong `benchmarks/hallusionbench/hallusionbench_colab.ipynb` (hoặc chạy CLI `python -m benchmarks.hallusionbench.evaluation --mode export-deliverables --overwrite-export --run-dirs smolvlm2=<dir1> qwen3_vl=<dir2> phi4_multimodal=<dir3> --dataset-json <HallusionBench.json> --data-root <DATA_ROOT>`). Tệp `README.md` này sẽ tự động được thay bằng bảng điểm thật của các mô hình đã chạy.
