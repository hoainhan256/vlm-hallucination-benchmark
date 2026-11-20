# HallusionBench Benchmark Adapter — Nhóm 2 (Pipeline A → B → C)

Module triển khai hoàn chỉnh đánh giá hiện tượng ảo giác (hallucination) cho 3 mô hình thị giác–ngôn ngữ (**SmolVLM2-2.2B-Instruct**, **Qwen3-VL-4B-Instruct**, **Phi-4-multimodal-instruct**) trên bộ dữ liệu **HallusionBench** (`tianyi-lab/HallusionBench` / bản phát hành HF `rayguan/HallusionBench`), tuân thủ đầy đủ chính sách trong `docs/` và [PHAN_CONG_NHOM_2_HALLUSIONBENCH_V2.md](PHAN_CONG_NHOM_2_HALLUSIONBENCH_V2.md).

---

## 1. Mục tiêu & Nguyên tắc cốt lõi

- **Bảo toàn cấu trúc gốc:** Giữ nguyên liên kết giữa ảnh, câu hỏi và các cặp câu hỏi kiểm soát (`group_or_pair_key`, `figure_key`, `pair_key`, `visual_input=0/1/2`); không tự ý chỉnh sửa câu hỏi, hình ảnh, nhãn gốc hoặc loại bỏ mẫu khó.
- **Tách bạch chỉ số:** Báo cáo đầy đủ 3 chỉ số chuẩn của HallusionBench (`aAcc`, `fAcc`, `qAcc`) kèm phân tách theo nhóm **VD (Visual Dependent)** vs. **VS (Visual Supplement)**, mức độ **Easy** vs. **Hard**, và loại đầu vào `visual_input` (`0`/`1`/`2`). Không gộp điểm chung với POPE hay MMHal-Bench.
- **Tách biệt lỗi kỹ thuật và ảo giác:** Các mẫu lỗi hạ tầng/VRAM (`status != "COMPLETED"`) được ghi vào `errors.jsonl` và **không** tính vào mẫu số điểm. Mọi câu trả lời sai đều phải được phân loại theo 6 nhãn chuẩn của `docs/evaluation_policy.md` thay vì mặc định là ảo giác.

---

## 2. Cấu trúc module (A + B + C đã tích hợp)

| Phần | Tệp | Chức năng chính |
|---|---|---|
| **A — DATA** | [schema.py](schema.py), [dataset.py](dataset.py), [DATASET_NOTES.md](DATASET_NOTES.md) | Kiểm chứng schema, tạo `sample_id` ổn định, bảo toàn nhóm/hình/cặp & `visual_input=0/1/2`, audit dataset thật, xuất & kiểm chứng tập `pilot_ids.json` trọn nhóm. |
| **B — INFERENCE** | [protocol.py](protocol.py), [inference.py](inference.py), [KAGGLE_RUNBOOK.md](KAGGLE_RUNBOOK.md), [hallusionbench_colab.ipynb](hallusionbench_colab.ipynb) | Chuẩn bị prompt không rò rỉ nhãn, tách chặt ảnh vs text-only, runner hỗ trợ resume & `--pilot-file`, kiểm soát chính sách GPU/BF16/không quantization, xuất đủ 7 artifact theo `docs/result_schema.md`. |
| **C — EVALUATION** | [scoring.py](scoring.py), [analysis.py](analysis.py), [EVALUATION_NOTES.md](EVALUATION_NOTES.md) | Parser Yes/No (`alternative_evaluation_rule_based`), tính `aAcc / fAcc / qAcc` kèm `VD/VS`, `Easy/Hard`, `visual_input`, chấm trực tiếp thư mục run của B (`score_run_directory`) & làm mới `manifest.json`, trích xuất bảng ca lỗi 6 nhãn, so sánh bất đồng 3 model & đối chứng Before/After. |
| **Tích hợp chung** | [benchmark.py](benchmark.py), [`__init__.py`](__init__.py) | Lớp `HallusionBenchBenchmark(BaseBenchmark)` kết nối A → B → C theo interface `benchmarks/base.py`. |
| **Kiểm thử CPU-only** | `tests/test_dataset.py`, `tests/test_protocol.py`, `tests/test_scoring.py`, `tests/test_analysis.py`, `tests/test_pipeline_e2e.py` | 35 kiểm thử đơn vị và tích hợp đầu-cuối A → B → C (không tải model/dataset thật). |

---

## 3. Hợp đồng dữ liệu dùng chung (Data Contract A ↔ B ↔ C)

### 3.1. Đầu ra chuẩn hóa từ Thành viên A (`HallusionBenchDataset.load_index()`)
Mỗi bản ghi ánh xạ theo `sample_id` (`cat{category}_sub{subcategory}_set{set_id}_fig{figure_id}_q{question_id}_vis{visual_input_raw}`) cung cấp:
- `sample_id` (`str`): Mã định danh ghép ổn định, duy nhất.
- `reference_answer` (`"yes"` / `"no"`) & `gt_answer` (`"1"` / `"0"`): Nhãn chuẩn (được loại bỏ hoàn toàn trong `to_model_safe_dict()` trước khi đưa cho B).
- `category` (`"VD"` / `"VS"`), `subcategory` (`str`), `difficulty` (`"easy"` / `"hard"`).
- `figure_key` (`str | None`): Khóa cấp hình ảnh cho `fAcc` (`None` với mẫu text-only `visual_input=0` để không làm nhiễu `fAcc`).
- `pair_key` (`str`): Khóa cấp cặp câu hỏi kiểm soát cho `qAcc` (khớp `question_pair_key`).
- `visual_input` (`bool`), `visual_input_raw` (`"0"` / `"1"` / `"2"`), `image_path` (`str | None`), `question` (`str`).

### 3.2. Đầu ra từ Thành viên B (`predictions.jsonl` & 7 artifact)
Mỗi run lưu tại `results/<experiment_id>/<run_id>/` gồm đủ 7 tệp theo `docs/result_schema.md`:
1. `predictions.jsonl` (`sample_id`, `prediction`, `raw_prediction`, `status`, `run_config_hash`, `error` nếu `FAILED`, `timing_ms`)
2. `metrics.json`
3. `run_config.yaml`
4. `run_metadata.json`
5. `errors.jsonl`
6. `run_status.json` (`PENDING`, `RUNNING`, `COMPLETED`, `BLOCKED`, `FAILED`)
7. `manifest.json` (có SHA-256 checksum, tự làm mới khi C chạy `score_run_directory`)

### 3.3. Khung 6 nhãn phân loại lỗi (`analysis.py`)
1. `incorrect_prediction`: Câu trả lời chốt khác nhãn tham chiếu.
2. `hallucination`: Mâu thuẫn bằng chứng thị giác (bịa đối tượng, bỏ sót/nhận diện sai, suy luận sai quan hệ/thuộc tính, thiên kiến ngôn ngữ).
3. `format_violation`: Trả lời nước đôi hoặc không trích xuất được Yes/No.
4. `technical_error`: Lỗi phần cứng, CUDA OOM, timeout hoặc I/O ở bước inference.
5. `label_data_error`: Lỗi từ câu hỏi, nhãn gốc hoặc tệp ảnh của dataset.
6. `undetermined`: Thiếu bằng chứng để kết luận chắc chắn.

---

## 4. Quy trình chạy đầu-cuối A → B → C

### Bước 1 (Phần A) — Audit dataset thật (ngoài Git) & xuất tập Pilot cố định

```powershell
python -m benchmarks.hallusionbench.dataset `
  --mode audit `
  --dataset-json "$env:DATA_ROOT\HallusionBench.json" `
  --data-root "$env:DATA_ROOT" `
  --dataset-source "rayguan/HallusionBench" `
  --dataset-revision "2067dfdce5abc7efdbb0927f4721ae282f0b6c51" `
  --output "tmp\hallusionbench_audit_report.json"

python -m benchmarks.hallusionbench.dataset `
  --mode export-pilot `
  --dataset-json "$env:DATA_ROOT\HallusionBench.json" `
  --data-root "$env:DATA_ROOT" `
  --target-groups 6 `
  --output "benchmarks\hallusionbench\pilot_ids.json"
```

### Bước 2 (Phần B) — Chạy `dry-run` (CPU) hoặc chạy 3 mô hình thật trên Colab/Kaggle

- **Google Colab (1-click lưu dataset lên Drive + chạy 3 model + tải kết quả `.zip`):** Mở [hallusionbench_colab.ipynb](hallusionbench_colab.ipynb).
- **Kaggle (GPU T4 ×2):** Xem [KAGGLE_RUNBOOK.md](KAGGLE_RUNBOOK.md).

```powershell
# Kiểm tra pipeline trên CPU (Fake Model)
python -m benchmarks.hallusionbench.inference `
  --mode dry-run `
  --model-key smolvlm2 `
  --stage pilot `
  --dtype float16 `
  --dataset-json "$env:DATA_ROOT\HallusionBench.json" `
  --data-root "$env:DATA_ROOT" `
  --pilot-file "benchmarks\hallusionbench\pilot_ids.json" `
  --experiment-id hb_dryrun `
  --run-id dryrun_pilot_01 `
  --output-root "$env:OUTPUT_DIR"
```

### Bước 3 (Phần C) — Chấm điểm thư mục run của B, kiểm chứng artifact & phân tích lỗi

```powershell
python -m benchmarks.hallusionbench.scoring `
  --run-dir "$env:OUTPUT_DIR\hb_dryrun\dryrun_pilot_01" `
  --dataset-json "$env:DATA_ROOT\HallusionBench.json" `
  --data-root "$env:DATA_ROOT"

python scripts/check_results.py "$env:OUTPUT_DIR\hb_dryrun\dryrun_pilot_01\predictions.jsonl"
python scripts/check_manifest.py "$env:OUTPUT_DIR\hb_dryrun\dryrun_pilot_01"

python -m benchmarks.hallusionbench.analysis `
  --mode extract `
  --run-dir "$env:OUTPUT_DIR\hb_dryrun\dryrun_pilot_01" `
  --dataset-json "$env:DATA_ROOT\HallusionBench.json" `
  --data-root "$env:DATA_ROOT"
```

---

## 5. Kiểm thử trên CPU (35 tests)

```powershell
python -m pytest -v tests/ benchmarks/hallusionbench/tests/
# Hoặc bằng stdlib unittest (không cần cài pytest):
python -m unittest discover -s benchmarks/hallusionbench/tests -p "test_dataset.py" -v
python -m unittest discover -s benchmarks/hallusionbench/tests -p "test_pipeline_e2e.py" -v
```

---

## 6. Checklist trước khi mở khóa chạy Official

- [x] **Đồng bộ hạ tầng (Gate G0):** Đã đồng bộ nhánh `main` vào `benchmark/hallusionbench` (`benchmarks/base.py`, `vlm_benchmark/`, `scripts/`, `configs/`).
- [ ] **Audit & khóa dataset (Gate G1):** Chạy `--mode audit` trên dataset thật, điền bảng kiểm chứng trong `DATASET_NOTES.md` và xin trưởng nhóm khóa `revision`/`license` trong `configs/datasets.yaml`.
- [ ] **Chốt cấu hình mô hình & phần cứng (Gate G3/P0):** Thu `resolved_model_revision`, `resolved_processor_revision`, log T4/BF16 từ chạy `smoke_test`/`pilot` và xin duyệt `dtype` / `max_new_tokens`.
- [ ] **Chốt phương pháp chấm (Gate G2/P1):** Đối chiếu `scoring.py` (`alternative_evaluation_rule_based`, xem `EVALUATION_NOTES.md`) với `evaluation.py` gốc trên kết quả thật.
