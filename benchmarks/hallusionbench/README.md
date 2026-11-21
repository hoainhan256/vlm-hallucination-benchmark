# HallusionBench — Nhóm 2 (A + B + C)

Đánh giá **zero-shot baseline (0 epoch — KHÔNG train/fine-tune)** hiện tượng ảo giác của 3 mô hình VLM (**SmolVLM2-2.2B-Instruct**, **Qwen3-VL-4B-Instruct**, **Phi-4-multimodal-instruct**) trên **HallusionBench**.

---

## 1. Cấu trúc thư mục siêu gọn

```text
benchmarks/hallusionbench/
├── README.md                  # Tài liệu duy nhất của nhóm (dữ liệu, quy trình, cách chạy)
├── __init__.py                # Export chính + lớp HallusionBenchBenchmark(BaseBenchmark)
├── dataset.py                 # Phần A: Schema, loader chỉ đọc, audit thật, chọn pilot đủ 8/8 dạng bài
├── inference.py               # Phần B: Prompt không lộ nhãn, runner tự resume, nạp 3 model HF, xuất 7 artifact + log/JSON giống MMHal
├── evaluation.py              # Phần C: Chấm aAcc/fAcc/qAcc + VD/VS + Easy/Hard + bảng lỗi 6 nhãn + xuất gói deliverables chuẩn repo
├── hallusionbench_colab.ipynb # Notebook Google Colab chia riêng từng model (5A/5B/5C) + xuất zip chuẩn cấu trúc repo
├── results/                   # Nơi lưu kết quả inference & evaluation chạy thật theo từng model (7 artifact chuẩn + format MMHal)
└── tests/
    └── test_hallusionbench.py # Toàn bộ unit test + E2E test CPU-only (chạy được cả unittest & pytest)
```

---

## 2. Dữ liệu HallusionBench (Thống kê kiểm chứng THẬT ngày 20/11/2026)

- **Nguồn thực tải:** `rayguan/HallusionBench` @ `2067dfdce5abc7efdbb0927f4721ae282f0b6c51` (License: `BSD-3-Clause`; chờ nhóm trưởng khóa vào `configs/datasets.yaml`).
- **`HallusionBench.json` SHA-256:** `ca6e0fb677dd56e5fb0e2700d06c58d01f8a6ae66e0f1773b2ac48c0a095d051`.
- **Trạng thái audit (`--mode audit`):** **`is_clean = true`** — **1.129/1.129** câu hợp lệ, **0** lỗi schema, **0** trùng `sample_id`, **346/346** ảnh duy nhất đọc được qua PIL (**0** ảnh thiếu, **0** ảnh hỏng), **34** mẫu `.PNG` lệch chữ hoa/thường được loader tự xử lý trên Linux.
- **Phân bố đầu vào (`visual_input`):**
  - `"0"` (**Text-only, không ảnh**): **178** câu (`image_path = None` tuyệt đối)
  - `"1"` (**Ảnh gốc**): **447** câu
  - `"2"` (**Ảnh đã chỉnh sửa để bẫy**): **504** câu
- **Phân bố nhóm & 8 dạng bài (`subcategory`):**
  - **`VD` (Visual Dependent):** **591** câu | **`VS` (Visual Supplement):** **538** câu
  - 8 dạng bài: `chart` (206), `ocr` (170), `video` (170), `table` (155), `illusion` (144), `math` (108), `map` (96), `figure` (80)
  - Đáp án chuẩn (`gt_answer`): `no` (`"0"`): **645** | `yes` (`"1"`): **484**
  - Cấu trúc nhóm: **167** set, **346** visual figures, **455** question pairs.

---

## 3. Kết quả thực nghiệm đã có (Pilot 8/8 dạng bài — Tesla T4 `float16`)

> Chi tiết đầy đủ tại [`results/README.md`](results/README.md) và artifact chuẩn schema tại `results/hb_colab_t4_pilot/`. Kết quả chạy thật 0-epoch trên 34 câu hỏi (15 hình ảnh, 14 cặp kiểm soát phủ đủ 8/8 subcategory: `chart`, `figure`, `illusion`, `map`, `math`, `ocr`, `table`, `video`), **0 lỗi kỹ thuật / 0 lỗi định dạng**.

| Model | Revision | Stage / Status | Hoàn thành | aAcc | fAcc | qAcc | Báo cáo chi tiết |
|---|---|---|---:|---:|---:|---:|---|
| `smolvlm2` (`HuggingFaceTB/SmolVLM2-2.2B-Instruct`) | `be92141c8d` | `pilot` / `COMPLETED` | 34/34 | **58.82%** | **40.00%** | **7.14%** | [`results/smolvlm2/README.md`](results/smolvlm2/README.md) |
| `qwen3_vl` (`Qwen/Qwen3-VL-4B-Instruct`) | `ebb281ec70` | `pilot` / `COMPLETED` | 34/34 | **67.65%** | **60.00%** | **21.43%** | [`results/qwen3_vl/README.md`](results/qwen3_vl/README.md) |
| `phi4_multimodal` (`microsoft/Phi-4-multimodal-instruct`) | `93f923e1a7` | `pilot` / `COMPLETED` | 34/34 | **61.76%** | **40.00%** | **21.43%** | [`results/phi4_multimodal/README.md`](results/phi4_multimodal/README.md) |

- **Bất đồng giữa 3 mô hình đã chạy:** **18 ca** ([`results/cross_model_disagreements.json`](results/cross_model_disagreements.json)).

---

## 4. Quy tắc Benchmark chuẩn chỉ (Không train — 0 epoch)

1. **Không train / không sửa trọng số / không lượng tử hóa (`INT4`/`INT8`)**, giải mã tất định `do_sample = False`, dùng chung prompt `"<question> Please answer yes or no."`.
2. **GPU T4 không hỗ trợ `bfloat16` native** (`sm_75`): chạy `float16` ở chế độ `smoke_test`, `pilot` (8 nhóm trọn vẹn phủ đủ **8/8 dạng bài**), hoặc `full_baseline` (**1.129 câu**). Không gọi là `official` trước khi nhóm trưởng duyệt khóa `configs/models.yaml` & `configs/datasets.yaml`.
3. **3 chỉ số chấm điểm (`evaluation.py`, gắn nhãn minh bạch `alternative_evaluation_rule_based`):**
   - **`aAcc`:** Tỷ lệ câu hỏi đúng.
   - **`fAcc`:** Tỷ lệ hình ảnh (chỉ tính ảnh thật, loại `text-only`) mà model đúng **tất cả** câu hỏi gắn với hình đó.
   - **`qAcc`:** Tỷ lệ cặp câu hỏi kiểm soát (`>= 2` biến thể ảnh gốc/sửa hoặc không ảnh/có ảnh) mà model đúng **cả cặp**.
4. **6 nhãn phân loại lỗi (`docs/evaluation_policy.md`):** `incorrect_prediction`, `hallucination`, `format_violation`, `technical_error` (loại khỏi mẫu số điểm), `label_data_error`, `undetermined`. Không mặc định mọi câu sai là ảo giác — phải mở ảnh thật kiểm chứng.
5. **Lưu kết quả kết hợp 2 chuẩn (chuẩn repo + dễ đọc giống MMHal):**
   - **7 artifact chuẩn (`docs/result_schema.md`):** `predictions.jsonl`, `metrics.json`, `run_config.yaml`, `run_metadata.json`, `errors.jsonl`, `run_status.json`, `manifest.json` (pass `scripts/check_results.py` & `scripts/check_manifest.py`).
   - **Bộ tệp JSON/log/README giống nhóm MMHal (tự sinh & tự backfill cho run cũ):** `inference_config.json`, `inference_summary.json`, `inference_log.txt`, `<model>_hallusionbench_responses.json`, `<model>_hallusionbench_evaluation.json`, `<model>_hallusionbench_summary.json`, `analysis/error_cases.md`, `README.md` từng model + `README.md` tổng hợp 3 model tại `benchmarks/hallusionbench/results/`.

---

## 5. Cách chạy (Colab chia riêng từng model / Kaggle 2×T4 / CLI)

### 5.1. Chạy nhanh & chính xác nhất trên Google Colab (Khuyến nghị)
Mở [hallusionbench_colab.ipynb](hallusionbench_colab.ipynb) trên Colab, chọn **Runtime → GPU (T4 / L4)**:
- **Cell 1:** Mount Google Drive + tự liệt kê các run đã có trên Drive để biết ngay thiếu model nào.
- **Cell 2:** Clone/pull nhánh `benchmark/hallusionbench` & cài môi trường chính (`transformers` mới cho **SmolVLM2** & **Qwen3-VL**).
- **Cell 4:** Tự tải dataset 1 lần rồi lưu `.tar` lên Google Drive, chạy `audit` & tạo `pilot_ids_8sub.json` phủ đủ **8/8 subcategory**.
- **Cell 5 + Cell 5A / 5B / 5C:** Chọn `STAGE` (`"smoke_test"` | `"pilot"` | `"full_baseline"`) rồi chạy từng model hoặc bấm **Run all** một lần cho đủ cả 3 model! Riêng **Cell 5C (`phi4_multimodal`)** tự tạo venv tách biệt `/content/venv_phi4` (`transformers==4.48.2`, `accelerate==1.3.0`, `peft==0.13.2`, kế thừa PyTorch/CUDA của Colab) và bật CPU offload trên T4 (`--max-gpu-memory 8GiB --max-cpu-memory 10GiB --offload-folder /content/phi4_offload`) nên không xung đột với `transformers 5.x` và không bị OOM.
- **Cell 6–8:** Tự phát hiện mọi model đã chạy trên Drive (kể cả các run cũ đã chạy từ trước), chấm điểm, bổ sung đầy đủ file format MMHal, chạy `scripts/check_results.py` & `scripts/check_manifest.py`, xuất đúng cấu trúc `benchmarks/hallusionbench/results/` và tải 1 file `.zip` về giải nén thẳng vào gốc repo để `git add` & `git push`.

### 5.2. Chạy bằng lệnh CLI (Kaggle 2×T4 / Linux / PowerShell)

```bash
# 1. Audit & xuất tập pilot phủ đủ 8/8 dạng bài (tải dataset vào DATA_ROOT ngoài Git trước)
python -m benchmarks.hallusionbench.dataset --mode audit --dataset-json "$DATA_ROOT/HallusionBench.json" --data-root "$DATA_ROOT"
python -m benchmarks.hallusionbench.dataset --mode export-pilot --dataset-json "$DATA_ROOT/HallusionBench.json" --data-root "$DATA_ROOT" --target-groups 8 --output pilot_ids_8sub.json

# 2. Chạy inference riêng từng model (bỏ --pilot-file & đổi --stage full_baseline để chạy đủ 1.129 câu)
# - SmolVLM2 / Qwen3-VL (cần transformers mới):
python -m benchmarks.hallusionbench.inference --mode run --allow-model-load --model-key smolvlm2 --stage pilot --dtype float16 --attn-implementation sdpa --dataset-json "$DATA_ROOT/HallusionBench.json" --data-root "$DATA_ROOT" --dataset-revision 2067dfdce5abc7efdbb0927f4721ae282f0b6c51 --pilot-file pilot_ids_8sub.json --experiment-id hb_pilot --run-id pilot_smolvlm2_fp16 --output-root "$OUTPUT_DIR"
# - Phi-4-multimodal (cần transformers==4.48.2 accelerate==1.3.0 peft==0.13.2; thêm CPU offload trên T4):
python -m benchmarks.hallusionbench.inference --mode run --allow-model-load --model-key phi4_multimodal --trust-remote-code --stage pilot --dtype float16 --attn-implementation eager --max-gpu-memory 10GiB --max-cpu-memory 12GiB --offload-folder /tmp/phi4_offload --dataset-json "$DATA_ROOT/HallusionBench.json" --data-root "$DATA_ROOT" --dataset-revision 2067dfdce5abc7efdbb0927f4721ae282f0b6c51 --pilot-file pilot_ids_8sub.json --experiment-id hb_pilot --run-id pilot_phi4_multimodal_fp16 --output-root "$OUTPUT_DIR"

# 3. Chấm điểm, xuất gói chuẩn repo (7 artifact + format MMHal) vào benchmarks/hallusionbench/results/ & kiểm chứng
python -m benchmarks.hallusionbench.evaluation --mode export-deliverables --overwrite-export --run-dirs smolvlm2="$OUTPUT_DIR/hb_pilot/pilot_smolvlm2_fp16" qwen3_vl="$OUTPUT_DIR/hb_pilot/pilot_qwen3_vl_fp16" phi4_multimodal="$OUTPUT_DIR/hb_pilot/pilot_phi4_multimodal_fp16" --dataset-json "$DATA_ROOT/HallusionBench.json" --data-root "$DATA_ROOT"
python scripts/check_results.py benchmarks/hallusionbench/results/smolvlm2/predictions.jsonl
python scripts/check_manifest.py benchmarks/hallusionbench/results/smolvlm2
```

### 5.3. Chạy kiểm thử CPU-only

```bash
python scripts/check_repository.py
python -m unittest discover -s benchmarks/hallusionbench/tests -p "test_hallusionbench.py" -v
# Hoặc sau khi cài requirements.txt:
python -m pytest -v tests/ benchmarks/hallusionbench/tests/
```
