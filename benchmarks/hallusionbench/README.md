# HallusionBench — Nhóm 2 (A + B + C)

Đánh giá **zero-shot baseline (0 epoch — KHÔNG train/fine-tune)** hiện tượng ảo giác của 3 mô hình VLM (**SmolVLM2-2.2B-Instruct**, **Qwen3-VL-4B-Instruct**, **Phi-4-multimodal-instruct**) trên **HallusionBench**.

---

## 1. Cấu trúc thư mục siêu gọn

```text
benchmarks/hallusionbench/
├── README.md                  # Tài liệu duy nhất của nhóm (dữ liệu, quy trình, cách chạy)
├── __init__.py                # Export chính + lớp HallusionBenchBenchmark(BaseBenchmark)
├── dataset.py                 # Phần A: Schema, loader chỉ đọc, audit thật, chọn pilot đủ 8/8 dạng bài
├── inference.py               # Phần B: Prompt không lộ nhãn, runner tự resume, nạp 3 model HF, xuất 7 artifact
├── evaluation.py              # Phần C: Chấm aAcc/fAcc/qAcc + VD/VS + Easy/Hard + bảng lỗi 6 nhãn & bất đồng 3 model
├── hallusionbench_colab.ipynb # Notebook 1-click Google Colab (tự lưu dataset lên Drive + chạy 3 model + tải zip)
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

## 3. Quy tắc Benchmark chuẩn chỉ (Không train — 0 epoch)

1. **Không train / không sửa trọng số / không lượng tử hóa (`INT4`/`INT8`)**, giải mã tất định `do_sample = False`, dùng chung prompt `"<question> Please answer yes or no."`.
2. **GPU T4 không hỗ trợ `bfloat16` native** (`sm_75`): chạy `float16` ở chế độ `smoke_test`, `pilot` (8 nhóm trọn vẹn phủ đủ **8/8 dạng bài**), hoặc `full_baseline` (**1.129 câu**). Không gọi là `official` trước khi nhóm trưởng duyệt khóa `configs/models.yaml` & `configs/datasets.yaml`.
3. **3 chỉ số chấm điểm (`evaluation.py`, gắn nhãn minh bạch `alternative_evaluation_rule_based`):**
   - **`aAcc`:** Tỷ lệ câu hỏi đúng.
   - **`fAcc`:** Tỷ lệ hình ảnh (chỉ tính ảnh thật, loại `text-only`) mà model đúng **tất cả** câu hỏi gắn với hình đó.
   - **`qAcc`:** Tỷ lệ cặp câu hỏi kiểm soát (`>= 2` biến thể ảnh gốc/sửa hoặc không ảnh/có ảnh) mà model đúng **cả cặp**.
4. **6 nhãn phân loại lỗi (`docs/evaluation_policy.md`):** `incorrect_prediction`, `hallucination`, `format_violation`, `technical_error` (loại khỏi mẫu số điểm), `label_data_error`, `undetermined`. Không mặc định mọi câu sai là ảo giác — phải mở ảnh thật kiểm chứng.
5. **Bộ 7 artifact bắt buộc mỗi run:** `predictions.jsonl`, `metrics.json`, `run_config.yaml`, `run_metadata.json`, `errors.jsonl`, `run_status.json`, `manifest.json` (tự làm mới SHA-256 khi chấm điểm).

---

## 4. Cách chạy (Colab 1-click / Kaggle 2×T4 / CLI)

### 4.1. Chạy nhanh nhất trên Google Colab (Khuyến nghị)
Mở [hallusionbench_colab.ipynb](hallusionbench_colab.ipynb) trên Colab, chọn **Runtime → GPU (T4 / L4)** và chạy **Cell 1 → Cell 8**:
- **Cell 4:** Tự tải dataset 1 lần rồi lưu `.tar` lên Google Drive (các lần sau giải nén trong ~2 giây), chạy `audit` & tạo `pilot_ids_8sub.json` phủ đủ **8/8 subcategory**.
- **Cell 5:** Chọn 1 dòng `STAGE`:
  - `"smoke_test"` (2 câu — kiểm tra kỹ thuật ~1 phút)
  - `"pilot"` (8 nhóm trọn vẹn phủ đủ **8/8 dạng bài** — ~5 phút/model)
  - `"full_baseline"` (**toàn bộ 1.129 câu** — tự lưu từng mẫu vào Drive & tự resume nếu ngắt)
- **Cell 6–8:** Tự chấm điểm, chạy `scripts/check_results.py` & `scripts/check_manifest.py`, xuất `error_cases.md` + `cross_model_disagreements.json` và tải file `.zip` kết quả về máy.

### 4.2. Chạy bằng lệnh CLI (Kaggle 2×T4 / Linux / PowerShell)

```bash
# 1. Audit & xuất tập pilot phủ đủ 8/8 dạng bài (tải dataset vào DATA_ROOT ngoài Git trước)
python -m benchmarks.hallusionbench.dataset --mode audit --dataset-json "$DATA_ROOT/HallusionBench.json" --data-root "$DATA_ROOT"
python -m benchmarks.hallusionbench.dataset --mode export-pilot --dataset-json "$DATA_ROOT/HallusionBench.json" --data-root "$DATA_ROOT" --target-groups 8 --output pilot_ids_8sub.json

# 2. Chạy inference (thay --model-key: smolvlm2 | qwen3_vl | phi4_multimodal; thêm --trust-remote-code --attn-implementation eager cho phi4_multimodal; bỏ --pilot-file & đổi --stage full_baseline để chạy đủ 1.129 câu)
python -m benchmarks.hallusionbench.inference --mode run --allow-model-load --model-key smolvlm2 --stage pilot --dtype float16 --attn-implementation sdpa --dataset-json "$DATA_ROOT/HallusionBench.json" --data-root "$DATA_ROOT" --dataset-revision 2067dfdce5abc7efdbb0927f4721ae282f0b6c51 --pilot-file pilot_ids_8sub.json --experiment-id hb_pilot --run-id pilot_smolvlm2_fp16 --output-root "$OUTPUT_DIR"

# 3. Chấm điểm, kiểm chứng artifact & trích bảng ca lỗi
python -m benchmarks.hallusionbench.evaluation --mode score --run-dir "$OUTPUT_DIR/hb_pilot/pilot_smolvlm2_fp16" --dataset-json "$DATA_ROOT/HallusionBench.json" --data-root "$DATA_ROOT"
python scripts/check_results.py "$OUTPUT_DIR/hb_pilot/pilot_smolvlm2_fp16/predictions.jsonl"
python scripts/check_manifest.py "$OUTPUT_DIR/hb_pilot/pilot_smolvlm2_fp16"
python -m benchmarks.hallusionbench.evaluation --mode extract --run-dir "$OUTPUT_DIR/hb_pilot/pilot_smolvlm2_fp16" --dataset-json "$DATA_ROOT/HallusionBench.json" --data-root "$DATA_ROOT"
```

### 4.3. Chạy kiểm thử CPU-only

```bash
python scripts/check_repository.py
python -m unittest discover -s benchmarks/hallusionbench/tests -p "test_hallusionbench.py" -v
# Hoặc sau khi cài requirements.txt:
python -m pytest -v tests/ benchmarks/hallusionbench/tests/
```
