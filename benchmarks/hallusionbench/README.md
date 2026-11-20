# HallusionBench Benchmark Adapter — Nhóm 2

Module triển khai đánh giá hiện tượng ảo giác (hallucination) cho 3 mô hình thị giác–ngôn ngữ (**Qwen3-VL-4B**, **SmolVLM2-2.2B**, **Phi-4-multimodal**) trên bộ dữ liệu **HallusionBench** (`tianyi-lab/HallusionBench`).

---

## 1. Mục tiêu & Nguyên tắc cốt lõi

- **Bảo toàn cấu trúc gốc:** Giữ nguyên liên kết giữa ảnh, câu hỏi và các cặp câu hỏi kiểm soát (`figure_key`, `pair_key`); không tự ý chỉnh sửa câu hỏi, hình ảnh, nhãn gốc hoặc loại bỏ mẫu khó.
- **Tách bạch chỉ số:** Báo cáo đầy đủ 3 chỉ số chuẩn của HallusionBench (`aAcc`, `fAcc`, `qAcc`) kèm phân tách theo nhóm **VD (Visual Dependent)** vs. **VS (Visual Supplement)** và mức độ **Easy** vs. **Hard**. Không gộp điểm chung với POPE hay MMHal-Bench.
- **Tách biệt lỗi kỹ thuật và ảo giác:** Các mẫu lỗi hạ tầng/VRAM (`status != "COMPLETED"`) được ghi vào `errors.jsonl` và **không** tính vào mẫu số điểm. Mọi câu trả lời sai đều phải được phân loại theo 6 nhãn chuẩn của dự án thay vì mặc định là ảo giác.

---

## 2. Phân công thành viên Nhóm 2

| Vai trò | Phụ trách chính | Tệp sở hữu trong `benchmarks/hallusionbench/` |
|---|---|---|
| **Thành viên A** (Data & Loader) | Xác minh nguồn/revision/license dataset, kiểm tra cặp kiểm soát & media, chuẩn hóa bản ghi đầu vào. | `dataset.py`, `schema.py` |
| **Thành viên B** (Prompt & Inference) | Xác minh prompt gốc, kết nối `BaseBenchmark`, chạy Pilot & Official (Kaggle/Local), xuất `predictions.jsonl` và quản lý 7 artifact. | `prompt.py`, `adapter.py` |
| **Thành viên C** (Scoring & QA) | Tính chỉ số (`aAcc`, `fAcc`, `qAcc`, `VD/VS`, `Easy/Hard`), kiểm tra resume an toàn, trích xuất & phân tích ca lỗi, chủ trì đề xuất giảm ảo giác. | `scoring.py`, `analysis.py`, `EVALUATION_NOTES.md`, `tests/` |

---

## 3. Cấu trúc thư mục & Tiến độ hiện tại

```text
benchmarks/hallusionbench/
├── README.md                  # Tài liệu tổng quan & hợp đồng phối hợp nhóm
├── EVALUATION_NOTES.md        # [Đã xong] Chi tiết quy tắc chấm rule-based, metric & 6 nhãn lỗi
├── scoring.py                 # [Đã xong] Tính aAcc/fAcc/qAcc, tách VD/VS & Easy/Hard, xuất artifact
├── analysis.py                # [Đã xong] Trích xuất ca lỗi, tìm bất đồng 3 model, đối chứng Before/After
├── dataset.py                 # [Chờ A] Đọc & chuẩn hóa dữ liệu từ nguồn đã khóa
├── schema.py                  # [Chờ A] Kiểm tra tính toàn vẹn cặp kiểm soát & schema bản ghi
├── prompt.py                  # [Chờ B] Mẫu câu lệnh chuẩn theo protocol gốc
├── adapter.py                 # [Chờ B & C] Kế thừa BaseBenchmark (benchmarks/base.py)
└── tests/
    ├── test_scoring.py        # [Đã xong - 6 tests] Kiểm thử tính điểm, thiếu cặp & chống ghi đè resume
    └── test_analysis.py       # [Đã xong - 4 tests] Kiểm thử trích xuất lỗi & đối chứng Before/After
```

---

## 4. Hợp đồng dữ liệu dùng chung (Data Contract A ↔ B ↔ C)

Để mã nguồn của A, B và C ghép nối trơn tru mà không cần sửa lại logic chấm điểm, dữ liệu truyền giữa các bước phải thống nhất các tên trường sau:

### 4.1. Đầu ra chuẩn hóa từ Thành viên A (`dataset_records`)
Mỗi bản ghi ánh xạ theo `sample_id` cần cung cấp:
- `sample_id` (`str`): Mã định danh duy nhất của mẫu.
- `reference_answer` (`str`): Nhãn chuẩn (`"yes"` hoặc `"no"`).
- `category` (`str`): `"VD"` (Visual Dependent) hoặc `"VS"` (Visual Supplement).
- `subcategory` (`str`, tùy chọn): Phân nhóm con của benchmark.
- `difficulty` (`str`, tùy chọn): `"easy"` hoặc `"hard"`.
- `figure_key` & `pair_key` (`str`): Khóa gộp cấp hình ảnh (`fAcc`) và cấp cặp câu hỏi kiểm soát (`qAcc`). *(Nếu chưa tạo sẵn 2 khóa này, `scoring.py` hỗ trợ tự ghép từ bộ trường gốc `category`, `subcategory`, `set_id`, `figure_id`, `question_id`).*
- `visual_input` (`bool`), `image_path` (`str`), `question` (`str`): Phục vụ hiển thị chứng cứ trong `analysis.py`.

### 4.2. Đầu ra từ bước chạy mô hình của Thành viên B (`predictions.jsonl`)
Mỗi dòng JSON bắt buộc có đủ 4 trường theo `docs/result_schema.md`:
- `sample_id` (`str`): Khớp với `sample_id` của dataset.
- `prediction` (`str | None`): Câu trả lời dùng để chấm điểm.
- `status` (`str`): `"COMPLETED"` nếu suy luận thành công; `"FAILED"` / `"BLOCKED"` nếu gặp lỗi kỹ thuật (OOM, timeout, lỗi đọc ảnh).
- `run_config_hash` (`str`): Mã băm cấu hình chạy (tạo bằng `stable_config_hash` trong `vlm_benchmark/results.py`).
- Trường khuyến nghị bổ sung: `raw_prediction` (chuỗi sinh gốc trước chuẩn hóa), `error` (mô tả lỗi nếu `status != "COMPLETED"`), `timing_ms`.

---

## 5. Quy chuẩn lưu kết quả & Phân loại lỗi

### 5.1. Bộ 7 tệp bắt buộc cho mỗi Run
Mọi lần chạy (Pilot và Official) lưu tại `results/<experiment_id>/<run_id>/` và phải có đủ 7 tệp:
1. `predictions.jsonl`
2. `metrics.json`
3. `run_config.yaml`
4. `run_metadata.json`
5. `errors.jsonl`
6. `run_status.json` (`PENDING`, `RUNNING`, `COMPLETED`, `BLOCKED`, `FAILED`)
7. `manifest.json`

**Lưu ý:** Không tạo dữ liệu giả trong thư mục `results/`; không đẩy trọng số mô hình hoặc tập ảnh lớn lên Git; hàm `export_hallusionbench_artifacts` tự động chặn ghi đè lên các mẫu đã có trạng thái `COMPLETED` khi chạy resume.

### 5.2. Khung 6 nhãn phân loại lỗi (`analysis.py`)
Khi phân tích kết quả sai, từng mẫu được phân loại vào đúng 1 trong 6 nhãn theo `docs/evaluation_policy.md`:
1. `incorrect_prediction`: Câu trả lời chốt khác nhãn tham chiếu.
2. `hallucination`: Câu trả lời mâu thuẫn bằng chứng thị giác (bịa đối tượng, bỏ sót/nhận diện sai, sai quan hệ/thuộc tính, thiên kiến ngôn ngữ).
3. `format_violation`: Trả lời nước đôi hoặc không trích xuất được Yes/No.
4. `technical_error`: Lỗi phần cứng, CUDA OOM, timeout hoặc I/O ở bước inference.
5. `label_data_error`: Lỗi từ câu hỏi, nhãn gốc hoặc tệp ảnh của dataset.
6. `undetermined`: Thiếu bằng chứng để kết luận chắc chắn.

---

## 6. Kiểm thử trên CPU (Không cần GPU)

Chạy toàn bộ 10 unit test kiểm tra bộ chấm điểm, phân tách nhóm `VD/VS`, xử lý lỗi kỹ thuật và an toàn resume:

```bash
python -m pytest benchmarks/hallusionbench/tests -v
```

---

## 7. Checklist trước khi mở khóa chạy Official

- [ ] **Đồng bộ hạ tầng (Gate G0):** Đồng bộ nhánh `main` vào `benchmark/hallusionbench` để lấy `benchmarks/base.py` và `vlm_benchmark/results.py`.
- [ ] **Khóa cấu hình & phiên bản:** Chốt `revision` của dataset `tianyi-lab/HallusionBench`, `model_id`/`revision` của 3 mô hình, `max_new_tokens` và chế độ giải mã (`do_sample: false` nếu tương thích).
- [ ] **Chốt phương pháp chấm:** Xác nhận sử dụng giao thức chấm rule-based (`alternative_evaluation`, xem chi tiết tại `EVALUATION_NOTES.md`) và đối chiếu khớp công thức gộp cặp với `evaluation.py` gốc.
- [ ] **Hoàn tất Pilot:** Chạy thử trên tập con cố định, kiểm tra đủ 7 artifact và xác nhận logic resume trước khi chạy toàn tập Official.