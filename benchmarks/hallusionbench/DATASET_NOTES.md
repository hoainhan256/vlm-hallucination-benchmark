# HallusionBench — Dataset Notes & Verification Guide (Thành viên A: DATA)

> **Phạm vi:** Tài liệu dữ liệu, đối chiếu nguồn/revision/license, schema chuẩn hóa, hợp đồng bàn giao A → B → C, và quy trình chạy audit thực tế + chốt tập pilot cố định cho **Nhóm 2 — HallusionBench**.
>
> **Nguyên tắc trung thực (`docs/dataset_policy.md`):**
> - Không chép số liệu bài báo (346 ảnh / 1.129 câu hỏi) thay cho thống kê đọc được từ file thực tế.
> - Không commit ảnh/dataset nặng lên Git; tải vào thư mục ngoài repo thông qua biến môi trường `DATA_ROOT`.
> - Không sửa `configs/datasets.yaml` trực tiếp khi chưa có bằng chứng audit và phê duyệt của trưởng nhóm.

---

## 1. Đối chiếu nguồn gốc với `configs/datasets.yaml`

| Hạng mục | Khai báo hiện tại trong `configs/datasets.yaml` | Thông tin xác minh từ nguồn chính thức | Trạng thái khóa |
|---|---|---|---|
| **Benchmark** | `HallusionBench` | Bài báo CVPR 2024: *HallusionBench: An Advanced Diagnostic Suite for Entangled Language Hallucination and Visual Illusion in Large Vision-Language Models* (Guan et al.) | — |
| **Nguồn GitHub gốc** | `tianyi-lab/HallusionBench` | `https://github.com/tianyi-lab/HallusionBench` (chứa `HallusionBench.json`, `evaluation.py`, `utils.py`; ảnh được dẫn qua link Google Drive `hallusion_bench.zip` nên thường gặp giới hạn quota trên Kaggle/Colab) | `PENDING_VERIFICATION` |
| **Bản phát hành Hugging Face của tác giả đầu** | *(chưa khai báo trong YAML)* | `https://huggingface.co/datasets/rayguan/HallusionBench` — ứng viên revision đang dùng trong runbook của Thành viên B: `2067dfdce5abc7efdbb0927f4721ae282f0b6c51` (cần đối chiếu SHA-256 của `HallusionBench.json` với `tianyi-lab/HallusionBench` trước khi xin trưởng nhóm chốt) | `PENDING_VERIFICATION` |
| **License** | `null` | **BSD-3-Clause** (khai báo tại `tianyi-lab/HallusionBench` và metadata `rayguan/HallusionBench`) | Chờ trưởng nhóm duyệt cập nhật YAML |
| **Revision** | `null` | Chưa khóa chính thức trong `configs/datasets.yaml` | `PENDING_VERIFICATION` |

> **Yêu cầu P0 gửi nhóm trưởng:** Sau khi chạy lệnh audit ở Mục 4 trên bản tải về thực tế, gửi `source URL`, `commit/revision`, `json_sha256`, `license` (`BSD-3-Clause`) và báo cáo thống kê để trưởng nhóm duyệt khóa `configs/datasets.yaml`.

---

## 2. Schema gốc (`HallusionBench.json`) & Hợp đồng chuẩn hóa (A → B → C)

### 2.1. Các trường trong file gốc `HallusionBench.json`

| Trường gốc | Bắt buộc | Kiểu / Giá trị hợp lệ | Ý nghĩa & lưu ý xử lý |
|---|---|---|---|
| `category` | Có | `"VD"` hoặc `"VS"` | `VD` = Visual Dependent (bắt buộc nhìn ảnh); `VS` = Visual Supplement (bổ trợ thị giác) |
| `subcategory` | Có | Chuỗi không rỗng | Phân nhóm chi tiết (vd: `illusion`, `math`, `figure`, `chart`, `table`, `map`, `ocr`, ...) |
| `visual_input` | Có | `"0"`, `"1"`, `"2"` | `"0"` = **text-only** (KHÔNG truyền ảnh); `"1"` = ảnh gốc; `"2"` = ảnh đã chỉnh sửa |
| `set_id` | Có | Chuỗi/số không rỗng | Mã nhóm bộ câu hỏi / cặp kiểm soát |
| `figure_id` | Có | Chuỗi/số không rỗng | Mã hình trong set |
| `question_id` | Có | Chuỗi/số không rỗng | Mã câu hỏi trong set (**KHÔNG duy nhất toàn cục**) |
| `question` | Có | Chuỗi không rỗng | Câu hỏi gốc; tuyệt đối không chỉnh sửa nội dung |
| `gt_answer` | Có | `"0"` hoặc `"1"` | Nhãn chuẩn (`"1"` → `"yes"`, `"0"` → `"no"`). **Cấm rò rỉ vào prompt model** |
| `filename` | Có nếu `visual_input != "0"` | Đường dẫn tương đối hoặc `null` | Với `visual_input = "0"`, loader **luôn đặt `image_path = None`** kể cả nếu `filename` có giá trị |
| `gt_answer_details` | Tùy chọn | Chuỗi hoặc `null` | Giải thích đáp án chuẩn; chỉ dùng cho Thành viên C phân tích lỗi, **cấm đưa vào prompt** |
| `sample_note` | Tùy chọn | Chuỗi hoặc `null` | Ghi chú mẫu của tác giả; **cấm đưa vào prompt** |

### 2.2. Quy tắc tạo `sample_id` ổn định & duy nhất

Vì `question_id` bị lặp giữa các `category`, `subcategory`, `set_id`, `figure_id` và biến thể `visual_input`, Thành viên A khóa quy tắc ID ghép đồng bộ 100% với `benchmarks/hallusionbench/protocol.py` của Thành viên B:

```text
cat{category}_sub{subcategory}_set{set_id}_fig{figure_id}_q{question_id}_vis{visual_input_raw}
```

Ví dụ: `catVD_subillusion_set0_fig0_q0_vis1` và `catVS_subchart_set3_fig1_q0_vis2`.

- Trong chế độ nạp chuẩn (`HallusionBenchDataset.load()`), nếu phát hiện trùng `sample_id`, loader **báo lỗi `DuplicateSampleIdError` ngay lập tức**, không âm thầm bỏ qua hay ghi đè.

### 2.3. Hợp đồng cấu trúc nhóm / cặp bảo toàn cho Thành viên B & C

Mỗi `HallusionSample` (định nghĩa trong [`schema.py`](schema.py)) bảo toàn nguyên vẹn toàn bộ trường gốc và cung cấp 3 khóa cấu trúc xác định (deterministic):

| Khóa dẫn xuất | Công thức | Vai trò bàn giao |
|---|---|---|
| `group_or_pair_key` | `{category}::{subcategory}::set_{set_id}` | Khóa gom trọn nhóm/cặp kiểm soát (dùng để chọn tập pilot không bị đứt cặp, khớp `protocol.make_group_or_pair_key`) |
| `figure_group_key` | `{category}::{subcategory}::set_{set_id}::fig_{figure_id}` | Khóa gom các câu hỏi cùng một hình (phục vụ kiểm tra tính nhất quán ảnh và hỗ trợ Thành viên C đối chiếu `fAcc`) |
| `question_pair_key` | `{category}::{subcategory}::set_{set_id}::q_{question_id}` | Khóa gom cùng một câu hỏi trên các biến thể hình/text-only (hỗ trợ Thành viên C đối chiếu `qAcc`) |

> **Lưu ý bắt buộc cho Thành viên C:** Ba khóa trên là khóa cấu trúc bảo toàn liên kết dữ liệu (không tự suy diễn công thức điểm). Thành viên C vẫn **phải đối chiếu trực tiếp** cách tính `aAcc`, `fAcc`, `qAcc`, `VD/VS`, `easy/hard` với `evaluation.py` và `utils.py` gốc của `tianyi-lab/HallusionBench`.

### 2.4. Những điểm dị biệt (quirks) đã xử lý trong `dataset.py`

1. **Khác biệt bố cục thư mục ảnh giữa 2 nguồn tải:**
   - Giải nén từ `hallusion_bench.zip` (GitHub/Google Drive): `<DATA_ROOT>/hallusion_bench/VD/...` hoặc `<DATA_ROOT>/VD/...`
   - Tải từ `rayguan/HallusionBench` (Hugging Face): `<DATA_ROOT>/data/VD/...` và `<DATA_ROOT>/data/VS/...`
   - `resolve_image_path()` hỗ trợ tìm kiếm xác định trên cả ba bố cục mà không sửa `HallusionBench.json`.
2. **Lỗi phân biệt chữ hoa/thường `.PNG` vs `.png` trên Linux (Kaggle/Colab):**
   - Một số ảnh `VS/chart` có đuôi `.PNG` viết hoa trên đĩa trong khi `HallusionBench.json` ghi `.png`.
   - Loader tự khớp case-insensitive xác định trong cùng thư mục **và ghi nhận cờ `image_case_mismatch = True`** vào báo cáo audit để minh bạch hóa mọi khác biệt.
3. **Cách ly tuyệt đối trường nhãn (`FORBIDDEN_MODEL_INPUT_FIELDS`):**
   - `HallusionSample.to_model_safe_dict()` loại bỏ hoàn toàn `gt_answer`, `gt_answer_details`, `reference_answer`, `reference`, `answer`, `label`, `sample_note` trước khi bàn giao cho pipeline suy luận.

---

## 3. Cách tải dataset (NGOÀI thư mục Git)

Không bao giờ tải dataset vào bên trong repository Git. Đặt biến môi trường `DATA_ROOT` trỏ ra thư mục ngoài repo.

### 3.1. Trên Kaggle / Colab / Linux (khuyến nghị dùng `rayguan/HallusionBench` đã khóa commit)

```python
from pathlib import Path
from huggingface_hub import snapshot_download

data_root = Path("/kaggle/working/data/HallusionBench")  # hoặc /content/data/HallusionBench
rev = "2067dfdce5abc7efdbb0927f4721ae282f0b6c51"
snapshot_download(
    repo_id="rayguan/HallusionBench",
    repo_type="dataset",
    revision=rev,
    local_dir=str(data_root),
)
```

### 3.2. Trên Windows PowerShell (máy cá nhân, ngoài thư mục repo)

```powershell
$env:DATA_ROOT = "D:\datasets\HallusionBench"
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='rayguan/HallusionBench', repo_type='dataset', revision='2067dfdce5abc7efdbb0927f4721ae282f0b6c51', local_dir=r'D:\datasets\HallusionBench')"
```

---

## 4. Lệnh chạy Audit thực tế & Chốt danh sách Pilot cố định

### 4.1. Chạy unit test Phần A (CPU-only, không cần tải dataset thật)

```powershell
# Với pytest (sau khi cài requirements.txt)
python -m pytest -q benchmarks/hallusionbench/tests/test_dataset.py

# Hoặc chạy trực tiếp bằng stdlib unittest (không cần cài thêm gói)
python -m unittest discover -s benchmarks/hallusionbench/tests -p "test_dataset.py" -v
```

### 4.2. Chạy audit toàn bộ dataset thật (bắt buộc trước khi đóng Gate G1)

```powershell
python -m benchmarks.hallusionbench.dataset `
  --mode audit `
  --dataset-json "$env:DATA_ROOT\HallusionBench.json" `
  --data-root "$env:DATA_ROOT" `
  --dataset-source "rayguan/HallusionBench" `
  --dataset-revision "2067dfdce5abc7efdbb0927f4721ae282f0b6c51" `
  --output "tmp\hallusionbench_audit_report.json"
```

*(Trên Linux/Kaggle/Colab dùng dấu `\` để xuống dòng và `$DATA_ROOT/HallusionBench.json`.)*

### 4.3. Xuất & kiểm chứng danh sách Pilot ID cố định (giữ trọn nhóm/cặp)

```powershell
python -m benchmarks.hallusionbench.dataset `
  --mode export-pilot `
  --dataset-json "$env:DATA_ROOT\HallusionBench.json" `
  --data-root "$env:DATA_ROOT" `
  --dataset-source "rayguan/HallusionBench" `
  --dataset-revision "2067dfdce5abc7efdbb0927f4721ae282f0b6c51" `
  --target-groups 6 `
  --output "benchmarks\hallusionbench\pilot_ids.json"

python -m benchmarks.hallusionbench.dataset `
  --mode verify-pilot `
  --dataset-json "$env:DATA_ROOT\HallusionBench.json" `
  --data-root "$env:DATA_ROOT" `
  --pilot-file "benchmarks\hallusionbench\pilot_ids.json"
```

Quy tắc chọn pilot mặc định (`select_frozen_pilot`) bảo đảm:
- Luôn giữ **trọn vẹn từng nhóm `(category, subcategory, set_id)`** — không cắt đôi cặp kiểm soát nên `aAcc`, `fAcc`, `qAcc` đều tính được.
- Bao phủ đồng thời: `VD`, `VS`, mẫu `text-only` (`visual_input="0"`), ảnh gốc (`visual_input="1"`), và ảnh chỉnh sửa (`visual_input="2"`).
- Ghi kèm `pilot_ids_sha256` để Thành viên B và Thành viên C đối chiếu chạy đúng cùng một tập mẫu.

---

## 5. Bảng thống kê kiểm chứng thực tế (đã chạy `--mode audit` ngày 20/11/2026)

> **Nguồn bằng chứng:** Chạy trực tiếp `python -m benchmarks.hallusionbench.dataset --mode audit` trên Google Colab (Linux x86_64, Python 3.13.16, Pillow 11.3.0, Git commit `a8130ebe6090604cb0370c757314ba2c66790407`), lưu tại `audit_and_pilot/hallusionbench_audit_report.json`.

| Chỉ số kiểm chứng | Giá trị thực đo (`--mode audit`) | Ghi chú |
|---|---|---|
| Nguồn & revision thực tải | `rayguan/HallusionBench` @ `2067dfdce5abc7efdbb0927f4721ae282f0b6c51` | Bản Hugging Face của tác giả đầu; chờ trưởng nhóm khóa `configs/datasets.yaml` |
| License | `BSD-3-Clause` | Khớp cả `tianyi-lab/HallusionBench` và `rayguan/HallusionBench` |
| `HallusionBench.json` SHA-256 | `ca6e0fb677dd56e5fb0e2700d06c58d01f8a6ae66e0f1773b2ac48c0a095d051` | Khóa toàn vẹn dữ liệu JSON |
| Tổng số dòng (`total_raw_records`) | **1.129** (`valid_records: 1.129`, `schema_error_records: 0`) | Khớp bài báo CVPR 2024; `is_clean = true` |
| Số `sample_id` trùng (`duplicate_sample_id_count`) | **0** | Quy tắc ID ghép duy nhất 100% |
| Phân bố `visual_input` (`"0"` / `"1"` / `"2"`) | `"0"` (text-only): **178** / `"1"` (ảnh gốc): **447** / `"2"` (ảnh sửa): **504** | Tổng mẫu có ảnh: **951**; `text_only_with_ignored_filename: 0` |
| Số ảnh thực tồn tại / thiếu / lỗi hỏng (PIL) | **346** ảnh duy nhất / **0** thiếu / **0** hỏng | Khớp 346 ảnh bài báo CVPR 2024 |
| Số ảnh lệch hoa/thường `.PNG` vs `.png` | **34 mẫu** (các file `VS/chart/*.PNG`) | Tự khớp case-insensitive xác định trên Linux |
| Số nhóm set / nhóm hình / cặp câu hỏi | **167** set / **403** figure (**346** visual figures) / **455** question pairs | Kích thước `question_pair`: `1: 3`, `2: 256`, `3: 170`, `4: 26` |
| Phân bố `VD` / `VS` và `subcategory` | **VD: 591** / **VS: 538** (`chart: 206`, `figure: 80`, `illusion: 144`, `map: 96`, `math: 108`, `ocr: 170`, `table: 155`, `video: 170`) | Phân bố `gt_answer`: `no ("0"): 645`, `yes ("1"): 484` |
| Pilot manifest ([pilot_ids.json](pilot_ids.json)) | **64 mẫu** (**6** nhóm trọn vẹn: `VS::chart::set_0..3`, `VD::illusion::set_0..1`; `52 VS / 12 VD`; `24` text-only / `26` ảnh gốc / `14` ảnh sửa) | `pilot_ids_sha256`: `9fa51b50ee851ac0fc46314620dd1d51cc2c34b83687759f0b97dce0b3649386` |

---

## 6. Checklist nghiệm thu Gate G1 (Thành viên A)

- [x] `schema.py`: định nghĩa đầy đủ trường gốc, khóa ghép `sample_id`, khóa cấu trúc nhóm/hình/cặp, kiểm tra lỗi schema và chặn rò rỉ nhãn.
- [x] `dataset.py`: loader read-only không tự tải mạng, hỗ trợ `visual_input=0/1/2`, kiểm tra tồn tại & toàn vẹn ảnh, audit toàn diện và chọn/kiểm chứng tập pilot trọn nhóm.
- [x] `tests/test_dataset.py`: unit test CPU-only cho record hợp lệ, thiếu trường, ảnh không tồn tại, ảnh hỏng, mẫu text-only, lệch chữ hoa/thường `.PNG`, trùng `sample_id`, lỗi liên kết nhóm/cặp, pilot trọn nhóm và CLI.
- [x] `DATASET_NOTES.md`: tài liệu đối chiếu nguồn, schema, quirks, hướng dẫn tái tạo và bảng thống kê kiểm chứng thực tế.
- [x] Đã chạy `--mode audit` trên dữ liệu thật ngoài repo (`is_clean: true`), điền bảng Mục 5 và khóa [pilot_ids.json](pilot_ids.json) (64 mẫu trọn 6 nhóm).
- [ ] Gửi bằng chứng nguồn/revision/license/checksum trong Mục 5 cho nhóm trưởng duyệt khóa `configs/datasets.yaml` (Blocker P0).
