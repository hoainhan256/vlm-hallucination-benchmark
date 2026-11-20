# KẾ HOẠCH & PHÂN CÔNG CÔNG VIỆC — NHÓM 2 HALLUSIONBENCH

> **Bản V2.0 — dùng để giao việc và theo dõi thực hiện**  
> **Ngày cập nhật:** 19/11/2026  
> **Đề tài chung:** Xây dựng pipeline dữ liệu đa phương thức phục vụ kiểm chứng thông tin và giảm ảo giác trong hệ thống trí tuệ nhân tạo.  
> **Nhóm phụ trách:** Nhóm 2 — **HallusionBench**, gồm **03 thành viên**.  
> **Repository:** https://github.com/hoainhan256/vlm-hallucination-benchmark  
> **Branch làm việc:** `benchmark/hallusionbench`  
> **Thư mục code được ưu tiên sở hữu:** `benchmarks/hallusionbench/`  
> **Trạng thái tài liệu:** **Kế hoạch đề xuất**; các quyết định về model/dataset revision, dtype, prompt, scoring, quyền sửa mã dùng chung và chạy official phải được người phụ trách dự án xác nhận.

---

## 0. ĐỌC PHẦN NÀY TRƯỚC — NHÓM MÌNH CẦN LÀM GÌ?

**Mục tiêu riêng của nhóm 2:** Xây dựng và thực nghiệm **một benchmark duy nhất là HallusionBench** trên ba mô hình đã được nhóm trưởng đề xuất. Từ kết quả thật, nhóm phải chỉ ra **mô hình sai ở đâu, loại lỗi gì, bằng chứng cụ thể**, sau đó có cơ sở nghiên cứu cách **phát hiện và giảm ảo giác**.

**Không làm:** POPE, MMHal-Bench, website toàn dự án, hay tự sửa `models/`, `configs/`, `scripts/` và các phần chung khi chưa được phân công/duyệt.

**Ba mô hình cần phối hợp đánh giá:**

| Model ID trong cấu hình chung | Mô hình | Vai trò |
|---|---|---|
| `qwen3_vl` | `Qwen/Qwen3-VL-4B-Instruct` | Mô hình nghiên cứu chính theo đề xuất nhóm trưởng |
| `smolvlm2` | `HuggingFaceTB/SmolVLM2-2.2B-Instruct` | Baseline gọn nhẹ, ưu tiên thử đầu tiên |
| `phi4_multimodal` | `microsoft/Phi-4-multimodal-instruct` | Mô hình so sánh kiến trúc/hành vi |

**Quy trình bắt buộc về logic:**

```text
Xác minh dữ liệu + giao thức chấm HallusionBench
                     ↓
Xây loader / chuẩn bị input / scorer + unit tests
                     ↓
Chạy PILOT bằng mô hình thật (ưu tiên SmolVLM2)
                     ↓
Chạy baseline chính thức khi cấu hình được phê duyệt
                     ↓
Tính metric gốc + trích xuất case sai có bằng chứng
                     ↓
Phân tích nguyên nhân → phát hiện lỗi → thử cải thiện
                     ↓
Báo cáo kết quả / bàn giao artifact cho nhóm tích hợp
```

> **Định nghĩa “đã làm xong”:** Có **file/code + test + log/đầu ra + cách chạy lại**. Một đoạn giải thích hoặc screenshot không thay thế kết quả thực nghiệm.

---

## 1. BẢNG GIAO VIỆC — CHIA 3 NGƯỜI KHÔNG TRÙNG NHAU

**Điền tên sau khi nhóm họp:**

| Mã | Người nhận | Vai trò chính | Sở hữu file/chức năng | Đầu ra bắt buộc | Phụ thuộc |
|---|---|---|---|---|---|
| **A — DATA** | **[Điền tên]** | Dataset, cấu trúc mẫu, cặp kiểm soát | `dataset.py`, `schema.py`, tài liệu dữ liệu và test | Loader hợp lệ, thống kê + danh sách ID pilot | Không phụ thuộc inference |
| **B — INFERENCE** | **[Điền tên]** | Kaggle, prompt, model integration, chạy baseline | `protocol.py`, `inference.py`, notebook/tài liệu chạy được duyệt | 3 run hoặc 3 báo cáo `BLOCKED`, raw predictions, môi trường | A hoàn thành schema; model loader dùng chung sẵn sàng |
| **C — EVALUATION** | **[Điền tên]** | Scorer, kiểm thử metric, phân tích case, báo cáo | `scoring.py`, `analysis.py`, tests, bảng kết quả | Metric đúng protocol, bảng case, báo cáo phân tích | A cung cấp nhóm/cặp; B cung cấp predictions |

**Người điều phối (có thể là một trong A/B/C):** **[Điền tên]**. Nhiệm vụ: xác nhận đầu ra, liên hệ nhóm trưởng khi có blocker, kiểm PR và tổng hợp báo cáo; đây **không phải người thứ tư**.

### Ranh giới rõ ràng

- **A không phải chạy 3 model**, và không tự chọn lại protocol chấm.
- **B không được tự sửa nhãn/dataset** để làm inference chạy; không tự đổi dtype hay lượng tử hóa trái chính sách.
- **C không tự thay câu trả lời gốc hoặc xóa mẫu lỗi** để “làm đẹp” điểm; không mặc định mọi câu sai đều là hallucination.
- Nếu công việc nằm ngoài `benchmarks/hallusionbench/` hoặc phụ thuộc file chung, **tạo yêu cầu với nhóm trưởng** thay vì sửa trực tiếp.
- Mọi thành viên đều review đầu vào/đầu ra của nhau tại các cổng nghiệm thu, nhưng **chỉ một người chịu trách nhiệm chính** cho mỗi module.

---

## 2. CÁC CỔNG NGHIỆM THU (GATES) VÀ PHỤ THUỘC

| Gate | Cần xác nhận điều gì? | Người chính | Khi nào được đi tiếp? |
|---|---|---|---|
| **G0 — Đồng bộ repo** | Branch đã lấy phần hạ tầng cần thiết từ `main` một cách an toàn | Điều phối + nhóm trưởng | Có commit, test cơ sở, không mất thay đổi |
| **G1 — Dataset locked** | Nguồn/revision, ảnh, nhãn, trường dữ liệu, nhóm/cặp, license được xác minh | A | Loader + schema tests pass |
| **G2 — Protocol locked** | Prompt, xử lý visual/text-only, câu trả lời gốc, bộ chấm và cách chuẩn hóa được xác minh | B + C | Unit tests + fixture pass; đã duyệt cách chấm |
| **G3 — Kaggle pilot** | Có inference thật, raw output, metadata, đủ mẫu pilot, không nhầm lỗi kỹ thuật thành dự đoán | B | Ít nhất 1 model chạy thành công trên tập pilot được chốt |
| **G4 — Baseline** | Ba model có run hợp lệ hoặc có blocker được chứng minh; official được duyệt riêng | B + C | Artifact đầy đủ; không gọi pilot là official |
| **G5 — Phân tích lỗi** | Có case thật gắn `sample_id`, ảnh/câu hỏi, raw response, bằng chứng phân loại | C | Báo cáo có thể truy ngược về run |
| **G6 — Phát hiện/cải thiện** | Phương pháp mới có thiết kế, test riêng, đối chứng và kết quả tái lập | C chủ trì, cả nhóm hỗ trợ | Không đánh giá trước/sau trên điều kiện khác nhau mà không nêu rõ |

**Không chờ cả 3 người xong mới bắt đầu:** A xác minh dataset, C nghiên cứu metric, B chuẩn bị Kaggle và đọc hướng dẫn model **có thể diễn ra song song**. Chỉ ghép hệ thống khi các interface đã thống nhất.

---

## 3. THÀNH VIÊN A — DATASET & LOADER

### 3.1. Việc phải làm (theo thứ tự)

- [ ] Đọc tài liệu HallusionBench gốc: bài báo, repository, file dữ liệu, luật sử dụng ảnh/dữ liệu.
- [ ] Đối chiếu nguồn gốc với `configs/datasets.yaml` (hiện khai báo `tianyi-lab/HallusionBench`; cần xác minh nguồn/revision có thể truy cập thực sự).
- [ ] Ghi **source URL, commit/revision, hash/checksum nếu phù hợp, license**; không chỉ ghi “đã tải được”.
- [ ] Kiểm tra bằng code **số dòng, số ảnh có thực, số mẫu text-only, số nhóm/cặp, trường thiếu, file ảnh lỗi, ID trùng**.
- [ ] Xác định chính xác trường gốc như `category`, `subcategory`, `visual_input`, `set_id`, `figure_id`, `question_id`, `question`, `gt_answer`, `filename` từ **phiên bản dữ liệu đã chốt**.
- [ ] Với `visual_input = 0`, giữ đúng **mẫu chỉ dùng văn bản**; không tự chèn hình không có trong mẫu.
- [ ] Định nghĩa `sample_id` **ổn định và duy nhất** theo schema thực tế. Nếu nguồn không có trường `sample_id`, không mặc định dùng `question_id` riêng lẻ (có thể trùng); lập quy tắc ID ghép có test.
- [ ] Bảo toàn liên kết **hình — nhóm câu hỏi — cặp kiểm soát — biến thể dễ/khó**; không shuffle rồi mất mapping.
- [ ] Chọn **một danh sách sample ID pilot cố định** gồm các nhóm/cặp trọn vẹn; tránh chọn ngẫu nhiên từng câu khiến qAcc/fAcc không đánh giá được.
- [ ] Viết unit test cho record tốt, thiếu trường, ảnh không tồn tại, mẫu text-only, ID trùng, liên kết nhóm/cặp lỗi.
- [ ] Viết `DATASET_NOTES.md` hoặc phần tài liệu tương đương ghi cách tải, quyền truy cập, schema và thống kê kiểm chứng.

### 3.2. File đề xuất do A sở hữu

```text
benchmarks/hallusionbench/dataset.py
benchmarks/hallusionbench/schema.py
benchmarks/hallusionbench/DATASET_NOTES.md
benchmarks/hallusionbench/tests/test_dataset.py
```

> Đây là **cấu trúc đề xuất**, chưa phải file đang tồn tại. Đổi tên khi nhóm thống nhất; không tạo lại interface chung `benchmarks/base.py`.

### 3.3. Định dạng dữ liệu nội bộ đề xuất (không phải schema gốc)

```python
# Đây chỉ là ví dụ mô tả hợp đồng giữa A, B, C.
# Tên trường cuối cùng phải được xác minh từ dataset đã khóa.
{
    "sample_id": "stable_unique_id",
    "question": "...",
    "image_path": None,         # text-only thì None
    "visual_input": False,
    "reference_answer": "yes", # TUYỆT ĐỐI không đưa vào prompt model
    "category": "VD_or_VS",
    "set_id": "...",
    "figure_id": "...",
    "question_id": "...",
    "group_or_pair_key": "...", # quy tắc nhóm đã được kiểm chứng
}
```

**Lưu ý:** `group_or_pair_key` chỉ là field nội bộ minh họa; **không tự suy diễn** công thức gộp chỉ từ tên trường. Cần xác minh mã đánh giá chính thức trước khi C dùng.

### 3.4. A được xem là hoàn thành khi

- [ ] Có thống kê dataset thực tế, không chỉ chép số 346 ảnh / 1.129 câu từ bài báo.
- [ ] Có test chứng minh không mất mẫu/nhãn/cặp và sample ID không trùng.
- [ ] B đọc được mẫu đã chuẩn hóa, bao gồm **image-dependent** và **text-only**.
- [ ] C nhận được mapping chính xác để tính score theo nhóm/cặp.
- [ ] Có đường dẫn nguồn/revision và hướng dẫn tái tạo loader.

**Ghi chú kiểm chứng:** Bài báo gốc nêu **346 ảnh và 1.129 câu hỏi**, nhưng repo HallusionBench có bảng snapshot cũ hơn; **không mặc định số lượng của file tải về khớp bài báo nếu không khóa phiên bản**.

---

## 4. THÀNH VIÊN B — KAGGLE, PROMPT & MODEL INFERENCE

### 4.1. Việc phải làm (theo thứ tự)

- [ ] Tạo Kaggle Notebook riêng phục vụ **HallusionBench**; chọn GPU khả dụng (ưu tiên T4 ×2 nếu tài khoản được cấp).
- [ ] Chạy `nvidia-smi` và ghi **GPU count, VRAM từng GPU, driver**. Hai T4 có bộ nhớ **tách biệt**, không có một khối VRAM 32 GB tự động.
- [ ] Chạy kiểm tra `torch.cuda.is_available()`, `torch.cuda.device_count()`, `torch.cuda.is_bf16_supported()`; ghi phiên bản `torch`, `transformers`, CUDA, Python.
- [ ] Xác minh cách nạp/processor/prompt của từng model từ model card/nhà phát hành và **mã model loader chung đã được phê duyệt**.
- [ ] Chỉ sử dụng đúng mẫu mà A cung cấp. Không để `reference_answer` hoặc `gt_answer_details` rò rỉ vào prompt mô hình.
- [ ] Xử lý riêng tình huống **có ảnh** và **không có ảnh** theo protocol; lưu prompt/template thực tế.
- [ ] Kiểm tra tải model và 1 sample trước, sau đó chạy pilot cố định (ví dụ **20–50 câu**, nhưng phải giữ trọn các nhóm/cặp liên quan).
- [ ] Ưu tiên thử **SmolVLM2** trước, tiếp theo **Qwen3-VL**, sau cùng **Phi-4**. Đây là thứ tự triển khai kỹ thuật, không thay thế danh sách 3 model được giao.
- [ ] Lưu **raw model response** trước mọi chuẩn hóa; mỗi record có ID, trạng thái, cấu hình và model tương ứng.
- [ ] Thử resume, chống trùng mẫu; ghi rõ `OOM`, lỗi import, thiếu GPU hoặc timeout thay vì xóa sample.
- [ ] Bàn giao cho C các file prediction chuẩn + run metadata; không gửi mỗi screenshot điểm số.

### 4.2. CẢNH BÁO BẮT BUỘC VỀ GPU T4 & CHÍNH SÁCH DỰ ÁN

- Kaggle T4 ×2: thông thường **2 GPU, 16 GB VRAM/GPU**; khả năng cấp máy và hạn mức thay đổi theo tài khoản/từng lúc.
- **T4 không hỗ trợ BF16 native**; trong khi `configs/models.yaml` hiện ghi `target_dtype: bfloat16`, `approved_dtypes: []`, `dtype_status: PENDING_VERIFICATION` và chính sách **cấm tự fallback/quantization**.
- Vì vậy, **chưa được tự sửa thành FP16 rồi gọi đó là official run**. Phải gửi nhóm trưởng đề xuất phê duyệt **FP16 như cấu hình riêng**, hoặc dùng phần cứng BF16 phù hợp. Nếu không được duyệt thì ghi `BLOCKED`.
- **Không cam kết cả ba model chạy nổi** trên T4. Phi-4 có thể gặp ràng buộc VRAM hoặc backend; CPU offload chỉ khi được chính sách cho phép và ghi rõ.
- Trước khi chạy chính thức, model revision, processor revision, dtype, config generation (`max_new_tokens`, `do_sample`, v.v.) và tập mẫu phải được **khóa**.
- Không dùng việc giảm batch, đổi prompt, đổi dtype hoặc đổi image preprocessing giữa các model/run mà không ghi nhận và kiểm tra ảnh hưởng so sánh.

### 4.3. File đề xuất do B sở hữu

```text
benchmarks/hallusionbench/protocol.py
benchmarks/hallusionbench/inference.py
benchmarks/hallusionbench/KAGGLE_RUNBOOK.md
benchmarks/hallusionbench/tests/test_protocol.py
```

Nếu muốn lưu Kaggle Notebook bên ngoài `benchmarks/hallusionbench/`, phải được nhóm trưởng đồng ý vị trí (`notebooks/` là khu vực dùng chung). Tránh commit checkpoint/model cache.

### 4.4. Các thao tác kiểm tra Kaggle có thể chạy ngay

```python
import sys, torch
print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
print("BF16 supported:", torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i),
          round(torch.cuda.get_device_properties(i).total_memory / 2**30, 2), "GiB")
```

```bash
nvidia-smi
```

**Đây chỉ là kiểm tra môi trường**, không phải lệnh chạy HallusionBench. Hiện tại `scripts/run_benchmark.py --mode check-config` trong repo **chỉ kiểm cấu hình**, chưa tải model, chưa suy luận và chưa chấm điểm.

### 4.5. B được xem là hoàn thành khi

- [ ] Có notebook/runbook với lệnh thực sự chạy được hoặc log lỗi rõ ràng.
- [ ] Mỗi model có `run_id` và trạng thái **`COMPLETED`, `BLOCKED` hoặc `FAILED` đúng sự thật**; không đánh đồng pilot với official.
- [ ] Với run `COMPLETED`: số sample đầu vào/đầu ra khớp phạm vi chạy; không trùng `sample_id`.
- [ ] Có raw predictions, cấu hình, Git SHA, phiên bản model/dataset, GPU/dtype và thời gian.
- [ ] C có thể dùng output của B để chấm mà không phải nạp lại model.

---

## 5. THÀNH VIÊN C — SCORING, KIỂM CHỨNG, PHÂN TÍCH LỖI

### 5.1. Việc phải làm (theo thứ tự)

- [ ] Đọc **mã gốc** `evaluation.py` và tài liệu công bố của HallusionBench; xác minh cách chuẩn hóa output và cách chấm, **không tự định nghĩa metric chỉ từ tên gọi**.
- [ ] Xây parser cho câu trả lời Yes/No và xác định xử lý trường hợp giải thích dài, mơ hồ, từ chối, không trả lời hoặc format sai.
- [ ] Nếu protocol chấm dùng **GPT-4 judge**, phải xác minh version/chi phí/quyền truy cập; nếu dùng chấm rule-based hoặc chấm tay, gắn rõ nhãn **alternative evaluation**, không giả nhận tương đương leaderboard gốc.
- [ ] Bảo toàn **VD (Visual Dependent)**, **VS (Visual Supplement)**, easy/hard và các nhóm/cặp khi tính điểm.
- [ ] Tính và báo cáo **aAcc (question), fAcc (figure), qAcc (question pair)** cùng các chỉ số phân tách có trong protocol đã chốt.
- [ ] Viết fixture kiểm thử: toàn đúng, toàn sai, một câu trong cặp sai, một câu trong hình sai, thiếu cặp, duplicate sample, lỗi parse, thiếu prediction.
- [ ] Đối chiếu scorer tự viết với output từ script gốc trên **cùng fixture/sample**; nếu lệch, phải ghi rõ và xử lý **trước khi báo cáo điểm chính thức**.
- [ ] Lấy raw predictions do B xuất và tạo bảng case sai, không chỉ ghi tỉ lệ tổng.
- [ ] Phân biệt **incorrect prediction** với **hallucination**, **format violation**, **technical error**, **label/data error**, **undetermined**.
- [ ] Phân tích ít nhất một số trường hợp **cùng một sample/cặp nhưng ba model trả lời khác nhau**; đưa bằng chứng ảnh/câu hỏi/câu trả lời gốc.
- [ ] Tổng hợp báo cáo baseline; sau đó cùng A/B đề xuất detector hoặc bước kiểm chứng giảm lỗi.

### 5.2. Metric bắt buộc giữ đúng ý nghĩa

| Metric | Đánh giá | Điều không được làm |
|---|---|---|
| `aAcc` — Accuracy per Question | Tỉ lệ câu hỏi được chấm đúng | Không gọi đây là toàn bộ kết quả HallusionBench |
| `fAcc` — Accuracy per Figure | Tính đúng/nhất quán ở cấp hình theo protocol | Không tách các câu cùng hình thành mẫu độc lập khi gộp điểm |
| `qAcc` — Accuracy per Question Pair | Tính đúng ở cấp cặp câu hỏi/điều kiện thị giác | Không tính qAcc từ một nửa cặp hoặc tự chế công thức |
| Easy/Hard, VD/VS | Phân tích theo các phân nhóm nguồn | Không trộn các tập con mà không ghi rõ mẫu số |

### 5.3. File đề xuất do C sở hữu

```text
benchmarks/hallusionbench/scoring.py
benchmarks/hallusionbench/analysis.py
benchmarks/hallusionbench/EVALUATION_NOTES.md
benchmarks/hallusionbench/tests/test_scoring.py
```

### 5.4. Bảng ghi ca lỗi (sau khi có kết quả THẬT)

| `run_id` | `sample_id` / group | Model | Ảnh & câu hỏi | Ground truth | Raw response | Chấm đúng/sai | Loại lỗi & chứng cứ |
|---|---|---|---|---|---|---|---|
| [Điền sau] | [Điền sau] | [Điền sau] | [Link có quyền truy cập] | [Điền sau] | [Điền sau] | [Điền sau] | [Điền sau] |

**Không tự động gắn mọi câu sai là ảo giác.** Cần giải thích cụ thể câu trả lời nào mâu thuẫn với bằng chứng trong ảnh, hoặc vì sao chưa thể kết luận.

### 5.5. C được xem là hoàn thành khi

- [ ] Scorer có unit test và đã được đối chiếu với protocol/implementation gốc.
- [ ] Có metrics tính từ **dữ liệu dự đoán thật** với `run_id`/config/hash truy nguyên được.
- [ ] Có bảng phân loại ca lỗi và các case minh chứng kiểm chứng được.
- [ ] Có báo cáo so sánh 3 model trên cùng benchmark và chỉ ra cả hạn chế của phương pháp chấm.
- [ ] Có hướng đề xuất phát hiện/giảm lỗi với tiêu chí đánh giá rõ ràng; chưa chạy thì ghi `PROPOSED`, không báo là cải thiện thành công.

---

## 6. QUY ƯỚC PHỐI HỢP GIỮA 3 NGƯỜI

### 6.1. Hợp đồng dữ liệu (phải chốt trước khi code ghép)

**A bàn giao cho B và C:**

- Danh sách sample ID ổn định, metadata gốc, image path hoặc text-only, quan hệ group/pair, nguồn/revision.
- Danh sách pilot đóng băng, phạm vi đầy đủ và thống kê kiểm tra.

**B bàn giao cho C:**

- `sample_id`, `model_id`, `raw_prediction`, `status`, `run_config_hash`, thời gian và lỗi kỹ thuật nếu có.
- Mỗi model một run riêng, cùng danh sách sample khi so sánh.

**C bàn giao cho nhóm trưởng/website:**

- `metrics.json`, bảng case có `run_id`/`sample_id`, giải thích metric và các báo cáo.
- Không tự biến kết quả script tạm hoặc case minh họa thành số liệu benchmark chính thức.

### 6.2. Nơi lưu kết quả theo chính sách dự án

```text
results/<experiment_id>/<run_id>/
├── predictions.jsonl
├── metrics.json
├── run_config.yaml
├── run_metadata.json
├── errors.jsonl
├── run_status.json
└── manifest.json
```

**Bộ 7 artifact này được mô tả trong `docs/result_schema.md` trên `main`.** Thực nghiệm phải lưu đủ các tệp cần thiết; không push dữ liệu nặng, model weight, token hoặc cache lên Git. Nếu policy không cho commit trong `results/`, lưu ở kho artifact được nhóm trưởng chấp thuận và dẫn link/manifest trong báo cáo.

### 6.3. Quy tắc làm việc trên branch chung

**Chỉ 1 branch benchmark chung, nhưng 3 người phải tránh cùng sửa 1 file.** Khuyến nghị mỗi người có branch con theo tên việc rồi mở PR về `benchmark/hallusionbench` nếu nhóm trưởng cho phép; ví dụ:

```text
benchmark/hallusionbench              # branch tích hợp nhóm
    ├── task/hallusion-data            # A
    ├── task/hallusion-inference       # B
    └── task/hallusion-evaluation      # C
```

Nếu nhóm trưởng yêu cầu **chỉ dùng đúng một branch**, các thành viên vẫn commit riêng vào **file thuộc phần mình**, `git pull --ff-only` trước khi push và không dùng `git push --force`.

**Vùng có quyền sửa mặc định:** `benchmarks/hallusionbench/`. Thay đổi `models/`, `configs/`, `scripts/`, `benchmarks/base.py`, `vlm_benchmark/`, tài liệu policy, `tests/` dùng chung hoặc `notebooks/` phải báo người phụ trách tích hợp trước.

### 6.4. Các lệnh Git mẫu (PowerShell)

```powershell
git clone https://github.com/hoainhan256/vlm-hallucination-benchmark.git
cd .\vlm-hallucination-benchmark
git fetch --prune origin
git switch --track origin/benchmark/hallusionbench
git status
```

Nếu local branch đã tồn tại, dùng `git switch benchmark/hallusionbench` thay cho `--track`.

```powershell
# Kiểm tra hạ tầng sau khi nhánh đã được đồng bộ có review
python -m pip install -r requirements.txt
python scripts/check_repository.py
python scripts/check_config.py --mode config
python scripts/run_benchmark.py --mode check-config
python -m pytest -q
```

> **Các lệnh này không chạy inference.** Tại thời điểm kiểm tra, branch HallusionBench chưa mang các script/hạ tầng đang có ở `main`, nên **phải đồng bộ branch an toàn trước**. Dữ liệu local chưa commit phải được bảo toàn; không `reset --hard` hoặc `force push` để giải quyết xung đột.

```powershell
# Ví dụ commit phạm vi A (điều chỉnh path theo file đã thống nhất)
git status
git add benchmarks/hallusionbench/dataset.py benchmarks/hallusionbench/schema.py
git diff --cached
git commit -m "feat(hallusionbench): add dataset loader and schema validation"
git push origin benchmark/hallusionbench
```

### 6.5. Checklist Pull Request / commit bàn giao

- [ ] Chỉ thay đổi file thuộc phạm vi được phân công hoặc đã được chấp thuận.
- [ ] Mô tả rõ **đã triển khai gì / chưa triển khai gì**.
- [ ] Nêu source/revision/prompt/metric thay đổi (nếu có).
- [ ] Liệt kê cách chạy test, kết quả test và môi trường.
- [ ] Có log thực nghiệm hoặc fixture minh họa **đúng nhãn** (không nói dữ liệu synthetic là kết quả thật).
- [ ] Không lộ API key, trọng số, dataset lớn, đường dẫn máy cá nhân.
- [ ] Chỉ merge khi người review đã xem; không sửa code nhóm khác để vượt lỗi.

---

## 7. LỘ TRÌNH TRIỂN KHAI THỰC TẾ (KHÔNG GÁN DEADLINE KHI CHƯA THỐNG NHẤT)

| Chặng | A — DATA | B — INFERENCE | C — EVALUATION | Điều kiện chốt |
|---|---|---|---|---|
| **Chặng 0: Khởi động** | Đọc dataset nguồn | Chuẩn bị Kaggle, đọc model card | Đọc scorer nguồn | Thống nhất scope và quyền sửa file |
| **Chặng 1: Có module độc lập** | Loader + schema test | Prompt + 1-sample smoke test khi có model loader | Metric/parser + synthetic tests | G1, G2 |
| **Chặng 2: Pilot** | Pilot IDs, audit mẫu | SmolVLM2 pilot, rồi model khác nếu được duyệt | Pilot metric + đối chiếu script gốc | G3 |
| **Chặng 3: Baseline** | Kiểm tính đủ của dataset | 3 model official hoặc blocker có log | Bảng metric, case lỗi, phân tích | G4, G5 |
| **Chặng 4: Hướng cải thiện** | Hỗ trợ phân nhóm dữ liệu | Tích hợp detector/thử cải thiện đã duyệt | Thiết kế kiểm định, so sánh before/after | G6 |

### Các mốc kiểm tra cần điền sau buổi họp

| Mốc | Hạn nộp | Người xác nhận | Trạng thái |
|---|---|---|---|
| G0: Đồng bộ branch | [Ngày] | [Tên] | ⬜ Chưa chốt |
| G1: Loader & schema | [Ngày] | [Tên] | ⬜ Chưa chốt |
| G2: Prompt & scorer | [Ngày] | [Tên] | ⬜ Chưa chốt |
| G3: Kaggle pilot | [Ngày] | [Tên] | ⬜ Chưa chốt |
| G4: Baseline | [Ngày] | [Tên] | ⬜ Chưa chốt |
| G5: Báo cáo case | [Ngày] | [Tên] | ⬜ Chưa chốt |
| G6: Detector/cải thiện | [Ngày] | [Tên] | ⬜ Chưa chốt |

---

## 8. NHỮNG VIỆC PHẢI HỎI NHÓM TRƯỞNG NGAY (BLOCKERS)

| Ưu tiên | Cần xác nhận | Vì sao chặn công việc? | Người gửi |
|---|---|---|---|
| **P0** | Ai đồng bộ `main` vào `benchmark/hallusionbench`? | Branch hiện thiếu interface và các file policy mới có trên main | Điều phối |
| **P0** | Ai chịu trách nhiệm **model loader chung** và inference interface dùng chung? | `models/base.py` hiện chỉ là abstract interface; B không thể gọi runner chính thức nếu chưa có adapter | B |
| **P0** | Dùng Kaggle T4 thì **được phép FP16 hay không**? | Config đang là BF16 chưa phê duyệt; T4 không hỗ trợ BF16 native | B |
| **P0** | Khóa model/dataset revision và phương pháp scoring nào? | Không thể công bố điểm chính thức nếu version/protocol chưa cố định | A + C |
| **P1** | Giải quyết judge GPT-4/chi phí hay chấp thuận phương án chấm khác? | Chấm khác script gốc có thể làm điểm không tương đương | C |
| **P1** | Nhóm được lưu notebook/test/manifest ngoài folder HallusionBench tại đâu? | Tránh sửa cấu trúc chung và thất lạc artifact | Điều phối |
| **P1** | Ai phát triển website và giao diện đọc artifact? | Nhóm 2 chỉ có trách nhiệm **bàn giao dữ liệu/metric/case** cho tích hợp | Điều phối |

**Quy tắc xử lý blocker:** Có bằng chứng thì ghi `BLOCKED` + mô tả + link issue/log; **không tự thay đổi protocol, model, dataset hay dtype để làm cho bằng được**.

---

## 9. TIÊU CHÍ BÁO CÁO CHO THẦY — ĐỦ BẰNG CHỨNG, KHÔNG BỊ HỎI VẶN

Khi nhóm báo cáo, trả lời được 6 câu sau:

1. **Dùng bản HallusionBench nào?** Nguồn, revision, số mẫu thực đọc, số mẫu có ảnh/text-only, nhóm/cặp.
2. **Đã thực sự chạy mô hình nào?** Model ID/revision, môi trường, dtype, số mẫu, log và run ID.
3. **Điểm được tính như thế nào?** aAcc/qAcc/fAcc, easy/hard, VD/VS, phương pháp chuẩn hóa/chấm, đối chiếu script gốc.
4. **Sai ở đâu?** Ảnh, câu hỏi, nhãn, câu trả lời gốc, loại lỗi, vì sao kết luận là ảo giác hay chưa đủ căn cứ.
5. **Có phát hiện/cải thiện được không?** Thuật toán, thước đo, điều kiện so sánh, cả ca cải thiện và ca xấu đi.
6. **Có thể chạy lại không?** Lệnh, commit, versions, prompt, config, output, manifest và hạn chế phần cứng.

### Mẫu bảng tổng hợp 3 mô hình (chỉ điền kết quả sau khi chạy)

| Model | Run ID | Scope/số mẫu | Dtype/GPU | aAcc | fAcc | qAcc | Trạng thái | Bằng chứng |
|---|---|---|---|---|---|---|---|---|
| SmolVLM2-2.2B | Chưa có | Chưa chạy | Chờ duyệt | — | — | — | `PENDING` | — |
| Qwen3-VL-4B | Chưa có | Chưa chạy | Chờ duyệt | — | — | — | `PENDING` | — |
| Phi-4 multimodal | Chưa có | Chưa chạy | Chờ duyệt | — | — | — | `PENDING` | — |

> Không cam kết điểm số trước khi chạy. Nếu chỉ chấm subset pilot, tiêu đề bảng phải ghi rõ **PILOT** và **không so sánh với leaderboard full benchmark** như một kết quả tương đương.

---

## 10. NỘI DUNG NHẮN NHÓM — CÓ THỂ COPY ĐỂ PHÂN CÔNG

> Mọi người ơi, nhóm mình chỉ phụ trách **HallusionBench** trên branch `benchmark/hallusionbench`, không làm phần POPE/MMHal của nhóm khác. Mục tiêu gần nhất là làm được dataset loader, input/model inference trên Kaggle, scorer theo protocol gốc, rồi chạy baseline 3 mô hình để phân tích lỗi thật.  
>  
> Mình đề xuất chia 3 mảng:  
> **A — Dataset:** xác minh dữ liệu, cặp câu hỏi/ảnh, viết loader và test.  
> **B — Kaggle/Inference:** chuẩn bị notebook, prompt, kết nối model loader chung, chạy pilot rồi baseline và lưu raw output.  
> **C — Scoring/Analysis:** đối chiếu metric gốc aAcc/fAcc/qAcc, viết scorer/test, thống kê các case ảo giác và báo cáo.  
>  
> Mỗi người phải bàn giao **code + test + hướng dẫn + đầu ra cụ thể**, và chỉ sửa file trong `benchmarks/hallusionbench/` nếu chưa có phê duyệt ngoài phạm vi. Trước khi chạy official phải hỏi nhóm trưởng về đồng bộ branch, model loader chung, version và GPU T4/FP16. Mọi người chốt giúp tên nhận A/B/C và mốc kiểm tra nhé.

---

## 11. TRẠNG THÁI REPOSITORY KIỂM TRA NGÀY 19/11/2026

Đã kiểm tra trực tiếp hai nhánh từ GitHub:

- `main` tại commit `aadda6f1fa5b766ca5dc32d476e7c5cf0ef1e431`: có `START_HERE.md`, `docs/guides/hallusionbench_team.md`, `benchmarks/base.py`, policy, config, checker và tests hạ tầng.
- `benchmark/hallusionbench` tại commit `ecdc1c185bf4d14ac211f6db6fb098408dd73844`: chỉ có skeleton; `benchmarks/hallusionbench/README.md` rỗng, chưa có loader/scorer/inference chạy thật.
- Ở lần so sánh trước, branch HallusionBench **behind `main` 9 / ahead 3 commits**; số đếm có thể thay đổi theo commit mới, cần kiểm tra lại lúc thực hiện merge.
- `scripts/run_benchmark.py` trên `main` là **configuration-only**, **không** nạp model, dataset, chạy inference hoặc scoring.
- `configs/models.yaml` đang có các mục revision/dtype `PENDING_VERIFICATION`; **không có run official đã được xác nhận** trong những file đã xem.

**Điều này chỉ mô tả nội dung GitHub công khai quan sát được; không phủ định code hay kết quả các thành viên có thể đang giữ ở máy/Kaggle riêng.**

---

## 12. TÀI LIỆU PHẢI ĐỌC VÀ ĐƯỜNG DẪN THAM KHẢO

**Tài liệu nội bộ dự án (nhánh main):**

1. [START_HERE.md](https://github.com/hoainhan256/vlm-hallucination-benchmark/blob/main/START_HERE.md)
2. [HallusionBench team guide](https://github.com/hoainhan256/vlm-hallucination-benchmark/blob/main/docs/guides/hallusionbench_team.md)
3. [Experiment protocol](https://github.com/hoainhan256/vlm-hallucination-benchmark/blob/main/docs/experiment_protocol.md)
4. [Dataset policy](https://github.com/hoainhan256/vlm-hallucination-benchmark/blob/main/docs/dataset_policy.md)
5. [Model policy](https://github.com/hoainhan256/vlm-hallucination-benchmark/blob/main/docs/model_policy.md)
6. [Evaluation policy](https://github.com/hoainhan256/vlm-hallucination-benchmark/blob/main/docs/evaluation_policy.md)
7. [Result schema](https://github.com/hoainhan256/vlm-hallucination-benchmark/blob/main/docs/result_schema.md)
8. [Git workflow](https://github.com/hoainhan256/vlm-hallucination-benchmark/blob/main/docs/git_workflow.md)

**Nguồn HallusionBench và phần cứng:**

- [HallusionBench GitHub chính thức](https://github.com/tianyi-lab/HallusionBench) — nguồn dữ liệu, định nghĩa `visual_input`, evaluation/metric và ví dụ schema.
- [HallusionBench evaluation.py](https://github.com/tianyi-lab/HallusionBench/blob/main/evaluation.py) — script chấm phải đối chiếu.
- [Bài báo HallusionBench, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Guan_HallusionBench_An_Advanced_Diagnostic_Suite_for_Entangled_Language_Hallucination_and_CVPR_2024_paper.pdf).
- [Kaggle Notebooks](https://www.kaggle.com/docs/notebooks) — môi trường GPU có thể được cấp.
- [Hugging Face T4 guide](https://github.com/huggingface/kernels/blob/main/kernel-builder/skills/cuda-kernels/references/t4-optimization-guide.md) — T4 không hỗ trợ BF16 native.

**Nguồn yêu cầu thầy:** Do người dùng chuyển lại trong hội thoại: chạy benchmark thật → phân tích lỗi có case → nghiên cứu phát hiện → cải thiện; website phát triển song song ở phạm vi toàn dự án. **Phân công A/B/C và thứ tự công việc trong tài liệu là đề xuất tổ chức nhóm**, chưa phải quy định nguyên văn của thầy hay nhóm trưởng.

---

**Chốt:** Nhóm 2 chỉ cần làm thật tốt HallusionBench. Khi **A có loader đáng tin**, **B có inference chạy thật**, **C có scorer đúng + case phân tích**, nhóm mới sở hữu một baseline vững để bước tiếp sang phát hiện và giảm ảo giác.
