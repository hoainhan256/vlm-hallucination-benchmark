# KẾ HOẠCH VÀ PHÂN CÔNG NHIỆM VỤ
## NHÓM 3 — THỰC NGHIỆM ĐÁNH GIÁ ẢO GIÁC THỊ GIÁC TRÊN POPE

**Phiên bản:** 1.1  
**Ngày cập nhật:** 20/11/2026  
**Đề tài chung:** Xây dựng pipeline dữ liệu đa phương thức phục vụ kiểm chứng thông tin và giảm thiểu hiện tượng ảo giác trong hệ thống trí tuệ nhân tạo.  
**Nhóm thực hiện:** Nhóm 3 — 03 thành viên  
**Bộ dữ liệu đánh giá:** POPE (Polling-based Object Probing Evaluation)

---

## 1. MỤC TIÊU VÀ PHẠM VI THỰC HIỆN

### 1.1. Mục tiêu

Nhóm 3 phụ trách thực nghiệm trên bộ dữ liệu **POPE** nhằm đánh giá khả năng nhận diện vật thể trong hình ảnh và khảo sát hiện tượng **object hallucination** ở các mô hình ngôn ngữ đa phương thức (VLM).

Thông qua các câu hỏi dạng Có/Không về sự xuất hiện của một vật thể trong ảnh, nhóm sẽ kiểm tra mức độ chính xác của mô hình, xác định những trường hợp mô hình khẳng định có vật thể không thực sự xuất hiện, đồng thời so sánh kết quả giữa ba mô hình được lựa chọn.

Kết quả thực nghiệm là cơ sở để nhóm phân tích nguyên nhân gây ra lỗi, nhận diện những tình huống mô hình dễ trả lời sai và đề xuất hướng phát hiện hoặc giảm thiểu hallucination.

### 1.2. Phạm vi

Nhóm tập trung vào một benchmark duy nhất là **POPE**, với ba nhóm câu hỏi:

| Nhóm dữ liệu | Đặc điểm | Mục đích đánh giá |
|---|---|---|
| **Random** | Vật thể được hỏi được lựa chọn theo cách ngẫu nhiên | Đánh giá khả năng nhận diện trong điều kiện cơ bản |
| **Popular** | Tập trung vào các vật thể phổ biến, thường xuất hiện trong dữ liệu hình ảnh | Kiểm tra xu hướng mô hình dự đoán vật thể quen thuộc |
| **Adversarial** | Câu hỏi hướng đến các vật thể dễ bị nhầm lẫn hoặc có liên quan về ngữ cảnh | Đánh giá khả năng mô hình bị ảnh hưởng bởi liên tưởng và ngữ cảnh |

Bản POPE thường được sử dụng trong thực nghiệm gồm 9.000 câu hỏi, chia đều cho ba nhóm trên. Tuy nhiên, số lượng mẫu chính thức của nhóm sẽ được xác nhận theo phiên bản dữ liệu thực tế tải về và sử dụng.

### 1.3. Các mô hình được lựa chọn

Nhóm tiến hành đánh giá ba mô hình sau trên cùng bộ dữ liệu POPE:

| Model ID | Tên mô hình | Vai trò trong thực nghiệm |
|---|---|---|
| `smolvlm2` | `HuggingFaceTB/SmolVLM2-2.2B-Instruct` | Mô hình baseline, ưu tiên triển khai thử nghiệm ban đầu |
| `qwen3_vl` | `Qwen/Qwen3-VL-4B-Instruct` | Mô hình chính để đánh giá và phân tích |
| `phi4_multimodal` | `microsoft/Phi-4-multimodal-instruct` | Mô hình đối chiếu kết quả |

Ba mô hình trên là lựa chọn của nhóm để thực hiện so sánh trên POPE, không phải các mô hình được quy định sẵn bởi benchmark. Phiên bản model, cấu hình chạy và khả năng đáp ứng phần cứng cần được xác minh trước khi thực nghiệm chính thức.

### 1.4. Quy trình thực hiện tổng quát

```text
Xác minh nguồn dữ liệu và cấu trúc POPE
                 ↓
Chuẩn hóa dữ liệu, ảnh, câu hỏi và nhãn
                 ↓
Thống nhất prompt và phương pháp đánh giá
                 ↓
Chạy thử nghiệm ban đầu (pilot)
                 ↓
Thực nghiệm ba mô hình trên cùng tập dữ liệu
                 ↓
Tổng hợp kết quả theo Random, Popular, Adversarial
                 ↓
Phân tích trường hợp dự đoán sai và object hallucination
                 ↓
Đề xuất hướng phát hiện hoặc giảm thiểu hallucination
                 ↓
Hoàn thiện báo cáo và bàn giao kết quả
```

**Nguyên tắc thực hiện:** Mọi kết quả phải được tạo ra từ quá trình chạy mô hình thực tế, có dữ liệu dự đoán, cấu hình và thông tin môi trường đi kèm. Kết quả chạy thử cần được ghi rõ là *pilot*, không trình bày thay cho kết quả đánh giá đầy đủ.

---

## 2. PHÂN CÔNG NHIỆM VỤ CHO BA THÀNH VIÊN

Để bảo đảm tiến độ và tránh chồng chéo công việc, Nhóm 3 chia nhiệm vụ thành ba phần: xử lý dữ liệu, triển khai mô hình và đánh giá kết quả.

| Thành viên | Phụ trách | Nhiệm vụ chính | Sản phẩm bàn giao |
|---|---|---|---|
| **Thành viên A — Dataset** | Dữ liệu POPE | Tìm hiểu nguồn dữ liệu, kiểm tra ảnh và nhãn, xây dựng bộ đọc dữ liệu, chuẩn hóa ba nhóm câu hỏi | Dataset loader, thống kê dữ liệu, tài liệu mô tả và kiểm thử |
| **Thành viên B — Model & Inference** | Triển khai mô hình | Chuẩn bị môi trường, thiết lập prompt, tải và chạy ba mô hình, lưu kết quả dự đoán | Notebook thực nghiệm, raw predictions, cấu hình và nhật ký chạy |
| **Thành viên C — Evaluation & Analysis** | Đánh giá và phân tích | Xây dựng bộ chấm điểm, tính các chỉ số, so sánh mô hình và phân tích các trường hợp hallucination | Mã đánh giá, bảng kết quả, phân tích lỗi và báo cáo |

**Người điều phối:** Một thành viên trong nhóm có thể kiêm nhiệm việc theo dõi tiến độ, tổng hợp kết quả và trao đổi với nhóm trưởng. Vai trò điều phối không làm phát sinh thêm thành viên hoặc thay thế trách nhiệm chuyên môn của từng người.

### 2.1. Nguyên tắc phối hợp

- Thành viên A chịu trách nhiệm bảo đảm dữ liệu đầu vào chính xác, đầy đủ và có thể truy xuất.
- Thành viên B sử dụng dữ liệu đã thống nhất để chạy mô hình; không tự ý thay đổi nhãn hoặc loại bỏ mẫu nhằm điều chỉnh kết quả.
- Thành viên C sử dụng dữ liệu dự đoán gốc để đánh giá; không chỉnh sửa câu trả lời của mô hình hoặc tự ý thay đổi quy tắc chấm.
- Khi phát sinh vấn đề liên quan đến nguồn dữ liệu, mô hình, cấu hình hoặc phương pháp đánh giá, thành viên phụ trách cần ghi nhận và trao đổi với nhóm trước khi thay đổi.
- Các thành viên cùng kiểm tra kết quả tại những bước bàn giao để bảo đảm dữ liệu và cách đánh giá được thống nhất.

---

## 3. THÀNH VIÊN A — TÌM HIỂU, XỬ LÝ VÀ CHUẨN BỊ DATASET POPE

### 3.1. Nội dung công việc

1. Tìm hiểu bài báo và repository chính thức của POPE để nắm mục tiêu, cách xây dựng dữ liệu và phương pháp đánh giá.
2. Xác định nguồn tải dữ liệu, phiên bản sử dụng và nguồn ảnh tương ứng.
3. Kiểm tra cấu trúc dữ liệu, số lượng mẫu trong từng nhóm Random, Popular và Adversarial.
4. Kiểm tra các trường thông tin cần thiết như mã ảnh, câu hỏi, nhãn chuẩn và nhóm dữ liệu dựa trên cấu trúc thực tế của tệp.
5. Đối chiếu đường dẫn ảnh, phát hiện ảnh bị thiếu, ảnh không đọc được, dữ liệu trùng lặp hoặc trường thông tin không hợp lệ.
6. Chuẩn hóa dữ liệu để các mô hình có thể sử dụng chung một định dạng đầu vào, đồng thời giữ lại thông tin gốc để đối chiếu.
7. Xây dựng một tập mẫu pilot cố định, có đại diện cho cả ba nhóm dữ liệu và hai nhãn Yes/No.
8. Viết kiểm thử cho quá trình đọc dữ liệu, kiểm tra ảnh, nhãn và mã định danh mẫu.
9. Hoàn thiện tài liệu mô tả nguồn dữ liệu, cấu trúc, cách tải và cách sử dụng.

### 3.2. Cấu trúc dữ liệu nội bộ đề xuất

```python
{
    "sample_id": "unique_sample_id",
    "image_id": "image_id",
    "image_path": "path/to/image.jpg",
    "question": "Is there a dog in the image?",
    "ground_truth": "yes",
    "split": "random"
}
```

Đây là định dạng nội bộ để các thành viên sử dụng thống nhất. Tên trường và cách ánh xạ phải được điều chỉnh theo dữ liệu POPE thực tế, không mặc định đây là cấu trúc gốc của bộ dữ liệu.

### 3.3. Các tệp bàn giao dự kiến

```text
benchmarks/pope/
├── dataset.py
├── schema.py
├── DATASET_NOTES.md
└── tests/
    └── test_dataset.py
```

### 3.4. Tiêu chí hoàn thành

- Xác định được nguồn và phiên bản dữ liệu sử dụng.
- Có thống kê số lượng mẫu hợp lệ ở từng nhóm.
- Loader đọc được dữ liệu và liên kết đúng câu hỏi với hình ảnh.
- Không làm sai lệch nhãn hoặc quan hệ giữa mẫu và nhóm dữ liệu.
- Có danh sách mẫu pilot cố định để thành viên B và C sử dụng.
- Có tài liệu hướng dẫn và kiểm thử cơ bản.

---

## 4. THÀNH VIÊN B — TRIỂN KHAI MÔ HÌNH VÀ THỰC NGHIỆM

### 4.1. Nội dung công việc

1. Chuẩn bị môi trường thực nghiệm trên nền tảng được nhóm thống nhất, ưu tiên môi trường có GPU.
2. Kiểm tra và ghi nhận phiên bản Python, PyTorch, Transformers, CUDA, loại GPU và dung lượng bộ nhớ.
3. Tìm hiểu hướng dẫn của từng mô hình để xác định cách tải model, processor và xử lý ảnh đầu vào.
4. Thiết lập prompt thống nhất về mục tiêu đánh giá: mô hình nhận một hình ảnh và một câu hỏi dạng Có/Không về sự xuất hiện của vật thể.
5. Không đưa nhãn chuẩn hoặc đáp án của dataset vào nội dung prompt.
6. Yêu cầu mô hình trả lời theo định dạng ngắn gọn Yes/No; đồng thời lưu lại phản hồi nguyên bản trước khi xử lý.
7. Kiểm tra từng mô hình bằng một số mẫu trước khi chạy tập pilot.
8. Ưu tiên chạy thử SmolVLM2, sau đó triển khai Qwen3-VL và Phi-4 multimodal.
9. Bảo đảm ba mô hình sử dụng cùng danh sách mẫu khi thực hiện so sánh.
10. Lưu câu trả lời gốc, kết quả sau khi chuẩn hóa, trạng thái chạy, thời gian và cấu hình thực nghiệm.
11. Ghi nhận riêng các lỗi kỹ thuật như thiếu bộ nhớ GPU, lỗi tải model, timeout hoặc lỗi đọc ảnh.
12. Bàn giao dữ liệu dự đoán và thông tin cấu hình cho thành viên C.

### 4.2. Danh sách mô hình thực nghiệm

| Thứ tự triển khai | Mô hình | Model ID |
|---|---|---|
| 1 | SmolVLM2-2.2B-Instruct | `HuggingFaceTB/SmolVLM2-2.2B-Instruct` |
| 2 | Qwen3-VL-4B-Instruct | `Qwen/Qwen3-VL-4B-Instruct` |
| 3 | Phi-4-multimodal-instruct | `microsoft/Phi-4-multimodal-instruct` |

Thứ tự trên là thứ tự triển khai đề xuất để kiểm tra khả năng vận hành trước, không phải thứ hạng về chất lượng mô hình.

### 4.3. Yêu cầu khi thực nghiệm

- Sử dụng cùng tập mẫu và cùng quy tắc đánh giá cho cả ba mô hình.
- Ghi rõ phiên bản model, processor, prompt và các tham số sinh câu trả lời.
- Không tự ý thay đổi dtype, lượng tử hóa hoặc cấu hình phần cứng khi chưa được thống nhất.
- Không chuyển lỗi kỹ thuật thành dự đoán Yes hoặc No.
- Những câu trả lời không rõ ràng, từ chối hoặc không đúng định dạng cần được lưu riêng để kiểm tra cách xử lý.
- Phải giữ lại câu trả lời gốc để việc đánh giá có thể kiểm tra và đối chiếu.

### 4.4. Các tệp bàn giao dự kiến

```text
benchmarks/pope/
├── protocol.py
├── inference.py
├── POPE_RUNBOOK.md
└── tests/
    └── test_protocol.py
```

### 4.5. Tiêu chí hoàn thành

- Có notebook hoặc hướng dẫn chạy thực nghiệm rõ ràng.
- Có kết quả chạy thử và kết quả của từng mô hình.
- Có raw predictions và thông tin cấu hình tương ứng.
- Ba mô hình được đánh giá trên cùng phạm vi mẫu.
- Trường hợp không thể chạy được phải có log và lý do cụ thể, không ghi nhận là đã hoàn thành.

---

## 5. THÀNH VIÊN C — ĐÁNH GIÁ KẾT QUẢ VÀ PHÂN TÍCH HALLUCINATION

### 5.1. Nội dung công việc

1. Tìm hiểu phương pháp đánh giá được sử dụng trong POPE và đối chiếu với mã đánh giá của nguồn dữ liệu.
2. Xây dựng bộ xử lý câu trả lời để chuyển phản hồi của mô hình về hai nhãn Yes/No theo quy tắc đã thống nhất.
3. Xác định cách xử lý các trường hợp trả lời dài, mơ hồ, từ chối hoặc không đúng định dạng.
4. Viết kiểm thử cho bộ chấm điểm bằng các trường hợp có kết quả dự kiến.
5. Tính toán các chỉ số đánh giá cho từng mô hình trên ba nhóm Random, Popular và Adversarial.
6. Lập confusion matrix để xác định số lượng dự đoán đúng và sai.
7. Phân tích các trường hợp mô hình trả lời Yes khi vật thể không xuất hiện trong ảnh (False Positive).
8. Phân tích các trường hợp mô hình trả lời No khi vật thể thực tế có trong ảnh (False Negative).
9. Lựa chọn các ví dụ tiêu biểu để đối chiếu hình ảnh, câu hỏi, nhãn chuẩn và phản hồi thực tế của mô hình.
10. So sánh kết quả giữa ba mô hình, xác định nhóm dữ liệu và tình huống mô hình dễ mắc lỗi.
11. Tổng hợp nhận xét, nêu hạn chế của quá trình đánh giá và đề xuất hướng phát hiện hoặc giảm thiểu hallucination.

### 5.2. Các chỉ số đánh giá

| Chỉ số | Ý nghĩa |
|---|---|
| **Accuracy** | Tỷ lệ dự đoán đúng trên tổng số mẫu |
| **Precision** | Tỷ lệ câu trả lời Yes đúng trong tổng số câu mô hình dự đoán Yes |
| **Recall** | Tỷ lệ vật thể có trong ảnh được mô hình nhận diện đúng |
| **F1-score** | Chỉ số cân bằng giữa Precision và Recall |
| **Yes Ratio** | Tỷ lệ mẫu được mô hình dự đoán là Yes |
| **Confusion Matrix** | Bảng thể hiện số lượng TP, FP, TN và FN |

Quy ước `Yes` là lớp dương:

- **TP (True Positive):** Vật thể có trong ảnh và mô hình trả lời Yes.
- **FP (False Positive):** Vật thể không có trong ảnh nhưng mô hình trả lời Yes. Đây là trường hợp cần chú ý khi phân tích object hallucination.
- **TN (True Negative):** Vật thể không có trong ảnh và mô hình trả lời No.
- **FN (False Negative):** Vật thể có trong ảnh nhưng mô hình trả lời No.

Công thức tính:

```text
Accuracy  = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1-score  = 2 × Precision × Recall / (Precision + Recall)
Yes Ratio = (TP + FP) / (TP + TN + FP + FN)
```

Trong trường hợp mẫu số bằng 0, cần thống nhất cách xử lý và ghi rõ trong báo cáo. Yes Ratio chỉ phản ánh tỷ lệ câu trả lời Yes, không được xem là tỷ lệ hallucination. Khi phân tích hiện tượng mô hình khẳng định vật thể không tồn tại trong ảnh, cần xem xét trực tiếp các trường hợp FP.

### 5.3. Bảng tổng hợp kết quả dự kiến

| Mô hình | Nhóm dữ liệu | Số mẫu hợp lệ | Accuracy | Precision | Recall | F1-score | Yes Ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| SmolVLM2 | Random | — | — | — | — | — | — |
| SmolVLM2 | Popular | — | — | — | — | — | — |
| SmolVLM2 | Adversarial | — | — | — | — | — | — |
| Qwen3-VL | Random | — | — | — | — | — | — |
| Qwen3-VL | Popular | — | — | — | — | — | — |
| Qwen3-VL | Adversarial | — | — | — | — | — | — |
| Phi-4 | Random | — | — | — | — | — | — |
| Phi-4 | Popular | — | — | — | — | — | — |
| Phi-4 | Adversarial | — | — | — | — | — | — |

Bảng chỉ được điền sau khi có kết quả dự đoán thực tế. Nếu nhóm chỉ chạy một phần dữ liệu, cần ghi rõ số lượng mẫu và đánh dấu kết quả là pilot.

### 5.4. Phân tích các trường hợp dự đoán sai

Mỗi trường hợp được lựa chọn để phân tích cần có các thông tin:

| Thông tin | Nội dung cần ghi nhận |
|---|---|
| Sample ID | Mã mẫu trong POPE |
| Model | Mô hình được kiểm tra |
| Split | Random, Popular hoặc Adversarial |
| Image | Hình ảnh đầu vào |
| Question | Câu hỏi được đưa vào mô hình |
| Ground truth | Nhãn chuẩn của dữ liệu |
| Raw response | Câu trả lời nguyên bản của mô hình |
| Prediction | Kết quả sau khi chuẩn hóa |
| Error type | FP, FN, lỗi định dạng hoặc lỗi kỹ thuật |
| Analysis | Nhận xét và bằng chứng cho trường hợp được phân tích |

Không mặc định mọi dự đoán sai đều là hallucination. Nhóm cần đối chiếu câu hỏi, hình ảnh và nhãn chuẩn để xác định bản chất của từng lỗi.

### 5.5. Các tệp bàn giao dự kiến

```text
benchmarks/pope/
├── scoring.py
├── analysis.py
├── EVALUATION_NOTES.md
└── tests/
    └── test_scoring.py
```

### 5.6. Tiêu chí hoàn thành

- Có bộ chấm điểm và kiểm thử.
- Có kết quả metric theo từng mô hình và từng nhóm dữ liệu.
- Có confusion matrix và phân tích FP/FN.
- Có ví dụ minh chứng truy xuất được về mẫu gốc.
- Có báo cáo so sánh ba mô hình và nhận xét dựa trên số liệu thực tế.

---

## 6. QUY ƯỚC BÀN GIAO VÀ LƯU TRỮ KẾT QUẢ

### 6.1. Bàn giao giữa các thành viên

**Thành viên A bàn giao cho B và C:**

- Dữ liệu POPE đã kiểm tra và chuẩn hóa.
- Thông tin nguồn, phiên bản và cấu trúc dữ liệu.
- Danh sách sample ID dùng cho pilot và thực nghiệm.
- Thống kê mẫu hợp lệ, mẫu lỗi và các trường hợp được xử lý.

**Thành viên B bàn giao cho C:**

- Câu trả lời gốc của từng mô hình.
- Kết quả dự đoán sau khi chuẩn hóa.
- Mã mẫu, tên mô hình và nhóm dữ liệu tương ứng.
- Prompt, cấu hình, phiên bản model và thông tin môi trường chạy.

**Thành viên C bàn giao cho nhóm:**

- Bảng metric và confusion matrix.
- Bảng so sánh kết quả giữa ba mô hình.
- Danh sách các trường hợp dự đoán sai kèm minh chứng.
- Báo cáo phân tích và đề xuất hướng cải thiện.

### 6.2. Định dạng kết quả đề xuất

```text
results/pope/<experiment_id>/<run_id>/
├── predictions.jsonl
├── metrics.json
├── run_config.yaml
├── run_metadata.json
├── errors.jsonl
├── run_status.json
└── manifest.json
```

Các tệp trên là cấu trúc lưu trữ đề xuất, cần điều chỉnh theo quy định của repository và nhóm trưởng. Không đưa API key, token, trọng số mô hình hoặc dữ liệu dung lượng lớn lên Git.

---

## 7. LỘ TRÌNH TRIỂN KHAI

| Giai đoạn | Thành viên A — Dataset | Thành viên B — Inference | Thành viên C — Evaluation |
|---|---|---|---|
| **Giai đoạn 1: Chuẩn bị** | Tìm hiểu nguồn và kiểm tra POPE | Chuẩn bị môi trường, tìm hiểu ba model | Nghiên cứu phương pháp đánh giá |
| **Giai đoạn 2: Xây dựng** | Hoàn thiện loader và schema | Xây dựng prompt, kiểm tra khả năng chạy model | Xây dựng parser, scorer và test |
| **Giai đoạn 3: Chạy thử** | Chốt tập mẫu pilot | Chạy pilot, ưu tiên SmolVLM2 | Kiểm tra kết quả và cách tính metric |
| **Giai đoạn 4: Thực nghiệm** | Theo dõi tính đầy đủ của dữ liệu | Chạy ba mô hình trên cùng tập mẫu | Tổng hợp metric theo từng split |
| **Giai đoạn 5: Phân tích** | Hỗ trợ truy xuất dữ liệu | Bổ sung kết quả hoặc chạy lại mẫu lỗi kỹ thuật | Phân tích FP/FN và so sánh mô hình |
| **Giai đoạn 6: Báo cáo** | Hoàn thiện mô tả dữ liệu | Bàn giao notebook và log | Hoàn thiện bảng kết quả, nhận xét và báo cáo |

Các thành viên có thể triển khai những phần việc độc lập song song. Việc chạy thực nghiệm chung chỉ bắt đầu khi dữ liệu, prompt và phương pháp đánh giá đã được thống nhất.

---

## 8. CÁC NỘI DUNG CẦN THỐNG NHẤT VỚI NHÓM TRƯỞNG

| Mức độ | Nội dung cần xác nhận | Mục đích |
|---|---|---|
| **Ưu tiên cao** | Danh sách ba mô hình và phiên bản sử dụng | Bảo đảm đúng phạm vi thực nghiệm |
| **Ưu tiên cao** | Nguồn và revision của POPE | Bảo đảm dữ liệu có thể kiểm tra và tái lập |
| **Ưu tiên cao** | Chạy toàn bộ dữ liệu hay chạy pilot trước | Chủ động kế hoạch tài nguyên |
| **Ưu tiên cao** | Nền tảng GPU, dtype và giới hạn bộ nhớ | Xác định khả năng triển khai từng mô hình |
| **Ưu tiên cao** | Prompt và quy tắc chuẩn hóa Yes/No | Bảo đảm việc so sánh nhất quán |
| **Ưu tiên** | Nơi lưu notebook, predictions và artifact | Thống nhất cách quản lý kết quả |
| **Ưu tiên** | Phương pháp phát hiện hoặc giảm hallucination cần nghiên cứu | Xác định phạm vi giai đoạn tiếp theo |

Khi có vấn đề về tài nguyên hoặc cấu hình, thành viên phụ trách cần báo cáo tình trạng và log liên quan. Không tự ý thay đổi mô hình, dữ liệu hoặc quy trình đánh giá để thay thế kết quả chưa thực hiện được.

---

## 9. YÊU CẦU ĐỐI VỚI BÁO CÁO KẾT QUẢ

Khi tổng hợp và trình bày, Nhóm 3 cần làm rõ các nội dung sau:

1. **Dữ liệu sử dụng:** Nguồn POPE, phiên bản, số lượng mẫu thực tế và cách chia ba nhóm Random, Popular, Adversarial.
2. **Mô hình thực nghiệm:** Tên và phiên bản của ba mô hình, môi trường chạy, cấu hình và số lượng mẫu đã xử lý.
3. **Quy trình đánh giá:** Cách xây dựng prompt, cách xử lý phản hồi Yes/No và phương pháp tính các chỉ số.
4. **Kết quả thực nghiệm:** Accuracy, Precision, Recall, F1-score và Yes Ratio của từng mô hình trên từng nhóm dữ liệu.
5. **Phân tích hallucination:** Những trường hợp mô hình dự đoán vật thể không có trong ảnh, các lỗi FP/FN và bằng chứng đi kèm.
6. **Hướng cải thiện:** Đề xuất hoặc thử nghiệm phương pháp phát hiện, giảm thiểu hallucination; nêu rõ kết quả thực tế và giới hạn.

Số liệu trong báo cáo phải khớp với tệp dự đoán và thông tin của lần chạy tương ứng. Không ước lượng hoặc điền trước kết quả khi chưa thực nghiệm.

---

## 10. NỘI DUNG THÔNG BÁO PHÂN CÔNG TRONG NHÓM

Nhóm 3 thống nhất thực hiện đánh giá hiện tượng ảo giác thị giác trên bộ dữ liệu **POPE** với ba mô hình SmolVLM2-2.2B-Instruct, Qwen3-VL-4B-Instruct và Phi-4-multimodal-instruct.

Công việc được chia thành ba phần:

- **Thành viên A — Dataset:** Phụ trách tìm hiểu, tải và kiểm tra dữ liệu POPE; xử lý ảnh, câu hỏi, nhãn; chuẩn hóa ba nhóm Random, Popular và Adversarial.
- **Thành viên B — Model & Inference:** Phụ trách chuẩn bị môi trường, thiết lập prompt, triển khai và chạy ba mô hình trên cùng tập dữ liệu; lưu câu trả lời gốc và thông tin cấu hình.
- **Thành viên C — Evaluation & Analysis:** Phụ trách xây dựng bộ đánh giá, tính các chỉ số Accuracy, Precision, Recall, F1-score và Yes Ratio; phân tích các trường hợp FP/FN và so sánh kết quả giữa các mô hình.

Trước tiên, nhóm sẽ hoàn thành việc chuẩn bị dữ liệu, kiểm tra khả năng chạy mô hình và thống nhất phương pháp đánh giá. Sau khi pilot đạt yêu cầu, nhóm tiến hành thực nghiệm trên phạm vi dữ liệu đã chốt, phân tích kết quả và đề xuất hướng phát hiện hoặc giảm thiểu hallucination.

Mỗi thành viên chịu trách nhiệm hoàn thành phần việc được giao và bàn giao đầy đủ mã nguồn, tài liệu, kết quả kiểm thử cùng các artifact liên quan để nhóm có thể kiểm tra và chạy lại khi cần.

---

## 11. TÀI LIỆU THAM KHẢO

1. POPE — Repository: https://github.com/RUCAIBox/POPE
2. POPE — Paper: https://arxiv.org/abs/2305.10355
3. SmolVLM2-2.2B-Instruct: https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct
4. Qwen3-VL-4B-Instruct: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
5. Phi-4-multimodal-instruct: https://huggingface.co/microsoft/Phi-4-multimodal-instruct

**Lưu ý:** Nhóm cần kiểm tra tài liệu model card, phiên bản dữ liệu, điều kiện sử dụng và khả năng chạy trên môi trường được cấp trước khi tiến hành thực nghiệm chính thức.

---

## 12. KẾT LUẬN

Nhóm 3 tập trung triển khai và đánh giá benchmark POPE trên ba mô hình đã lựa chọn. Việc phân công theo ba mảng dữ liệu, thực nghiệm mô hình và đánh giá kết quả giúp mỗi thành viên có phạm vi công việc rõ ràng, đồng thời bảo đảm các bước trong quy trình được liên kết thống nhất.

Thông qua kết quả trên ba nhóm Random, Popular và Adversarial, nhóm hướng đến việc xác định khả năng nhận diện vật thể của từng mô hình, phân tích các trường hợp object hallucination và tạo cơ sở cho những nghiên cứu tiếp theo về phát hiện, giảm thiểu hiện tượng này.
