# Experimental Protocol v0.1 — VLM Hallucination Benchmark

**Dự án:** Xây dựng pipeline dữ liệu đa phương thức phục vụ kiểm chứng thông tin và giảm ảo giác trong hệ thống trí tuệ nhân tạo.

**Trạng thái:** Dự thảo — cần trưởng nhóm và mentor phê duyệt trước thực nghiệm chính thức.

## 1. Mục tiêu

Đánh giá và phân tích hiện tượng ảo giác của ba mô hình thị giác–ngôn ngữ trên ba bộ dữ liệu chuẩn. Xác định các dạng lỗi thường gặp, xây dựng bộ kết quả có khả năng tái lập và tạo cơ sở nghiên cứu các phương pháp phát hiện, giảm thiểu ảo giác.

Giai đoạn này tập trung vào đánh giá VLM với dữ liệu hình ảnh và văn bản, chưa phải đánh giá trực tiếp khả năng bám sát bằng chứng truy xuất của hệ thống RAG (sinh câu trả lời có tăng cường truy xuất).

## 2. Phân công thực nghiệm

| Nhóm | Benchmark | Mô hình cần đánh giá |
|---|---|---|
| Nhóm 1 | POPE | Qwen3-VL, SmolVLM2, Phi-4 multimodal |
| Nhóm 2 | HallusionBench | Qwen3-VL, SmolVLM2, Phi-4 multimodal |
| Nhóm 3 | MMHal-Bench | Qwen3-VL, SmolVLM2, Phi-4 multimodal |

Mỗi nhóm chịu trách nhiệm chuẩn bị dữ liệu, chạy mô hình, chấm điểm, phân tích lỗi và báo cáo.

## 3. Quy định về mô hình

- Sử dụng mô hình và trọng số từ nguồn phát hành chính thức.
- Ghi đầy đủ `model_id`, `revision`, phiên bản processor (bộ xử lý đầu vào) và tokenizer (bộ tách văn bản thành token).
- Không tự ý tinh chỉnh mô hình trong giai đoạn đánh giá ban đầu.
- Không thay đổi trọng số, phương pháp lượng tử hóa hoặc cách tiền xử lý giữa các phần của cùng một thí nghiệm.
- Nếu phải thay đổi cấu hình do giới hạn phần cứng, tạo mã thực nghiệm riêng và báo cáo rõ sự khác biệt.

**Mã revision chính thức:** CHƯA CHỐT.

## 4. Quy định về dataset

- Tải từ nguồn đã được xác minh.
- Lưu mã phiên bản dữ liệu, tên tập con và số lượng mẫu.
- Không sửa câu hỏi, ảnh hoặc nhãn gốc.
- Mọi bước chuyển đổi định dạng phải có mã nguồn và kiểm thử.
- Bảo đảm cùng một mô hình được đánh giá trên cùng tập mẫu trong cùng benchmark.
- Không dùng nhãn hoặc đáp án tham chiếu làm đầu vào cho mô hình được đánh giá.
- Không tự ý loại bỏ các mẫu khó hoặc mẫu mô hình trả lời sai.

**Nguồn và phiên bản cụ thể của từng dataset:** CHƯA CHỐT.

## 5. Hai giai đoạn thực nghiệm

### Giai đoạn A — Pilot (chạy thử)

- Chọn một tập con nhỏ, có mã mẫu cố định.
- Kiểm tra nạp mô hình, xử lý ảnh, câu hỏi, suy luận, lưu kết quả và chấm điểm.
- Kiểm tra trên ít nhất một môi trường Kaggle và một môi trường máy cá nhân nếu có đủ tài nguyên.
- Các kết quả pilot chỉ phục vụ xác nhận kỹ thuật, không thay thế kết quả benchmark chính thức.

### Giai đoạn B — Official Evaluation (đánh giá chính thức)

- Khóa phiên bản mô hình, dataset, mã nguồn và cấu hình.
- Chạy toàn bộ tập đánh giá đã được thống nhất.
- Lưu câu trả lời gốc của mô hình trước khi chuẩn hóa hoặc chấm điểm.
- Lưu thông tin các mẫu bị lỗi kỹ thuật, không âm thầm bỏ qua.
- Tính chỉ số bằng bộ chấm điểm đã xác minh.
- Phân tích các trường hợp lỗi và ảo giác.

## 6. Quy định cấu hình suy luận

- Ưu tiên suy luận không lấy mẫu ngẫu nhiên (`do_sample: false`) nếu tương thích với mô hình và quy trình benchmark.
- Không áp đặt một giới hạn token duy nhất cho mọi benchmark khi nhiệm vụ yêu cầu độ dài câu trả lời khác nhau.
- Mỗi benchmark phải có mẫu câu lệnh đầu vào được lưu trong mã nguồn.
- Không thay đổi câu lệnh trong khi đang chạy chính thức.
- Ghi lại `max_new_tokens`, `do_sample`, các tham số giải mã thực sự được sử dụng, độ chính xác số học và cấu hình xử lý ảnh.
- Mọi thay đổi cấu hình đều phải tạo mã thực nghiệm mới.

**Giới hạn token và cấu hình chi tiết:** CHƯA CHỐT, cần chạy thử.

## 7. Quy định môi trường thực thi

Cho phép sử dụng Kaggle và máy cá nhân.

Mỗi lần chạy phải ghi lại:

- Hệ điều hành và phiên bản Python.
- Phiên bản PyTorch, Transformers, CUDA và các thư viện liên quan.
- Loại GPU, VRAM và số lượng GPU.
- Phiên bản Git commit của mã nguồn.
- Độ chính xác số học và phương pháp lượng tử hóa.
- Thời gian bắt đầu, kết thúc và trạng thái thực nghiệm.

Không được gộp kết quả của các cấu hình khác nhau thành một lần chạy thống nhất nếu chưa có cơ sở kiểm chứng tính tương đương.

## 8. Quy định lưu kết quả

Mỗi thí nghiệm có `run_id` duy nhất.

Bắt buộc lưu:

1. `predictions.jsonl`: Dự đoán gốc trên từng mẫu.
2. `metrics.json`: Chỉ số đánh giá.
3. `run_config.yaml`: Cấu hình thực tế.
4. `run_metadata.json`: Phiên bản và môi trường.
5. `errors.jsonl`: Các trường hợp lỗi được xác định.
6. Nhật ký chạy và danh sách mẫu thất bại nếu có.

Không ghi đè kết quả đã hoàn thành. Không đẩy trọng số mô hình, dữ liệu lớn hoặc thông tin bí mật lên GitHub.

## 9. Quy định đánh giá

- POPE: Báo cáo riêng theo ba chiến lược chọn mẫu và các chỉ số theo quy trình đánh giá.
- HallusionBench: Giữ nguyên cấu trúc nhóm câu hỏi, các cặp kiểm soát và phương pháp tính điểm của phiên bản đã chọn.
- MMHal-Bench: Lưu mô hình giám khảo, phiên bản, rubric (bảng tiêu chí chấm), câu lệnh và câu trả lời của giám khảo. Nếu thay đổi giám khảo, phải ghi rõ đây là phương pháp đánh giá khác.

Không so sánh trực tiếp điểm số khác loại giữa các benchmark như một thước đo thống nhất.

## 10. Phân tích lỗi

Mỗi nhóm phải lựa chọn và phân tích các trường hợp tiêu biểu, bao gồm:

- Câu trả lời không phù hợp với nội dung ảnh.
- Khẳng định có đối tượng không xuất hiện trong ảnh.
- Bỏ sót hoặc nhận diện sai đối tượng.
- Sai trong suy luận về quan hệ, thuộc tính hoặc thông tin thị giác.
- Câu trả lời không tuân thủ định dạng yêu cầu.
- Trường hợp khó kết luận hoặc có vấn đề về nhãn.

Phân biệt lỗi dự đoán, lỗi kỹ thuật, lỗi định dạng và ảo giác. Không tự động gán mọi câu trả lời sai thành ảo giác.

## 11. Điều kiện nghiệm thu

Một tổ hợp mô hình–benchmark chỉ được coi là hoàn thành khi:

- Xác nhận được phiên bản mô hình và dữ liệu.
- Có kết quả hoặc trạng thái lỗi cho tất cả mẫu thuộc phạm vi đánh giá.
- Có tệp dự đoán và cấu hình thực nghiệm.
- Có kết quả chấm điểm được kiểm tra.
- Có báo cáo phân tích lỗi.
- Có hướng dẫn tái lập.

## 12. Quy trình phê duyệt thay đổi

Mọi thay đổi ảnh hưởng kết quả thực nghiệm phải được ghi nhận qua GitHub Pull Request (yêu cầu hợp nhất mã nguồn) và được trưởng nhóm hoặc người được ủy quyền kiểm tra.

Các thay đổi quan trọng về mô hình, dataset hoặc phương pháp chấm phải được thống nhất trước khi thực hiện lại thí nghiệm.

**Phiên bản tài liệu:** 0.1 — chưa phê duyệt.
