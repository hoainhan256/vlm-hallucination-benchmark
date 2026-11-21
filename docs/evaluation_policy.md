# Chính sách đánh giá

## Phân loại kết quả

- **Sai dự đoán (incorrect prediction):** câu trả lời khác kết luận hoặc nhãn tham chiếu theo protocol của benchmark.
- **Ảo giác (hallucination):** nội dung không có căn cứ hoặc mâu thuẫn chứng cứ thị giác, phải được phân tích theo tiêu chí benchmark.
- **Không tuân thủ định dạng (format violation):** trả lời không đáp ứng schema hoặc cách chọn nhãn yêu cầu.
- **Lỗi kỹ thuật (technical error):** nạp model, xử lý, timeout, CUDA hoặc I/O làm không thể tạo prediction.
- **Lỗi nhãn hoặc dữ liệu (label/data error):** có bằng chứng sample, nhãn hoặc media không nhất quán.
- **Chưa thể kết luận (undetermined):** thiếu bằng chứng để phân loại chắc chắn.

Không đồng nhất mọi câu trả lời sai với ảo giác.

## Protocol riêng

- **POPE:** bảo toàn ba sampling strategies random, popular và adversarial; tính chỉ số theo protocol gốc đã xác minh.
- **HallusionBench:** giữ cấu trúc câu hỏi/cặp kiểm soát và aggregation của phiên bản đã chọn.
- **MMHal-Bench:** ưu tiên judge protocol gốc; trước triển khai phải xác minh judge, model/version truy cập, rubric, chi phí và cách ghi kết quả. Không thay judge bằng mô hình khác và coi là tương đương.
