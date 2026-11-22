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
- **MMHal-Bench:** hiện dùng GPT-6 tương tác trong Codex thay GPT-4 API theo yêu cầu người phụ trách ngày 19/11/2026. Giữ nguyên prompt, năm ví dụ, bốn trường đầu vào văn bản, rubric 0–6, ngưỡng <3 và công thức tổng hợp của script gốc. Lưu từng điểm/giải thích, prompt tái dựng, model family và giới hạn tái lập; không giả là API response hoặc temperature=0. Đây là phương án judge thay thế, không tương đương điểm GPT-4 gốc. Chi tiết tại [protocol GPT-6](mmhal_gpt6_protocol.md). Review ảnh là vòng bổ sung riêng; thiếu chi tiết trong reference không tự là hallucination.
