# Chính sách dataset

Nội dung dataset và revision phải được xác minh trước khi chạy chính thức. Mã nguồn, tập con, license (giấy phép) và revision được ghi trong `configs/datasets.yaml`.

- Không sửa câu hỏi, ảnh, nhãn tham chiếu hoặc chiến lược chọn mẫu.
- Không rò rỉ nhãn hoặc đáp án tham chiếu vào prompt đầu vào.
- Không loại bỏ mẫu sai, khó hoặc không mong muốn tùy ý.
- Không thay đổi phương pháp chấm; POPE, HallusionBench và MMHal-Bench giữ protocol riêng.
- Mọi chuyển đổi định dạng phải có mã nguồn, kiểm thử và ghi lại trong run metadata.
- License, revision và cấu trúc sample được coi là PENDING_VERIFICATION cho tới khi có nguồn chính thức xác nhận.
