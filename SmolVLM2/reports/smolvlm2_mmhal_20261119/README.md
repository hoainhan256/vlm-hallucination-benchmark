# Bàn giao báo cáo SmolVLM2 và MMHal

- [Báo cáo MMHal của ba mô hình](../../../reports/Bao_cao_tong_hop_MMHal.docx).
- `review_96_cases.csv`: nguyên văn câu hỏi/reference/output, điểm judge vòng một,
  nhãn và bằng chứng review ảnh vòng hai, cờ truncation và tên ảnh nguồn.
- `visual_review_summary.json`: tổng hợp review ảnh, không phải rescore chính thức.
- `integration_record.json`: bản ghi để ghép kết quả liên nhóm.
- `manifest.json`: checksum nguồn và các tệp bàn giao chính.

Ảnh minh họa xuất từ snapshot dữ liệu chính thức, không sinh mới hoặc sửa nội
dung. Review 96 ảnh bằng bảng tổng quan; năm ca khó 015/031/063/086/087 mở nguyên
ảnh. Không tuyên bố kiểm tra toàn bộ ảnh ở mức pixel hoặc có human review độc lập.

Vòng một: 3,28125/6; 34/96 điểm dưới 3. Vòng hai: 36 lỗi có bằng chứng, 37 phù hợp,
15 thiếu/mơ hồ, 8 chưa kết luận. Các con số thuộc hai protocol khác nhau, không
thay thế nhau. Điểm vòng một và run inference được giữ nguyên.

QA: renderer chuẩn không có LibreOffice trên Windows; đã thử và ghi nhận lỗi.
Sau đó xuất PDF nội bộ bằng COM Office, backend thực tế ghi WPS Docs trong PDF
metadata, dùng Poppler bundled rasterize đủ 16 trang và kiểm tra từng PNG.
Không bàn giao PDF/PNG QA như kết quả chính. Docx được dựng bằng python-docx từ
runtime bundled. Không gọi dịch vụ chấm API, không sửa trọng số hoặc chạy lại model.

Các ý phát hiện/cải thiện là định hướng thí nghiệm tiếp theo, chưa có benchmark
chứng minh hiệu quả. Tám ca uncertain và chấp thuận khoa học cuối cùng cần nhóm/
mentor quyết định; không giả tạo nhãn chắc chắn để làm đẹp thống kê.
