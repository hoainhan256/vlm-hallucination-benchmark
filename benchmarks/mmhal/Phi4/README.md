# Đánh giá Mô hình Phi-4-Multimodal-Instruct trên MMHal-Bench

Thư mục này chứa toàn bộ mã nguồn đánh giá, kết quả thực nghiệm và báo cáo chuyên sâu cho mô hình **Phi-4-Multimodal-Instruct** trên benchmark **MMHal-Bench** (96 mẫu).

---

## 1. Tóm tắt Kết quả Hiệu năng

| Chỉ số | Thang đo | Kết quả Phi-4-Multimodal |
| :--- | :---: | :---: |
| **Tổng số mẫu thử nghiệm** | N | **96** |
| **Điểm trung bình (Average Score)** | 0 - 6 | **3.72 / 6.0** |
| **Điểm chuẩn hóa (Normalized Score)** | 0 - 5 | **3.10 / 5.0** |
| **Tỉ lệ Ảo giác (Hallucination Rate)** | % | **26.04%** (25 / 96 mẫu) |
| **Độ chính xác theo dạng câu hỏi:** | | |
| - *Comparison (So sánh trực quan)* | Hal Rate | **0.0%** (0 / 12 mẫu lỗi) - Xuất sắc |
| - *Attribute (Nhận diện thuộc tính)* | Hal Rate | **16.7%** (2 / 12 mẫu lỗi) |
| - *Other (OCR / Đọc chữ, logo)* | Hal Rate | **16.7%** (2 / 12 mẫu lỗi) |
| - *Adversarial (Bẫy đối kháng)* | Hal Rate | **25.0%** (3 / 12 mẫu lỗi) |
| - *Environment (Môi trường)* | Hal Rate | **25.0%** (3 / 12 mẫu lỗi) |
| - *Relation (Quan hệ không gian)* | Hal Rate | **25.0%** (3 / 12 mẫu lỗi) |
| - *Holistic (Miêu tả toàn cảnh)* | Hal Rate | **41.7%** (5 / 12 mẫu lỗi) |
| - *Counting (Đếm số lượng)* | Hal Rate | **58.3%** (7 / 12 mẫu lỗi) - Điểm yếu lớn nhất |

---

## 2. Danh mục Tệp trong Thư mục

- `phi4_mmhal_responses.json`: Kết quả suy luận (inference) thô của mô hình Phi-4 trên 96 mẫu của MMHal-Bench.
- `evaluate_phi4.py`: Script tự động chấm điểm độc lập (LLM-as-a-Judge) theo đúng rubric 0–6 của MMHal-Bench.
- `phi4_mmhal_evaluated.json`: Kết quả chi tiết sau khi chấm điểm, chứa điểm số, nhãn ảo giác, phân loại taxonomy lỗi và lời giải thích cho từng mẫu.

- `phi4_mmhal_bench_report.docx`: Bản báo cáo định dạng Microsoft Word chuyên nghiệp.
