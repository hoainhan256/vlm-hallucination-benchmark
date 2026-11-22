# Hướng dẫn nhóm MMHal-Bench — cập nhật 19/11/2026

## Trạng thái và quyết định hiện tại

Đã triển khai loader, SmolVLM2 adapter, inference, artifact/resume, API judge và
import review tại `benchmarks/mmhal/`. Run `full-infer-001` hoàn tất 96/96 mẫu,
stage pilot/exploratory, không phải official evaluation đã nghiệm thu.

Theo yêu cầu người phụ trách dự án trong chat Codex, dùng GPT-6 tương tác để
chấm thay vì bắt buộc GPT-4 API. Phải bám sát phương pháp gốc: nguyên prompt
và năm ví dụ, bốn trường text, rubric 0–6, hallucination nếu <3, trung bình toàn
96 và tám loại theo thứ tự gốc. Xem [protocol đầy đủ](../mmhal_gpt6_protocol.md).

## Phân công

1. Dataset/inference: verify source snapshot và manifest, bảo toàn câu hỏi/ảnh,
   không rò reference vào model; theo dõi lỗi bộ nhớ và truncation.
2. Judge/QA: dùng prompt gốc, lưu từng điểm/giải thích và đánh dấu ca cần xem ảnh.
   Không giả API log, temperature, snapshot hoặc human review chưa thực hiện.
3. Tổng hợp/phân tích: import ledger, kiểm tra parser/công thức gốc, xuất metrics,
   bảng từng ca và báo cáo; phân biệt thiếu thông tin với hallucination.

## Quy trình chấm

- Verify run inference và đủ 96 mẫu COMPLETED.
- Dùng `original_template()` từ source đã khóa để dựng đúng input từng mẫu.
- GPT-6 đánh giá nội dung theo rubric gốc, ghi giải thích và một Rating: N.
- `import_review.py` chỉ kiểm tra/nhập điểm, không suy ra điểm bằng heuristic.
- Giữ nguyên inference và export run chấm mới, checksum nguồn và ledger.
- Báo cáo rõ judge thay thế, tương tác chung ngữ cảnh, không kiểm soát được
  temperature/snapshot. Không tuyên bố tương đương GPT-4 API.
- Xem ảnh/human review ở vòng bổ sung; nếu đổi điểm phải có artifact/version mới.

## Đầu ra và kiểm tra

`predictions.jsonl`, `metrics.json`, `run_config.yaml`, `run_metadata.json`,
`errors.jsonl`, `run_status.json`, `manifest.json`, `judge_inputs.jsonl`,
`judge_reviews.jsonl`, `review_ledger.tsv`, `cases.csv`, `report.md`.

Kiểm tra 96 ID đúng thứ tự, source checksum, parser original, tổng hợp modulo 8,
không trộn source/config, không ghi đè. Các ca chưa rõ phải lưu cờ review;
reference không hoàn chỉnh không phải căn cứ tự động gán ảo giác.

## Giới hạn nghiệm thu

Run inference hiện có Git dirty, 10 output holistic chạm giới hạn token.
Không che giấu các giới hạn này hoặc đổi pilot thành official sau thực tế.
Dataset/model revisions đã quan sát, nhưng các approval tham chiếu tài liệu
chưa tồn tại không phải bằng chứng trưởng nhóm đã ký. Protocol mới được đổi
theo yêu cầu người dùng; quyền chấp nhận kết quả khoa học thuộc nhóm/mentor.
API backend cũ chỉ là phương án đối chứng khi được yêu cầu riêng.

Thay đổi này chỉ áp dụng MMHal. POPE/HallusionBench giữ nguyên nguyên tắc.
