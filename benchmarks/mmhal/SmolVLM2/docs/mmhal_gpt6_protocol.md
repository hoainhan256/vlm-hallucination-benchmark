# MMHal: GPT-6 interactive judge, protocol v1

> Vòng review ảnh bổ sung đã thực hiện cho 96 mẫu theo yêu cầu hoàn thiện phân
> tích. Kết quả tại `reports/smolvlm2_mmhal_20261119/`; giữ nguyên score vòng text,
> không đổi thành official GPT-4 score hoặc human ground truth. Có 8 ca chưa kết
> luận; reviewer người độc lập vẫn chưa được thực hiện.

Quyết định theo yêu cầu người phụ trách dự án trong chat Codex ngày 19/11/2026:
dùng GPT-6 trong phiên tương tác để chấm thay GPT-4 API. Đây là **phương án judge
thay thế**, không tuyên bố tương đương điểm GPT-4 của tác giả. Không cần API key
hay thực hiện request trả phí để nhập kết quả chấm tương tác.

## Những phần giữ nguyên phương pháp tự động gốc

Nguồn chuẩn: `benchmarks/mmhal/vendor/eval_gpt4.py.txt`, sao chép nguyên vẹn tại
dataset revision `f5f49a938f45ed99e235b8519ba28f76832a2add`. Hàm
`benchmarks.mmhal.judge.original_template()` xác minh SHA-256 trước khi đọc.

- Giữ nguyên toàn bộ template tiếng Anh, năm ví dụ và bảy mức điểm.
- Mỗi mẫu dùng chính xác bốn trường: `', '.join(image_content)`, `question`,
  `gt_answer`, `model_answer`, nội suy bằng `template.format(...)`.
- Judge chấm trên văn bản như script gốc, không đưa ảnh gốc hoặc web search vào
  lượt chấm chính. Không dùng question_type/topic, truncation hoặc output của
  model khác để quyết định điểm. Các trường đó chỉ phân tích sau chấm.
- Giải thích trước, sau đó ghi đúng một `Rating: N`, với N nguyên từ 0 đến 6.
- 6: rất informative, có phân tích/suy luận tốt, không hallucination;
  5: rất informative, không hallucination;
  4: somewhat informative, không hallucination;
  3: not informative, không hallucination;
  2: rất informative, có hallucination;
  1: somewhat informative, có hallucination;
  0: not informative, có hallucination.
- Đáp án chuẩn có thể không đầy đủ. Chi tiết ngoài đáp án không tự động là sai.
  Thiếu chi tiết, từ chối hoặc thể hiện không chắc chắn không tự là hallucination.
  Các khẳng định số lượng/quan hệ/thuộc tính sai rõ so với chứng cứ vẫn được chấm
  là hallucination theo ví dụ gốc, đồng thời ghi loại lỗi riêng.
- Điểm tổng = tổng 96 điểm / 96. Hallucination rate = số điểm < 3 / 96.
  Từng loại câu hỏi = các chỉ số dòng i có i % 8 bằng chỉ số loại, đúng thứ tự
  attribute, adversarial, comparison, counting, relation, environment, holistic,
  other. Kiểm tra lại tên loại trên mỗi dòng, không sắp xếp lại nguồn.

## Những phần khác biệt bắt buộc công bố

Judge thuộc họ GPT-6 trong chat Codex này, không có lời gọi API GPT-4. Không có
API model snapshot, temperature hoặc seed truy cập được cho lượt tương tác;
phải ghi null/not controllable, không tự nhận temperature=0 hay snapshot cụ thể.
Các mẫu được đọc/chấm trong ngữ cảnh phiên chung, không phải 96 request độc lập.
Do đó giữ prompt/rubric/input/công thức không có nghĩa tái lập chính xác quy trình
vận hành API hoặc điểm số gốc. Không có independent second judge hay human review
nếu chưa thực hiện và ghi bằng chứng.

Script gốc gán 0 nếu tìm thấy không đúng một giá trị điểm. Phương án mới lưu raw
text và kiểm tra parser gốc; nếu bất kỳ phản hồi nào không hợp lệ, dừng tổng hợp
để sửa lỗi nhập/chấm, không biến lỗi kỹ thuật thành ảo giác. Khi có đủ phản hồi
hợp lệ, kết quả parser và công thức phải khớp script gốc. Không sửa run inference.

`needs_visual_review` là chú thích QA sau chấm, không thay điểm gốc. Đánh dấu này
không chứng minh nhãn sai hay model sai; cần kiểm tra ảnh và reviewer độc lập ở
vòng sau. Không gọi các nhãn do judge tạo là ground truth đã xác nhận. Nếu review
ảnh làm thay đổi điểm, xuất run bổ sung với protocol image-assisted riêng.

## Artifact và quy trình

1. Verify manifest inference, đủ 96 prediction COMPLETED và responses.json đúng
   từng dòng gốc, không trộn run. Giữ nguyên trạng thái pilot/exploratory của nguồn.
2. Lưu 96 prompt được tạo từ template gốc để audit đầu vào; prompt được lưu là
   bản dựng lại chính xác từ nguồn, không giả là log HTTP đã gửi.
3. Lưu từng nhận xét/điểm do GPT-6 đưa ra dưới dạng ledger cùng raw review text.
   Ledger là kết quả phán đoán trong phiên, mã import chỉ kiểm tra/tính toán,
   không phải thuật toán tự suy ra điểm từ so khớp đáp án.
4. Import bằng `python -m benchmarks.mmhal.import_review`, kiểm tra 96 ID, điểm,
   original parser, original aggregation, checksum và không ghi đè run.
5. Xuất prediction chấm, metrics, errors, config, metadata, status, manifest,
   bảng cases.csv và báo cáo. Gắn source run hash, source checksum, rubric hash,
   review ledger checksum và code checksum. Ngày giờ artifact không giả là
   thời gian từng request API; API cost/usage là not applicable.
6. Chỉ gọi kết quả là `MMHal-Bench / GPT-6 interactive / text-reference`.
   Không dùng nhãn `official GPT-4 score`. Tiêu chí nghiệm thu khoa học vẫn cần
   nhóm review; thay judge không tự chuyển một pilot run thành official run.

## Trạng thái hiện tại

Nguồn `full-infer-001`: 96/96 inference thành công trên RTX 5060, BF16, greedy,
256 token. Nguồn có stage pilot và Git dirty được lưu trung thực. Có 10 output
chạm giới hạn token, đều thuộc holistic; giữ nguyên và phân tích riêng, không
phạt thêm điểm chỉ vì bị cắt. Pilot 001 từng lỗi bộ nhớ, pilot 002 đã chạy đủ 8.
Đây không phải bằng chứng đã xác minh đủ mọi yêu cầu của official evaluation.

API runner cũ vẫn giữ để đối chứng khi được yêu cầu rõ; không cần sửa khóa
`judge` trong cấu hình inference đã lưu. Backend tương tác dùng config riêng
trong run chấm. POPE và HallusionBench giữ nguyên protocol.
