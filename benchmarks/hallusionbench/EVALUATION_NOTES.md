# HallusionBench — Tài liệu Giao thức Chấm điểm & Phân tích Lỗi (Evaluation Notes)

> **Phụ trách:** Nguyễn Tấn Vũ (Evaluation & QA) — Nhóm 2  
> **Trạng thái giao thức:** `unverified_draft_rule_based` (Chờ trưởng nhóm chốt Blocker P1 về phương pháp chấm và đối chiếu `evaluation.py` gốc trước khi chạy Official)

---

## 1. Quy ước phương pháp chấm điểm

- **Phân loại chế độ chấm:** `alternative_evaluation_rule_based`.
- **Lý do & Giới hạn:** Trong mã nguồn gốc của HallusionBench (`tianyi-lab/HallusionBench/evaluation.py`), các câu trả lời dài hoặc không khớp trực tiếp từ khóa có thể sử dụng mô hình giám khảo GPT-4. Do dự án hiện chưa cấp ngân sách/API GPT-4 judge, bộ chấm này sử dụng bộ trích xuất quy tắc (`parse_yes_no`) và được gắn nhãn rõ là **alternative evaluation**, không giả nhận tương đương trực tiếp với bảng xếp hạng dùng GPT-4 judge gốc.
- **Quy tắc chuẩn hóa (`parse_yes_no`):**
  1. Loại bỏ ngay các câu trả lời nước đôi trực tiếp chứa cụm `yes and no`, `yes or no`, `yes/no` (trả về `None` — ghi nhận `format_violation`).
  2. Khớp từ khóa `yes` hoặc `no` ngay đầu câu trả lời (không phân biệt hoa/thường).
  3. Với câu giải thích dài, chỉ chốt `yes` hoặc `no` nếu toàn bộ câu trả lời chỉ chứa một trong hai từ khóa khẳng định/phủ định (tránh nhận nhầm câu giải thích chứa cả hai vế).
  4. Các trường hợp không trích xuất được `yes`/`no` hợp lệ được ghi vào `errors.jsonl` với nhãn `format_violation` và tính `0` điểm trong mẫu số câu hỏi hợp lệ.

---

## 2. Định nghĩa các chỉ số (Metrics)

Mọi chỉ số đều chỉ tính trên các bản ghi có `status == "COMPLETED"` và được báo cáo kèm mẫu số cụ thể:

| Chỉ số | Tên đầy đủ | Quy tắc tính toán |
|---|---|---|
| **`aAcc`** | Accuracy per Question | Tỷ lệ câu hỏi trả lời đúng trên tổng số câu hỏi hợp lệ (`scored_questions`). |
| **`fAcc`** | Accuracy per Figure | Tỷ lệ hình ảnh (`figure_key`) mà mô hình trả lời đúng **toàn bộ** các câu hỏi gắn với hình đó. |
| **`qAcc`** | Accuracy per Question Pair | Tỷ lệ cặp câu hỏi kiểm soát (`pair_key`, yêu cầu đủ $\ge 2$ biến thể điều kiện thị giác) mà mô hình trả lời đúng **cả cặp**. Các cặp thiếu biến thể (ví dụ trong tập pilot chưa đủ cặp) được đếm vào `incomplete_pairs_skipped` và không làm sai lệch mẫu số `qAcc`. |
| **`by_category`** | Phân tách `VD` vs. `VS` | Báo cáo riêng bộ 3 chỉ số (`aAcc`, `fAcc`, `qAcc`) cho nhóm **Visual Dependent (`VD`)** và **Visual Supplement (`VS`)**. |
| **`by_difficulty`** | Phân tách `Easy` vs. `Hard` | Báo cáo riêng bộ 3 chỉ số theo độ khó của từng phân nhóm dữ liệu (nếu có trong metadata). |

---

## 3. Hợp đồng dữ liệu (Data Contract) với Thành viên A và B

- **Từ Thành viên A (`dataset_records`):** Mỗi `sample_id` cần cung cấp `reference_answer` (`yes`/`no`), `category` (`VD`/`VS`), `difficulty` (nếu có), cùng `figure_key` và `pair_key` (hoặc bộ trường gốc `subcategory`, `set_id`, `figure_id`, `question_id` để `scoring.py` tự ghép khóa).
- **Từ Thành viên B (`predictions.jsonl`):** Mỗi dòng bắt buộc có đủ 4 trường `sample_id`, `prediction`, `status`, `run_config_hash`. Các bản ghi có `status != "COMPLETED"` (như `FAILED` do OOM, timeout) được tách sang `errors.jsonl` với nhãn `technical_error` và **không** bị tính như dự đoán sai.
- **Bảo toàn dữ liệu khi chạy tiếp (Resume):** Hàm `export_hallusionbench_artifacts` sử dụng `merge_result_records` để ngăn chặn việc ghi đè lên các `sample_id` đã có trạng thái `COMPLETED`.

---

## 4. Khung 6 nhãn phân loại lỗi (Error Taxonomy — Gate G5)

Khi phân tích các trường hợp trả lời sai trong `analysis.py`, tuyệt đối không mặc định mọi câu sai đều là ảo giác; mỗi ca phải được phân loại vào đúng 1 trong 6 nhãn theo `docs/evaluation_policy.md`:

1. **`incorrect prediction`**: Câu trả lời chốt khác nhãn tham chiếu theo protocol.
2. **`hallucination`**: Nội dung không có căn cứ hoặc mâu thuẫn bằng chứng thị giác (bịa đối tượng không có trong ảnh, bỏ sót/nhận diện sai đối tượng, suy luận sai thuộc tính/quan hệ không gian, hoặc trả lời theo thiên kiến ngôn ngữ bất chấp ảnh).
3. **`format violation`**: Mô hình trả lời mơ hồ, từ chối trả lời hoặc không chốt được Yes/No.
4. **`technical error`**: Lỗi hạ tầng, tràn VRAM (CUDA OOM), timeout hoặc lỗi I/O từ bước inference.
5. **`label/data error`**: Có bằng chứng cho thấy câu hỏi, nhãn tham chiếu hoặc tệp ảnh của dataset bị lỗi.
6. **`undetermined`**: Thiếu bằng chứng để kết luận chắc chắn nguyên nhân.

---

## 5. Cách chạy kiểm thử trên CPU

```powershell
python -m pytest benchmarks/hallusionbench/tests/test_scoring.py -v