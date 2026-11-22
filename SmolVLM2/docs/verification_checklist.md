# Checklist xác minh trước benchmark thật

| Hạng mục | Trạng thái | Ghi chú |
|---|---|---|
| Git commit mã nguồn đã ghi | PENDING_VERIFICATION | |
| Revision ba mô hình | PENDING_VERIFICATION | Không suy đoán commit |
| Processor/tokenizer revision | PENDING_VERIFICATION | |
| BF16 tương thích từng mô hình/GPU | PENDING_VERIFICATION | Chưa phê duyệt |
| Dataset revision và license | PENDING_VERIFICATION | |
| Số lượng sample chính thức | PENDING_VERIFICATION | |
| POPE sampling và metric | PENDING_VERIFICATION | |
| HallusionBench aggregation | PENDING_VERIFICATION | |
| MMHal judge, rubric, truy cập và chi phí | INTERACTIVE_ALTERNATIVE_REVIEWED | GPT-6 tương tác theo yêu cầu 19/11/2026; rubric/prompt/công thức gốc được giữ; không gọi API, snapshot/temperature không truy cập được; human/visual review còn chờ. Xem `docs/mmhal_gpt6_protocol.md`. |
| max_new_tokens và generation config | PENDING_VERIFICATION | |
| Không lượng tử hóa | APPROVED | Theo chính sách cấu hình |
| Không fine-tuning | APPROVED | Theo chính sách cấu hình |
| GPU/CUDA/PyTorch tương thích | PENDING_VERIFICATION | |
| Resume/manifest/schema tests | PENDING_VERIFICATION | Chạy sau triển khai |

Trạng thái chỉ được đổi thành APPROVED khi có bằng chứng kiểm chứng hoặc phê duyệt ghi nhận.
