# Chính sách mô hình

## Trọng số và phiên bản

- Chỉ sử dụng trọng số chính thức (official weights) do nhà phát hành công bố.
- Cấu hình `revision` và `processor_revision` phải là revision đã khóa; không được để `null`.
- Không fine-tuning, không đổi trọng số và không thay thế mô hình bằng một phiên bản mới chưa phê duyệt.
- Cùng model ID, revision, processor và cấu hình dtype phải được dùng cho cả ba nhóm.

## Dtype và lượng tử hóa

- BF16 là ứng viên, chưa được phê duyệt. Trạng thái hiện tại phải là `PENDING_VERIFICATION`.
- Không lượng tử hóa INT8, INT4, GPTQ, AWQ hoặc phương pháp tương đương.
- Không tự chuyển sang FP16/FP32 để vượt lỗi phần cứng hoặc làm dtype fallback.
- Chỉ cho phép thêm dtype vào `approved_dtypes` sau khi có bằng chứng tương thích.

## CPU offload

CPU offload chỉ được phép khi không thay đổi trọng số và không thay đổi dtype đã phê duyệt. Thực nghiệm phải ghi rõ phân bổ thiết bị. Nếu không tương thích, chạy phải có trạng thái `BLOCKED` kèm lý do; không mặc nhiên coi kết quả giữa nền tảng tương đương.
