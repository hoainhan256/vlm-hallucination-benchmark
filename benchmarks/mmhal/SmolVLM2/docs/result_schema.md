# Schema lưu trữ kết quả

Mỗi run nằm dưới `results/<experiment_id>/<run_id>/` và không ghi đè run đã hoàn thành.

## Tệp bắt buộc

- `predictions.jsonl`: một record JSON cho mỗi sample: `sample_id`, `prediction`, `status`, `run_config_hash`; có thể có `raw_prediction`, `reference`, `prompt`, `timing_ms`.
- `metrics.json`: chỉ số benchmark; không tạo điểm tổng hợp 3 benchmark.
- `run_config.yaml`: cấu hình thực tế đã dùng.
- `run_metadata.json`: model/dataset revisions, Git commit, Python/PyTorch/Transformers/Accelerate/CUDA/GPU, device allocation và trạng thái.
- `errors.jsonl`: lỗi kỹ thuật, dữ liệu hoặc định dạng theo từng sample.
- `run_status.json`: `PENDING`, `RUNNING`, `COMPLETED`, `BLOCKED` hoặc `FAILED`.
- `manifest.json`: danh sách tệp, checksum, run/experiment identity và cấu hình chuẩn hóa.

## Trạng thái

- `PENDING`: chưa bắt đầu.
- `RUNNING`: đang xử lý.
- `COMPLETED`: đủ sample trong scope hoặc phần chạy đã được công nhận.
- `BLOCKED`: không thể chạy do cấu hình/phần cứng không tương thích.
- `FAILED`: có lỗi và không thể tiếp tục ở trạng thái hiện tại.

Không tạo dữ liệu/result giả trong thư mục chính thức. Fixture trong `tests/fixtures/` chỉ là dữ liệu giả lập.
