# Hướng dẫn nhóm POPE

## Mục tiêu

Xây dựng adapter (lớp kết nối) POPE giữ nguyên ba sampling strategies: `random`, `popular`, `adversarial`; chuẩn hóa dữ liệu, gọi model theo prompt đã xác minh, chấm điểm theo protocol gốc và xuất artifact tái lập được.

## Tệp hiện có liên quan

| Tệp | Chức năng hiện tại |
|---|---|
| `benchmarks/pope/README.md` | Tệp rỗng; chưa có triển khai POPE. |
| `benchmarks/base.py` | Interface benchmark bắt buộc phải hiện thực. |
| `configs/datasets.yaml` | Khai báo nguồn `lmms-lab/POPE`, ba subset, schema tối thiểu và trạng thái xác minh. |
| `configs/experiment.yaml` | Khai báo ba tổ hợp model–POPE và chính sách chạy. |
| `docs/evaluation_policy.md` | Yêu cầu giữ ba strategy và metric riêng. |
| `docs/dataset_policy.md` | Quy tắc revision, nhãn, chuyển đổi dữ liệu. |
| `docs/result_schema.md` | Schema artifact đầu ra. |
| `vlm_benchmark/results.py` | Kiểm tra prediction, hash cấu hình và resume. |
| `tests/test_results.py`, `tests/fixtures/synthetic_predictions.jsonl` | Test hạ tầng kết quả và fixture giả, không phải dữ liệu POPE. |

## Phân công gợi ý cho ba thành viên

1. **Dữ liệu/protocol:** xác minh nguồn, revision, license, schema và ba subset; viết loader/normalizer và test fixture nhỏ.
2. **Input/model integration:** hiện thực chuẩn bị ảnh, câu hỏi, prompt và adapter gọi interface; không sửa `models/` nếu chưa được duyệt.
3. **Scoring/QA:** hiện thực metric theo từng strategy, export kết quả, test lỗi/trùng mẫu và tài liệu tái lập.

## Thứ tự triển khai

1. Ghi nguồn, revision và bằng chứng protocol; cập nhật trạng thái sau phê duyệt.
2. Hiện thực đọc dữ liệu và kiểm tra schema mà không làm thay đổi dữ liệu gốc.
3. Chốt prompt/input và kiểm thử bằng fixture tổng hợp.
4. Chốt scorer và artifact; kiểm thử resume, manifest và lỗi.
5. Chạy pilot chỉ sau khi trưởng nhóm duyệt; official chỉ khi checklist đủ.

## Đầu ra phải bàn giao

Mã trong `benchmarks/pope/`, test tập trung, tài liệu protocol/prompt, bảng revision và license, cùng run artifact gồm `predictions.jsonl`, `metrics.json`, `run_config.yaml`, `run_metadata.json`, `errors.jsonl`, `run_status.json`, `manifest.json` theo `docs/result_schema.md`. Không nộp dataset hoặc trọng số vào repository.

## Tiêu chí kiểm thử

Kiểm tra đủ ba subset, không đổi `sample_id`/nhãn, không rò rỉ đáp án vào input, metric tách riêng theo strategy, prediction không trùng mẫu và cùng `run_config_hash`. Test phải bắt được record thiếu trường, JSONL hỏng và cố ghi đè mẫu đã `COMPLETED`.

## Làm việc với mã nguồn dùng chung

`benchmarks/base.py`, `models/`, `configs/` và `scripts/` dùng chung cho ba nhóm. Tạo PR nhỏ, báo trưởng nhóm trước khi đổi interface hoặc config; không đổi policy để làm POPE chạy riêng.

## Vấn đề chưa được xác minh

Dataset revision/license, số mẫu, schema thực tế, prompt, quy tắc nhãn/metric, revision processor/model, BF16, `max_new_tokens` và khả năng chạy GPU đều đang chờ xác minh. `benchmarks/pope/README.md` hiện rỗng và chưa có code loader/scorer.
