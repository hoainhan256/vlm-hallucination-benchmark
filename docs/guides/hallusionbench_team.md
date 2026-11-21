# Hướng dẫn nhóm HallusionBench

## Mục tiêu

Xây dựng adapter HallusionBench bảo toàn cấu trúc sample, các cặp kiểm soát và aggregation (cách gộp điểm) của phiên bản đã chọn; phân biệt lỗi dữ liệu, lỗi kỹ thuật và ảo giác.

## Tệp hiện có liên quan

| Tệp | Chức năng hiện tại |
|---|---|
| `benchmarks/hallusionbench/README.md` | Tệp rỗng; chưa có triển khai. |
| `benchmarks/base.py` | Interface đọc, chuẩn hóa, input, model, score và export. |
| `configs/datasets.yaml` | Khai báo nguồn `tianyi-lab/HallusionBench`, chưa khóa revision và chưa khai báo subset. |
| `configs/experiment.yaml` | Khai báo ba tổ hợp model–HallusionBench. |
| `docs/evaluation_policy.md` | Yêu cầu giữ câu hỏi/cặp kiểm soát và aggregation. |
| `docs/dataset_policy.md` | Chính sách không sửa sample/nhãn và phải ghi revision. |
| `docs/result_schema.md` | Schema và trạng thái run. |
| `vlm_benchmark/results.py` | Validation và merge kết quả. |
| `tests/test_results.py` | Test hạ tầng kết quả, không có test HallusionBench-specific. |

## Phân công gợi ý cho ba thành viên

1. **Cấu trúc dữ liệu:** xác minh sample, cặp kiểm soát, media, nhãn và loader/normalizer.
2. **Protocol input:** xác minh prompt, thứ tự câu hỏi, chuẩn bị input và tích hợp interface model.
3. **Aggregation/QA:** hiện thực scorer, phân tích lỗi theo cặp và test export/resume.

## Thứ tự triển khai

1. Khóa phiên bản benchmark và mô tả cấu trúc sample từ nguồn chính thức.
2. Viết loader/normalizer có kiểm tra cặp và giữ nguyên liên kết sample.
3. Chốt prompt và input bằng fixture, không dùng nhãn tham chiếu làm input.
4. Hiện thực aggregation và test các trường hợp thiếu cặp/sai schema.
5. Pilot rồi mới official sau checklist và phê duyệt.

## Đầu ra phải bàn giao

Mã và test trong `benchmarks/hallusionbench/`, tài liệu cấu trúc/cặp/aggregation, bằng chứng revision/license và hướng dẫn tái lập. Mỗi run phải có đủ bảy artifact theo `docs/result_schema.md`, gồm raw prediction và `errors.jsonl`; không đưa dataset lớn vào Git.

## Tiêu chí kiểm thử

Test phải xác nhận cấu trúc và cặp kiểm soát không bị đảo/mất, sample không bị trùng, aggregation dùng đúng version protocol, prediction giữ `sample_id` và hash cấu hình, lỗi kỹ thuật không bị tính như điểm. Kiểm tra cả run thiếu cặp và resume không ghi đè.

## Làm việc với mã nguồn dùng chung

Không tự sửa `models/`, `configs/`, `scripts/` hoặc `benchmarks/base.py`. Nếu cần interface hoặc field mới, mô tả ảnh hưởng tới POPE/MMHal-Bench trong PR và chờ trưởng nhóm duyệt.

## Vấn đề chưa được xác minh

Revision/license, số lượng mẫu, subset, cấu trúc cặp thực tế, prompt, quy tắc aggregation, model/processor revision, dtype, device và generation config đều chưa được khóa. `benchmarks/hallusionbench/README.md` rỗng; chưa có loader, adapter hay scorer.
