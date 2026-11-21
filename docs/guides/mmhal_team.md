# Hướng dẫn nhóm MMHal-Bench

## Mục tiêu

Xây dựng adapter MMHal-Bench giữ protocol judge (mô hình giám khảo) gốc, rubric (bảng tiêu chí chấm), prompt và cách ghi điểm; kiểm soát riêng quyền truy cập, phiên bản và chi phí của judge.

## Tệp hiện có liên quan

| Tệp | Chức năng hiện tại |
|---|---|
| `benchmarks/mmhal/README.md` | Tệp rỗng; chưa có triển khai MMHal-Bench. |
| `benchmarks/base.py` | Interface benchmark bắt buộc hiện thực. |
| `configs/datasets.yaml` | Khai báo nguồn `Shengcao1006/MMHal-Bench`, schema tối thiểu và judge protocol đang chờ xác minh. |
| `configs/experiment.yaml` | Khai báo ba tổ hợp model–MMHal. |
| `docs/evaluation_policy.md` | Yêu cầu dùng judge protocol gốc và không coi judge khác là tương đương. |
| `docs/experiment_protocol.md` | Yêu cầu lưu judge, version, rubric, prompt và judge response. |
| `docs/result_schema.md` | Quy định artifact và trạng thái run. |
| `vlm_benchmark/results.py` | Validation prediction, hash cấu hình và resume. |
| `tests/test_results.py` | Test hạ tầng; chưa có test judge/MMHal-specific. |

## Phân công gợi ý cho ba thành viên

1. **Dataset/protocol:** xác minh source, revision, sample schema, license và prompt dữ liệu.
2. **Judge/chi phí:** xác minh judge model/version, quyền truy cập, rubric, prompt, giá và lưu judge response.
3. **Scoring/QA:** hiện thực parser/score, export artifact, test lỗi judge, resume và phân tích mẫu khó.

## Thứ tự triển khai

1. Xác minh tính hợp pháp, quyền truy cập, version và chi phí judge; chưa gọi dịch vụ thật khi chưa được duyệt.
2. Khóa schema và input model, bảo đảm không đưa reference answer vào prompt model.
3. Thiết kế lưu raw model response và raw judge response trước khi chuẩn hóa.
4. Hiện thực scorer theo rubric gốc, test parser và lỗi/timeout.
5. Pilot có giới hạn sau phê duyệt; official chỉ khi chi phí, revision và checklist được ghi nhận.

## Đầu ra phải bàn giao

Mã/test trong `benchmarks/mmhal/`, tài liệu judge protocol, bằng chứng access/version/cost, rubric và prompt đã khóa. Run phải có đủ artifact bắt buộc, đồng thời metadata phải ghi judge model/version, request status, chi phí nếu có và raw responses theo chính sách.

## Tiêu chí kiểm thử

Test schema sample, không rò rỉ reference, lưu được raw response, parser xử lý đúng rubric và phản hồi lỗi, không trùng sample, không trộn `run_config_hash`, không ghi đè completed run. Có test judge unavailable/timeout ở chế độ không gọi dịch vụ thật.

## Làm việc với mã nguồn dùng chung

`models/`, `configs/`, `scripts/` và `benchmarks/base.py` là mã dùng chung. Mọi thay đổi field/config hoặc dependency phải được trưởng nhóm duyệt vì có thể ảnh hưởng hai benchmark còn lại; không hard-code token/API key.

## Vấn đề chưa được xác minh

Revision/license và số mẫu dataset, judge model/version, rubric, prompt, quyền truy cập, chi phí, rate limit, lưu response, model/processor revision, BF16 và generation config đều chưa được xác minh. `benchmarks/mmhal/README.md` rỗng; chưa có judge adapter hay scorer.
