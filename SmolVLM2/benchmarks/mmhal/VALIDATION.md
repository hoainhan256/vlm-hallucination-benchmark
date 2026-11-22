# Bằng chứng kiểm tra triển khai — 19/11/2026

> Cập nhật sau thực nghiệm cùng ngày: đã có run thật `full-infer-001` đủ 96/96,
> RTX 5060/BF16/CUDA 12.8; đã chấm bằng GPT-6 tương tác theo protocol mới
> `docs/mmhal_gpt6_protocol.md`. Bộ test hiện đạt 49 passed. Artifact tại
> `results/smolvlm2-mmhal-baseline/gpt6-interactive-001/` đã qua verify checksum,
> check_results (96 records), check_manifest. Phần chưa chạy inference/GPU phía
> dưới là ghi chép trước đó, không phải trạng thái hiện tại. API GPT-4 vẫn chưa
> được gọi; 24 ca cần review ảnh và nghiệm thu khoa học của nhóm còn chờ.

Phạm vi: mã mới trong `benchmarks/mmhal/`, nhánh cục bộ
`benchmark/mmhal-smolvlm2`. Nhánh này lấy nền tảng từ checkout ban đầu đã có
interface và utilities; nhánh `benchmark/mmhal` cũ chưa có hạ tầng tương ứng.
Chưa commit hoặc push thay đổi.

## Đã thực hiện

- Đọc protocol, model/dataset/evaluation policy, result schema, checklist,
  Git workflow và guide nhóm MMHal.
- Đọc model card, dataset card, schema template và scorer chính thức; tra
  Hugging Face API để lấy exact revisions ghi trong `config.yaml`.
- Lưu nguyên mã chấm upstream, checksum và giấy phép trong `vendor/`.
- `python -m pytest -q`: **46 passed**, gồm test chung ban đầu và test MMHal.
- `python -m benchmarks.mmhal check`: đạt kiểm tra cấu trúc config/checksum rubric.
- `python scripts/check_repository.py`: đạt.
- `python -m compileall -q benchmarks/mmhal`: đạt.
- `git diff --check`: không có lỗi whitespace; Git có cảnh báo LF/CRLF theo
  thiết lập checkout Windows.

Pytest 8.4.2 được cài riêng vào `.cache/test-deps`, không sửa dependency chung.
Lệnh test trong môi trường kiểm tra này dùng PowerShell:

```powershell
$env:PYTHONPATH = '.cache/test-deps'
python -m pytest -q
```

Test MMHal kiểm tra schema/thứ tự/chọn mẫu, không rò rỉ đáp án vào VLM, giữ
prompt judge, thang điểm/coverage, timeout/key thiếu, lưu artifacts/checksum,
chống ghi đè, một writer, resume không trộn config, export upstream và toàn
luồng inference/judge/review bằng fake adapters. Test adapter xác nhận decode
chỉ token mới và phân biệt EOS với truncation. Các fake artifact chỉ nằm trong
thư mục tạm của pytest, không nằm trong `results/`.

## Chưa xác nhận và điều kiện chạy thật

- PyTorch được cài báo `cuda_available=False`, `gpu_count=0` trong môi trường
  hiện tại. Chưa xác nhận driver/CUDA/native BF16/VRAM trên máy chạy thực tế.
- Chưa tải trọng số, dataset ảnh hoặc chạy inference thật. Chưa xác nhận tương
  thích model/processor với một môi trường GPU được khóa, thời gian và bộ nhớ.
- Chưa gọi API judge, chưa xác minh khả năng truy cập `gpt-4-0314`, chi phí,
  quota và độ ổn định chấm điểm. Không có điểm số khoa học để báo cáo.
- Các approval trong config mới vẫn null; exact revisions được quan sát từ
  nguồn chính thức không có nghĩa đã được nhóm phê duyệt.
- `python scripts/check_config.py --mode config` vẫn thất bại do config chung
  còn revision và max_new_tokens chưa khóa. Giữ nguyên config chung theo phạm
  vi nhóm; không tuyên bố bộ runner chung đã được hoàn thành.
- Cần chạy pilot trên GPU đã duyệt, đối chiếu input/ảnh/đầu ra và judge, review
  config/môi trường với nhóm, rồi mới commit cấu hình official và chạy đủ 96 mẫu.

Hướng dẫn tái lập và quy trình phân tích từng ca nằm trong `README.md` cùng
thư mục. Không có thay đổi nào phụ thuộc thư mục `benchmark/` bị tạo nhầm.
