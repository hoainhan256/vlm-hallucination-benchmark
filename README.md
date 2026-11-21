# VLM Hallucination Benchmark

Hạ tầng mã nguồn và quy trình nghiên cứu cho đánh giá ảo giác của Vision-Language Models (mô hình thị giác–ngôn ngữ).

Repository đang ở giai đoạn thiết lập infrastructure. Nó không tải model/dataset, không chạy inference và không tạo kết quả benchmark chính thức.

## Bắt đầu theo nhóm

Đọc [`START_HERE.md`](START_HERE.md) trước khi sửa mã nguồn. Sau đó đọc hướng dẫn nhóm tương ứng:

- [Nhóm POPE](docs/guides/pope_team.md)
- [Nhóm HallusionBench](docs/guides/hallusionbench_team.md)
- [Nhóm MMHal-Bench](docs/guides/mmhal_team.md)

## Phạm vi

- 3 mô hình: Qwen3-VL-4B-Instruct, SmolVLM2-2.2B-Instruct, Phi-4-multimodal-instruct
- 3 benchmark: POPE, HallusionBench, MMHal-Bench
- 9 tổ hợp model–benchmark
- 11 thành viên và một giảng viên hướng dẫn

## Cấu trúc chính

- `configs/`: model, dataset và experiment configuration
- `benchmarks/`: interface chung và thư mục protocol riêng
- `models/`: interface model abstraction
- `vlm_benchmark/`: config/result/environment utilities
- `scripts/`: công cụ kiểm tra không thực thi benchmark
- `docs/`: protocol, chính sách và quy trình vận hành
- `tests/`: CPU-only tests và synthetic fixtures

## Cài đặt kiểm thử

```bash
python -m pip install -r requirements.txt
python scripts/check_repository.py
python scripts/check_config.py --mode config
python scripts/run_benchmark.py --mode check-config
pytest -q
```

Chế độ official chỉ chạy kiểm tra cấu hình và từ chối cấu hình chưa được khóa:

```bash
python scripts/run_benchmark.py --mode official
```

Xem `docs/experiment_protocol.md` để biết quy trình pilot và official, và `docs/README.md` để điều hướng tài liệu.
