# Bắt đầu từ đây

## 1. Mục tiêu dự án

Repository này xây dựng hạ tầng và quy trình để đánh giá hiện tượng **ảo giác** của Vision-Language Model (VLM, mô hình thị giác–ngôn ngữ) trên ba benchmark: POPE, HallusionBench và MMHal-Bench. Có ba model được khai báo trong cấu hình, tạo thành chín tổ hợp model–benchmark.

**Trạng thái hiện tại:** nền tảng đã có cấu hình, interface (hợp đồng lớp), bộ kiểm tra và chính sách. Chưa có adapter model, loader dataset, pipeline inference (suy luận) hay scorer (bộ chấm điểm) hoàn chỉnh. Các revision model/dataset và nhiều thông số vẫn `PENDING_VERIFICATION`. Không coi repository đã chạy benchmark thật nếu chưa có bằng chứng kiểm thử.

## 2. Cấu trúc và chức năng

| Đường dẫn | Chức năng thực tế |
|---|---|
| `benchmarks/base.py` | Interface chung gồm đọc dữ liệu, chuẩn hóa, chuẩn bị input, gọi model, chấm điểm và xuất kết quả. |
| `benchmarks/pope/` | Thư mục dành cho POPE; hiện chỉ có `README.md` rỗng, chưa có mã benchmark. |
| `benchmarks/hallusionbench/` | Thư mục dành cho HallusionBench; hiện chỉ có `README.md` rỗng. |
| `benchmarks/mmhal/` | Thư mục dành cho MMHal-Bench; hiện chỉ có `README.md` rỗng. |
| `models/base.py` | Interface VLM; chưa tải trọng số hoặc chạy inference. |
| `vlm_benchmark/config.py` | Đọc YAML và kiểm tra chính sách model, dataset, experiment. |
| `vlm_benchmark/results.py` | Đọc/kiểm tra JSONL, chống trùng mẫu và chống ghi đè kết quả đã hoàn tất. |
| `vlm_benchmark/environment.py` | Thu thập metadata môi trường; không tải model/dataset. |
| `configs/` | `models.yaml`, `datasets.yaml`, `experiment.yaml`; các revision quan trọng hiện chưa khóa. |
| `scripts/` | `check_repository.py`, `check_config.py`, `check_manifest.py`, `check_results.py`, `run_benchmark.py`, `collect_environment.py`. Chúng là công cụ kiểm tra; `run_benchmark.py` hiện chỉ kiểm tra cấu hình. |
| `docs/` | Protocol, chính sách model/dataset, schema kết quả, Git và checklist xác minh. |
| `tests/` | Test CPU-only và fixture giả lập; không phải kết quả benchmark thật. |
| `results/` | Nơi quy định lưu artifact chạy; hiện chỉ có README rỗng. |
| `notebooks/` | Quy ước notebook; không có notebook thực tế. |

## 3. Tệp bắt buộc phải đọc

Tất cả thành viên đọc trước: `README.md`, `docs/experiment_protocol.md`, `docs/dataset_policy.md`, `docs/model_policy.md`, `docs/result_schema.md`, `docs/git_workflow.md`, `docs/verification_checklist.md` và `configs/experiment.yaml`.

Sau đó đọc tài liệu nhóm tương ứng:

- POPE: `docs/guides/pope_team.md`
- HallusionBench: `docs/guides/hallusionbench_team.md`
- MMHal-Bench: `docs/guides/mmhal_team.md`

## 4. Clone và chuyển sang nhánh nhóm

```bash
git clone <repository-url>
cd vlm-hallucination-benchmark
git fetch --prune origin
```

Nếu nhánh nhóm đã tồn tại trên remote:

```bash
git switch --track origin/benchmark/pope
# hoặc benchmark/hallusionbench, benchmark/mmhal
git pull --ff-only
```

Nếu chưa có nhánh trên remote, tạo nhánh cục bộ rồi đẩy lần đầu:

```bash
git switch -c benchmark/pope
git push -u origin benchmark/pope
```

Tại thời điểm rà soát, repository chỉ có `main` và `origin/main`; ba nhánh nhóm chưa xuất hiện. Trưởng nhóm cần thống nhất việc tạo nhánh trước khi phân công chính thức.

## 5. Xác định phần việc

Chọn đúng một nhóm benchmark và đọc guide của nhóm đó. Phần thuộc nhóm là `benchmarks/<tên benchmark>/` cùng dữ liệu, protocol và test trực tiếp của benchmark. `models/`, `configs/` và `scripts/` là mã dùng chung: hỏi trưởng nhóm trước khi sửa. Không tự đổi model, dataset, schema hoặc cách chấm để “cho chạy được”.

## 6. Kiểm tra và bàn giao

Sau khi cài dependency:

```bash
python -m pip install -r requirements.txt
python scripts/check_repository.py
python scripts/check_config.py --mode config
pytest -q
```

Kiểm tra artifact bằng `python scripts/check_results.py <đường-dẫn-predictions.jsonl>` và manifest bằng `python scripts/check_manifest.py <run_dir>` khi các tệp đó đã được triển khai. Xem `git diff`, ghi test đã chạy và các vấn đề chưa xác minh trong Pull Request (yêu cầu hợp nhất).

```bash
git status
git diff
git add <tệp-thuộc-phần-việc>
git commit -m "<mô tả ngắn>"
git push -u origin <nhánh-nhóm>
```

## 7. Những hành động bị cấm

- Không tự tải dataset, model hoặc chạy inference thật trong phạm vi công việc này.
- Không đẩy trọng số, dataset lớn, token/API key hoặc đường dẫn máy cá nhân lên Git.
- Không sửa câu hỏi, ảnh, nhãn, sampling strategy (chiến lược lấy mẫu) hoặc judge/rubric mà không có phê duyệt và bằng chứng.
- Không fine-tuning, lượng tử hóa hoặc tự fallback dtype để vượt lỗi phần cứng.
- Không ghi đè run đã hoàn thành, không xóa mẫu khó và không biến lỗi kỹ thuật thành prediction hợp lệ.
- Không tự ý sửa, xóa, di chuyển mã dùng chung hoặc dùng `git reset`/discard thay đổi chưa commit của người khác.

## 8. Kết quả rà soát khả năng chạy

`python -m compileall` đã chạy thành công. `check_repository.py` chạy thành công. `check_config.py`, chế độ `official` và `pytest` chưa chạy được trong môi trường hiện tại vì thiếu `PyYAML` và lệnh `pytest`; cần cài `requirements.txt` rồi chạy lại. Việc này không chứng minh benchmark đã hoạt động.
