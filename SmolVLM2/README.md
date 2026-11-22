# SmolVLM2 2.2B Instruct trên MMHal Bench

Gói bàn giao GitHub ngày 21/11/2026, chỉ gồm phần việc SmolVLM2 × MMHal.
Các bản làm việc ban đầu nằm ngoài gói vẫn giữ nguyên trên máy. Khi commit phần
này, chỉ chọn `SmolVLM2/`, không dùng `git add .` ở repo cha.

## Nội dung cần đưa lên GitHub

| Thư mục hoặc tệp | Vai trò |
|---|---|
| `benchmarks/mmhal/` | Loader, adapter, inference, judge/API tùy chọn, import review, tests, rubric gốc và license |
| `benchmarks/base.py`, `models/base.py`, `vlm_benchmark/` | Bản sao hạ tầng tối thiểu để chạy gói từ thư mục này |
| `configs/pilot.yaml`, `configs/full.yaml` | Cấu hình mẫu cho lần chạy mới; approval để trống, không giả là đã duyệt |
| `docs/` | Protocol, nguyên tắc, schema và hướng dẫn nhóm liên quan |
| `results/smolvlm2-mmhal-baseline/full-infer-001/` | Kết quả inference thực tế đủ 96 câu, output và môi trường |
| `results/smolvlm2-mmhal-baseline/gpt6-interactive-001/` | Điểm, giải thích, prompt tái dựng và manifest của GPT-6 judge |
| `results/smolvlm2-mmhal-baseline/pilot-infer-001/` | Bằng chứng pilot lỗi bộ nhớ, không phải prediction hợp lệ |
| `results/smolvlm2-mmhal-baseline/pilot-infer-002/` | Pilot 8 mẫu thành công |
| `reports/smolvlm2_mmhal_20261119/` | Review 96 ca, summary và bản ghi ghép liên nhóm |
| `PUBLICATION_NOTES.json` | Ghi nhận che đường dẫn máy và checksum gốc của những tệp có thay đổi |
| `FILE_INVENTORY.csv`, `PACKAGE_MANIFEST.json` | Danh sách tệp, kích thước và SHA-256 của gói |
| `requirements.txt`, `.gitignore` | Dependency kiểm thử và quy tắc loại cache/trọng số/key |

Không có `.venv`, cache Hugging Face, ảnh dataset rời, trọng số, API key, PNG/PDF
render QA hoặc các bản xuất ảnh trung gian. Báo cáo Word có bốn ảnh minh họa từ
dataset và trích output/reference; ghi nguồn, không coi đây là ảnh do dự án sở hữu.
Giấy phép nguồn và nghĩa vụ của bên sử dụng vẫn áp dụng; license trong vendor
chỉ mô tả source upstream, không tự cấp phép cho toàn bộ repository.

## Cách đọc kết quả

Xem [báo cáo MMHal của ba mô hình](../reports/Bao_cao_tong_hop_MMHal.docx).
Điểm text judge: 3,28125/6; 34/96 dưới 3. Review ảnh riêng: 36 lỗi có bằng chứng,
37 phù hợp, 15 thiếu/mơ hồ, 8 chưa kết luận. Không trộn hai phương pháp thành một
metric; chưa có người review độc lập. Judge GPT-6 tương tác là phương án thay
thế, không phải 96 request GPT-4 API hoặc official score gốc.

Các run là bản lưu trữ để công bố, không dùng `--resume` trên chúng. Những đường
dẫn máy riêng được che bằng `<REPO_ROOT>`/`<USER_HOME>`. Manifest checksum tệp được
cập nhật cho bản công bố; semantic run/source hashes vẫn mô tả run gốc. Mã nguồn
đã thay vị trí nên checksum code hiện tại không cần bằng checksum của run cũ.

## Kiểm thử từ gói độc lập

Chạy từ repo cha:

```powershell
cd SmolVLM2
python -m pip install -r requirements.txt
python -m pytest -q benchmarks/mmhal
python -m benchmarks.mmhal check --config configs/full.yaml
python -m benchmarks.mmhal verify --predictions-run results/smolvlm2-mmhal-baseline/full-infer-001
python -m benchmarks.mmhal verify --predictions-run results/smolvlm2-mmhal-baseline/gpt6-interactive-001
```

Để chạy mới, cài PyTorch/CUDA phù hợp GPU và
`pip install -r benchmarks/mmhal/requirements.txt`. Đọc protocol và điền approval
bằng chứng thực tế trong config của lần chạy mới. Các tài liệu ghi đường dẫn
`.cache/mmhal-results/...` mô tả lịch sử local; output lưu trữ tương ứng giờ nằm
dưới `results/...` trong gói này. Cache model/dataset phải tải lại qua prepare.

```powershell
python -m benchmarks.mmhal prepare --config configs/full.yaml --cache-dir .cache/mmhal-hub
# DATA_ROOT lấy từ Data snapshot do prepare in ra
python -m benchmarks.mmhal infer --config configs/full.yaml --data-root $env:DATA_ROOT --cache-dir .cache/mmhal-hub --output-dir .cache/new-results --run-id new-full-infer-001
```

`build_report.py` là script dựng báo cáo lịch sử; cần ảnh từ snapshot local,
không cần chạy khi chỉ đọc báo cáo. `import_review.py` nhập ledger đã có, không
tự sinh điểm mới; đổi source cần lượt chấm mới.

## Commit có chọn lọc

Từ repo cha, xem danh sách rồi stage duy nhất gói:

```powershell
git status --short -- SmolVLM2
git add -- SmolVLM2
git diff --cached --stat
# Kiểm tra không có tệp đã stage từ trước ngoài phạm vi mong muốn
git diff --cached --name-only
git commit -m "Add SmolVLM2 MMHal benchmark and research report"
```

Chưa tự commit hoặc push. Các thay đổi cũ ngoài `SmolVLM2/` không cần stage
cho gói bàn giao này. Không reset/discard chúng khi chưa quyết định riêng.
