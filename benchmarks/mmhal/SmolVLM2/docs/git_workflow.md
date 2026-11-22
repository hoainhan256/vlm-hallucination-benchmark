# Quy trình Git

Các nhóm làm việc trực tiếp trên nhánh benchmark:

- `benchmark/pope`
- `benchmark/hallusionbench`
- `benchmark/mmhal`

```bash
git clone <repository-url>
cd vlm-hallucination-benchmark
git fetch --prune origin
git switch benchmark/pope
git pull --ff-only origin benchmark/pope
```

Sau chỉnh sửa:

```bash
git status
git diff
git add <các-tệp-cần-thêm>
git commit -m "<mô tả ngắn>"
git push origin benchmark/pope
```

Khi xung đột, giữ thay đổi của cả hai phía khi phù hợp, kiểm tra lại config và tests, rồi commit giải quyết. Không reset/discard thay đổi chưa commit. Trưởng nhóm kiểm tra thay đổi và mở/duyệt Pull Request (yêu cầu hợp nhất) về `main`; repository không tự tạo PR hoặc thay đổi quyền truy cập.
