# Kaggle Runbook — Nhóm 2 HallusionBench (Thành viên B: Inference)

> **Phạm vi:** Hướng dẫn kiểm tra môi trường Kaggle, tải dataset HallusionBench vào vùng ngoài Git, chạy `dry-run`, `smoke_test` (2 mẫu) và `pilot` (các nhóm/cặp trọn vẹn) cho 3 mô hình theo thứ tự kỹ thuật: **SmolVLM2 → Qwen3-VL → Phi-4**.
>
> **Cảnh báo chính sách bắt buộc:**
> - **KHÔNG** gọi bất kỳ lượt chạy `float16` trên T4 hay lượt chạy chưa khóa `revision` nào là `official`.
> - **KHÔNG** bật lượng tử hóa (`INT4`/`INT8`), **KHÔNG** tự sửa `configs/models.yaml` / `configs/datasets.yaml` khi chưa được nhóm trưởng duyệt.
> - **KHÔNG** commit dataset lớn, ảnh, model weight hoặc file kết quả nặng lên GitHub.

---

## Cell 1 — Kiểm tra phần cứng Kaggle (BẮT BUỘC chạy đầu tiên để lấy bằng chứng P0)

Bật GPU trong Kaggle Notebook (**Accelerator: GPU T4 x2** hoặc GPU khác nếu có) và bật **Internet: On**, sau đó chạy:

```bash
!nvidia-smi
```

```python
import sys, torch

print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
print("BF16 supported:", torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"GPU {i}: {torch.cuda.get_device_name(i)} | {round(props.total_memory / 2**30, 2)} GiB")
```

> Lưu lại kết quả Cell 1 để báo nhóm trưởng về việc T4 có `BF16 supported: False` (xin duyệt `float16` cho nhánh T4 hoặc chuyển GPU có BF16).

---

## Cell 2 — Clone nhánh `benchmark/hallusionbench` & cài đặt thư viện

```bash
%cd /kaggle/working
!git clone -b benchmark/hallusionbench https://github.com/hoainhan256/vlm-hallucination-benchmark.git
%cd /kaggle/working/vlm-hallucination-benchmark
!git rev-parse HEAD
```

Cài đặt dependency chung của repo + bộ thư viện inference VLM trên Kaggle:

```bash
!pip install -q -r requirements.txt
!pip install -q --upgrade transformers accelerate qwen-vl-utils num2words peft backoff soundfile
```

Đặt biến môi trường theo [docs/environment_setup.md](../../docs/environment_setup.md) (tránh để dữ liệu/cache nằm trong thư mục Git):

```python
import os

os.environ["DATA_ROOT"] = "/kaggle/working/data/HallusionBench"
os.environ["HF_HOME"] = "/kaggle/working/hf_cache"
os.environ["MODEL_CACHE_DIR"] = "/kaggle/working/hf_cache"
os.environ["OUTPUT_DIR"] = "/kaggle/working/results"
```

---

## Cell 3 — Kiểm tra cấu trúc repo & chạy toàn bộ 35 unit tests (A + B + C + E2E, CPU-only)

```bash
!python scripts/check_repository.py
!python -m pytest -q tests/ benchmarks/hallusionbench/tests/
!python -m benchmarks.hallusionbench.inference --mode check-env
```

Chạy thử `dry-run` (dùng Fake Model trên CPU để kiểm chứng xuất đủ 7 artifact và pass toàn bộ checker của repo):

```bash
!python -m benchmarks.hallusionbench.inference \
  --mode dry-run \
  --model-key smolvlm2 \
  --stage smoke_test \
  --dtype float16 \
  --experiment-id hb_dryrun \
  --run-id dryrun_01 \
  --output-root /kaggle/working/results

!python scripts/check_results.py /kaggle/working/results/hb_dryrun/dryrun_01/predictions.jsonl
!python scripts/check_manifest.py /kaggle/working/results/hb_dryrun/dryrun_01
```

---

## Cell 4 (Phần A thật) — Tải dataset HallusionBench vào `/kaggle/working/data/HallusionBench`, chạy Audit & xuất `pilot_ids.json`

Tải trực tiếp từ Hugging Face chính chủ của tác giả đầu (`rayguan/HallusionBench` @ `2067dfdce5abc7efdbb0927f4721ae282f0b6c51`, không bị lỗi quota Google Drive):

```python
import os
from pathlib import Path
from huggingface_hub import snapshot_download

data_root = Path("/kaggle/working/data/HallusionBench")
rev = "2067dfdce5abc7efdbb0927f4721ae282f0b6c51"
snapshot_download(
    repo_id="rayguan/HallusionBench",
    repo_type="dataset",
    revision=rev,
    local_dir=str(data_root),
)
os.environ["DATASET_REV"] = rev
print("Downloaded rayguan/HallusionBench @", rev)
```

Chạy kiểm chứng toàn bộ dataset thật bằng Phần A (`audit`, kiểm tra PIL pixel integrity, `.PNG` case mismatch, `json_sha256`) và chốt danh sách `pilot_ids.json` trọn nhóm cho cả 3 model:

```bash
!python -m benchmarks.hallusionbench.dataset \
  --mode audit \
  --dataset-json /kaggle/working/data/HallusionBench/HallusionBench.json \
  --data-root /kaggle/working/data/HallusionBench \
  --dataset-source rayguan/HallusionBench \
  --dataset-revision "$DATASET_REV" \
  --output /kaggle/working/hallusionbench_audit_report.json

!python -m benchmarks.hallusionbench.dataset \
  --mode export-pilot \
  --dataset-json /kaggle/working/data/HallusionBench/HallusionBench.json \
  --data-root /kaggle/working/data/HallusionBench \
  --dataset-source rayguan/HallusionBench \
  --dataset-revision "$DATASET_REV" \
  --target-groups 6 \
  --output /kaggle/working/pilot_ids.json

!python -m benchmarks.hallusionbench.dataset \
  --mode verify-pilot \
  --dataset-json /kaggle/working/data/HallusionBench/HallusionBench.json \
  --data-root /kaggle/working/data/HallusionBench \
  --pilot-file /kaggle/working/pilot_ids.json
```

---

## Cell 5 — Smoke-test (2 mẫu: 1 có ảnh + 1 text-only) & Pilot theo thứ tự 3 mô hình

> **Lưu ý trên Kaggle T4:** Do T4 không hỗ trợ `bfloat16` native, các lệnh bên dưới dùng `--dtype float16` và gắn rõ `--stage smoke_test` / `--stage pilot` để kiểm tra tương thích phần cứng và lấy bằng chứng `resolved_model_revision`, `resolved_processor_revision`, `max_new_tokens` gửi nhóm trưởng.
> Nếu bạn chạy trên GPU hỗ trợ BF16 (L4/A100), đổi `--dtype bfloat16`.

Lấy `DATASET_REV` bằng Python để biến môi trường tồn tại xuyên suốt tất cả các cell sau trên Kaggle:

```python
import os, subprocess

dataset_rev = subprocess.check_output(
    ["git", "-C", "/kaggle/working/data/HallusionBench", "rev-parse", "HEAD"],
    text=True,
).strip()
os.environ["DATASET_REV"] = dataset_rev
print("Dataset Git revision:", dataset_rev)
```

### 5.1. Ưu tiên 1 — `SmolVLM2-2.2B-Instruct`

**Smoke-test 2 mẫu (1 có ảnh + 1 text-only):**

```bash
!python -m benchmarks.hallusionbench.inference \
  --mode run \
  --allow-model-load \
  --model-key smolvlm2 \
  --stage smoke_test \
  --dtype float16 \
  --attn-implementation sdpa \
  --max-new-tokens 128 \
  --dataset-json /kaggle/working/data/HallusionBench/HallusionBench.json \
  --data-root /kaggle/working/data/HallusionBench \
  --dataset-revision "$DATASET_REV" \
  --max-samples 2 \
  --experiment-id hb_kaggle_t4_smoke \
  --run-id smoke_smolvlm2_fp16 \
  --output-root /kaggle/working/results
```

**Pilot (dùng đúng `pilot_ids.json` trọn nhóm của Phần A để cả 3 model chạy cùng 1 tập mẫu và Phần C tính được `aAcc/fAcc/qAcc`):**

```bash
!python -m benchmarks.hallusionbench.inference \
  --mode run \
  --allow-model-load \
  --model-key smolvlm2 \
  --stage pilot \
  --dtype float16 \
  --attn-implementation sdpa \
  --max-new-tokens 128 \
  --dataset-json /kaggle/working/data/HallusionBench/HallusionBench.json \
  --data-root /kaggle/working/data/HallusionBench \
  --dataset-revision "$DATASET_REV" \
  --pilot-file /kaggle/working/pilot_ids.json \
  --experiment-id hb_kaggle_t4_pilot \
  --run-id pilot_smolvlm2_fp16 \
  --output-root /kaggle/working/results
```

### 5.2. Ưu tiên 2 — `Qwen3-VL-4B-Instruct`

```bash
!python -m benchmarks.hallusionbench.inference \
  --mode run \
  --allow-model-load \
  --model-key qwen3_vl \
  --stage smoke_test \
  --dtype float16 \
  --attn-implementation sdpa \
  --max-new-tokens 128 \
  --dataset-json /kaggle/working/data/HallusionBench/HallusionBench.json \
  --data-root /kaggle/working/data/HallusionBench \
  --dataset-revision "$DATASET_REV" \
  --max-samples 2 \
  --experiment-id hb_kaggle_t4_smoke \
  --run-id smoke_qwen3_vl_fp16 \
  --output-root /kaggle/working/results
```

*(Sau khi smoke-test ổn, đổi `--stage pilot --max-groups 5` và `--run-id pilot_qwen3_vl_fp16`.)*

### 5.3. Ưu tiên 3 — `Phi-4-multimodal-instruct` (Cảnh báo T4!)

> **Cảnh báo:** Phi-4 bắt buộc `--trust-remote-code` và trên T4 **bắt buộc** `--attn-implementation eager` (hoặc `sdpa`) vì T4 không hỗ trợ `flash_attention_2`. Nếu bị OOM trên T4 ở FP16 (khi không lượng tử hóa), runner sẽ tự ghi `BLOCKED` hoặc `errors.jsonl` — giữ nguyên bằng chứng đó để báo nhóm trưởng.

```bash
!python -m benchmarks.hallusionbench.inference \
  --mode run \
  --allow-model-load \
  --model-key phi4_multimodal \
  --trust-remote-code \
  --stage smoke_test \
  --dtype float16 \
  --attn-implementation eager \
  --max-new-tokens 128 \
  --dataset-json /kaggle/working/data/HallusionBench/HallusionBench.json \
  --data-root /kaggle/working/data/HallusionBench \
  --dataset-revision "$DATASET_REV" \
  --max-samples 2 \
  --experiment-id hb_kaggle_t4_smoke \
  --run-id smoke_phi4_fp16 \
  --output-root /kaggle/working/results
```

---

## Cell 6 (Phần C thật) — Chấm điểm `aAcc / fAcc / qAcc`, kiểm chứng 7 artifact & trích xuất bảng lỗi

```python
import os

os.environ["RUN_DIR"] = "/kaggle/working/results/hb_kaggle_t4_pilot/pilot_smolvlm2_fp16"
```

```bash
# 1. Chấm điểm & tự làm mới manifest.json
!python -m benchmarks.hallusionbench.scoring \
  --run-dir "$RUN_DIR" \
  --dataset-json /kaggle/working/data/HallusionBench/HallusionBench.json \
  --data-root /kaggle/working/data/HallusionBench

# 2. Kiểm tra hợp lệ theo checker chung của repo
!python scripts/check_results.py "$RUN_DIR/predictions.jsonl"
!python scripts/check_manifest.py "$RUN_DIR"
!cat "$RUN_DIR/metrics.json"

# 3. Trích xuất bảng ca lỗi 6 nhãn (Gate G5)
!python -m benchmarks.hallusionbench.analysis \
  --mode extract \
  --run-dir "$RUN_DIR" \
  --dataset-json /kaggle/working/data/HallusionBench/HallusionBench.json \
  --data-root /kaggle/working/data/HallusionBench
!head -n 15 "$RUN_DIR/analysis/error_cases.md"

# 4. So sánh bất đồng khi đã chạy xong >= 2 mô hình (SmolVLM2, Qwen3-VL, Phi-4)
!python -m benchmarks.hallusionbench.analysis \
  --mode disagreements \
  --run-dirs \
    smolvlm2=/kaggle/working/results/hb_kaggle_t4_pilot/pilot_smolvlm2_fp16 \
    qwen3_vl=/kaggle/working/results/hb_kaggle_t4_pilot/pilot_qwen3_vl_fp16 \
  --dataset-json /kaggle/working/data/HallusionBench/HallusionBench.json \
  --data-root /kaggle/working/data/HallusionBench \
  --output /kaggle/working/results/hb_kaggle_t4_pilot/cross_model_disagreements.json
```

### Checklist bàn giao từ Kaggle (Member B)
1. **Gửi nhóm trưởng:** log Cell 1 (GPU/VRAM/BF16), `resolved_model_revision` + `resolved_processor_revision` trích từ `run_metadata.json`, đề xuất `max_new_tokens`, và bằng chứng nếu Phi-4 bị `BLOCKED`/OOM trên T4.
2. **Gửi thành viên A:** `DATASET_REV` đã tải từ `tianyi-lab/HallusionBench`, nguồn `hallusion_bench.zip` và số liệu thống kê ở Cell 4 để A đối chiếu khóa dataset.
3. **Gửi thành viên C:** trọn bộ thư mục run (đặc biệt là `predictions.jsonl`, `errors.jsonl`, `run_config.yaml`, `run_metadata.json`) để C đối chiếu và chấm `aAcc / fAcc / qAcc`.
