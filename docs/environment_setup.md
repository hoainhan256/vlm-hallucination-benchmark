# Thiết lập môi trường Kaggle và máy cá nhân

## Biến đường dẫn

- `DATA_ROOT`: thư mục dataset.
- `MODEL_CACHE_DIR`: cache trọng số.
- `OUTPUT_DIR`: thư mục kết quả.
- `HF_HOME`: cache Hugging Face.

Không ghi đường dẫn tuyệt đối của máy cá nhân vào Repository. Không đặt token hoặc API key trong mã nguồn.

## Python và dependencies

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

Trên Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
```

## Kiểm tra

```bash
python scripts/check_config.py --mode config
python scripts/collect_environment.py
pytest -q
```

## Kaggle

Tạo environment hoặc Notebook, đặt biến dữ liệu/cache bằng environment variable. Không tải mô hình/dataset bằng các lệnh benchmark trong giai đoạn setup. Ghi CUDA, GPU, VRAM và device allocation.

## Windows hoặc Linux

Chạy trên GPU NVIDIA chỉ sau khi xác minh driver/CUDA/PyTorch tương thích. Không giả định BF16 hoạt động trên mọi GPU. CPU offload phải được ghi cấu hình; nếu chạy không tương thích, trạng thái là `BLOCKED`.
