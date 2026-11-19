Notebook mẫu phải dùng `vlm_benchmark.config` để kiểm tra cấu hình và không được tự tải model/dataset khi mở Notebook.

Ví dụ:

```python
from vlm_benchmark.config import validate_all

errors = validate_all(
    "configs/models.yaml",
    "configs/datasets.yaml",
    "configs/experiment.yaml",
)
print(errors)
```