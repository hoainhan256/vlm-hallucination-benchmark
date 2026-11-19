from pathlib import Path


def test_local_and_absolute_repository_paths_can_be_resolved(tmp_path):
    data_root = tmp_path / "data"
    model_cache = tmp_path / "model-cache"
    output_dir = tmp_path / "results"

    assert all(isinstance(path, Path) for path in (data_root, model_cache, output_dir))
    assert data_root.name == "data"
