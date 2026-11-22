"""Synthetic CPU-only tests. No dataset/model downloads or paid API calls."""

from copy import deepcopy
import json
from pathlib import Path
from urllib.error import URLError

import pytest
import yaml

from vlm_benchmark.results import stable_config_hash
from .__main__ import approved, main, scoped_records, validate_config
from .artifacts import Run, verify_run
from .dataset import MMHalBenchmark, TYPES
from .judge import aggregate, make_request, original_template, parse_rating


@pytest.fixture
def config():
    return yaml.safe_load(Path(__file__).with_name("config.yaml").read_text(encoding="utf-8"))


@pytest.fixture
def dataset(tmp_path):
    rows = [{"image_id": f"image{i}", "image_src": f"https://example.invalid/{i}.jpg",
             "question": f"Question {i}?", "gt_answer": f"SECRET_REFERENCE_{i}",
             "image_content": ["SECRET_CONTENT"], "question_type": TYPES[i % 8],
             "question_topic": f"topic{i // 8}", "model_answer": ""} for i in range(96)]
    (tmp_path / "response_template.json").write_text(json.dumps(rows), encoding="utf-8")
    return MMHalBenchmark(tmp_path)


def test_config_is_valid_but_not_approved(config):
    validate_config(config)
    with pytest.raises(ValueError, match="dtype_hardware"):
        approved(config, "dtype_hardware")


@pytest.mark.parametrize("change", [
    lambda c: c["model"].update(revision="main"),
    lambda c: c["model"].update(dtype="float16"),
    lambda c: c["generation"].update(do_sample=True),
    lambda c: c["judge"].update(model="another-judge"),
    lambda c: c.update(stage="official"),
    lambda c: c.update(sample_ids=["mmhal-000", "mmhal-000"]),
])
def test_invalid_config(config, change):
    change(config)
    with pytest.raises(ValueError):
        validate_config(config)


def test_loader_order_and_no_reference_leak(dataset, config):
    records = dataset.read_dataset()
    assert len(records) == 96
    assert records[0]["sample_id"] == "mmhal-000"
    assert len(scoped_records(dataset, config)) == 8
    inputs = dataset.prepare_inputs(records[0])
    assert set(inputs) == {"image_path", "question"}
    assert "SECRET" not in json.dumps(inputs)
    assert records[0]["gt_answer"] == "SECRET_REFERENCE_0"


def test_loader_rejects_reordering_and_missing_fields(dataset):
    path = dataset.root / "response_template.json"
    records = json.loads(path.read_text())
    records[0], records[1] = records[1], records[0]
    path.write_text(json.dumps(records))
    with pytest.raises(ValueError, match="order"):
        dataset.read_dataset()
    records[0].pop("gt_answer")
    path.write_text(json.dumps(records))
    with pytest.raises(ValueError, match="gt_answer"):
        dataset.read_dataset()


def test_prompt_is_upstream_and_references_only_reach_judge(dataset, config):
    record = dataset.read_dataset()[0]
    request = make_request(record, "prediction", config["judge"])
    assert len(original_template()) > 5000
    assert request["temperature"] == 0.0
    assert request["messages"][0]["content"] == original_template().format(
        "SECRET_CONTENT", record["question"], record["gt_answer"], "prediction")


@pytest.mark.parametrize("score", range(7))
def test_parser_all_ratings(score):
    assert parse_rating(f"Explanation.\nRating: {score}") == score


@pytest.mark.parametrize("text", ["timeout", "Rating: 0 or Rating: 6", "Rating: 10", "Rating: 2.5", "Rating: -1"])
def test_parse_failures_are_not_zero(text):
    with pytest.raises(ValueError):
        parse_rating(text)


def test_aggregate_coverage_and_threshold():
    rows = [{"status": "COMPLETED", "rating": i, "question_type": TYPES[i], "question_topic": "topic"}
            for i in range(7)]
    metrics = aggregate(rows)
    assert metrics["average_score"] == 3
    assert metrics["hallucination_rate"] == 3 / 7
    rows.append({"status": "FAILED", "rating": None, "question_type": "other", "question_topic": "topic"})
    metrics = aggregate(rows)
    assert metrics["average_score"] is None
    assert metrics["valid_only_diagnostic"]["average_score"] == 3
    assert metrics["scored"] == 7


def make_run(path, config, identity=None, resume=False):
    return Run(path, config, identity or {"config": config}, {"started_at": "synthetic"}, resume)


def prediction(sample="mmhal-000"):
    return {"sample_id": sample, "prediction": "synthetic", "raw_prediction": "synthetic", "status": "COMPLETED"}


def test_artifacts_complete_immutable_and_tamper_detected(tmp_path, config):
    path = tmp_path / "test"
    with make_run(path, config) as run:
        run.add(prediction())
    rows, manifest = verify_run(path)
    assert rows[0]["run_config_hash"] == stable_config_hash({"config": config})
    with pytest.raises(ValueError, match="overwrite completed"):
        with make_run(path, config, resume=True):
            pass
    (path / "predictions.jsonl").write_text("tampered")
    with pytest.raises(ValueError, match="checksum"):
        verify_run(path)


def test_resume_and_incompatible_identity(tmp_path, config):
    path = tmp_path / "test"
    with pytest.raises(RuntimeError):
        with make_run(path, config) as run:
            run.add(prediction())
            raise RuntimeError("interrupted")
    with pytest.raises(ValueError, match="mismatch"):
        with make_run(path, config, identity={"config": "different"}, resume=True):
            pass
    with make_run(path, config, resume=True) as run:
        assert len(run.records) == 1
        with pytest.raises(ValueError, match="Duplicate"):
            run.add(prediction())
        run.add(prediction("mmhal-001"))
    assert len(verify_run(path)[0]) == 2


def test_single_writer(tmp_path, config):
    path = tmp_path / "test"
    with make_run(path, config):
        with pytest.raises(FileExistsError):
            with make_run(path, config):
                pass


def test_upstream_export_order_and_failures(dataset, tmp_path):
    records = dataset.read_dataset()
    predictions = [dict(prediction(r["sample_id"]), run_config_hash="synthetic") for r in records]
    dataset.export_results(reversed(predictions), tmp_path)
    exported = json.loads((tmp_path / "responses.json").read_text())
    assert exported[0]["image_id"] == "image0"
    assert exported[95]["model_answer"] == "synthetic"
    with pytest.raises(FileExistsError):
        dataset.export_results(predictions, tmp_path)
    with pytest.raises(ValueError, match="all 96"):
        dataset.export_results(predictions[:-1], tmp_path)


def test_api_missing_and_timeout(monkeypatch):
    from . import judge
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="missing"):
        judge.call_judge({}, 1)
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-not-a-key")
    def timeout(*args, **kwargs):
        raise URLError("synthetic timeout")
    monkeypatch.setattr(judge, "urlopen", timeout)
    with pytest.raises(URLError):
        judge.call_judge({}, 1)


def test_pending_config_blocks_real_run(tmp_path):
    assert main(["infer", "--data-root", str(tmp_path), "--run-id", "not-run"]) == 2
    assert not (tmp_path / "predictions.jsonl").exists()


def test_end_to_end_inference_judge_and_review_with_fakes(dataset, config, tmp_path, monkeypatch):
    """Exercise real orchestration and artifacts, replacing GPU and network only."""
    import sys
    from types import SimpleNamespace
    from . import __main__ as cli
    from .artifacts import checksum, write_json
    config["approvals"] = {k: "synthetic-test-evidence" for k in config["approvals"]}
    (dataset.root / "images").mkdir()
    records = dataset.read_dataset()
    for record in records:
        Path(record["image_path"]).write_bytes(b"synthetic-image-not-a-real-jpeg")
    files = [dataset.root / "response_template.json"] + [Path(r["image_path"]) for r in records]
    write_json(dataset.root / "mmhal_provenance.json", {
        "source": config["dataset"]["source"], "revision": config["dataset"]["revision"],
        "files": {p.relative_to(dataset.root).as_posix(): checksum(p) for p in files}})
    monkeypatch.setattr(cli, "runtime", lambda: {"synthetic": True})
    monkeypatch.setattr(cli.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout=""))
    fake_torch = SimpleNamespace(manual_seed=lambda s: None, use_deterministic_algorithms=lambda x: None,
        backends=SimpleNamespace(cudnn=SimpleNamespace(), cuda=SimpleNamespace(matmul=SimpleNamespace())))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class FakeModel:
        def __init__(self, *args):
            pass
        def load_model(self):
            pass
        def initialize_processor(self):
            self.processor = SimpleNamespace(image_processor=SimpleNamespace(to_dict=lambda: {}),
                                              tokenizer=object(), chat_template="synthetic")
        def infer(self, inputs, generation):
            for item in inputs:
                assert set(item) == {"image_path", "question"}
                yield {"raw_prediction": "synthetic answer", "prompt": item["question"], "hit_token_limit": False}
        def release_resources(self):
            pass
    monkeypatch.setattr(cli, "SmolVLM2", FakeModel)
    args = SimpleNamespace(data_root=str(dataset.root), output_dir=str(tmp_path / "runs"),
                           run_id="infer", resume=False, cache_dir="unused", predictions_run=None)
    assert cli.run(args, config, "infer") == 0
    source = Path(args.output_dir) / config["experiment_id"] / "infer"
    assert len(verify_run(source)[0]) == 8
    args.predictions_run = str(source)
    args.run_id = "judge"
    def fake_judge(request, timeout):
        assert "SECRET_REFERENCE" in request["messages"][0]["content"]
        return {"model": "synthetic", "usage": {"total_tokens": 1},
                "choices": [{"finish_reason": "stop", "message": {"content": "Explanation. Rating: 2"}}]}
    monkeypatch.setattr(cli, "call_judge", fake_judge)
    assert cli.run(args, config, "judge") == 0
    judge_root = source.parent / "judge"
    judged, _ = verify_run(judge_root)
    assert len(judged) == 8
    assert json.loads((judge_root / "metrics.json").read_text())["hallucination_rate"] == 1
    args.predictions_run = str(judge_root)
    args.analysis_dir = str(tmp_path / "review")
    cli.analyze(args)
    assert (Path(args.analysis_dir) / "cases.csv").exists()
    args.predictions_run = str(source)
    args.run_id = "judge-timeout"
    monkeypatch.setattr(cli, "call_judge", lambda *a: (_ for _ in ()).throw(TimeoutError()))
    assert cli.run(args, config, "judge") == 1
    failed_root = source.parent / "judge-timeout"
    assert len(verify_run(failed_root)[0]) == 8
    metrics = json.loads((failed_root / "metrics.json").read_text())
    assert metrics["average_score"] is None
    assert metrics["scored"] == 0
    assert json.loads((failed_root / "run_status.json").read_text())["status"] == "FAILED"


def test_model_decodes_only_continuation(tmp_path, monkeypatch):
    """Exercise actual adapter input/template/decode paths without GPU/weights."""
    import sys
    from contextlib import nullcontext
    from types import SimpleNamespace
    import numpy as np
    from PIL import Image
    from .model import SmolVLM2
    image = tmp_path / "tiny.png"
    Image.new("RGB", (2, 2)).save(image)
    events = {}
    class Inputs(dict):
        def to(self, device, dtype):
            events["placement"] = (device, dtype)
            return self
    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            assert messages == [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": "original question"}]}]
            return "official template prompt"
        def __call__(self, **kwargs):
            assert kwargs["images"][0].mode == "RGB"
            return Inputs(input_ids=np.array([[1, 2, 3]]))
        def decode(self, tokens, **kwargs):
            assert tokens.tolist() == [4, 9]
            return "generated answer"
    class FakeConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    class Model:
        device = "cuda:0"
        generation_config = SimpleNamespace(bos_token_id=1, eos_token_id=9, pad_token_id=0)
        def generate(self, **kwargs):
            assert kwargs["generation_config"].do_sample is False
            return np.array([[1, 2, 3, 4, 9]])
    fake_torch = SimpleNamespace(bfloat16="bf16", cuda=SimpleNamespace(synchronize=lambda: None),
                                 inference_mode=nullcontext)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(GenerationConfig=FakeConfig))
    adapter = SmolVLM2({}, None)
    adapter.processor, adapter.model = Processor(), Model()
    response = next(adapter.infer([{"image_path": str(image), "question": "original question"}], {"max_new_tokens": 2}))
    assert response["raw_prediction"] == "generated answer"
    assert response["generated_token_ids"] == [4, 9]
    assert response["stopped_on_eos"] is True
    assert response["hit_token_limit"] is False
    assert events["placement"] == ("cuda:0", "bf16")
