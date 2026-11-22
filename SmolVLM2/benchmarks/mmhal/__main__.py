"""python -m benchmarks.mmhal: prepare, infer, judge, analyze, verify."""

import argparse
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import re
import subprocess

import yaml

from vlm_benchmark.results import stable_config_hash
from .artifacts import Run, append_jsonl, checksum, verify_run, write_json
from .dataset import MMHalBenchmark
from .judge import aggregate, call_judge, make_request, parse_rating
from .model import SmolVLM2

ROOT = Path(__file__).resolve().parents[2]


def approved(config, *names):
    for name in names:
        evidence = config.get("approvals", {}).get(name)
        if not isinstance(evidence, str) or not evidence.strip() or evidence == "PENDING_VERIFICATION":
            raise ValueError(f"Missing approval/evidence reference: approvals.{name}")


def validate_config(config):
    for section in ("model", "dataset", "generation", "judge", "approvals"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing section: {section}")
    model, dataset = config["model"], config["dataset"]
    if model["model_id"] != "HuggingFaceTB/SmolVLM2-2.2B-Instruct":
        raise ValueError("Only the requested SmolVLM2 model is supported")
    if dataset["source"] != "Shengcao1006/MMHal-Bench" or dataset["expected_count"] != 96:
        raise ValueError("Unexpected MMHal source/count")
    for rev in (model["revision"], model["processor_revision"], dataset["revision"]):
        if not isinstance(rev, str) or not re.fullmatch(r"[a-f0-9]{40}", rev):
            raise ValueError("Exact 40-character commit revisions required")
    if model["dtype"] != "bfloat16" or not re.fullmatch(r"cuda:\d+", model["device"]):
        raise ValueError("This baseline requires explicit CUDA device and approved BF16")
    if model["attn_implementation"] not in ("sdpa", "eager"):
        raise ValueError("Unsupported attention implementation")
    generation = config["generation"]
    if set(generation) != {"do_sample", "max_new_tokens"} or generation["do_sample"] is not False:
        raise ValueError("Only greedy baseline generation is supported")
    if type(generation["max_new_tokens"]) is not int or generation["max_new_tokens"] <= 0:
        raise ValueError("max_new_tokens must be positive")
    ids = config["sample_ids"]
    if config["stage"] not in ("pilot", "official"):
        raise ValueError("stage must be pilot or official")
    if config["stage"] == "official" and ids is not None:
        raise ValueError("Official evaluation requires sample_ids: null (all 96)")
    if config["stage"] == "pilot" and (not isinstance(ids, list) or not ids or len(set(ids)) != len(ids)):
        raise ValueError("Pilot needs explicit unique sample_ids")
    judge = config["judge"]
    if judge["protocol"] not in ("original", "alternative"):
        raise ValueError("Unknown judge protocol")
    if judge["protocol"] == "original" and judge["model"] != "gpt-4-0314":
        raise ValueError("Changed judge must be marked protocol: alternative")
    if judge["temperature"] != 0.0 or not 0 < judge["timeout_seconds"] <= 300:
        raise ValueError("Invalid judge temperature/timeout")
    if type(config["seed"]) is not int:
        raise ValueError("seed must be an integer")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", config["experiment_id"]):
        raise ValueError("Invalid experiment_id")


def scoped_records(benchmark, config):
    records = list(benchmark.read_dataset())
    ids = config["sample_ids"]
    if ids is not None:
        if set(ids) - {r["sample_id"] for r in records}:
            raise ValueError("Unknown pilot sample IDs")
        records = [r for r in records if r["sample_id"] in ids]
    return records


def prepare(args, config):
    approved(config, "inference", "dataset_license")
    from huggingface_hub import snapshot_download
    dataset = config["dataset"]
    root = Path(snapshot_download(dataset["source"], repo_type="dataset", revision=dataset["revision"],
                                 cache_dir=args.cache_dir, allow_patterns=["response_template.json", "images/*"]))
    benchmark = MMHalBenchmark(root)
    records = benchmark.read_dataset()
    files = [root / "response_template.json"] + [Path(r["image_path"]) for r in records]
    provenance = {"source": dataset["source"], "revision": dataset["revision"],
                  "files": {p.relative_to(root).as_posix(): checksum(p) for p in files}}
    write_json(root / "mmhal_provenance.json", provenance)
    model = config["model"]
    for revision in {model["revision"], model["processor_revision"]}:
        snapshot_download(model["model_id"], revision=revision, cache_dir=args.cache_dir,
                          allow_patterns=["*.json", "*.jinja", "*.txt", "*.model", "*.safetensors"])
    print(f"Data snapshot: {root}\nModel cache: {args.cache_dir}")


def provenance(root, config):
    root = Path(root)
    lock = json.loads((root / "mmhal_provenance.json").read_text(encoding="utf-8"))
    if (lock["source"], lock["revision"]) != (config["dataset"]["source"], config["dataset"]["revision"]):
        raise ValueError("Dataset provenance revision mismatch; use prepare")
    required = {"response_template.json"} | {
        Path(r["image_path"]).relative_to(root).as_posix() for r in MMHalBenchmark(root).read_dataset()}
    if set(lock["files"]) != required:
        raise ValueError("Dataset provenance is incomplete")
    for name, digest in lock["files"].items():
        if checksum(root / name) != digest:
            raise ValueError(f"Dataset checksum mismatch: {name}")
    return stable_config_hash(lock)


def runtime():
    import torch
    packages = {}
    for package in ("torch", "transformers", "accelerate", "huggingface-hub", "Pillow", "num2words", "PyYAML"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    import platform
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages,
            "cuda": torch.version.cuda,
            "gpu": [{"name": torch.cuda.get_device_name(i),
                     "vram_bytes": torch.cuda.get_device_properties(i).total_memory,
                     "capability": list(torch.cuda.get_device_capability(i))}
                    for i in range(torch.cuda.device_count())]}


def code_identity():
    paths = list((ROOT / "benchmarks/mmhal").rglob("*.py"))
    paths += [ROOT / "benchmarks/mmhal/vendor/eval_gpt4.py.txt", ROOT / "models/base.py", ROOT / "benchmarks/base.py"]
    paths += list((ROOT / "vlm_benchmark").glob("*.py"))
    return {p.relative_to(ROOT).as_posix(): checksum(p) for p in sorted(paths)}


def run(args, config, mode):
    approved(config, "inference", "dtype_hardware", "generation", "dataset_license")
    if config["stage"] == "official":
        approved(config, "official")
    if mode == "judge":
        approved(config, "judge_access_cost")
    benchmark = MMHalBenchmark(args.data_root, config["generation"])
    records = scoped_records(benchmark, config)
    dataset_hash = provenance(args.data_root, config)
    environment = runtime()
    identity = {"config": config, "mode": mode, "data_hash": dataset_hash,
                "code": code_identity(), "environment": environment}
    source = None
    if mode == "judge":
        source, manifest = verify_run(args.predictions_run)
        meta = json.loads((Path(args.predictions_run) / "run_metadata.json").read_text(encoding="utf-8"))
        def inference_settings(c):
            return {k: v for k, v in c.items() if k not in ("judge", "approvals")}
        if meta["identity"]["mode"] != "infer" or inference_settings(meta["identity"]["config"]) != inference_settings(config):
            raise ValueError("Judge config must match the source inference config")
        if meta["identity"]["data_hash"] != dataset_hash:
            raise ValueError("Judge dataset differs from inference")
        if set(r["sample_id"] for r in source) != set(r["sample_id"] for r in records):
            raise ValueError("Source inference scope incomplete")
        identity["source_predictions_sha256"] = checksum(Path(args.predictions_run) / "predictions.jsonl")
        identity["source_run_hash"] = manifest["run_config_hash"]
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    identity["git_commit"] = git
    metadata = {"started_at": datetime.now(timezone.utc).isoformat(), "git_commit": git,
                "git_dirty": bool(dirty), "environment": environment, "device_allocation": config["model"]["device"],
                "quantization": None, "model_revision": config["model"]["revision"],
                "processor_revision": config["model"]["processor_revision"],
                "dataset_revision": config["dataset"]["revision"], "expected_sample_ids": [r["sample_id"] for r in records]}
    if config["stage"] == "official" and dirty:
        raise ValueError("Official run requires a clean committed checkout")
    adapter = None
    with Run(Path(args.output_dir) / config["experiment_id"] / args.run_id,
             config, identity, metadata, args.resume) as output:
        seen = {r["sample_id"] for r in output.records}
        if seen - set(metadata["expected_sample_ids"]):
            raise ValueError("Resume includes out-of-scope IDs")
        if mode == "infer":
            import torch
            import random
            import numpy as np
            torch.manual_seed(config["seed"])
            random.seed(config["seed"])
            np.random.seed(config["seed"])
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.use_deterministic_algorithms(True)
            adapter = SmolVLM2(config["model"], args.cache_dir)
            try:
                adapter.load_model()
                adapter.initialize_processor()
            except BaseException:
                adapter.release_resources()
                raise
            (output.root / "pip_freeze.txt").write_text(subprocess.run(
                [os.sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True).stdout,
                encoding="utf-8")
            write_json(output.root / "processor_config.json", {
                "class": type(adapter.processor).__name__,
                "image_processor": adapter.processor.image_processor.to_dict(),
                "tokenizer_class": type(adapter.processor.tokenizer).__name__,
                "chat_template": adapter.processor.chat_template,
                "preprocessing": "PIL RGB; official processor defaults; no external resizing",
                "decoding": {"num_beams": 1, "do_sample": False, "use_cache": True,
                             "max_new_tokens": config["generation"]["max_new_tokens"]}})
        try:
            for record in records:
                if record["sample_id"] in seen:
                    continue
                result = {k: record[k] for k in ("sample_id", "question_type", "question_topic", "question", "image_id")}
                result.update(reference=record["gt_answer"], image_sha256=checksum(record["image_path"]))
                if mode == "infer":
                    try:
                        response = benchmark.call_model(adapter, benchmark.prepare_inputs(record))
                        if not response["raw_prediction"].strip():
                            raise ValueError("Empty generated response")
                        result.update(response, prediction=response["raw_prediction"], status="COMPLETED")
                    except Exception as exc:
                        output.error(record["sample_id"], exc, "inference_error")
                        result.update(prediction=None, raw_prediction=None, status="FAILED")
                        output.add(result)
                        # Abort on GPU/model error instead of repeating a broken context.
                        raise
                else:
                    prediction = next(r for r in source if r["sample_id"] == record["sample_id"])
                    result.update(prediction=prediction["prediction"], raw_prediction=prediction.get("raw_prediction"))
                    if prediction["status"] != "COMPLETED":
                        result.update(status="FAILED", rating=None)
                        output.error(record["sample_id"], ValueError(), "source_inference_failed")
                    else:
                        request = make_request(record, prediction["raw_prediction"], config["judge"])
                        # Persist exact request before network and raw response before parsing.
                        append_jsonl(output.root / "judge_requests.jsonl", {"sample_id": record["sample_id"], "request": request})
                        try:
                            response = call_judge(request, config["judge"]["timeout_seconds"])
                            append_jsonl(output.root / "judge_responses.jsonl", {"sample_id": record["sample_id"], "response": response})
                            text = response["choices"][0]["message"]["content"]
                            if response["choices"][0].get("finish_reason") != "stop":
                                raise ValueError("Incomplete judge response")
                            result.update(rating=parse_rating(text), judge_explanation=text,
                                          judge_model_returned=response.get("model"), status="COMPLETED")
                        except Exception as exc:
                            result.update(status="FAILED", rating=None)
                            output.error(record["sample_id"], exc, "judge_error")
                output.add(result)
                print(f"{record['sample_id']}: {result['status']}", flush=True)
            if mode == "judge":
                metrics = aggregate(output.records)
                metrics.update(stage=config["stage"], judge=config["judge"],
                               comparable_to_original=config["judge"]["protocol"] == "original")
            else:
                metrics = {"status": "NOT_SCORED", "expected": len(records),
                           "completed": sum(r["status"] == "COMPLETED" for r in output.records)}
            write_json(output.root / "metrics.json", metrics)
            if mode == "infer" and len(records) == 96 and all(r["status"] == "COMPLETED" for r in output.records):
                benchmark.export_results(output.records, output.root)
        finally:
            if adapter is not None:
                adapter.release_resources()
    return 0 if all(r["status"] == "COMPLETED" for r in output.records) else 1


def analyze(args):
    import csv
    records, manifest = verify_run(args.predictions_run)
    target = Path(args.analysis_dir)
    target.mkdir(parents=True, exist_ok=False)
    columns = ["sample_id", "question_type", "question_topic", "image_id", "question", "reference",
               "raw_prediction", "status", "rating", "judge_explanation", "hit_token_limit",
               "candidate_hallucination", "human_label", "unsupported_claim", "visual_evidence",
               "error_category", "cause_hypothesis", "reviewer", "notes"]
    with (target / "cases.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in records:
            row = {k: record.get(k, "") for k in columns}
            rating = record.get("rating")
            row["candidate_hallucination"] = rating < 3 if type(rating) is int else "undetermined"
            row["human_label"] = "unreviewed"
            # Prevent spreadsheet formula injection when reviewing model text.
            writer.writerow({k: "'" + v if isinstance(v, str) and v.startswith(("=", "+", "-", "@")) else v
                             for k, v in row.items()})
    write_json(target / "source.json", manifest)
    write_json(target / "summary.json", {"records": len(records),
               "technical_failures": sum(r["status"] != "COMPLETED" for r in records),
               "judge_flagged": sum(type(r.get("rating")) is int and r["rating"] < 3 for r in records),
               "human_review": "PENDING", "causal_claims": "Not established by baseline benchmarking"})
    print(f"Review sheet: {target / 'cases.csv'}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "prepare", "infer", "judge", "analyze", "verify"))
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--data-root")
    parser.add_argument("--cache-dir", default=os.environ.get("MODEL_CACHE_DIR", ".cache/mmhal-hub"))
    parser.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR", "results"))
    parser.add_argument("--run-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--predictions-run")
    parser.add_argument("--analysis-dir")
    args = parser.parse_args(argv)
    try:
        if args.command in ("verify", "analyze"):
            if not args.predictions_run:
                parser.error("--predictions-run is required")
            if args.command == "verify":
                verify_run(args.predictions_run)
                print("Artifact checksums and prediction identity passed")
            else:
                if not args.analysis_dir:
                    parser.error("--analysis-dir is required")
                analyze(args)
            return 0
        config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        validate_config(config)
        if args.command == "check":
            from .judge import original_template
            original_template()
            print("Config structure and original rubric checksum passed; approvals are not implied")
        elif args.command == "prepare":
            prepare(args, config)
        else:
            if not args.data_root or not args.run_id or not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id):
                parser.error("--data-root and safe --run-id are required")
            if args.command == "judge" and not args.predictions_run:
                parser.error("judge requires --predictions-run")
            return run(args, config, args.command)
        return 0
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
