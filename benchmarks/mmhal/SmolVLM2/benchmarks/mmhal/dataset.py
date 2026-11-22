"""Read official response_template.json and local images, without dataset scripts."""

import json
import re
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

from benchmarks.base import BaseBenchmark

TYPES = ("attribute", "adversarial", "comparison", "counting", "relation",
         "environment", "holistic", "other")


class MMHalBenchmark(BaseBenchmark):
    def __init__(self, data_root, generation=None):
        self.root = Path(data_root)
        self.generation = generation or {}

    def read_dataset(self):
        rows = json.loads((self.root / "response_template.json").read_text(encoding="utf-8"))
        if not isinstance(rows, list) or len(rows) != 96:
            raise ValueError("Official MMHal template must contain exactly 96 records")
        records = [self.normalize_record(dict(row, source_index=i)) for i, row in enumerate(rows)]
        if Counter(r["question_type"] for r in records) != Counter({t: 12 for t in TYPES}):
            raise ValueError("Expected 12 questions per type")
        for i, record in enumerate(records):
            if record["question_type"] != TYPES[i % 8]:
                raise ValueError("Official question order changed")
        return records

    def normalize_record(self, record):
        for key in ("image_id", "image_src", "question", "gt_answer", "question_type", "question_topic"):
            if not isinstance(record.get(key), str) or not record[key].strip():
                raise ValueError(f"Invalid field: {key}")
        if record["question_type"] not in TYPES:
            raise ValueError("Unknown question_type")
        if not isinstance(record.get("image_content"), list) or not all(
            isinstance(v, str) for v in record["image_content"]
        ):
            raise ValueError("image_content must be a string list")
        name = Path(urlparse(record["image_src"]).path).name
        if not re.fullmatch(r"[A-Za-z0-9_.-]+\.(jpg|jpeg|png)", name, re.IGNORECASE):
            raise ValueError("Unsafe/unsupported image filename")
        result = dict(record)
        result["sample_id"] = f"mmhal-{record['source_index']:03d}"
        result["image_path"] = str(self.root / "images" / name)
        return result

    def prepare_inputs(self, record):
        # Deliberate allowlist: no gt_answer, image_content, type or topic.
        return {"image_path": record["image_path"], "question": record["question"]}

    def call_model(self, model, prepared_input):
        return next(iter(model.infer([prepared_input], self.generation)))

    def score(self, predictions):
        from .judge import aggregate
        return aggregate(list(predictions))

    def export_results(self, predictions, output_dir):
        """Export in original order, retaining all original source fields."""
        from vlm_benchmark.results import validate_predictions
        predictions = list(predictions)
        errors = validate_predictions(predictions)
        if errors:
            raise ValueError(errors)
        records = self.read_dataset()
        by_id = {p["sample_id"]: p for p in predictions}
        if set(by_id) != {r["sample_id"] for r in records}:
            raise ValueError("Upstream export requires all 96 samples")
        output = []
        for record in records:
            prediction = by_id[record["sample_id"]]
            if prediction["status"] != "COMPLETED":
                raise ValueError("Cannot export technical failures as model responses")
            output.append({k: v for k, v in record.items()
                           if k not in ("sample_id", "source_index", "image_path")})
            output[-1]["model_answer"] = prediction["raw_prediction"]
        path = Path(output_dir) / "responses.json"
        with path.open("x", encoding="utf-8") as handle:
            json.dump(output, handle, indent=2, ensure_ascii=False)
