"""Original prompt, auditable requests, and coverage-aware scoring."""

import ast
import hashlib
import json
import os
import re
from pathlib import Path
from urllib.request import Request, urlopen

SOURCE_SHA256 = "5ba718be65d03cb9e80321ad9eb08b37a0eb0040372e1fa20f78eaf525fe8f7c"


def original_template():
    source = Path(__file__).with_name("vendor") / "eval_gpt4.py.txt"
    data = source.read_bytes()
    if hashlib.sha256(data).hexdigest() != SOURCE_SHA256:
        raise ValueError("Upstream judge source checksum mismatch")
    for statement in ast.parse(data.decode("utf-8")).body:
        if isinstance(statement, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "template" for t in statement.targets
        ):
            return ast.literal_eval(statement.value)
    raise ValueError("Upstream template missing")


def make_request(record, prediction, judge_config):
    prompt = original_template().format(
        ", ".join(record["image_content"]), record["question"], record["gt_answer"], prediction,
    )
    return {"model": judge_config["model"], "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0}


def call_judge(request, timeout):
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is missing")
    # Fixed TLS endpoint; no token passed through CLI or persisted in artifacts.
    req = Request("https://api.openai.com/v1/chat/completions",
                  data=json.dumps(request).encode("utf-8"), method="POST",
                  headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urlopen(req, timeout=timeout) as response:
        return json.load(response)


def parse_rating(text):
    # More conservative than upstream's ambiguous -> 0 fallback. Parse failures
    # are recorded as errors and never counted as hallucinated predictions.
    found = set(re.findall(r"\brating:\s*([0-6])(?![\d.])", text.lower()))
    if len(found) != 1:
        raise ValueError("Judge response contains zero or conflicting ratings")
    return int(found.pop())


def aggregate(records):
    good = [r for r in records if r.get("status") == "COMPLETED" and
            type(r.get("rating")) is int and 0 <= r["rating"] <= 6]
    def stats(rows):
        return {"count": len(rows),
                "average_score": sum(r["rating"] for r in rows) / len(rows) if rows else None,
                "hallucination_rate": sum(r["rating"] < 3 for r in rows) / len(rows) if rows else None}
    partial = stats(good)
    complete = len(good) == len(records) and bool(records)
    return {"expected": len(records), "scored": len(good), "complete": complete,
            "average_score": partial["average_score"] if complete else None,
            "hallucination_rate": partial["hallucination_rate"] if complete else None,
            "valid_only_diagnostic": partial,
            "by_question_type": {t: stats([r for r in good if r["question_type"] == t])
                                 for t in sorted({r["question_type"] for r in records})},
            "by_question_topic": {t: stats([r for r in good if r["question_topic"] == t])
                                  for t in sorted({r["question_topic"] for r in records})},
            "parse_policy": "strict_no_score_on_error"}
