"""Check interactive import guards and original scoring semantics, no API."""

import csv
from pathlib import Path

import pytest

from .import_review import load_ledger, upstream_parse
from .judge import aggregate


def test_review_complete_and_original_formulas():
    rows = load_ledger(Path(__file__).with_name("gpt6_review_full_infer_001.tsv"))
    records = [{"rating": upstream_parse(r["rationale"] + "\nRating: " + r["rating"]),
                "status": "COMPLETED", "question_type": str(i % 8), "question_topic": str(i // 8)}
               for i, r in enumerate(rows)]
    scores = [r["rating"] for r in records]
    metrics = aggregate(records)
    assert metrics["average_score"] == sum(scores) / 96
    assert metrics["hallucination_rate"] == sum(s < 3 for s in scores) / 96
    for i in range(8):
        assert metrics["by_question_type"][str(i)]["average_score"] == sum(scores[i::8]) / 12


def test_original_parser_rejects_missing_or_conflicting_scores():
    with pytest.raises(ValueError):
        upstream_parse("No rating supplied")
    with pytest.raises(ValueError):
        upstream_parse("Rating: 1 or Rating: 5")


def test_duplicate_review_indices_rejected(tmp_path):
    source = Path(__file__).with_name("gpt6_review_full_infer_001.tsv")
    with source.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    rows[1]["index"] = "0"
    target = tmp_path / "invalid.tsv"
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="indices"):
        load_ledger(target)
