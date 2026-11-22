"""Import an explicitly authored interactive review; never generates scores."""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from .artifacts import Run, append_jsonl, checksum, verify_run, write_json
from .dataset import TYPES
from .judge import SOURCE_SHA256, aggregate, original_template


def load_ledger(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 96 or [int(r["index"]) for r in rows] != list(range(96)):
        raise ValueError("Review must contain exactly indices 0..95 in original order")
    for row in rows:
        rating = int(row["rating"])
        if not 0 <= rating <= 6 or row["visual_review"] not in ("true", "false") or not row["rationale"].strip():
            raise ValueError("Invalid rating/review record")
    return rows


def upstream_parse(text):
    # Exact original substring logic. Refuse upstream's ambiguous -> 0 fallback.
    values = [s for s in range(7) if f"rating: {s}" in text.lower()]
    if len(values) != 1:
        raise ValueError("Original parser found zero or conflicting ratings")
    return values[0]


def import_review(source_dir, ledger_path, output_dir):
    source_dir, ledger_path, output_dir = map(Path, (source_dir, ledger_path, output_dir))
    predictions, source_manifest = verify_run(source_dir)
    responses = json.loads((source_dir / "responses.json").read_text(encoding="utf-8"))
    if len(responses) != 96 or len(predictions) != 96:
        raise ValueError("Full 96-sample inference required")
    by_id = {r["sample_id"]: r for r in predictions}
    if set(by_id) != {f"mmhal-{i:03d}" for i in range(96)}:
        raise ValueError("Inference sample IDs incomplete")
    ledger = load_ledger(ledger_path)
    lock_path = ledger_path.with_suffix(".source.json")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if (lock.get("source_responses_sha256") != checksum(source_dir / "responses.json") or
            lock.get("source_predictions_sha256") != checksum(source_dir / "predictions.jsonl") or
            lock.get("ledger_sha256") != checksum(ledger_path) or lock.get("count") != 96):
        raise ValueError("Review ledger is bound to a different source/content; a new review is required")
    template = original_template()
    now = datetime.now(timezone.utc).isoformat()
    source_meta = json.loads((source_dir / "run_metadata.json").read_text(encoding="utf-8"))
    config = {"experiment_id": source_manifest["experiment_id"], "stage": "exploratory_evaluation",
              "protocol": "mmhal-gpt6-interactive-text-v1", "source_run_id": source_dir.name,
              "judge": {"family": "GPT-6", "execution": "Codex interactive conversation",
                        "exact_model_snapshot": None, "temperature": None, "seed": None,
                        "independent_api_requests": False, "images_provided": False},
              "rubric": "unmodified original MMHal template", "rating_threshold": 3,
              "metric_scope": "all_96", "source_stage": source_meta["identity"]["config"]["stage"]}
    identity = {"config": config, "source_run_hash": source_manifest["run_config_hash"],
                "source_predictions_sha256": checksum(source_dir / "predictions.jsonl"),
                "source_responses_sha256": checksum(source_dir / "responses.json"),
                "review_ledger_sha256": checksum(ledger_path), "rubric_source_sha256": SOURCE_SHA256,
                "import_code_sha256": checksum(__file__)}
    metadata = {"started_at": now, "expected_sample_ids": [f"mmhal-{i:03d}" for i in range(96)],
                "judge": config["judge"], "api_requests": 0, "api_cost": None,
                "api_usage": None, "cost_note": "No judge API calls; chat subscription cost not measured",
                "review_provenance": "Ratings and explanations authored by GPT-6 in this Codex conversation; importer only validates and aggregates",
                "prompt_provenance": "Exact reconstructed upstream inputs, not HTTP request logs",
                "source_environment": source_meta["environment"], "human_review": "PENDING",
                "source_git_dirty": source_meta.get("git_dirty"), "source_git_commit": source_meta.get("git_commit")}
    if output_dir.exists():
        raise FileExistsError("Review output must be a new directory")
    with Run(output_dir, config, identity, metadata) as run:
        shutil.copyfile(ledger_path, output_dir / "review_ledger.tsv")
        write_json(output_dir / "source_manifest.json", source_manifest)
        for i, (response, review) in enumerate(zip(responses, ledger)):
            sample_id = f"mmhal-{i:03d}"
            prediction = by_id[sample_id]
            if prediction["status"] != "COMPLETED" or prediction["raw_prediction"] != response["model_answer"]:
                raise ValueError(f"Prediction/source mismatch: {sample_id}")
            if response["question_type"] != TYPES[i % 8] or response["question"] != prediction["question"]:
                raise ValueError(f"Original question order mismatch: {sample_id}")
            raw_review = review["rationale"] + "\nRating: " + review["rating"]
            rating = upstream_parse(raw_review)
            prompt = template.format(", ".join(response["image_content"]), response["question"],
                                     response["gt_answer"], response["model_answer"])
            append_jsonl(output_dir / "judge_inputs.jsonl", {"sample_id": sample_id,
                         "messages": [{"role": "user", "content": prompt}],
                         "provenance": "reconstructed_original_prompt"})
            append_jsonl(output_dir / "judge_reviews.jsonl", {"sample_id": sample_id,
                         "text": raw_review, "provenance": "serialized_interactive_review_not_api_response"})
            run.add({"sample_id": sample_id, "prediction": prediction["prediction"],
                     "raw_prediction": prediction["raw_prediction"], "status": "COMPLETED",
                     "rating": rating, "judge_explanation": raw_review, "reference": response["gt_answer"],
                     "question": response["question"], "question_type": response["question_type"],
                     "question_topic": response["question_topic"], "image_id": response["image_id"],
                     "image_content": response["image_content"], "image_sha256": prediction["image_sha256"],
                     "hit_token_limit": prediction.get("hit_token_limit", False),
                     "generated_tokens": prediction.get("generated_tokens"),
                     "error_category": review["category"],
                     "needs_visual_review": review["visual_review"] == "true",
                     "human_review_status": "PENDING", "judge_label": "hallucination" if rating < 3 else "no_hallucination"})
        metrics = aggregate(run.records)
        scores = [r["rating"] for r in run.records]
        upstream_means = [sum(scores[j::8]) / 12 for j in range(8)]
        assert all(metrics["by_question_type"][t]["average_score"] == upstream_means[j]
                   for j, t in enumerate(TYPES))
        metrics.update(judge=config["judge"], protocol=config["protocol"],
                       comparable_to_original=False, original_formula_verified=True,
                       parse_policy="original_substring_parser_all_96_valid_no_fallback_used",
                       human_review="PENDING", needs_visual_review=sum(r["needs_visual_review"] for r in run.records),
                       hit_token_limit=sum(r["hit_token_limit"] for r in run.records),
                       rating_histogram={str(s): scores.count(s) for s in range(7)},
                       source_stage=config["source_stage"])
        write_json(output_dir / "metrics.json", metrics)
        columns = ["sample_id", "question_type", "question_topic", "question", "reference", "raw_prediction",
                   "rating", "judge_label", "error_category", "judge_explanation", "needs_visual_review", "hit_token_limit"]
        with (output_dir / "cases.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for record in run.records:
                values = {k: record[k] for k in columns}
                writer.writerow({k: "'" + v if isinstance(v, str) and v.startswith(("=", "+", "-", "@")) else v
                                 for k, v in values.items()})
        report = ["# MMHal-Bench — GPT-6 interactive, text-reference", "",
                  "Exploratory inference; alternative judge; human review pending. No GPT-4 API calls.", "",
                  f"- Scored: {len(scores)}/96", f"- Average score: {metrics['average_score']:.6f}/6",
                  f"- Hallucination rate: {metrics['hallucination_rate']:.6%}",
                  f"- Judge-flagged cases: {sum(s < 3 for s in scores)}",
                  f"- Needs visual review: {metrics['needs_visual_review']}",
                  f"- Token-limit outputs: {metrics['hit_token_limit']}", "",
                  "Same original prompt/rubric/inputs/formulas. Different judge and interactive execution;",
                  "no controllable API snapshot/temperature, no independent per-sample API calls.",
                  "These are judge assessments, not visually confirmed ground truth or official GPT-4 scores.", "",
                  "| Type | Count | Mean score | Hallucination rate |", "|---|---:|---:|---:|"]
        for t in TYPES:
            group = metrics["by_question_type"][t]
            report.append(f"| {t} | {group['count']} | {group['average_score']:.4f} | {group['hallucination_rate']:.2%} |")
        report += ["", "## Cases", ""]
        for record in run.records:
            report += [f"### {record['sample_id']} — {record['rating']}/6 ({record['error_category']})", "",
                       record["judge_explanation"], ""]
        (output_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    verify_run(output_dir)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    metrics = import_review(args.source_run, args.ledger, args.output_dir)
    print(json.dumps({k: metrics[k] for k in ("scored", "average_score", "hallucination_rate", "needs_visual_review")}, indent=2))


if __name__ == "__main__":
    main()
