"""YAML configuration loading and policy validation."""

from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


APPROVED_STATES = {"PENDING_VERIFICATION", "APPROVED", "BLOCKED"}


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return data


def validate_model_config(data: Mapping[str, Any], official_stage: bool = False) -> list[str]:
    errors: list[str] = []
    policy = data.get("policy", {})
    models = data.get("models")
    if not isinstance(models, dict):
        return ["Missing required field: models"]

    for key, model in models.items():
        if not isinstance(model, dict):
            errors.append(f"Model {key}: configuration must be a mapping")
            continue
        for field in (
            "model_id",
            "revision",
            "processor_revision",
            "target_dtype",
            "dtype_status",
            "quantization",
            "verification_status",
        ):
            if field not in model:
                errors.append(f"Model {key}: missing required field: {field}")
        if model.get("revision") is None:
            errors.append(f"Model {key}: revision is not locked")
        if model.get("processor_revision") is None:
            errors.append(f"Model {key}: processor_revision is not locked")
        if model.get("quantization") not in (None, "none"):
            errors.append(f"Model {key}: quantization is not allowed")
        if model.get("verification_status") not in APPROVED_STATES:
            errors.append(f"Model {key}: invalid verification_status")
        if model.get("dtype_status") not in APPROVED_STATES:
            errors.append(f"Model {key}: invalid dtype_status")
        if model.get("dtype_status") == "APPROVED" and model.get("target_dtype") not in policy.get("approved_dtypes", []):
            errors.append(f"Model {key}: APPROVED dtype is not in approved_dtypes")

        if official_stage:
            if model.get("verification_status") != "APPROVED":
                errors.append(f"Model {key}: official mode requires APPROVED verification_status")
            if model.get("dtype_status") != "APPROVED":
                errors.append(f"Model {key}: official mode requires APPROVED dtype_status")

    if policy.get("allow_quantization") is not False:
        errors.append("Policy: quantization must be disabled")
    if policy.get("allow_finetuning") is not False:
        errors.append("Policy: fine-tuning must be disabled")
    if policy.get("allow_dtype_fallback") is not False:
        errors.append("Policy: automatic dtype fallback must be disabled")
    return errors


def validate_dataset_config(data: Mapping[str, Any], official_stage: bool = False) -> list[str]:
    errors: list[str] = []
    datasets = data.get("datasets")
    if not isinstance(datasets, dict):
        return ["Missing required field: datasets"]
    required = ("name", "source", "revision", "sample_schema", "scoring_method", "verification_status")
    for key, dataset in datasets.items():
        if not isinstance(dataset, dict):
            errors.append(f"Dataset {key}: configuration must be a mapping")
            continue
        for field in required:
            if field not in dataset:
                errors.append(f"Dataset {key}: missing required field: {field}")
        if dataset.get("revision") is None:
            errors.append(f"Dataset {key}: revision is not locked")
        if dataset.get("verification_status") not in APPROVED_STATES:
            errors.append(f"Dataset {key}: invalid verification_status")
        if official_stage and dataset.get("verification_status") != "APPROVED":
            errors.append(f"Dataset {key}: official mode requires APPROVED verification_status")
    return errors


def validate_experiment_config(
    data: Mapping[str, Any],
    model_data: Mapping[str, Any],
    dataset_data: Mapping[str, Any],
    official_stage: bool = False,
) -> list[str]:
    errors: list[str] = []
    experiment = data.get("experiment")
    if not isinstance(experiment, dict):
        return ["Missing required field: experiment"]

    for field in (
        "protocol_version",
        "stage",
        "seed",
        "batch_size",
        "generation_config",
        "device_policy",
        "output_policy",
        "resume_policy",
        "logging_policy",
    ):
        if field not in experiment:
            errors.append(f"Experiment: missing required field: {field}")

    generation = experiment.get("generation_config", {})
    if not isinstance(generation, dict):
        errors.append("Experiment: generation_config must be a mapping")
    elif generation.get("max_new_tokens") is None:
        errors.append("Experiment: max_new_tokens is pending approval")

    combinations = experiment.get("combinations")
    if not isinstance(combinations, list) or not combinations:
        errors.append("Experiment: combinations must be a non-empty list")
    else:
        model_names = set(model_data.get("models", {}))
        dataset_names = set(dataset_data.get("datasets", {}))
        seen: set[tuple[str, str]] = set()
        for combo in combinations:
            if not isinstance(combo, dict):
                errors.append("Experiment: each combination must be a mapping")
                continue
            model = combo.get("model")
            benchmark = combo.get("benchmark")
            if model not in model_names:
                errors.append(f"Experiment: unknown model {model}")
            if benchmark not in dataset_names:
                errors.append(f"Experiment: unknown benchmark {benchmark}")
            pair = (str(model), str(benchmark))
            if pair in seen:
                errors.append(f"Experiment: duplicate combination {pair}")
            seen.add(pair)

    stage = experiment.get("stage")
    if official_stage or stage == "official":
        errors.extend(validate_model_config(model_data, official_stage=True))
        errors.extend(validate_dataset_config(dataset_data, official_stage=True))
    return sorted(set(errors))


def validate_all(
    models_path: str | Path,
    datasets_path: str | Path,
    experiment_path: str | Path,
    official_stage: bool = False,
) -> list[str]:
    models = load_yaml(models_path)
    datasets = load_yaml(datasets_path)
    experiment = load_yaml(experiment_path)
    errors = validate_model_config(models, official_stage)
    errors.extend(validate_dataset_config(datasets, official_stage))
    errors.extend(validate_experiment_config(experiment, models, datasets, official_stage))
    return sorted(set(errors))
