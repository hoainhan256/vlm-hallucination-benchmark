from copy import deepcopy

import pytest

from vlm_benchmark.config import (
    validate_dataset_config,
    validate_experiment_config,
    validate_model_config,
)


def model_config():
    return {
        "policy": {
            "allow_quantization": False,
            "allow_finetuning": False,
            "allow_dtype_fallback": False,
            "approved_dtypes": ["bfloat16"],
        },
        "models": {
            "demo": {
                "model_id": "demo/model",
                "revision": "locked",
                "processor_revision": "locked",
                "target_dtype": "bfloat16",
                "dtype_status": "APPROVED",
                "quantization": None,
                "verification_status": "APPROVED",
            }
        },
    }


def dataset_config():
    return {
        "datasets": {
            "demo": {
                "name": "Demo",
                "source": "demo/source",
                "revision": "locked",
                "sample_schema": {"required_fields": ["sample_id"]},
                "scoring_method": "demo_protocol",
                "verification_status": "APPROVED",
            }
        }
    }


def experiment_config():
    return {
        "experiment": {
            "protocol_version": "0.1",
            "stage": "official",
            "seed": 0,
            "batch_size": 1,
            "generation_config": {"max_new_tokens": 32},
            "device_policy": {},
            "output_policy": {},
            "resume_policy": {},
            "logging_policy": {},
            "combinations": [{"model": "demo", "benchmark": "demo"}],
        }
    }


def test_valid_configs_pass():
    assert validate_model_config(model_config(), official_stage=True) == []
    assert validate_dataset_config(dataset_config(), official_stage=True) == []
    assert validate_experiment_config(
        experiment_config(), model_config(), dataset_config(), official_stage=True
    ) == []


def test_missing_fields_are_detected():
    data = deepcopy(model_config())
    del data["models"]["demo"]["revision"]
    assert any("missing required field: revision" in error for error in validate_model_config(data))


def test_locked_revision_is_required():
    data = deepcopy(model_config())
    data["models"]["demo"]["revision"] = None
    assert any("revision is not locked" in error for error in validate_model_config(data))


def test_quantization_is_rejected():
    data = deepcopy(model_config())
    data["models"]["demo"]["quantization"] = "int4"
    assert any("quantization is not allowed" in error for error in validate_model_config(data))


def test_unapproved_dtype_is_rejected_in_official_mode():
    data = deepcopy(model_config())
    data["models"]["demo"]["target_dtype"] = "float32"
    data["models"]["demo"]["dtype_status"] = "PENDING_VERIFICATION"
    assert any("official mode requires APPROVED dtype_status" in error
               for error in validate_model_config(data, official_stage=True))


def test_duplicate_combinations_are_rejected():
    data = experiment_config()
    data["experiment"]["combinations"].append({"model": "demo", "benchmark": "demo"})
    assert any("duplicate combination" in error
               for error in validate_experiment_config(data, model_config(), dataset_config(), True))


def test_pending_max_new_tokens_is_rejected():
    data = experiment_config()
    data["experiment"]["generation_config"]["max_new_tokens"] = None
    assert any("max_new_tokens is pending approval" in error
               for error in validate_experiment_config(data, model_config(), dataset_config()))
