from __future__ import annotations

import json
from pathlib import Path

import pytest

from lisai.config import load_yaml
from lisai.config.json_schema import experiment_json_schema, write_experiment_json_schema
from lisai.config.models import ExperimentConfig


@pytest.mark.parametrize(
    "config_path",
    [
        Path("configs/experiments/hdn_training.yml"),
        Path("configs/experiments/upsamp_training.yml"),
    ],
)
def test_current_experiment_yaml_validates_against_authoring_schema(config_path: Path):
    cfg = load_yaml(config_path)

    validated = ExperimentConfig.model_validate(cfg)

    assert validated.experiment.exp_name


def test_experiment_json_schema_describes_authoring_shape_only():
    schema = experiment_json_schema()

    data_ref = schema["properties"]["data"]["$ref"].split("/")[-1]
    data_properties = schema["$defs"][data_ref]["properties"]
    assert "data_dir" not in data_properties
    assert "dataset_info" not in data_properties
    assert "volumetric" not in data_properties
    assert "masking" not in data_properties

    load_model_ref = schema["properties"]["load_model"]["$ref"].split("/")[-1]
    load_model_properties = schema["$defs"][load_model_ref]["properties"]
    assert "canonical_load" in load_model_properties
    assert "checkpoint" not in load_model_properties


def test_write_experiment_json_schema_writes_json_file(tmp_path: Path):
    output_path = tmp_path / "experiment.schema.json"

    written_path = write_experiment_json_schema(output_path)

    assert written_path == output_path
    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data["title"] == "ExperimentConfig"
