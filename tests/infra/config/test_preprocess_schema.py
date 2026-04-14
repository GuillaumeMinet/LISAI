from __future__ import annotations

import json
from pathlib import Path

from lisai.config import load_yaml
from lisai.config.json_schema import preprocess_json_schema, write_preprocess_json_schema
from lisai.preprocess.core import PreprocessConfig
from lisai.preprocess.pipelines import PIPELINES_REGISTRY


def _branch_for_pipeline(schema: dict, pipeline_name: str) -> dict:
    for branch in schema.get("oneOf", []):
        pipeline_property = branch.get("properties", {}).get("pipeline", {})
        if pipeline_property.get("const") == pipeline_name:
            return branch
    raise AssertionError(f"Did not find preprocess schema branch for pipeline '{pipeline_name}'.")


def test_preprocess_yaml_example_validates_against_preprocess_config_model():
    cfg = load_yaml(Path("configs/preprocess/preprocess.yml"))

    validated = PreprocessConfig.model_validate(cfg)

    assert validated.dataset_name
    assert validated.pipeline


def test_preprocess_json_schema_exposes_described_pipeline_specific_branches():
    schema = preprocess_json_schema()

    assert "oneOf" in schema
    assert len(schema["oneOf"]) == len(PIPELINES_REGISTRY)

    branch_names = {branch["properties"]["pipeline"]["const"] for branch in schema["oneOf"]}
    assert branch_names == set(PIPELINES_REGISTRY)

    single_branch = _branch_for_pipeline(schema, "single_recon")
    shared_properties = single_branch["properties"]
    pipeline_cfg_properties = shared_properties["pipeline_cfg"]["properties"]

    assert shared_properties["dataset_name"]["description"]
    assert shared_properties["pipeline"]["description"]
    assert shared_properties["split"]["$ref"] == "#/$defs/PreprocessSplitConfig"
    assert pipeline_cfg_properties["dump_subfolder"]["description"]
    assert pipeline_cfg_properties["crop_size"]["description"]
    assert schema["$defs"]["PreprocessSplitConfig"]["properties"]["mode"]["description"]
    assert schema["$defs"]["PreprocessLogConfig"]["properties"]["enabled"]["description"]


def test_write_preprocess_json_schema_writes_json_file(tmp_path: Path):
    output_path = tmp_path / "preprocess.schema.json"

    written_path = write_preprocess_json_schema(output_path)

    assert written_path == output_path
    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data["title"] == "PreprocessConfig"
    assert data["oneOf"]
