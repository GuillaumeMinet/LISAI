from __future__ import annotations

import copy
import json
from dataclasses import fields as dataclass_fields, is_dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, TypeAdapter

from lisai.preprocess.core import PreprocessConfig
from lisai.preprocess.core.config import (
    PreprocessLogConfig,
    PreprocessSplitConfig,
    PreprocessSplitManualConfig,
    PreprocessSplitRandomConfig,
    PreprocessSplitReuseConfig,
)
from lisai.preprocess.pipelines import PIPELINES_REGISTRY

PREPROCESS_MODEL_TYPES: tuple[type[BaseModel], ...] = (
    PreprocessConfig,
    PreprocessLogConfig,
    PreprocessSplitConfig,
    PreprocessSplitRandomConfig,
    PreprocessSplitManualConfig,
    PreprocessSplitReuseConfig,
)
SHARED_PROPERTY_KEYS = ("dataset_name", "pipeline", "data_type", "fmt", "log", "split")


def _ensure_model_field_descriptions() -> None:
    missing: dict[str, list[str]] = {}
    for model_type in PREPROCESS_MODEL_TYPES:
        missing_fields = [
            name for name, model_field in model_type.model_fields.items() if not (model_field.description or "").strip()
        ]
        if missing_fields:
            missing[model_type.__name__] = missing_fields

    if missing:
        details = ", ".join(f"{model}: {fields}" for model, fields in missing.items())
        raise ValueError(f"Missing preprocess schema descriptions for: {details}")


def _ensure_pipeline_cfg_descriptions(pipeline_name: str, cfg_type: type[Any]) -> None:
    if not is_dataclass(cfg_type):
        raise TypeError(f"Pipeline '{pipeline_name}' Config must be a dataclass type.")

    missing = [field.name for field in dataclass_fields(cfg_type) if not str(field.metadata.get("description", "")).strip()]
    if missing:
        raise ValueError(f"Pipeline '{pipeline_name}' is missing config field descriptions for: {missing}")


def _merge_defs(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in source.items():
        existing = target.get(key)
        if existing is not None and existing != value:
            raise ValueError(f"Conflicting preprocess schema definition for '{key}'.")
        target[key] = value


def _pipeline_cfg_schema(pipeline_name: str, pipeline_cls: type[Any]) -> dict[str, Any]:
    cfg_type = getattr(pipeline_cls, "Config", None)
    if cfg_type is None:
        raise TypeError(f"Pipeline '{pipeline_name}' does not define a Config dataclass.")

    _ensure_pipeline_cfg_descriptions(pipeline_name, cfg_type)
    schema = TypeAdapter(cfg_type).json_schema()
    if schema.get("type") == "object":
        schema.setdefault("additionalProperties", False)
    return schema


def preprocess_json_schema() -> dict:
    _ensure_model_field_descriptions()

    base_schema = PreprocessConfig.model_json_schema()
    shared_properties = {key: copy.deepcopy(base_schema["properties"][key]) for key in SHARED_PROPERTY_KEYS}
    pipeline_property = copy.deepcopy(base_schema["properties"]["pipeline"])
    pipeline_cfg_description = str(base_schema["properties"]["pipeline_cfg"].get("description", "")).strip()
    required = list(base_schema.get("required", []))
    defs = copy.deepcopy(base_schema.get("$defs", {}))

    branches: list[dict[str, Any]] = []
    for pipeline_name, pipeline_cls in sorted(PIPELINES_REGISTRY.items()):
        pipeline_cfg_schema = _pipeline_cfg_schema(pipeline_name, pipeline_cls)
        _merge_defs(defs, pipeline_cfg_schema.pop("$defs", {}))
        pipeline_cfg_schema.setdefault("description", pipeline_cfg_description)

        branch_pipeline_property = copy.deepcopy(pipeline_property)
        branch_pipeline_property.pop("enum", None)
        branch_pipeline_property["const"] = pipeline_name
        branch_pipeline_property["default"] = pipeline_name

        branch_properties = copy.deepcopy(shared_properties)
        branch_properties["pipeline"] = branch_pipeline_property
        branch_properties["pipeline_cfg"] = pipeline_cfg_schema

        branches.append(
            {
                "type": "object",
                "title": f"PreprocessConfig[{pipeline_name}]",
                "additionalProperties": False,
                "properties": branch_properties,
                "required": required,
            }
        )

    schema: dict[str, Any] = {
        "title": base_schema.get("title", "PreprocessConfig"),
        "type": "object",
        "oneOf": branches,
    }
    if defs:
        schema["$defs"] = defs
    return schema


def write_preprocess_json_schema(output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = preprocess_json_schema()
    path.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    return path
