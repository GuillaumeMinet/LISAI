from __future__ import annotations

import json
from pathlib import Path

from ..models.inference import InferenceOverrides, ResolvedInferenceConfig


def inference_overrides_json_schema() -> dict:
    """Return the JSON schema for user-authored inference YAML overrides."""
    return InferenceOverrides.model_json_schema()


def inference_defaults_json_schema() -> dict:
    """Return the JSON schema for fully resolved inference defaults YAML files."""
    return ResolvedInferenceConfig.model_json_schema()


def write_inference_overrides_json_schema(output_path: str | Path) -> Path:
    """Write the inference overrides JSON schema to disk and return the output path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = inference_overrides_json_schema()
    path.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    return path


def write_inference_defaults_json_schema(output_path: str | Path) -> Path:
    """Write the inference defaults JSON schema to disk and return the output path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = inference_defaults_json_schema()
    path.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    return path


# Backward-compatible aliases kept while the schema flow is split between
# sparse override files and the fully resolved defaults file.
inference_json_schema = inference_overrides_json_schema
write_inference_json_schema = write_inference_overrides_json_schema
