from __future__ import annotations

import json
from pathlib import Path

from .experiment import ExperimentConfig


def experiment_json_schema() -> dict:
    return ExperimentConfig.model_json_schema()


def write_experiment_json_schema(output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = experiment_json_schema()
    path.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    return path
