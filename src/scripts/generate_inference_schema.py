from pathlib import Path

from lisai.config.json_schema import (
    write_inference_defaults_json_schema,
    write_inference_overrides_json_schema,
)


def main():
    repo_root = Path(__file__).resolve().parents[2]
    schema_dir = repo_root / "configs" / "schema"
    defaults_path = schema_dir / "inference-defaults.schema.json"
    overrides_path = schema_dir / "inference.schema.json"
    write_inference_defaults_json_schema(defaults_path)
    write_inference_overrides_json_schema(overrides_path)
    print(defaults_path)
    print(overrides_path)


if __name__ == "__main__":
    main()
