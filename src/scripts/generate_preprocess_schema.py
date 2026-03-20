from pathlib import Path

from lisai.config.json_schema import write_preprocess_json_schema


def main():
    repo_root = Path(__file__).resolve().parents[2]
    output_path = repo_root / "configs" / "schema" / "preprocess.schema.json"
    write_preprocess_json_schema(output_path)
    print(output_path)


if __name__ == "__main__":
    main()
