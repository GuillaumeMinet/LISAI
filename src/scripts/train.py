import argparse
from pathlib import Path

from lisai.training.run_training import run_training


def _resolve_config_path(config_arg: str) -> Path:
    config_path = Path(config_arg)
    if config_path.exists():
        return config_path

    repo_root = Path(__file__).resolve().parents[2]
    experiment_config_path = repo_root / "configs" / "experiments" / config_arg
    if experiment_config_path.exists():
        return experiment_config_path

    return config_path


def main():
    parser = argparse.ArgumentParser(description="Train a model using a YAML config")
    parser.add_argument("--config", "-c", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    run_training(_resolve_config_path(args.config))

if __name__ == "__main__":
    main()
