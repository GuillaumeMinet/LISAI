# src/scripts/train.py
import argparse

from lisai.training.run_training import run_training


def main():
    parser = argparse.ArgumentParser(description="Train a model using a YAML config")
    parser.add_argument("--config", "-c", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    # Just pass the path. Let run_training handle the loading/resolving.
    run_training(args.config)

if __name__ == "__main__":
    main()