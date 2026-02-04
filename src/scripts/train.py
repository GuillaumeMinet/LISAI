import argparse
from lisai.lib.utils import config_utils
from lisai.training.run_training import run_training

def main():
    parser = argparse.ArgumentParser(description="Train a model using a YAML config")
    parser.add_argument("--config", "-c", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    # Load config
    cfg = config_utils.load_yaml_config(args.config)

    # Run training
    run_training(cfg)

if __name__ == "__main__":
    main()
