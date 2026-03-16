from __future__ import annotations

import argparse

from lisai.data.preprocess import PreprocessRun
from lisai.config import load_yaml, settings
from lisai.infra.paths import Paths


def main(cfg_path: str):
    cfg = load_yaml(cfg_path)
    run = PreprocessRun.from_cfg(cfg, paths=Paths(settings))

    result = run.execute()
    print(
        f"Preprocess completed: n_files={result.n_files}, "
        f"n_frames={result.n_frames}, snr_levels={result.snr_levels}"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LISAI preprocessing from a YAML config.")
    parser.add_argument("config", help="Path to preprocess config YAML file.")
    args = parser.parse_args()
    main(args.config)
