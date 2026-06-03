from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


def build_parser(*, prog: str = "lisai-runner train") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Run a LISAI training job through the public LISAI API.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the resolved training config snapshot.",
    )
    return parser


def run_training(config: str | Path) -> int:
    from lisai.api import train

    train(Path(config))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return run_training(args.config)


if __name__ == "__main__":
    raise SystemExit(main())
