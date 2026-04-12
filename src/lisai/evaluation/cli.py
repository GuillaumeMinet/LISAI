from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import yaml

from lisai.runs.cli import add_run_filter_arguments
from lisai.runs.listing import filter_runs, render_runs_table, write_invalid_run_warnings
from lisai.runs.scanner import scan_runs
from lisai.runs.selection import resolve_ambiguous_run_matches

from .defaults import UNSET
from .run_apply_model import run_apply_model
from .run_evaluate import run_evaluate


def _parse_run_ref(run_ref: str) -> tuple[str, str, str]:
    parts = [part for part in run_ref.replace("\\", "/").split("/") if part]
    if len(parts) < 2:
        raise ValueError(
            "Run reference must be 'dataset/exp_name' or 'dataset/subfolder/exp_name'."
        )
    dataset_name = parts[0]
    model_name = parts[-1]
    model_subfolder = "/".join(parts[1:-1])
    return dataset_name, model_subfolder, model_name


def _parse_csv_list(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("Expected a comma-separated list with at least one value.")
    return items


def _parse_crop_size(value: str) -> int | tuple[int, int]:
    items = _parse_csv_list(value)
    if len(items) == 1:
        return int(items[0])
    if len(items) == 2:
        return int(items[0]), int(items[1])
    raise argparse.ArgumentTypeError("crop_size must be 'N' or 'H,W'.")


def _parse_key_value_overrides(values: list[str] | None, parser: argparse.ArgumentParser) -> dict | object:
    if not values:
        return UNSET

    out: dict[str, object] = {}
    for value in values:
        key, sep, raw = value.partition("=")
        if not sep:
            parser.error(f"Expected KEY=VALUE override, got: {value}")
        key = key.strip()
        if not key:
            parser.error(f"Override key cannot be empty: {value}")
        out[key] = yaml.safe_load(raw)
    return out


def _maybe_unset(value):
    return UNSET if value is None else value


def add_apply_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("run", help="Run reference: dataset[/subfolder]/exp_name.")
    parser.add_argument("data_path", help="Input file or directory to process.")
    parser.add_argument(
        "-c",
        "--config",
        help="Inference config path, or a config name from configs/inference with or without .yml/.yaml. Defaults to defaults.yml.",
    )
    parser.add_argument("--save-folder", "--save_folder", dest="save_folder")
    parser.add_argument("--in-place", "--in_place", dest="in_place", action=argparse.BooleanOptionalAction)
    parser.add_argument("--epoch-number", "--epoch_number", dest="epoch_number", type=int)
    parser.add_argument("--best-or-last", "--best_or_last", dest="best_or_last", choices=["best", "last", "both"])
    parser.add_argument("--filters", type=_parse_csv_list)
    parser.add_argument("--skip-if-contain", "--skip_if_contain", dest="skip_if_contain", type=_parse_csv_list)
    parser.add_argument("--crop-size", "--crop_size", dest="crop_size", type=_parse_crop_size)
    parser.add_argument(
        "--keep-original-shape",
        "--keep_original_shape",
        dest="keep_original_shape",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument("--tiling-size", "--tiling_size", dest="tiling_size", type=int)
    parser.add_argument("--stack-selection-idx", "--stack_selection_idx", dest="stack_selection_idx", type=int)
    parser.add_argument("--timelapse-max", "--timelapse_max", dest="timelapse_max", type=int)
    parser.add_argument("--lvae-num-samples", "--lvae_num_samples", dest="lvae_num_samples", type=int)
    parser.add_argument(
        "--lvae-save-samples",
        "--lvae_save_samples",
        dest="lvae_save_samples",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--denormalize-output",
        "--denormalize_output",
        dest="denormalize_output",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument("--save-inp", "--save_inp", dest="save_inp", action=argparse.BooleanOptionalAction)
    parser.add_argument("--downsamp", type=int)
    parser.add_argument(
        "--apply-color-code",
        "--apply_color_code",
        dest="apply_color_code",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--color-code-option",
        "--color_code_option",
        dest="color_code_option",
        action="append",
        metavar="KEY=VALUE",
        help="Override nested apply.color_code_prm values, for example 'saturation=0.5'.",
    )
    parser.add_argument(
        "--dark-frame-context-length",
        "--dark_frame_context_length",
        dest="dark_frame_context_length",
        action=argparse.BooleanOptionalAction,
    )
    return parser


def add_evaluate_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "run",
        nargs="?",
        help=(
            "Run selector: dataset[/subfolder]/run_name, or run_name when paired with run_index. "
            "Use --run-id as an alternative selector."
        ),
    )
    parser.add_argument("run_index", nargs="?", type=int, help="Run index used with a run_name selector.")
    parser.add_argument("--run-id", help="Stable run identifier to evaluate.")
    add_run_filter_arguments(parser, include_identity=False, include_status=False)
    parser.add_argument(
        "-c",
        "--config",
        help="Inference config path, or a config name from configs/inference with or without .yml/.yaml. Defaults to defaults.yml.",
    )
    parser.add_argument("--best-or-last", "--best_or_last", dest="best_or_last", choices=["best", "last", "both"])
    parser.add_argument("--epoch-number", "--epoch_number", dest="epoch_number", type=int)
    parser.add_argument("--tiling-size", "--tiling_size", dest="tiling_size", type=int)
    parser.add_argument("--crop-size", "--crop_size", dest="crop_size", type=_parse_crop_size)
    parser.add_argument("--metrics", type=_parse_csv_list)
    parser.add_argument("--lvae-num-samples", "--lvae_num_samples", dest="lvae_num_samples", type=int)
    parser.add_argument("--save-folder", "--save_folder", dest="save_folder")
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction)
    parser.add_argument("--eval-gt", "--eval_gt", dest="eval_gt")
    parser.add_argument(
        "--data-option",
        "--data_option",
        dest="data_option",
        action="append",
        metavar="KEY=VALUE",
        help="Override nested evaluate.data_prm_update values, for example 'data_dir=/tmp/data'.",
    )
    parser.add_argument("--ch-out", "--ch_out", dest="ch_out", type=int)
    parser.add_argument("--split")
    parser.add_argument("--limit-n-imgs", "--limit_n_imgs", dest="limit_n_imgs", type=int)
    return parser


def run_apply_from_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    try:
        dataset_name, model_subfolder, model_name = _parse_run_ref(args.run)
    except ValueError as exc:
        parser.error(str(exc))
        raise AssertionError("argparse.error should raise SystemExit")

    run_apply_model(
        model_dataset=dataset_name,
        model_subfolder=model_subfolder,
        model_name=model_name,
        data_path=Path(args.data_path),
        config=args.config,
        save_folder=_maybe_unset(args.save_folder),
        in_place=_maybe_unset(args.in_place),
        epoch_number=_maybe_unset(args.epoch_number),
        best_or_last=_maybe_unset(args.best_or_last),
        filters=_maybe_unset(args.filters),
        skip_if_contain=_maybe_unset(args.skip_if_contain),
        crop_size=_maybe_unset(args.crop_size),
        keep_original_shape=_maybe_unset(args.keep_original_shape),
        tiling_size=_maybe_unset(args.tiling_size),
        stack_selection_idx=_maybe_unset(args.stack_selection_idx),
        timelapse_max=_maybe_unset(args.timelapse_max),
        lvae_num_samples=_maybe_unset(args.lvae_num_samples),
        lvae_save_samples=_maybe_unset(args.lvae_save_samples),
        denormalize_output=_maybe_unset(args.denormalize_output),
        save_inp=_maybe_unset(args.save_inp),
        downsamp=_maybe_unset(args.downsamp),
        apply_color_code=_maybe_unset(args.apply_color_code),
        color_code_prm=_parse_key_value_overrides(args.color_code_option, parser),
        dark_frame_context_length=_maybe_unset(args.dark_frame_context_length),
    )
    return 0


def run_evaluate_from_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    resolved = _resolve_evaluate_run_selector(args, parser=parser)
    if resolved is None:
        return 1
    dataset_name, model_subfolder, model_name = resolved

    run_evaluate(
        dataset_name=dataset_name,
        model_subfolder=model_subfolder,
        model_name=model_name,
        config=args.config,
        best_or_last=_maybe_unset(args.best_or_last),
        epoch_number=_maybe_unset(args.epoch_number),
        tiling_size=_maybe_unset(args.tiling_size),
        crop_size=_maybe_unset(args.crop_size),
        metrics_list=_maybe_unset(args.metrics),
        lvae_num_samples=_maybe_unset(args.lvae_num_samples),
        save_folder=_maybe_unset(args.save_folder),
        overwrite=_maybe_unset(args.overwrite),
        eval_gt=_maybe_unset(args.eval_gt),
        data_prm_update=_parse_key_value_overrides(args.data_option, parser),
        ch_out=_maybe_unset(args.ch_out),
        split=_maybe_unset(args.split),
        limit_n_imgs=_maybe_unset(args.limit_n_imgs),
    )
    return 0


def _resolve_evaluate_run_selector(
    args: argparse.Namespace,
    *,
    parser: argparse.ArgumentParser,
) -> tuple[str, str, str] | None:
    out = sys.stdout
    err = sys.stderr

    run = args.run
    run_index = args.run_index
    run_id = args.run_id
    dataset = args.dataset
    model_subfolder = args.model_subfolder

    if run_id is not None and (run is not None or run_index is not None):
        print("Use either <run_name> <run_index> or --run-id, not both.", file=err)
        return None

    if run_id is None:
        if run is None:
            print(
                "Missing run selector. Use dataset[/subfolder]/run_name, <run_name> <run_index>, or --run-id <run_id>.",
                file=err,
            )
            return None

        normalized = run.replace("\\", "/")
        has_run_ref_separator = "/" in normalized
        if run_index is None and has_run_ref_separator:
            if dataset is not None or model_subfolder is not None:
                print(
                    "--dataset/--subfolder can only be used with <run_name> <run_index> or --run-id selectors.",
                    file=err,
                )
                return None
            try:
                return _parse_run_ref(run)
            except ValueError as exc:
                parser.error(str(exc))
                raise AssertionError("argparse.error should raise SystemExit")

        if run_index is None:
            print(
                "Missing run_index for run_name selector. Use <run_name> <run_index>, "
                "or pass dataset[/subfolder]/run_name.",
                file=err,
            )
            return None
        if run_index < 0:
            print("run_index must be >= 0.", file=err)
            return None
        if has_run_ref_separator:
            print(
                "run_index cannot be combined with dataset[/subfolder]/run_name selectors.",
                file=err,
            )
            return None

    scan_result = scan_runs()
    matches = filter_runs(
        scan_result.runs,
        run_id=run_id,
        run_name=run if run_id is None else None,
        run_index=run_index if run_id is None else None,
        dataset=dataset,
        model_subfolder=model_subfolder,
    )

    if not matches:
        if run_id is not None:
            selector_desc = f"run_id={run_id!r}"
        else:
            selector_desc = f"run_name={run!r}, run_index={run_index}"
        print(f"No matching run found for {selector_desc}.", file=err)
        print("Use 'lisai runs list' to inspect available runs.", file=err)
        write_invalid_run_warnings(scan_result.invalid, stderr=err)
        return None

    selected = resolve_ambiguous_run_matches(
        matches,
        stdin=sys.stdin,
        stdout=out,
        stderr=err,
        rerun_hint="Rerun with --dataset/--subfolder or with --run-id to disambiguate.",
    )
    if selected is None:
        write_invalid_run_warnings(scan_result.invalid, stderr=err)
        return None

    print("Selected run:", file=out)
    print(render_runs_table([selected]), file=out)
    write_invalid_run_warnings(scan_result.invalid, stderr=err)
    return selected.dataset, selected.model_subfolder, selected.run_dir.name


def add_apply_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]):
    parser = subparsers.add_parser(
        "apply",
        help="Apply a trained model to one file or directory.",
        description="Apply a trained model to image file(s)",
    )
    add_apply_arguments(parser)
    parser.set_defaults(handler=lambda args, p=parser: run_apply_from_args(args, p))
    return parser


def add_evaluate_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]):
    parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate a trained model on a dataset split.",
        description="Evaluate a trained model on a dataset split",
    )
    add_evaluate_arguments(parser)
    parser.set_defaults(handler=lambda args, p=parser: run_evaluate_from_args(args, p))
    return parser


def build_apply_parser(*, prog: str = "lisai apply") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply a trained model to image file(s)", prog=prog)
    add_apply_arguments(parser)
    return parser


def build_evaluate_parser(*, prog: str = "lisai evaluate") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained model on a dataset split", prog=prog)
    add_evaluate_arguments(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lisai evaluation")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True
    add_apply_subparser(subparsers)
    add_evaluate_subparser(subparsers)
    args = parser.parse_args(list(argv) if argv is not None else None)
    return args.handler(args)
