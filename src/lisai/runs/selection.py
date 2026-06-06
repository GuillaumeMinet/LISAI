from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from typing import TypeVar

from .listing import filter_runs, matches_exp_name, render_runs_table, write_invalid_run_warnings
from .scanner import DiscoveredRun, ScanResults, scan_runs

_T = TypeVar("_T")
_RUN_SELECTOR_HINT = (
    "Rerun with --dataset/--subfolder or with --run-id to disambiguate."
)


def resolve_ambiguous_matches(
    matches: Sequence[_T],
    *,
    render_matches: Callable[[Sequence[_T]], str],
    heading: str,
    rerun_hint: str,
    selection_name: str = "item",
    stdin=None,
    stdout=None,
    stderr=None,
) -> _T | None:
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]

    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr
    in_stream = sys.stdin if stdin is None else stdin

    print(heading, file=out)
    print(render_matches(matches), file=out)

    selected_idx = _prompt_selection_index(
        len(matches),
        stdin=in_stream,
        stdout=out,
        stderr=err,
        selection_name=selection_name,
    )
    if selected_idx is None:
        print(rerun_hint, file=err)
        return None
    return matches[selected_idx]


def resolve_ambiguous_run_matches(
    matches: Sequence[DiscoveredRun],
    *,
    stdin=None,
    stdout=None,
    stderr=None,
    rerun_hint: str = _RUN_SELECTOR_HINT,
) -> DiscoveredRun | None:
    return resolve_ambiguous_matches(
        matches,
        render_matches=lambda entries: render_runs_table(entries, include_selection_index=True),
        heading="Multiple matching runs found:",
        rerun_hint=rerun_hint,
        selection_name="run",
        stdin=stdin,
        stdout=stdout,
        stderr=stderr,
    )


def resolve_discovered_run_selector(
    *,
    selector: str | None = None,
    run_id: str | None = None,
    dataset: str | None = None,
    model_subfolder: str | None = None,
    scan_result: ScanResults | None = None,
    allow_partial_exp_name: bool = True,
    emit_selected: bool = True,
    stdin=None,
    stdout=None,
    stderr=None,
    rerun_hint: str = _RUN_SELECTOR_HINT,
) -> DiscoveredRun | None:
    """Resolve a public CLI run selector to a discovered run.

    Supported selectors:
    - ``--run-id`` via ``run_id``
    - ``dataset[/subfolder]/run_dir_name`` via ``selector``
    - bare ``run_dir_name`` via ``selector``
    - bare partial semantic experiment name when no exact run directory matches
    """
    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr

    normalized_selector = None if selector is None else selector.strip()
    if run_id is not None and normalized_selector:
        print("Use either a run selector or --run-id, not both.", file=err)
        return None

    if run_id is None and not normalized_selector:
        print(
            "Missing run selector. Use --run-id <run_id>, dataset[/subfolder]/run_dir_name, or run_dir_name.",
            file=err,
        )
        return None

    # find all runs
    resolved_scan = scan_runs() if scan_result is None else scan_result

    # filter by run_id if provided
    if run_id is not None:
        matches = filter_runs(
            resolved_scan.runs,
            run_id=run_id,
            dataset=dataset,
            model_subfolder=model_subfolder,
        )
        selector_description = f"run_id={run_id!r}"
    
    # public selector filtering such as exp_name or run_dir
    else:
        assert normalized_selector is not None
        matches, selector_description = _select_runs_by_public_selector(
            resolved_scan.runs,
            normalized_selector,
            dataset=dataset,
            model_subfolder=model_subfolder,
            allow_partial_exp_name=allow_partial_exp_name,
            stderr=err,
        )
        if matches is None:
            return None

    # 0 match case
    if not matches:
        print(f"No matching run found for {selector_description}.", file=err)
        print("Use 'lisai runs list' to inspect available runs.", file=err)
        write_invalid_run_warnings(resolved_scan.invalid, stderr=err)
        return None
    
    # 1 match only
    if len(matches) == 1:
        selected = matches[0]
    
    # more than 1 match: user-driven ambiguous resolution
    else:
        selected = resolve_ambiguous_run_matches(
            matches,
            stdin=sys.stdin if stdin is None else stdin,
            stdout=out,
            stderr=err,
            rerun_hint=rerun_hint,
        )
        if selected is None:
            write_invalid_run_warnings(resolved_scan.invalid, stderr=err)
            return None

    if emit_selected:
        print("Selected run:", file=out)
        print(render_runs_table([selected]), file=out)
    write_invalid_run_warnings(resolved_scan.invalid, stderr=err)
    return selected


def _select_runs_by_public_selector(
    runs: Sequence[DiscoveredRun],
    selector: str,
    *,
    dataset: str | None,
    model_subfolder: str | None,
    allow_partial_exp_name: bool,
    stderr,
) -> tuple[list[DiscoveredRun] | None, str]:
    normalized = selector.replace("\\", "/")

    # check if selector is of type dataset[/subfolder]/run_dir_name
    has_path_separator = "/" in normalized
    if has_path_separator:
        if dataset is not None or model_subfolder is not None:
            print(
                "--dataset/--subfolder cannot be combined with dataset[/subfolder]/run_dir_name selectors.",
                file=stderr,
            )
            return None, f"run={selector!r}"

        parts = [part for part in normalized.split("/") if part]
        if len(parts) < 2:
            print(
                "Run selector must be dataset[/subfolder]/run_dir_name.",
                file=stderr,
            )
            return None, f"run={selector!r}"
        
        # split dataset[/subfolder]/run_dir_name into dataset, [subfolder(s)], run_dir
        selector_dataset = parts[0]
        selector_model_subfolder = "/".join(parts[1:-1])
        run_dir_name = parts[-1]

        # filter
        matches = filter_runs(
            runs,
            run_dir_name=run_dir_name,
            dataset=selector_dataset,
            model_subfolder=selector_model_subfolder,
        )
        return matches, f"run={selector!r}"

    # cases where selector is the full run directory name, or the partial exp_name
    # fist we look for exact_matches
    exact_matches = filter_runs(
        runs,
        run_dir_name=selector,
        dataset=dataset,
        model_subfolder=model_subfolder,
    )
    if exact_matches or not allow_partial_exp_name:
        return exact_matches, f"run={selector!r}"

    # if not found, we look for patial matches
    scoped_runs = filter_runs(
        runs,
        dataset=dataset,
        model_subfolder=model_subfolder,
    )
    return [
        run
        for run in scoped_runs
        if matches_exp_name(run, selector)
    ], f"exp_name~={selector!r}"


def _prompt_selection_index(
    count: int,
    *,
    stdin,
    stdout,
    stderr,
    selection_name: str = "item",
) -> int | None:
    if count <= 1:
        return 0 if count == 1 else None

    is_tty = getattr(stdin, "isatty", None)
    if not callable(is_tty) or not is_tty():
        return None

    while True:
        print(
            f"Select {selection_name} number from '#' (for example 01), or press Enter to cancel: ",
            end="",
            file=stdout,
            flush=True,
        )
        answer = stdin.readline()
        if answer == "":
            return None
        choice = answer.strip()
        if choice == "":
            return None
        if not choice.isdigit():
            print("Invalid selection. Enter a number from the '#' column.", file=stderr)
            continue

        selected = int(choice)
        if 1 <= selected <= count:
            return selected - 1
        print(f"Selection out of range. Enter a value between 1 and {count}.", file=stderr)


__all__ = [
    "resolve_ambiguous_matches",
    "resolve_ambiguous_run_matches",
    "resolve_discovered_run_selector",
]
