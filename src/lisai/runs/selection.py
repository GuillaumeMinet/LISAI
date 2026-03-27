from __future__ import annotations

import sys
from collections.abc import Sequence

from .listing import render_runs_table
from .scanner import DiscoveredRun


def resolve_ambiguous_run_matches(
    matches: Sequence[DiscoveredRun],
    *,
    stdin=None,
    stdout=None,
    stderr=None,
    rerun_hint: str = "Rerun with --dataset/--subfolder or with --run-id to disambiguate.",
) -> DiscoveredRun | None:
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]

    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr
    in_stream = sys.stdin if stdin is None else stdin

    print("Multiple matching runs found:", file=out)
    print(render_runs_table(matches, include_selection_index=True), file=out)

    selected_idx = _prompt_selection_index(
        len(matches),
        stdin=in_stream,
        stdout=out,
        stderr=err,
    )
    if selected_idx is None:
        print(rerun_hint, file=err)
        return None
    return matches[selected_idx]


def _prompt_selection_index(
    count: int,
    *,
    stdin,
    stdout,
    stderr,
) -> int | None:
    if count <= 1:
        return 0 if count == 1 else None

    is_tty = getattr(stdin, "isatty", None)
    if not callable(is_tty) or not is_tty():
        return None

    while True:
        print(
            "Select run number from '#' (for example 01), or press Enter to cancel: ",
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


__all__ = ["resolve_ambiguous_run_matches"]
