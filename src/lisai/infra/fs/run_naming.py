import re
from pathlib import Path

from lisai.config import settings


def run_dir_index_width() -> int:
    return int(getattr(settings.NAMING, "run_dir_index_width", 2))


def format_run_dir_name(run_name: str, run_index: int, *, width: int | None = None) -> str:
    base_name = run_name.strip()
    if not base_name:
        raise ValueError("run_name must not be empty.")
    if run_index < 0:
        raise ValueError("run_index must be >= 0.")

    digits = run_dir_index_width() if width is None else int(width)
    if digits < 1:
        raise ValueError("run_dir index width must be >= 1.")
    return f"{base_name}_{run_index:0{digits}d}"


def parse_run_dir_name(run_dir_name: str) -> tuple[str, int]:
    name = run_dir_name.strip()
    if not name:
        raise ValueError("run_dir_name must not be empty.")

    match = re.match(r"^(?P<run_name>.+)_(?P<run_index>\d+)$", name)
    if not match:
        return name, 0

    run_name = match.group("run_name").strip()
    if not run_name:
        raise ValueError(f"Invalid run directory name: {run_dir_name!r}")
    return run_name, int(match.group("run_index"))


def next_run_index(save_dir: Path, run_name: str) -> int:
    root = Path(save_dir)
    if not root.exists():
        return 0

    base_name = run_name.strip()
    if not base_name:
        raise ValueError("run_name must not be empty.")

    pattern = re.compile(rf"^{re.escape(base_name)}_(\d+)$")
    seen: list[int] = []
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        match = pattern.match(entry.name)
        if match is None:
            continue
        seen.append(int(match.group(1)))

    if not seen:
        return 0
    return max(seen) + 1


def allocate_run_dir_name(save_dir: Path, run_name: str, *, width: int | None = None) -> tuple[str, int]:
    run_index = next_run_index(save_dir, run_name)
    return format_run_dir_name(run_name, run_index, width=width), run_index


def get_unique_exp_name(save_dir: Path, exp_name: str) -> str:
    """
    Calculates the next available exp name (e.g., 'exp1_02') by scanning save_dir.

    Note: this helper is kept for non-run outputs (e.g. evaluation folders).
    Training run directories now use explicit run_name/run_index allocation.
    """
    save_dir = Path(save_dir)

    if not save_dir.exists():
        return exp_name

    base_exp_path = save_dir / exp_name
    if not base_exp_path.exists():
        return exp_name

    fmt = settings.NAMING.exp_name_format

    existing_names = [p.name for p in save_dir.iterdir() if p.is_dir()]
    max_id = 0

    for name in existing_names:
        if not name.startswith(exp_name):
            continue

        suffix = name[len(exp_name):]
        clean_suffix = re.sub(r'^[^0-9]+', '', suffix)

        if clean_suffix.isdigit():
            idx = int(clean_suffix)
            if idx > max_id:
                max_id = idx

    return fmt.format(name=exp_name, id=max_id + 1)
