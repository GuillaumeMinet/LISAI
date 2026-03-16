import re
from pathlib import Path

from lisai.config import settings


def get_unique_exp_name(save_dir: Path, exp_name: str) -> str:
    """
    Calculates the next available exp name (e.g., 'exp1_02') by scanning save_dir.
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
