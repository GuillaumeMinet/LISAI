from __future__ import annotations

import json
from pathlib import Path

from tifffile import imwrite

from lisai.data.utils import get_saving_shape
from lisai.infra.fs import ensure_folder
from lisai.infra.fs.run_naming import get_unique_exp_name


def create_save_folder(path: Path, overwrite: bool = False, parent_exists_check: bool = False) -> Path | None:
    path = Path(path)
    parent = path.parent

    if parent_exists_check and not parent.exists():
        return None

    ensure_folder(parent, mode="exist_ok")

    if path.exists():
        if overwrite:
            return ensure_folder(path, mode="overwrite")
        unique_name = get_unique_exp_name(parent, path.name)
        path = parent / unique_name

    return ensure_folder(path, mode="strict")


def ensure_save_folder(path: Path) -> Path:
    path = Path(path)
    ensure_folder(path, mode="exist_ok")
    return path


def resolve_prediction_inputs(
    data_path: Path,
    *,
    filters: list[str] | str,
    skip_if_contain: list[str] | None = None,
) -> tuple[Path, list[str], str | None]:
    data_path = Path(data_path)
    if isinstance(filters, str):
        filters = [filters]
    filters = [f.lower() for f in filters]

    if data_path.is_dir():
        list_files = []
        for file_name in sorted(data_path.iterdir()):
            if not file_name.is_file():
                continue
            suffix = file_name.suffix.lower().replace(".", "")
            if suffix not in filters:
                continue
            if skip_if_contain is not None and any(skip in file_name.name for skip in skip_if_contain):
                continue
            list_files.append(file_name.name)

        if not list_files:
            raise FileNotFoundError(f"No file found in {data_path} with filters={filters}.")
        return data_path, list_files, None

    if data_path.is_file():
        return data_path, [""], data_path.name

    if data_path.suffix.lower() in {".tif", ".tiff"}:
        suffix_to_try = ".tif" if data_path.suffix.lower() == ".tiff" else ".tiff"
        path_to_try = data_path.with_suffix(suffix_to_try)
        if path_to_try.is_file():
            return path_to_try, [""], path_to_try.name

    raise FileNotFoundError(f"Input path not found: {data_path}")


def save_metrics_json(save_folder: Path, results: dict) -> None:
    save_folder = Path(save_folder)
    with open(save_folder / "metrics.json", "w") as f:
        json.dump(results, f, indent=4)


def save_outputs(tosave: dict, save_folder: Path, img_name: str, no_suffix: bool = False) -> None:
    """
    Save inference outputs as TIFF files.
    """
    for key, item in tosave.items():
        if item is None:
            continue
        if len(tosave) == 1 and no_suffix:
            path = Path(save_folder) / f"{img_name}.tif"
        else:
            path = Path(save_folder) / f"{img_name}_{key}.tif"

        if key == "pred_colorCoded":
            imwrite(path, item, photometric="rgb")
        else:
            shape = get_saving_shape(item)
            imwrite(path, item, imagej=True, metadata={"axes": shape})
