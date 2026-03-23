"""Evaluation data reconstruction helpers.

This module owns the evaluation-side data preparation logic that depends on a
`SavedTrainingRun`: resolving the dataset location, applying evaluation-only
overrides, and building a test loader for `run_evaluate`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from lisai.config import load_yaml, settings
from lisai.config.io import deep_merge
from lisai.config.models.training import DataSection
from lisai.data.data_prep import make_test_loader
from lisai.infra.paths import Paths

from .saved_run import SavedTrainingRun



def resolve_dataset_info(dataset_name: str | None) -> dict[str, Any] | None:
    """Load dataset-registry metadata for a dataset name when available."""
    if not dataset_name:
        return None

    paths = Paths(settings)
    try:
        registry = load_yaml(paths.dataset_registry_path())
    except FileNotFoundError:
        return None

    info = registry.get(dataset_name)
    if isinstance(info, Mapping):
        return dict(info)
    return None



def resolve_eval_data_dir(saved_run: SavedTrainingRun, data_cfg: Mapping[str, Any]) -> Path | None:
    """Resolve the dataset directory used for evaluation from saved config and overrides."""
    for key in ("data_dir", "full_data_path"):
        value = data_cfg.get(key)
        if value:
            return Path(value)

    if data_cfg.get("canonical_load") is False:
        return None

    dataset_name = data_cfg.get("dataset_name") or saved_run.dataset_name
    if not dataset_name:
        return None

    subfolder = data_cfg.get("subfolder")
    if subfolder is None:
        subfolder = saved_run.data_subfolder

    paths = Paths(settings)
    return paths.dataset_dir(dataset_name=dataset_name, data_subfolder=subfolder or "")



def build_eval_loader(
    saved_run: SavedTrainingRun,
    *,
    split: str = "test",
    crop_size: int | tuple[int, int] | None = None,
    eval_gt=None,
    data_prm_update: Mapping[str, Any] | None = None,
):
    """Build the evaluation dataloader for a saved run and a set of eval overrides."""
    data_cfg = dict(saved_run.data_cfg)
    model_norm_prm = dict(saved_run.model_norm_prm) if saved_run.model_norm_prm is not None else None

    if eval_gt is not None and data_cfg.get("paired") is False:
        data_cfg["paired"] = True
        data_cfg["target"] = eval_gt
        if model_norm_prm is None:
            model_norm_prm = {}
        model_norm_prm["data_mean_gt"] = 0
        model_norm_prm["data_std_gt"] = 1

    if crop_size is not None:
        data_cfg["initial_crop"] = crop_size

    if data_prm_update is not None:
        data_cfg = deep_merge(data_cfg, dict(data_prm_update))

    data_dir = resolve_eval_data_dir(saved_run, data_cfg)
    if data_dir is None:
        raise ValueError(
            "Could not resolve `data_dir` for evaluation. "
            "Provide it through `data_prm_update={\'data_dir\': \'...path...\'}`."
        )

    dataset_info = resolve_dataset_info(data_cfg.get("dataset_name") or saved_run.dataset_name)
    prep_cfg = DataSection.model_validate(data_cfg).resolved(
        data_dir=Path(data_dir),
        norm_prm=saved_run.data_norm_prm,
        dataset_info=dataset_info,
        model_norm_prm=model_norm_prm,
        split=split,
    )
    return make_test_loader(config=prep_cfg)



__all__ = ["build_eval_loader", "resolve_dataset_info", "resolve_eval_data_dir"]
