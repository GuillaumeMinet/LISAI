from __future__ import annotations

from pathlib import Path

import pytest

import lisai.infra.config.resolver as resolver_mod
from lisai.infra.config import save_yaml, settings
from lisai.infra.config.resolver import prune_config_for_saving, resolve_config
from lisai.infra.config.schema import ResolvedExperiment
from lisai.infra.paths import Paths

PROJECT_CFG = Path("configs/project/project.yml")
DATA_CFG = Path("configs/data/data.yml")


def test_resolve_config_train_mode_disables_load_model(tmp_path: Path):
    exp_cfg = tmp_path / "exp_train.yml"
    save_yaml(
        {
            "experiment": {"mode": "train", "exp_name": "exp_train"},
            "data": {"dataset_name": "ds_train"},
            "model": {"architecture": "unet", "parameters": {}},
            "load_model": {
                "canonical_load": False,
                "model_full_path": str(tmp_path / "ignored_origin"),
                "load_method": "state_dict",
            },
        },
        exp_cfg,
    )

    cfg = resolve_config(
        experiment_cfg_path=exp_cfg,
        project_cfg_path=PROJECT_CFG,
        data_cfg_path=DATA_CFG,
    )

    assert cfg.experiment.mode == "train"
    assert cfg.load_model.enabled is False
    assert cfg.load_model.run_dir is None
    assert cfg.experiment.origin_run_dir is None


def test_apply_mode_resolution_continue_training_loads_origin_and_applies_overrides(tmp_path: Path):
    origin_run_dir = tmp_path / "origin_run"
    origin_run_dir.mkdir(parents=True, exist_ok=True)

    origin_cfg_path = Paths(settings).cfg_train_path(run_dir=origin_run_dir)
    save_yaml(
        {
            "experiment": {"mode": "train", "exp_name": "origin_exp"},
            "data": {"dataset_name": "origin_ds", "patch_size": 64},
            "model": {"architecture": "unet", "parameters": {"channels": 8}},
            "training": {"n_epochs": 50, "batch_size": 2},
        },
        origin_cfg_path,
    )

    user_cfg = {
        "experiment": {
            "mode": "continue_training",
            "exp_name": "exp_continue",
            "origin_run_dir": str(origin_run_dir),
        },
        "training": {"n_epochs": 3},
        "load_model": {"checkpoint": {"method": "state_dict", "selector": "last"}},
    }

    merged = resolver_mod._apply_mode_resolution(
        user_cfg=user_cfg,
        mode="continue_training",
        paths=Paths(settings),
    )

    assert resolver_mod._dget(merged, "experiment.mode") == "continue_training"
    assert resolver_mod._dget(merged, "experiment.exp_name") == "exp_continue"
    assert Path(resolver_mod._dget(merged, "experiment.origin_run_dir")).resolve() == origin_run_dir.resolve()

    # Comes from origin config
    assert resolver_mod._dget(merged, "model.architecture") == "unet"
    assert resolver_mod._dget(merged, "model.parameters.channels") == 8

    # Comes from user override
    assert resolver_mod._dget(merged, "training.n_epochs") == 3
    assert resolver_mod._dget(merged, "load_model.checkpoint.method") == "state_dict"
    assert resolver_mod._dget(merged, "load_model.checkpoint.selector") == "last"


def test_resolve_config_continue_training_requires_load_model_section(tmp_path: Path):
    exp_cfg = tmp_path / "exp_invalid_continue.yml"
    save_yaml(
        {
            "experiment": {"mode": "continue_training", "exp_name": "broken_continue"},
        },
        exp_cfg,
    )

    with pytest.raises(ValueError, match="requires 'load_model' section"):
        resolve_config(
            experiment_cfg_path=exp_cfg,
            project_cfg_path=PROJECT_CFG,
            data_cfg_path=DATA_CFG,
        )


def test_prune_config_for_saving_drops_train_only_and_disabled_sections():
    cfg = ResolvedExperiment.model_validate(
        {
            "experiment": {
                "mode": "train",
                "exp_name": "exp_prune",
                "origin_run_dir": "C:/tmp/origin",
            },
            "data": {"dataset_name": "ds"},
            "model": {"architecture": "unet", "parameters": {}},
            "saving": {"enabled": False},
            "tensorboard": {"enabled": False},
            "load_model": {
                "enabled": True,
                "source": "path",
                "run_dir": "C:/tmp/origin",
                "checkpoint": {"method": "state_dict"},
            },
        }
    )

    out = prune_config_for_saving(cfg)

    assert "load_model" not in out
    assert "origin_run_dir" not in out["experiment"]
    assert "saving" not in out
    assert "tensorboard" not in out
