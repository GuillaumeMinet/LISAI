from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

import lisai.config.io.resolver as resolver_mod
from lisai.config import save_yaml, settings
from lisai.config.io.resolver import prune_config_for_saving, resolve_config
from lisai.config.models import ResolvedExperiment
from lisai.infra.paths import Paths

PROJECT_CFG = Path("configs/project_config.yml")
DATA_CFG = Path("configs/data_config.yml")


def test_resolve_config_train_mode_forbids_load_model_section(tmp_path: Path):
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

    with pytest.raises(ValidationError, match="load_model"):
        resolve_config(
            experiment_cfg_path=exp_cfg,
            project_cfg_path=PROJECT_CFG,
            data_cfg_path=DATA_CFG,
        )



def test_apply_mode_resolution_continue_training_loads_origin_and_applies_sparse_overrides(tmp_path: Path):
    origin_run_dir = tmp_path / "origin_run"
    origin_run_dir.mkdir(parents=True, exist_ok=True)

    origin_cfg_path = Paths(settings).cfg_train_path(run_dir=origin_run_dir)
    save_yaml(
        {
            "experiment": {"mode": "train", "exp_name": "origin_exp"},
            "data": {"dataset_name": "origin_ds", "patch_size": 64},
            "model": {"architecture": "unet", "parameters": {"feat": 8}},
            "training": {"n_epochs": 50, "batch_size": 2},
        },
        origin_cfg_path,
    )

    user_cfg = {
        "experiment": {
            "mode": "continue_training",
            "post_training_inference": False,
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
    assert resolver_mod._dget(merged, "experiment.exp_name") == "origin_exp"
    assert resolver_mod._dget(merged, "experiment.post_training_inference") is False
    assert Path(resolver_mod._dget(merged, "experiment.origin_run_dir")).resolve() == origin_run_dir.resolve()

    assert resolver_mod._dget(merged, "model.architecture") == "unet"
    assert resolver_mod._dget(merged, "model.parameters.feat") == 8

    assert resolver_mod._dget(merged, "training.n_epochs") == 3
    assert resolver_mod._dget(merged, "training.batch_size") == 2
    assert resolver_mod._dget(merged, "load_model.checkpoint.method") == "state_dict"
    assert resolver_mod._dget(merged, "load_model.checkpoint.selector") == "last"



def test_resolve_config_continue_training_requires_load_model_section(tmp_path: Path):
    exp_cfg = tmp_path / "exp_invalid_continue.yml"
    save_yaml(
        {
            "experiment": {"mode": "continue_training"},
        },
        exp_cfg,
    )

    with pytest.raises(ValidationError, match="load_model"):
        resolve_config(
            experiment_cfg_path=exp_cfg,
            project_cfg_path=PROJECT_CFG,
            data_cfg_path=DATA_CFG,
        )



def test_resolve_config_continue_training_forbids_exp_name_override(tmp_path: Path):
    exp_cfg = tmp_path / "exp_continue_with_name.yml"
    save_yaml(
        {
            "experiment": {"mode": "continue_training", "exp_name": "new_name"},
            "load_model": {
                "canonical_load": False,
                "model_full_path": str(tmp_path / "origin_run"),
                "load_method": "state_dict",
            },
        },
        exp_cfg,
    )

    with pytest.raises(ValidationError, match="exp_name"):
        resolve_config(
            experiment_cfg_path=exp_cfg,
            project_cfg_path=PROJECT_CFG,
            data_cfg_path=DATA_CFG,
        )



def test_resolve_config_continue_training_forbids_loss_function_override(tmp_path: Path):
    exp_cfg = tmp_path / "exp_continue_with_loss.yml"
    save_yaml(
        {
            "experiment": {"mode": "continue_training"},
            "load_model": {
                "canonical_load": False,
                "model_full_path": str(tmp_path / "origin_run"),
                "load_method": "state_dict",
            },
            "loss_function": {"name": "CharEdge_loss", "CharEdge_loss_prm": {"alpha": 0.1}},
        },
        exp_cfg,
    )

    with pytest.raises(ValidationError, match="loss_function"):
        resolve_config(
            experiment_cfg_path=exp_cfg,
            project_cfg_path=PROJECT_CFG,
            data_cfg_path=DATA_CFG,
        )



def test_resolve_config_retrain_allows_data_and_normalization_overrides(tmp_path: Path):
    origin_run_dir = tmp_path / "origin_run"
    origin_run_dir.mkdir(parents=True, exist_ok=True)

    origin_cfg_path = Paths(settings).cfg_train_path(run_dir=origin_run_dir)
    save_yaml(
        {
            "experiment": {"mode": "train", "exp_name": "origin_exp"},
            "routing": {"models_subfolder": "origin_subfolder"},
            "data": {"dataset_name": "origin_ds", "patch_size": 64},
            "model": {"architecture": "unet", "parameters": {"feat": 8}},
            "normalization": {"load_from_noise_model": True},
            "noise_model": {"name": "origin_noise"},
            "training": {"n_epochs": 50},
        },
        origin_cfg_path,
    )

    exp_cfg = tmp_path / "retrain.yml"
    save_yaml(
        {
            "experiment": {"mode": "retrain", "exp_name": "retrain_exp"},
            "routing": {"models_subfolder": "retrain_subfolder"},
            "data": {"dataset_name": "target_ds", "patch_size": 32},
            "normalization": {"load_from_noise_model": False},
            "noise_model": {"name": "target_noise"},
            "load_model": {
                "canonical_load": False,
                "model_full_path": str(origin_run_dir),
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

    assert cfg.experiment.mode == "retrain"
    assert cfg.experiment.exp_name == "retrain_exp"
    assert Path(cfg.experiment.origin_run_dir).resolve() == origin_run_dir.resolve()
    assert cfg.routing.models_subfolder == "retrain_subfolder"
    assert cfg.data.dataset_name == "target_ds"
    assert cfg.data.patch_size == 32
    assert cfg.model.architecture == "unet"
    assert cfg.normalization.load_from_noise_model is False
    assert cfg.noise_model.name == "target_noise"
    assert cfg.load_model.enabled is True
    assert Path(cfg.load_model.run_dir).resolve() == origin_run_dir.resolve()



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