from __future__ import annotations

import pytest

import lisai.training.setup.model as model_mod
from lisai.config.models import ResolvedExperiment



def _training_cfg(**overrides) -> ResolvedExperiment:
    base = {
        "experiment": {
            "mode": "continue_training",
            "exp_name": "exp1",
            "origin_run_dir": "C:/tmp/origin_run",
        },
        "data": {
            "dataset_name": "ds",
            "patch_size": 64,
            "downsampling": {
                "downsamp_factor": 2,
                "downsamp_method": "random",
            },
        },
        "model": {
            "architecture": "lvae",
            "parameters": {"num_latents": 3},
        },
        "noise_model": {"name": "noise_A"},
        "load_model": {
            "enabled": True,
            "checkpoint": {
                "method": "state_dict",
                "selector": "last",
                "epoch": 7,
                "filename": "model_last_state_dict.pt",
            },
        },
    }

    base.update(overrides)
    return ResolvedExperiment.model_validate(base)



def test_training_model_spec_extracts_model_load_fields():
    cfg = _training_cfg()

    spec = model_mod.TrainingModelSpec.from_config(cfg)

    assert spec.architecture == "lvae"
    assert spec.parameters == {"num_latents": 3}
    assert spec.mode == "continue_training"
    assert spec.patch_size == 64
    assert spec.downsamp_factor == 2
    assert str(spec.origin_run_dir).replace("\\", "/").endswith("/origin_run")
    assert spec.checkpoint_method == "state_dict"
    assert spec.checkpoint_selector == "last"
    assert spec.checkpoint_epoch == 7
    assert spec.checkpoint_filename == "model_last_state_dict.pt"
    assert spec.noise_model_name == "noise_A"



def test_training_model_spec_uses_val_patch_size_when_needed():
    cfg = _training_cfg(
        experiment={"mode": "train", "exp_name": "exp2"},
        data={"dataset_name": "ds", "val_patch_size": 128},
        model={"architecture": "unet", "parameters": {}},
        noise_model=None,
        load_model={},
    )

    spec = model_mod.TrainingModelSpec.from_config(cfg)

    assert spec.patch_size == 128
    assert spec.downsamp_factor == 1
    assert spec.origin_run_dir is None
    assert spec.noise_model_name is None



def test_build_model_uses_training_model_spec(monkeypatch: pytest.MonkeyPatch):
    cfg = _training_cfg(
        experiment={"mode": "train", "exp_name": "exp3"},
        model={"architecture": "unet", "parameters": {"depth": 4}},
        noise_model=None,
        load_model={},
    )
    captured = {}
    model_obj = object()
    state = {"epoch": 3}

    def fake_prepare_model_for_training(*, spec, device, model_norm_prm=None, noise_model=None):
        captured["spec"] = spec
        captured["device"] = device
        captured["model_norm_prm"] = model_norm_prm
        captured["noise_model"] = noise_model
        return model_obj, state

    monkeypatch.setattr(model_mod, "prepare_model_for_training", fake_prepare_model_for_training)

    out_model, out_state = model_mod.build_model(
        cfg,
        device="cpu",
        model_norm_prm={"std": 2.0},
        noise_model=None,
    )

    assert out_model is model_obj
    assert out_state == state
    assert isinstance(captured["spec"], model_mod.TrainingModelSpec)
    assert captured["spec"].architecture == "unet"
    assert captured["spec"].parameters == {"depth": 4}
    assert captured["device"] == "cpu"
    assert captured["model_norm_prm"] == {"std": 2.0}
    assert captured["noise_model"] is None
