from __future__ import annotations

from pathlib import Path

import pytest
import torch

import lisai.evaluation.runtime as runtime_mod



def test_initialize_runtime_builds_inference_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    raw_cfg = {"noise_model": {"name": "noise_A"}}
    normalized_cfg = {
        "model_architecture": "lvae",
        "data_prm": {"patch_size": 64},
        "model_norm_prm": {"data_mean": 1.0},
        "normalization": {"norm_prm": {"clip": 0}},
    }
    fake_paths = object()
    model_obj = object()

    monkeypatch.setattr(runtime_mod, "Paths", lambda _settings: fake_paths)
    monkeypatch.setattr(runtime_mod, "load_training_cfg_from_run", lambda run_dir: raw_cfg)
    monkeypatch.setattr(runtime_mod, "normalize_training_cfg_for_inference", lambda cfg: dict(normalized_cfg))
    monkeypatch.setattr(runtime_mod, "extract_model_architecture", lambda cfg: "lvae")
    monkeypatch.setattr(runtime_mod, "extract_data_prm", lambda cfg: {"patch_size": 64})
    monkeypatch.setattr(runtime_mod, "extract_patch_size_and_downsamp_factor", lambda data_prm: (64, 2))
    monkeypatch.setattr(runtime_mod, "extract_noise_model_name", lambda cfg: "noise_A")
    monkeypatch.setattr(runtime_mod, "extract_norm_prm", lambda cfg, data_prm: {"clip": 0})
    monkeypatch.setattr(runtime_mod, "resolve_tiling_size", lambda training_cfg, user_tiling_size: 128)
    monkeypatch.setattr(runtime_mod, "resolve_upsampling_factor", lambda training_cfg: 4)
    monkeypatch.setattr(runtime_mod, "resolve_context_length", lambda training_cfg: 5)
    monkeypatch.setattr(
        runtime_mod,
        "_load_model_from_run",
        lambda **kwargs: ("state_dict", tmp_path / "checkpoint.pt", model_obj),
    )

    runtime = runtime_mod.initialize_runtime(
        model_folder=tmp_path,
        device="cpu",
        best_or_last="best",
        epoch_number=None,
        tiling_size=128,
    )

    assert runtime.paths is fake_paths
    assert runtime.device == torch.device("cpu")
    assert runtime.run_dir == tmp_path
    assert runtime.model is model_obj
    assert runtime.architecture == "lvae"
    assert runtime.is_lvae is True
    assert runtime.data_prm == {"patch_size": 64}
    assert runtime.data_norm_prm == {"clip": 0}
    assert runtime.model_norm_prm == {"data_mean": 1.0}
    assert runtime.noise_model_name == "noise_A"
    assert runtime.patch_size == 64
    assert runtime.downsamp_factor == 2
    assert runtime.tiling_size == 128
    assert runtime.upsampling_factor == 4
    assert runtime.context_length == 5
    assert runtime.checkpoint_path == tmp_path / "checkpoint.pt"
    assert runtime.load_method == "state_dict"



def test_initialize_runtime_requires_architecture(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(runtime_mod, "Paths", lambda _settings: object())
    monkeypatch.setattr(runtime_mod, "load_training_cfg_from_run", lambda run_dir: {})
    monkeypatch.setattr(runtime_mod, "normalize_training_cfg_for_inference", lambda cfg: {})
    monkeypatch.setattr(runtime_mod, "extract_model_architecture", lambda cfg: None)

    with pytest.raises(ValueError, match="architecture"):
        runtime_mod.initialize_runtime(model_folder=tmp_path, device="cpu")
