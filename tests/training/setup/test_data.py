from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import lisai.training.setup.data as data_mod



def test_prepare_data_returns_prepared_training_data(monkeypatch):
    resolved_kwargs = {}

    def fake_resolved(**kwargs):
        resolved_kwargs.update(kwargs)
        return "resolved_data_cfg"

    cfg = SimpleNamespace(
        model=SimpleNamespace(architecture="unet"),
        normalization={"norm_prm": {"mean": 1.0}},
        data=SimpleNamespace(dataset_name="demo", resolved=fake_resolved),
        routing=SimpleNamespace(data_subfolder="raw"),
    )
    runtime = SimpleNamespace(
        paths=SimpleNamespace(
            dataset_dir=lambda dataset_name, data_subfolder: Path("/tmp/data"),
            dataset_registry_path=lambda: Path("/tmp/registry.yaml"),
        )
    )

    monkeypatch.setattr(data_mod, "load_yaml", lambda path: {"demo": {"shape": [1, 2, 3]}})
    monkeypatch.setattr(
        data_mod,
        "make_training_loaders",
        lambda config: ("train_loader", "val_loader", {"std": 2.0}, {"patch": 64}),
    )

    prepared = data_mod.prepare_data(cfg, runtime)

    assert isinstance(prepared, data_mod.PreparedTrainingData)
    assert prepared.train_loader == "train_loader"
    assert prepared.val_loader == "val_loader"
    assert prepared.data_norm_prm == {"mean": 1.0}
    assert prepared.model_norm_prm == {"std": 2.0}
    assert prepared.patch_info == {"patch": 64}
    assert resolved_kwargs == {
        "data_dir": Path("/tmp/data"),
        "norm_prm": {"mean": 1.0},
        "dataset_info": {"shape": [1, 2, 3]},
        "volumetric": False,
    }



def test_prepare_data_uses_noise_model_metadata_for_lvae(monkeypatch):
    resolved_kwargs = {}
    calls = []

    def fake_resolved(**kwargs):
        resolved_kwargs.update(kwargs)
        return "resolved_data_cfg"

    cfg = SimpleNamespace(
        model=SimpleNamespace(architecture="lvae"),
        normalization={"norm_prm": {"mean": 0.0}, "load_from_noise_model": True},
        data=SimpleNamespace(dataset_name="demo", resolved=fake_resolved),
        routing=SimpleNamespace(data_subfolder="raw"),
    )
    runtime = SimpleNamespace(
        paths=SimpleNamespace(
            dataset_dir=lambda dataset_name, data_subfolder: Path("/tmp/data"),
            dataset_registry_path=lambda: Path("/tmp/registry.yaml"),
        )
    )

    def fake_resolve_noise_model_metadata(cfg_arg, paths_arg):
        calls.append((cfg_arg, paths_arg))
        return {"mean": 3.0}

    monkeypatch.setattr(data_mod, "resolve_noise_model_metadata", fake_resolve_noise_model_metadata)
    monkeypatch.setattr(data_mod, "load_yaml", lambda path: {})
    monkeypatch.setattr(
        data_mod,
        "make_training_loaders",
        lambda config: ("train_loader", "val_loader", {"std": 2.0}, None),
    )

    prepared = data_mod.prepare_data(cfg, runtime)

    assert calls == [(cfg, runtime.paths)]
    assert prepared.data_norm_prm == {"mean": 3.0}
    assert resolved_kwargs["norm_prm"] == {"mean": 3.0}
