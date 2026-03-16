from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import lisai.training.setup.run_dir as run_dir_mod


class FakePaths:
    def checkpoints_dir(self, *, run_dir):
        return Path(run_dir) / "checkpoints"

    def validation_images_dir(self, *, run_dir):
        return Path(run_dir) / "validation_images"

    def retrain_origin_dir(self, *, run_dir):
        return Path(run_dir) / "retrain_origin"

    def loss_file_path(self, *, run_dir):
        return Path(run_dir) / "loss.txt"

    def log_file_path(self, *, run_dir):
        return Path(run_dir) / "train_log.log"

    def cfg_train_path(self, *, run_dir):
        return Path(run_dir) / "config_train.yaml"

    def retrain_origin_loss_path(self, *, run_dir):
        return self.retrain_origin_dir(run_dir=run_dir) / "origin_loss.txt"

    def retrain_origin_log_path(self, *, run_dir):
        return self.retrain_origin_dir(run_dir=run_dir) / "origin_log.log"

    def retrain_origin_cfg_path(self, *, run_dir):
        return self.retrain_origin_dir(run_dir=run_dir) / "origin_config.yaml"


def _make_cfg(*, mode: str, saving_enabled: bool, origin_run_dir: str | None = None, overwrite: bool = False):
    return SimpleNamespace(
        saving=SimpleNamespace(enabled=saving_enabled),
        experiment=SimpleNamespace(
            mode=mode,
            exp_name="exp_name",
            origin_run_dir=origin_run_dir,
            overwrite=overwrite,
        ),
        data=SimpleNamespace(dataset_name="dataset_a"),
        routing=SimpleNamespace(models_subfolder="subfolder_a"),
    )


def test_prepare_run_dir_returns_none_when_saving_disabled(tmp_path: Path):
    cfg = _make_cfg(mode="train", saving_enabled=False)
    ctx = SimpleNamespace(paths=FakePaths())

    run_dir, exp_name = run_dir_mod.prepare_run_dir(cfg, ctx)

    assert run_dir is None
    assert exp_name == "exp_name"


def test_prepare_run_dir_continue_training_requires_origin_path():
    cfg = _make_cfg(mode="continue_training", saving_enabled=True, origin_run_dir=None)
    ctx = SimpleNamespace(paths=FakePaths())

    with pytest.raises(ValueError, match="requires experiment.origin_run_dir"):
        run_dir_mod.prepare_run_dir(cfg, ctx)


def test_prepare_run_dir_continue_training_reuses_origin_folder(tmp_path: Path):
    origin = tmp_path / "origin_run"
    cfg = _make_cfg(mode="continue_training", saving_enabled=True, origin_run_dir=str(origin))
    ctx = SimpleNamespace(paths=FakePaths())

    run_dir, exp_name = run_dir_mod.prepare_run_dir(cfg, ctx)

    assert run_dir == origin
    assert exp_name == origin.name


def test_prepare_run_dir_retrain_copies_origin_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    paths = FakePaths()

    origin = tmp_path / "origin_run"
    origin.mkdir(parents=True, exist_ok=True)
    paths.loss_file_path(run_dir=origin).write_text("loss history", encoding="utf-8")
    paths.log_file_path(run_dir=origin).write_text("train log", encoding="utf-8")
    paths.cfg_train_path(run_dir=origin).write_text("config body", encoding="utf-8")

    new_run_dir = tmp_path / "new_run_dir"
    captured = {}

    def fake_create_run_dir(*, paths, ds_name, exp_name, subfolder, overwrite):
        captured["create_run_dir"] = {
            "ds_name": ds_name,
            "exp_name": exp_name,
            "subfolder": subfolder,
            "overwrite": overwrite,
        }
        return new_run_dir, "exp_name_01"

    monkeypatch.setattr(run_dir_mod, "create_run_dir", fake_create_run_dir)

    cfg = _make_cfg(
        mode="retrain",
        saving_enabled=True,
        origin_run_dir=str(origin),
        overwrite=True,
    )
    ctx = SimpleNamespace(paths=paths)

    out_run_dir, out_exp_name = run_dir_mod.prepare_run_dir(cfg, ctx)

    assert out_run_dir == new_run_dir
    assert out_exp_name == "exp_name_01"

    assert paths.checkpoints_dir(run_dir=new_run_dir).is_dir()
    assert paths.validation_images_dir(run_dir=new_run_dir).is_dir()
    assert paths.retrain_origin_dir(run_dir=new_run_dir).is_dir()

    assert paths.retrain_origin_loss_path(run_dir=new_run_dir).read_text(encoding="utf-8") == "loss history"
    assert paths.retrain_origin_log_path(run_dir=new_run_dir).read_text(encoding="utf-8") == "train log"
    assert paths.retrain_origin_cfg_path(run_dir=new_run_dir).read_text(encoding="utf-8") == "config body"

    assert captured["create_run_dir"] == {
        "ds_name": "dataset_a",
        "exp_name": "exp_name",
        "subfolder": "subfolder_a",
        "overwrite": True,
    }


def test_save_training_config_writes_clean_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    paths = FakePaths()
    run_dir = tmp_path / "new_run_dir"
    run_dir.mkdir(parents=True, exist_ok=True)
    captured = {}

    def fake_prune_config_for_saving(cfg):
        return {"saved": True}

    def fake_save_yaml(cfg, path):
        captured["save_yaml"] = {"cfg": cfg, "path": Path(path)}

    monkeypatch.setattr(run_dir_mod, "prune_config_for_saving", fake_prune_config_for_saving)
    monkeypatch.setattr(run_dir_mod, "save_yaml", fake_save_yaml)

    cfg = _make_cfg(mode="retrain", saving_enabled=True, origin_run_dir=str(tmp_path / 'origin'))
    runtime = SimpleNamespace(paths=paths, run_dir=run_dir)

    out_path = run_dir_mod.save_training_config(
        cfg,
        runtime,
        data_norm_prm={"mean": 1.0},
        model_norm_prm={"std": 2.0},
    )

    assert out_path == paths.cfg_train_path(run_dir=run_dir)
    assert captured["save_yaml"]["cfg"] == {
        "saved": True,
        "normalization": {"norm_prm": {"mean": 1.0}},
        "model_norm_prm": {"std": 2.0},
    }
    assert captured["save_yaml"]["path"] == paths.cfg_train_path(run_dir=run_dir)
