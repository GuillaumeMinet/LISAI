from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import lisai.training.run_training as run_training_mod
from lisai.training.setup.data import PreparedTrainingData


class DummyWriter:
    def __init__(self):
        self.close_calls = 0

    def close(self):
        self.close_calls += 1


class DummyLogger:
    def __init__(self):
        self.errors = []

    def error(self, msg, exc_info=False):
        self.errors.append((msg, exc_info))


class DummyTrainer:
    def __init__(self, raise_on_train: Exception | None = None):
        self.raise_on_train = raise_on_train
        self.train_calls = 0

    def train(self):
        self.train_calls += 1
        if self.raise_on_train is not None:
            raise self.raise_on_train



def test_run_training_happy_path_builds_and_trains(monkeypatch: pytest.MonkeyPatch):
    cfg = SimpleNamespace(model=SimpleNamespace(architecture="unet"))
    writer = DummyWriter()
    logger = DummyLogger()
    runtime = SimpleNamespace(
        device="cpu",
        run_dir=Path("run_a"),
        writer=writer,
        callbacks=["cb"],
        console_filter="console_filter",
        file_filter="file_filter",
        logger=logger,
        paths="paths",
    )
    prepared_data = PreparedTrainingData(
        train_loader="train_loader",
        val_loader="val_loader",
        data_norm_prm={"data_mean": 0.0},
        model_norm_prm={"data_mean": 0.0},
        patch_info=None,
    )
    trainer = DummyTrainer()
    captured = {}

    def fake_build_model(cfg_arg, device, lisai_paths, model_norm_prm):
        captured["build_model_args"] = (cfg_arg, device, lisai_paths, model_norm_prm)
        return "model_obj", {"epoch": 0}

    fake_setup = SimpleNamespace(
        prepare_data=lambda c, x: prepared_data,
        save_training_config=lambda *args, **kwargs: None,
        build_model=fake_build_model,
    )

    def fake_get_trainer(**kwargs):
        captured["trainer_kwargs"] = kwargs
        return trainer

    monkeypatch.setattr(run_training_mod, "resolve_config", lambda path: cfg)
    monkeypatch.setattr(run_training_mod, "initialize_runtime", lambda c: runtime)
    monkeypatch.setattr(run_training_mod, "setup", fake_setup)
    monkeypatch.setattr(run_training_mod, "get_trainer", fake_get_trainer)

    out = run_training_mod.run_training("configs/experiments/hdn_training.yml")

    assert out is trainer
    assert trainer.train_calls == 1
    assert writer.close_calls == 1
    assert captured["build_model_args"] == (cfg, "cpu", "paths", {"data_mean": 0.0})
    assert captured["trainer_kwargs"]["architecture"] == "unet"
    assert captured["trainer_kwargs"]["model"] == "model_obj"
    assert captured["trainer_kwargs"]["train_loader"] == "train_loader"
    assert captured["trainer_kwargs"]["val_loader"] == "val_loader"
    assert captured["trainer_kwargs"]["state_dict"] == {"epoch": 0}
    assert captured["trainer_kwargs"]["patch_info"] is None



def test_run_training_logs_and_reraises_on_training_crash(monkeypatch: pytest.MonkeyPatch):
    cfg = SimpleNamespace(model=SimpleNamespace(architecture="unet"))
    writer = DummyWriter()
    logger = DummyLogger()
    runtime = SimpleNamespace(
        device="cpu",
        run_dir=Path("run_b"),
        writer=writer,
        callbacks=[],
        console_filter=None,
        file_filter=None,
        logger=logger,
        paths="paths",
    )
    prepared_data = PreparedTrainingData(
        train_loader="train_loader",
        val_loader="val_loader",
        data_norm_prm={"data_mean": 0.0},
        model_norm_prm={"data_mean": 0.0},
        patch_info=None,
    )
    trainer = DummyTrainer(raise_on_train=RuntimeError("boom"))

    fake_setup = SimpleNamespace(
        prepare_data=lambda c, x: prepared_data,
        save_training_config=lambda *args, **kwargs: None,
        build_model=lambda cfg_arg, device, lisai_paths, model_norm_prm: ("model_obj", None),
    )

    monkeypatch.setattr(run_training_mod, "resolve_config", lambda path: cfg)
    monkeypatch.setattr(run_training_mod, "initialize_runtime", lambda c: runtime)
    monkeypatch.setattr(run_training_mod, "setup", fake_setup)
    monkeypatch.setattr(run_training_mod, "get_trainer", lambda **kwargs: trainer)

    with pytest.raises(RuntimeError, match="boom"):
        run_training_mod.run_training("configs/experiments/hdn_training.yml")

    assert trainer.train_calls == 1
    assert writer.close_calls == 1
    assert logger.errors == [("Training crashed", True)]
