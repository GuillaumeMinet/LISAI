from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import lisai.training.run_training as run_training_mod
from lisai.runs.io import read_run_metadata
from lisai.training.trainers.base import BaseTrainer
from lisai.training.setup.data import PreparedTrainingData


class DummyWriter:
    def __init__(self):
        self.close_calls = 0

    def close(self):
        self.close_calls += 1


class DummyLogger:
    def __init__(self):
        self.errors = []
        self.infos = []
        self.warnings = []

    def error(self, msg, exc_info=False):
        self.errors.append((msg, exc_info))

    def info(self, msg):
        self.infos.append(msg)

    def warning(self, msg):
        self.warnings.append(msg)


class DummyTrainer:
    def __init__(self, raise_on_train: Exception | None = None, outcome=None):
        self.raise_on_train = raise_on_train
        self.outcome = outcome
        self.train_calls = 0

    def train(self):
        self.train_calls += 1
        if self.raise_on_train is not None:
            raise self.raise_on_train
        return self.outcome


class CallbackTrainer:
    def __init__(self, callbacks, *, outcome=None, raise_on_train: Exception | None = None):
        self.callbacks = callbacks
        self.outcome = outcome
        self.raise_on_train = raise_on_train
        self.train_calls = 0

    def train(self):
        self.train_calls += 1
        if self.raise_on_train is not None:
            raise self.raise_on_train
        for callback in self.callbacks:
            callback.on_validation_batch_end(self, 3, "x", "y", "prediction")
        for callback in self.callbacks:
            callback.on_validation_images_end(self, 3, [("x", "y", "prediction")])
        for callback in self.callbacks:
            callback.on_epoch_end(self, 3, {"train_loss": 1.2, "val_loss": 0.4})
        return self.outcome


def _make_cfg(*, post_training_inference: bool = False):
    return SimpleNamespace(
        model=SimpleNamespace(architecture="unet"),
        experiment=SimpleNamespace(post_training_inference=post_training_inference, mode="train"),
        data=SimpleNamespace(dataset_name="dataset_a"),
        routing=SimpleNamespace(models_subfolder="Upsamp"),
        training=SimpleNamespace(n_epochs=10),
    )


def _make_runtime(*, writer, logger, run_dir: Path):
    return SimpleNamespace(
        device="cpu",
        run_dir=run_dir,
        writer=writer,
        callbacks=[],
        console_filter="console_filter",
        file_filter="file_filter",
        logger=logger,
        paths="paths",
    )


def _make_prepared_data():
    return PreparedTrainingData(
        train_loader="train_loader",
        val_loader="val_loader",
        data_norm_prm={"data_mean": 0.0},
        model_norm_prm={"data_mean": 0.0},
        patch_info=None,
    )


def test_run_training_happy_path_builds_and_trains(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    cfg = _make_cfg()
    writer = DummyWriter()
    logger = DummyLogger()
    runtime = _make_runtime(writer=writer, logger=logger, run_dir=tmp_path / "run_a")
    prepared_data = _make_prepared_data()
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

    out = run_training_mod.run_training("configs/training/hdn_training.yml")

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


def test_run_training_logs_and_reraises_on_training_crash(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    cfg = _make_cfg()
    writer = DummyWriter()
    logger = DummyLogger()
    runtime = _make_runtime(writer=writer, logger=logger, run_dir=tmp_path / "run_b")
    prepared_data = _make_prepared_data()
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
        run_training_mod.run_training("configs/training/hdn_training.yml")

    assert trainer.train_calls == 1
    assert writer.close_calls == 1
    assert logger.errors == [("Training crashed", True)]


def test_run_training_triggers_post_training_evaluation_on_completion(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    cfg = _make_cfg(post_training_inference=True)
    writer = DummyWriter()
    logger = DummyLogger()
    runtime = _make_runtime(writer=writer, logger=logger, run_dir=tmp_path / "runs" / "dataset_a" / "Upsamp" / "run_c")
    prepared_data = _make_prepared_data()
    trainer = DummyTrainer(outcome=SimpleNamespace(reason="completed", last_completed_epoch=12))
    captured = {}

    fake_setup = SimpleNamespace(
        prepare_data=lambda c, x: prepared_data,
        save_training_config=lambda *args, **kwargs: None,
        build_model=lambda cfg_arg, device, lisai_paths, model_norm_prm: ("model_obj", None),
    )

    monkeypatch.setattr(run_training_mod, "resolve_config", lambda path: cfg)
    monkeypatch.setattr(run_training_mod, "initialize_runtime", lambda c: runtime)
    monkeypatch.setattr(run_training_mod, "setup", fake_setup)
    monkeypatch.setattr(run_training_mod, "get_trainer", lambda **kwargs: trainer)
    monkeypatch.setattr(run_training_mod, "run_evaluate", lambda **kwargs: captured.update(kwargs))

    run_training_mod.run_training("configs/training/hdn_training.yml")

    assert captured == {
        "dataset_name": "dataset_a",
        "model_name": "run_c",
        "model_subfolder": "Upsamp",
        "config": "post_training",
    }


def test_run_training_prompts_before_post_training_evaluation_on_interrupt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    cfg = _make_cfg(post_training_inference=True)
    writer = DummyWriter()
    logger = DummyLogger()
    runtime = _make_runtime(writer=writer, logger=logger, run_dir=tmp_path / "runs" / "dataset_a" / "Upsamp" / "run_d")
    prepared_data = _make_prepared_data()
    trainer = DummyTrainer(outcome=SimpleNamespace(reason="interrupted", last_completed_epoch=4))
    captured = {}

    fake_setup = SimpleNamespace(
        prepare_data=lambda c, x: prepared_data,
        save_training_config=lambda *args, **kwargs: None,
        build_model=lambda cfg_arg, device, lisai_paths, model_norm_prm: ("model_obj", None),
    )

    monkeypatch.setattr(run_training_mod, "resolve_config", lambda path: cfg)
    monkeypatch.setattr(run_training_mod, "initialize_runtime", lambda c: runtime)
    monkeypatch.setattr(run_training_mod, "setup", fake_setup)
    monkeypatch.setattr(run_training_mod, "get_trainer", lambda **kwargs: trainer)
    monkeypatch.setattr(run_training_mod, "_prompt_yes_no", lambda prompt: True)
    monkeypatch.setattr(run_training_mod, "run_evaluate", lambda **kwargs: captured.update(kwargs))

    run_training_mod.run_training("configs/training/hdn_training.yml")

    assert captured["model_name"] == "run_d"
    assert captured["config"] == "post_training"


def test_run_training_skips_post_training_evaluation_when_interrupt_prompt_declined(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    cfg = _make_cfg(post_training_inference=True)
    writer = DummyWriter()
    logger = DummyLogger()
    runtime = _make_runtime(writer=writer, logger=logger, run_dir=tmp_path / "runs" / "dataset_a" / "Upsamp" / "run_e")
    prepared_data = _make_prepared_data()
    trainer = DummyTrainer(outcome=SimpleNamespace(reason="interrupted", last_completed_epoch=4))
    calls = []

    fake_setup = SimpleNamespace(
        prepare_data=lambda c, x: prepared_data,
        save_training_config=lambda *args, **kwargs: None,
        build_model=lambda cfg_arg, device, lisai_paths, model_norm_prm: ("model_obj", None),
    )

    monkeypatch.setattr(run_training_mod, "resolve_config", lambda path: cfg)
    monkeypatch.setattr(run_training_mod, "initialize_runtime", lambda c: runtime)
    monkeypatch.setattr(run_training_mod, "setup", fake_setup)
    monkeypatch.setattr(run_training_mod, "get_trainer", lambda **kwargs: trainer)
    monkeypatch.setattr(run_training_mod, "_prompt_yes_no", lambda prompt: False)
    monkeypatch.setattr(run_training_mod, "run_evaluate", lambda **kwargs: calls.append(kwargs))

    run_training_mod.run_training("configs/training/hdn_training.yml")

    assert calls == []


def test_run_training_writes_and_finalizes_run_metadata_on_completion(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    cfg = _make_cfg()
    writer = DummyWriter()
    logger = DummyLogger()
    run_dir = tmp_path / "datasets" / "dataset_a" / "models" / "Upsamp" / "run_complete"
    runtime = _make_runtime(writer=writer, logger=logger, run_dir=run_dir)
    prepared_data = _make_prepared_data()

    fake_setup = SimpleNamespace(
        prepare_data=lambda c, x: prepared_data,
        save_training_config=lambda *args, **kwargs: None,
        build_model=lambda cfg_arg, device, lisai_paths, model_norm_prm: ("model_obj", None),
    )

    def fake_get_trainer(**kwargs):
        return CallbackTrainer(
            kwargs["callbacks"],
            outcome=SimpleNamespace(reason="completed", last_completed_epoch=3),
        )

    monkeypatch.setattr(run_training_mod, "resolve_config", lambda path: cfg)
    monkeypatch.setattr(run_training_mod, "initialize_runtime", lambda c: runtime)
    monkeypatch.setattr(run_training_mod, "setup", fake_setup)
    monkeypatch.setattr(run_training_mod, "get_trainer", fake_get_trainer)

    run_training_mod.run_training("configs/training/hdn_training.yml")
    metadata = read_run_metadata(run_dir)

    assert metadata.status == "completed"
    assert metadata.closed_cleanly is True
    assert metadata.last_epoch == 3
    assert metadata.max_epoch == 10
    assert metadata.best_val_loss == pytest.approx(0.4)
    assert metadata.training_signature is not None
    assert metadata.training_signature.architecture == "unet"
    assert metadata.training_signature.batch_size == 1
    assert metadata.training_signature.patch_size is None
    assert metadata.ended_at is not None


def test_run_training_finalizes_run_metadata_as_stopped(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    cfg = _make_cfg()
    writer = DummyWriter()
    logger = DummyLogger()
    run_dir = tmp_path / "datasets" / "dataset_a" / "models" / "Upsamp" / "run_stopped"
    runtime = _make_runtime(writer=writer, logger=logger, run_dir=run_dir)
    prepared_data = _make_prepared_data()

    fake_setup = SimpleNamespace(
        prepare_data=lambda c, x: prepared_data,
        save_training_config=lambda *args, **kwargs: None,
        build_model=lambda cfg_arg, device, lisai_paths, model_norm_prm: ("model_obj", None),
    )

    def fake_get_trainer(**kwargs):
        return CallbackTrainer(
            kwargs["callbacks"],
            outcome=SimpleNamespace(reason="interrupted", last_completed_epoch=3),
        )

    monkeypatch.setattr(run_training_mod, "resolve_config", lambda path: cfg)
    monkeypatch.setattr(run_training_mod, "initialize_runtime", lambda c: runtime)
    monkeypatch.setattr(run_training_mod, "setup", fake_setup)
    monkeypatch.setattr(run_training_mod, "get_trainer", fake_get_trainer)

    run_training_mod.run_training("configs/training/hdn_training.yml")
    metadata = read_run_metadata(run_dir)

    assert metadata.status == "stopped"
    assert metadata.closed_cleanly is True
    assert metadata.ended_at is not None


def test_run_training_finalizes_run_metadata_as_failed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    cfg = _make_cfg()
    writer = DummyWriter()
    logger = DummyLogger()
    run_dir = tmp_path / "datasets" / "dataset_a" / "models" / "Upsamp" / "run_failed"
    runtime = _make_runtime(writer=writer, logger=logger, run_dir=run_dir)
    prepared_data = _make_prepared_data()

    fake_setup = SimpleNamespace(
        prepare_data=lambda c, x: prepared_data,
        save_training_config=lambda *args, **kwargs: None,
        build_model=lambda cfg_arg, device, lisai_paths, model_norm_prm: ("model_obj", None),
    )

    def fake_get_trainer(**kwargs):
        return CallbackTrainer(kwargs["callbacks"], raise_on_train=RuntimeError("boom"))

    monkeypatch.setattr(run_training_mod, "resolve_config", lambda path: cfg)
    monkeypatch.setattr(run_training_mod, "initialize_runtime", lambda c: runtime)
    monkeypatch.setattr(run_training_mod, "setup", fake_setup)
    monkeypatch.setattr(run_training_mod, "get_trainer", fake_get_trainer)

    with pytest.raises(RuntimeError, match="boom"):
        run_training_mod.run_training("configs/training/hdn_training.yml")

    metadata = read_run_metadata(run_dir)

    assert metadata.status == "failed"
    assert metadata.closed_cleanly is True
    assert metadata.ended_at is not None


def test_run_training_persists_peak_gpu_memory_stats_when_cuda_available(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    cfg = _make_cfg()
    writer = DummyWriter()
    logger = DummyLogger()
    run_dir = tmp_path / "datasets" / "dataset_a" / "models" / "Upsamp" / "run_cuda"
    runtime = _make_runtime(writer=writer, logger=logger, run_dir=run_dir)
    runtime.device = SimpleNamespace(type="cuda", index=0)
    prepared_data = _make_prepared_data()

    fake_setup = SimpleNamespace(
        prepare_data=lambda c, x: prepared_data,
        save_training_config=lambda *args, **kwargs: None,
        build_model=lambda cfg_arg, device, lisai_paths, model_norm_prm: ("model_obj", None),
    )

    def fake_get_trainer(**kwargs):
        return CallbackTrainer(
            kwargs["callbacks"],
            outcome=SimpleNamespace(reason="completed", last_completed_epoch=3),
        )

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def current_device():
            return 0

        @staticmethod
        def reset_peak_memory_stats(index):
            return None

        @staticmethod
        def max_memory_allocated(index):
            return 10 * 1024 * 1024

    monkeypatch.setattr(run_training_mod, "resolve_config", lambda path: cfg)
    monkeypatch.setattr(run_training_mod, "initialize_runtime", lambda c: runtime)
    monkeypatch.setattr(run_training_mod, "setup", fake_setup)
    monkeypatch.setattr(run_training_mod, "get_trainer", fake_get_trainer)
    monkeypatch.setattr(run_training_mod.torch, "cuda", FakeCuda)

    run_training_mod.run_training("configs/training/hdn_training.yml")
    metadata = read_run_metadata(run_dir)

    assert metadata.runtime_stats is not None
    assert metadata.runtime_stats.peak_gpu_mem_mb == 10


def test_display_epoch_is_one_based():
    assert BaseTrainer._display_epoch(0) == 1
    assert BaseTrainer._display_epoch(99) == 100
