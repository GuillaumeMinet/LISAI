from __future__ import annotations

from collections import deque
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch

import lisai.training.trainers.base as base_mod


def _fake_trainer_for_train_failure(exc: BaseException):
    saved_safe_state = {}

    fake_self = SimpleNamespace(
        _align_safe_resume_epoch_with_metadata=lambda: None,
        state_dict={"epoch": -1},
        _initialize_log_file=lambda: None,
        n_epochs=2,
        logger=SimpleNamespace(info=lambda *_: None, error=lambda *_: None),
        pbar=False,
        saving_prm={},
        train_epoch=lambda epoch: (_ for _ in ()).throw(exc),
        validate=lambda epoch, save_imgs=False: {"loss": 0.0},
        _log_epoch_metrics=lambda *args, **kwargs: None,
        model=SimpleNamespace(state_dict=lambda: {}),
        optimizer=SimpleNamespace(state_dict=lambda: {}),
        _optimizer_step_count=0,
        _auto_stop_best_metric=None,
        _auto_stop_bad_epochs=0,
        is_lvae=False,
        scheduler=None,
        callbacks=[],
        _check_auto_stop=lambda **kwargs: False,
        debug_stop=False,
        early_stop=False,
        _display_epoch=lambda epoch: base_mod.BaseTrainer._display_epoch(epoch),
        _log_keyboard_interrupt=lambda *args, **kwargs: None,
        _save_last_safe_training_state=lambda **kwargs: saved_safe_state.update(kwargs),
        _log_training_finished=lambda *args, **kwargs: None,
    )
    return fake_self, saved_safe_state


def test_train_returns_retryable_outcome_on_hdn_divergence():
    fake_self, saved_safe_state = _fake_trainer_for_train_failure(base_mod.HDNDivergenceError("diverged"))

    outcome = base_mod.BaseTrainer.train(fake_self)

    assert isinstance(outcome, base_mod.TrainingOutcome)
    assert outcome.reason == "failed_retryable_hdn_divergence"
    assert outcome.retry_eligible is True
    assert outcome.failure_reason == "diverged"
    assert saved_safe_state == {"epoch": 0, "cause": "diverged"}


def test_train_returns_nonretryable_outcome_on_generic_exception():
    fake_self, saved_safe_state = _fake_trainer_for_train_failure(RuntimeError("boom"))

    outcome = base_mod.BaseTrainer.train(fake_self)

    assert isinstance(outcome, base_mod.TrainingOutcome)
    assert outcome.reason == "failed_nonretryable"
    assert outcome.retry_eligible is False
    assert outcome.failure_reason == "RuntimeError: boom"
    assert saved_safe_state == {}


def test_train_returns_interrupted_outcome_on_keyboard_interrupt():
    fake_self, saved_safe_state = _fake_trainer_for_train_failure(KeyboardInterrupt())

    outcome = base_mod.BaseTrainer.train(fake_self)

    assert isinstance(outcome, base_mod.TrainingOutcome)
    assert outcome.reason == "interrupted"
    assert outcome.retry_eligible is False
    assert outcome.failure_reason is None
    assert saved_safe_state == {}


def test_is_hdn_safe_resume_active_matches_recovery_checkpoint_even_when_status_running(
    monkeypatch: pytest.MonkeyPatch,
):
    fake_cfg = SimpleNamespace(
        recovery=SimpleNamespace(
            hdn_safe_resume=SimpleNamespace(enabled=True),
        ),
        experiment=SimpleNamespace(origin_run_dir="/tmp/origin_run"),
        load_model=SimpleNamespace(
            checkpoint=SimpleNamespace(filename="safe_on_divergence.pth"),
        ),
    )
    fake_self = SimpleNamespace(mode="continue_training", cfg=fake_cfg)

    monkeypatch.setattr(
        base_mod,
        "read_run_metadata",
        lambda _: SimpleNamespace(
            status="running",
            recovery_checkpoint_filename="safe_on_divergence.pth",
        ),
    )

    assert base_mod.BaseTrainer._is_hdn_safe_resume_active(fake_self) is True


def test_apply_recovery_overrides_updates_optimizer_lr_from_compounded_fail_count_and_max_grad_norm():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    logger = SimpleNamespace(warning=lambda _: None)

    safe_cfg = SimpleNamespace(
        enabled=True,
        lr_scale=0.2,
        min_lr=1.0e-8,
        max_compound_steps=None,
        force_grad_clip_max_norm=1.5,
    )
    fake_self = SimpleNamespace(
        cfg=SimpleNamespace(
            recovery=SimpleNamespace(hdn_safe_resume=safe_cfg),
            experiment=SimpleNamespace(origin_run_dir="/tmp/origin_run"),
        ),
        optimizer=optimizer,
        _base_learning_rate_from_config=1.0e-4,
        training_prm={},
        logger=logger,
    )
    fake_self._is_hdn_safe_resume_active = lambda: True
    fake_self._read_safe_resume_fail_count = lambda _: 2

    base_mod.BaseTrainer._apply_recovery_overrides(fake_self)

    assert optimizer.param_groups[0]["lr"] == pytest.approx(4.0e-6)
    assert fake_self.training_prm["max_grad_norm"] == pytest.approx(1.5)


def test_apply_recovery_overrides_applies_compound_cap_and_min_lr_floor():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    logger = SimpleNamespace(warning=lambda _: None)

    safe_cfg = SimpleNamespace(
        enabled=True,
        lr_scale=0.1,
        min_lr=1.0e-6,
        max_compound_steps=2,
        force_grad_clip_max_norm=None,
    )
    fake_self = SimpleNamespace(
        cfg=SimpleNamespace(
            recovery=SimpleNamespace(hdn_safe_resume=safe_cfg),
            experiment=SimpleNamespace(origin_run_dir="/tmp/origin_run"),
        ),
        optimizer=optimizer,
        _base_learning_rate_from_config=1.0e-5,
        training_prm={},
        logger=logger,
    )
    fake_self._is_hdn_safe_resume_active = lambda: True
    fake_self._read_safe_resume_fail_count = lambda _: 9

    base_mod.BaseTrainer._apply_recovery_overrides(fake_self)

    # Capped compound steps -> 1e-5 * 0.1^2 = 1e-7, then floored to min_lr=1e-6
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0e-6)


def test_drop_optimizer_scheduler_state_on_safe_resume_removes_optimizer_and_scheduler_entries():
    warnings = []
    fake_self = SimpleNamespace(
        state_dict={
            "epoch": 1,
            "optimizer_state_dict": {"foo": "bar"},
            "scheduler_state_dict": {"s": 1},
            "scheduler": {"legacy": True},
        },
        cfg=SimpleNamespace(
            recovery=SimpleNamespace(
                hdn_safe_resume=SimpleNamespace(
                    drop_optimizer_scheduler_state_on_safe_resume=True,
                )
            )
        ),
        logger=SimpleNamespace(warning=lambda msg: warnings.append(msg)),
    )
    fake_self._is_hdn_safe_resume_active = lambda: True

    base_mod.BaseTrainer._drop_optimizer_scheduler_state_on_safe_resume(fake_self)

    assert "optimizer_state_dict" not in fake_self.state_dict
    assert "scheduler_state_dict" not in fake_self.state_dict
    assert "scheduler" not in fake_self.state_dict
    assert warnings


def test_drop_optimizer_scheduler_state_on_safe_resume_respects_disabled_flag():
    fake_self = SimpleNamespace(
        state_dict={
            "epoch": 1,
            "optimizer_state_dict": {"foo": "bar"},
            "scheduler_state_dict": {"s": 1},
        },
        cfg=SimpleNamespace(
            recovery=SimpleNamespace(
                hdn_safe_resume=SimpleNamespace(
                    drop_optimizer_scheduler_state_on_safe_resume=False,
                )
            )
        ),
        logger=SimpleNamespace(warning=lambda _: None),
    )
    fake_self._is_hdn_safe_resume_active = lambda: True

    base_mod.BaseTrainer._drop_optimizer_scheduler_state_on_safe_resume(fake_self)

    assert "optimizer_state_dict" in fake_self.state_dict
    assert "scheduler_state_dict" in fake_self.state_dict


def test_select_safe_training_state_for_persistence_uses_rewinded_confirmed_snapshot():
    fake_self = SimpleNamespace(
        _safe_state_rewind_steps=1,
        _confirmed_safe_training_states=deque(
            [
                {"epoch": 0, "batch_id": 0},
                {"epoch": 1, "batch_id": 10},
                {"epoch": 2, "batch_id": 20},
            ]
        ),
        _last_safe_training_state={"epoch": 3, "batch_id": 30},
    )

    selected_state, source = base_mod.BaseTrainer._select_safe_training_state_for_persistence(fake_self)

    assert source == "confirmed_buffer"
    assert selected_state["epoch"] == 1
    assert selected_state["batch_id"] == 10


def test_select_safe_training_state_for_persistence_falls_back_to_latest_when_no_confirmed():
    fake_self = SimpleNamespace(
        _safe_state_rewind_steps=1,
        _confirmed_safe_training_states=deque(),
        _last_safe_training_state={"epoch": 3, "batch_id": 30},
    )

    selected_state, source = base_mod.BaseTrainer._select_safe_training_state_for_persistence(fake_self)

    assert source == "latest_fallback"
    assert selected_state["epoch"] == 3
    assert selected_state["batch_id"] == 30


def test_save_last_safe_training_state_clamps_epoch_to_last_completed(monkeypatch: pytest.MonkeyPatch):
    captured_state = {}
    recovery_update = {}

    class DummyCheckpointManager:
        def save_emergency_safe_state(self, *, state_dict, model, tag):
            captured_state["state_dict"] = state_dict
            captured_state["model"] = model
            captured_state["tag"] = tag
            return "safe_on_divergence.pth"

    def _capture_recovery_update(run_dir, **kwargs):
        recovery_update["run_dir"] = run_dir
        recovery_update.update(kwargs)
        return None

    monkeypatch.setattr(base_mod, "update_run_recovery_info", _capture_recovery_update)

    fake_self = SimpleNamespace(
        _last_safe_training_state={"epoch": 2, "batch_id": -1},
        _confirmed_safe_training_states=deque(),
        _safe_state_rewind_steps=1,
        state_dict={"epoch": 1},
        saving_prm={"enabled": True},
        run_dir=Path("/tmp/fake_run"),
        ckpt=DummyCheckpointManager(),
        model=object(),
        logger=SimpleNamespace(warning=lambda _: None),
    )
    fake_self._select_safe_training_state_for_persistence = (
        lambda: base_mod.BaseTrainer._select_safe_training_state_for_persistence(fake_self)
    )
    fake_self._read_safe_resume_fail_count = lambda _: 0

    base_mod.BaseTrainer._save_last_safe_training_state(
        fake_self,
        epoch=3,
        cause="diverged",
    )

    state_dict = captured_state["state_dict"]
    assert state_dict["safe_epoch"] == 2
    assert state_dict["epoch"] == 1
    assert state_dict["failure_epoch"] == 3
    assert state_dict["failure_cause"] == "diverged"
    assert state_dict["safe_state_source"] == "latest_fallback"
    assert recovery_update["last_safe_epoch"] == 1
    assert recovery_update["safe_resume_fail_count"] == 1


def test_align_safe_resume_epoch_with_metadata_clamps_ahead_checkpoint_epoch(
    monkeypatch: pytest.MonkeyPatch,
):
    warnings = []

    fake_self = SimpleNamespace(
        state_dict={"epoch": 2},
        cfg=SimpleNamespace(experiment=SimpleNamespace(origin_run_dir="/tmp/origin_run")),
        logger=SimpleNamespace(warning=lambda msg: warnings.append(msg)),
    )
    fake_self._is_hdn_safe_resume_active = lambda: True

    monkeypatch.setattr(
        base_mod,
        "read_run_metadata",
        lambda _: SimpleNamespace(last_epoch=1),
    )

    base_mod.BaseTrainer._align_safe_resume_epoch_with_metadata(fake_self)

    assert fake_self.state_dict["epoch"] == 1
    assert warnings


def test_save_last_safe_training_state_uses_confirmed_buffer_with_rewind(monkeypatch: pytest.MonkeyPatch):
    captured_state = {}
    recovery_update = {}

    class DummyCheckpointManager:
        def save_emergency_safe_state(self, *, state_dict, model, tag):
            captured_state["state_dict"] = state_dict
            return "safe_on_divergence.pth"

    def _capture_recovery_update(run_dir, **kwargs):
        recovery_update.update(kwargs)
        return None

    monkeypatch.setattr(base_mod, "update_run_recovery_info", _capture_recovery_update)

    fake_self = SimpleNamespace(
        _last_safe_training_state={"epoch": 3, "batch_id": 30},
        _confirmed_safe_training_states=deque(
            [
                {"epoch": 0, "batch_id": 0},
                {"epoch": 1, "batch_id": 10},
                {"epoch": 2, "batch_id": 20},
            ]
        ),
        _safe_state_rewind_steps=1,
        state_dict={"epoch": 2},
        saving_prm={"enabled": True},
        run_dir=Path("/tmp/fake_run"),
        ckpt=DummyCheckpointManager(),
        model=object(),
        logger=SimpleNamespace(warning=lambda _: None),
    )
    fake_self._select_safe_training_state_for_persistence = (
        lambda: base_mod.BaseTrainer._select_safe_training_state_for_persistence(fake_self)
    )
    fake_self._read_safe_resume_fail_count = lambda _: 2

    base_mod.BaseTrainer._save_last_safe_training_state(fake_self, epoch=3, cause="diverged")

    state_dict = captured_state["state_dict"]
    assert state_dict["safe_state_source"] == "confirmed_buffer"
    assert state_dict["safe_epoch"] == 1
    assert state_dict["epoch"] == 1
    assert recovery_update["last_safe_epoch"] == 1
    assert recovery_update["last_safe_batch_id"] == 10
    assert recovery_update["safe_resume_fail_count"] == 3


def test_configure_warmup_enables_for_reduce_on_plateau():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=1.0e-4)

    logs = {"info": [], "warning": []}
    fake_self = SimpleNamespace(
        _warmup_enabled_requested=True,
        _warmup_steps=5,
        _warmup_start_factor=0.1,
        _warmup_active=False,
        _warmup_base_lrs=[],
        _warmup_ignore_warned=False,
        _optimizer_step_count=0,
        _scheduler_name="ReduceLROnPlateau",
        optimizer=optimizer,
        _base_learning_rate_from_config=1.0e-4,
        logger=SimpleNamespace(
            info=lambda msg: logs["info"].append(msg),
            warning=lambda msg: logs["warning"].append(msg),
        ),
    )
    fake_self._warmup_factor_for_step = lambda step: base_mod.BaseTrainer._warmup_factor_for_step(fake_self, step)
    fake_self._set_warmup_lrs_for_step = lambda step: base_mod.BaseTrainer._set_warmup_lrs_for_step(fake_self, step)

    base_mod.BaseTrainer._configure_warmup(fake_self)

    assert fake_self._warmup_active is True
    assert fake_self._warmup_base_lrs == [pytest.approx(1.0e-4)]
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0e-5)
    assert logs["warning"] == []


def test_configure_warmup_ignores_non_plateau_scheduler():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=1.0e-4)

    warnings = []
    fake_self = SimpleNamespace(
        _warmup_enabled_requested=True,
        _warmup_steps=5,
        _warmup_start_factor=0.1,
        _warmup_active=False,
        _warmup_base_lrs=[],
        _warmup_ignore_warned=False,
        _optimizer_step_count=0,
        _scheduler_name="StepLR",
        optimizer=optimizer,
        _base_learning_rate_from_config=1.0e-4,
        logger=SimpleNamespace(info=lambda _: None, warning=lambda msg: warnings.append(msg)),
    )

    base_mod.BaseTrainer._configure_warmup(fake_self)

    assert fake_self._warmup_active is False
    assert warnings


def test_apply_manual_warmup_if_needed_scales_lr_with_optimizer_step_counter():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=1.0)

    fake_self = SimpleNamespace(
        _warmup_active=True,
        _warmup_steps=3,
        _warmup_start_factor=0.1,
        _warmup_base_lrs=[1.0],
        _optimizer_step_count=0,
        optimizer=optimizer,
    )
    fake_self._warmup_factor_for_step = lambda step: base_mod.BaseTrainer._warmup_factor_for_step(fake_self, step)
    fake_self._set_warmup_lrs_for_step = lambda step: base_mod.BaseTrainer._set_warmup_lrs_for_step(fake_self, step)
    fake_self._set_warmup_base_lrs = lambda: base_mod.BaseTrainer._set_warmup_base_lrs(fake_self)

    base_mod.BaseTrainer._apply_manual_warmup_if_needed(fake_self)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1)

    fake_self._optimizer_step_count = 1
    base_mod.BaseTrainer._apply_manual_warmup_if_needed(fake_self)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.55)

    fake_self._optimizer_step_count = 2
    base_mod.BaseTrainer._apply_manual_warmup_if_needed(fake_self)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0)


def test_check_auto_stop_uses_minimize_rule_with_patience():
    fake_self = SimpleNamespace(
        _auto_stop_enabled=True,
        _auto_stop_metric="val_loss",
        _auto_stop_patience=2,
        _auto_stop_best_metric=None,
        _auto_stop_bad_epochs=0,
    )

    assert base_mod.BaseTrainer._check_auto_stop(fake_self, train_loss=2.0, val_loss=1.0) is False
    assert fake_self._auto_stop_best_metric == pytest.approx(1.0)
    assert fake_self._auto_stop_bad_epochs == 0

    assert base_mod.BaseTrainer._check_auto_stop(fake_self, train_loss=2.0, val_loss=1.1) is False
    assert fake_self._auto_stop_bad_epochs == 1

    assert base_mod.BaseTrainer._check_auto_stop(fake_self, train_loss=2.0, val_loss=1.2) is True
    assert fake_self._auto_stop_bad_epochs == 2


def test_check_auto_stop_can_monitor_train_loss():
    fake_self = SimpleNamespace(
        _auto_stop_enabled=True,
        _auto_stop_metric="loss",
        _auto_stop_patience=1,
        _auto_stop_best_metric=None,
        _auto_stop_bad_epochs=0,
    )

    assert base_mod.BaseTrainer._check_auto_stop(fake_self, train_loss=0.5, val_loss=0.7) is False
    assert base_mod.BaseTrainer._check_auto_stop(fake_self, train_loss=0.6, val_loss=0.6) is True


def test_capture_safe_training_state_persists_optimizer_step_and_auto_stop_state():
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0e-3)

    fake_self = SimpleNamespace(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        state_dict={"train_loss": 1.2, "val_loss": 0.9},
        _optimizer_step_count=42,
        _auto_stop_best_metric=0.9,
        _auto_stop_bad_epochs=3,
        _last_safe_training_state=None,
        _pending_safe_training_states=deque(),
        _confirmed_safe_training_states=deque(),
        _safe_state_confirmation_lag=1,
        _safe_state_rewind_steps=1,
    )
    fake_self._cpu_state_dict = lambda state: base_mod.BaseTrainer._cpu_state_dict(fake_self, state)

    base_mod.BaseTrainer._capture_safe_training_state(fake_self, epoch=5, batch_id=12)

    snapshot = fake_self._last_safe_training_state
    assert snapshot is not None
    assert snapshot["optimizer_step_count"] == 42
    assert snapshot["auto_stop_best_metric"] == pytest.approx(0.9)
    assert snapshot["auto_stop_bad_epochs"] == 3
