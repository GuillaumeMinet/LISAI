from __future__ import annotations

from collections import deque
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch

import lisai.training.trainers.base as base_mod


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
