from __future__ import annotations

from pathlib import Path

import pytest

import lisai.cli as root_cli
import lisai.training.cli as training_cli


def test_resolve_config_path_supports_experiment_short_name():
    repo_root = Path(__file__).resolve().parents[2]
    expected = (repo_root / "configs" / "experiments" / "hdn_training.yml").resolve()

    assert training_cli.resolve_config_path("hdn_training.yml") == expected


def test_training_cli_main_accepts_legacy_config_flag(monkeypatch: pytest.MonkeyPatch):
    captured = {}

    def fake_run_training(config_path):
        captured["config_path"] = config_path

    monkeypatch.setattr(training_cli, "run_training", fake_run_training)

    exit_code = training_cli.main(["--config", "hdn_training.yml"])

    assert exit_code == 0
    assert captured["config_path"].name == "hdn_training.yml"


def test_root_cli_train_dispatches_to_training(monkeypatch: pytest.MonkeyPatch):
    captured = {}

    def fake_run_training(config_path):
        captured["config_path"] = config_path

    monkeypatch.setattr(training_cli, "run_training", fake_run_training)

    exit_code = root_cli.main(["train", "hdn_training.yml"])

    assert exit_code == 0
    assert captured["config_path"].name == "hdn_training.yml"
