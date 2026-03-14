from __future__ import annotations

from pathlib import Path

import pytest

import lisai.cli as root_cli
import lisai.training.cli as training_cli


def test_resolve_config_path_supports_experiment_short_name():
    repo_root = Path(__file__).resolve().parents[2]
    expected = (repo_root / "configs" / "experiments" / "hdn_training.yml").resolve()

    assert training_cli.resolve_config_path("hdn_training.yml") == expected


def test_resolve_config_path_supports_experiment_short_name_without_extension():
    repo_root = Path(__file__).resolve().parents[2]
    expected = (repo_root / "configs" / "experiments" / "upsamp_training.yml").resolve()

    assert training_cli.resolve_config_path("upsamp_training") == expected


def test_resolve_config_path_lists_available_configs_when_missing():
    with pytest.raises(FileNotFoundError, match="Training config not found: missing_training_config") as exc_info:
        training_cli.resolve_config_path("missing_training_config")

    message = str(exc_info.value)
    assert "Available configs:" in message
    assert "hdn_training.yml" in message
    assert "upsamp_training.yml" in message


def test_training_cli_main_accepts_legacy_config_flag(monkeypatch: pytest.MonkeyPatch):
    captured = {}

    def fake_run_training(config_path):
        captured["config_path"] = config_path

    monkeypatch.setattr(training_cli, "run_training", fake_run_training)

    exit_code = training_cli.main(["--config", "hdn_training.yml"])

    assert exit_code == 0
    assert captured["config_path"].name == "hdn_training.yml"


def test_training_cli_main_accepts_extensionless_config_name(monkeypatch: pytest.MonkeyPatch):
    captured = {}

    def fake_run_training(config_path):
        captured["config_path"] = config_path

    monkeypatch.setattr(training_cli, "run_training", fake_run_training)

    exit_code = training_cli.main(["upsamp_training"])

    assert exit_code == 0
    assert captured["config_path"].name == "upsamp_training.yml"


def test_root_cli_train_dispatches_extensionless_config_to_training(monkeypatch: pytest.MonkeyPatch):
    captured = {}

    def fake_run_training(config_path):
        captured["config_path"] = config_path

    monkeypatch.setattr(training_cli, "run_training", fake_run_training)

    exit_code = root_cli.main(["train", "upsamp_training"])

    assert exit_code == 0
    assert captured["config_path"].name == "upsamp_training.yml"
