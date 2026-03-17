from __future__ import annotations

from pathlib import Path

import pytest

import lisai.cli as root_cli
import lisai.training.cli as training_cli


def test_resolve_config_path_supports_training_short_name():
    repo_root = Path(__file__).resolve().parents[2]
    expected = (repo_root / "configs" / "training" / "hdn.yml").resolve()

    assert training_cli.resolve_config_path("hdn.yml") == expected


def test_resolve_config_path_supports_training_short_name_without_extension():
    repo_root = Path(__file__).resolve().parents[2]
    expected = (repo_root / "configs" / "training" / "upsamp.yml").resolve()

    assert training_cli.resolve_config_path("upsamp") == expected


def test_resolve_config_path_lists_available_configs_when_missing():
    with pytest.raises(FileNotFoundError, match="Training config not found: missing_training_config") as exc_info:
        training_cli.resolve_config_path("missing_training_config")

    message = str(exc_info.value)
    assert "Available configs:" in message
    assert "hdn.yml" in message
    assert "upsamp.yml" in message


def test_training_cli_main_accepts_config_flag(monkeypatch: pytest.MonkeyPatch):
    captured = {}

    def fake_run_training(config_path):
        captured["config_path"] = config_path

    monkeypatch.setattr(training_cli, "run_training", fake_run_training)

    exit_code = training_cli.main(["--config", "hdn.yml"])

    assert exit_code == 0
    assert captured["config_path"].name == "hdn.yml"


def test_training_cli_main_accepts_extensionless_config_name(monkeypatch: pytest.MonkeyPatch):
    captured = {}

    def fake_run_training(config_path):
        captured["config_path"] = config_path

    monkeypatch.setattr(training_cli, "run_training", fake_run_training)

    exit_code = training_cli.main(["upsamp"])

    assert exit_code == 0
    assert captured["config_path"].name == "upsamp.yml"


def test_root_cli_train_dispatches_extensionless_config_to_training(monkeypatch: pytest.MonkeyPatch):
    captured = {}

    def fake_run_training(config_path):
        captured["config_path"] = config_path

    monkeypatch.setattr(training_cli, "run_training", fake_run_training)

    exit_code = root_cli.main(["train", "upsamp"])

    assert exit_code == 0
    assert captured["config_path"].name == "upsamp.yml"
