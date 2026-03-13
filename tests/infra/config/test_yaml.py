from pathlib import Path

import pytest

from lisai.infra.config.yaml import load_yaml, save_yaml


def test_yaml_round_trip(tmp_path: Path):
    cfg = {
        "experiment": {"mode": "train", "exp_name": "smoke"},
        "training": {"n_epochs": 2},
    }
    p = tmp_path / "cfg.yml"

    save_yaml(cfg, p)
    loaded = load_yaml(p)

    assert loaded == cfg


def test_load_yaml_raises_for_missing_file(tmp_path: Path):
    missing = tmp_path / "missing.yml"
    with pytest.raises(FileNotFoundError):
        load_yaml(missing)


def test_load_yaml_raises_when_root_is_not_mapping(tmp_path: Path):
    p = tmp_path / "bad.yml"
    p.write_text("- item1\n- item2\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_yaml(p)
