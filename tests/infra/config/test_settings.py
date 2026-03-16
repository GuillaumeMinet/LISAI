from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from string import Formatter

import pytest

CONFIG_DIR = Path(__file__).resolve().parents[3] / "src" / "lisai" / "config"
CONFIG_PACKAGE = "lisai.config"


if CONFIG_PACKAGE not in sys.modules:
    config_pkg = types.ModuleType(CONFIG_PACKAGE)
    config_pkg.__path__ = [str(CONFIG_DIR)]
    sys.modules[CONFIG_PACKAGE] = config_pkg

settings_mod = importlib.import_module("lisai.config.settings")
yaml_mod = importlib.import_module("lisai.config.io.yaml")
Settings = settings_mod.Settings
load_yaml = yaml_mod.load_yaml


DATA_YAML = Path("configs/data/data.yml")
TEMPLATE_VALUES = {
    "sample_id": "sample_name",
    "snr_level": "snr_10",
    "n_timepoints": 5,
    "timepoint": "01",
}


def _build_kwargs(template: str) -> dict[str, object]:
    field_names = {
        field_name
        for _, field_name, _, _ in Formatter().parse(template)
        if field_name
    }
    missing_fields = sorted(field_names - TEMPLATE_VALUES.keys())
    assert not missing_fields, f"Missing sample values for placeholders: {missing_fields}"
    return {field_name: TEMPLATE_VALUES[field_name] for field_name in field_names}


def _format_cases() -> list[tuple[str, str, str]]:
    cfg = load_yaml(DATA_YAML)
    return [
        (fmt, data_type, template)
        for fmt, type_map in cfg["format"].items()
        for data_type, template in type_map.items()
    ]


@pytest.fixture(scope="module")
def settings_obj() -> Settings:
    return Settings()


@pytest.mark.parametrize(("fmt", "data_type", "template"), _format_cases())
def test_get_data_filename_matches_data_yml(
    settings_obj: Settings, fmt: str, data_type: str, template: str
):
    kwargs = _build_kwargs(template)
    expected = Path(template.format(**kwargs))

    actual = settings_obj.get_data_filename(fmt=fmt, data_type=data_type, **kwargs)

    assert actual == expected
