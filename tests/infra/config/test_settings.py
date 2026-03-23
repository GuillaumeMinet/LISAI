from __future__ import annotations

from pathlib import Path
from string import Formatter

import pytest

from lisai.config import load_yaml
from lisai.config.settings import Settings


DATA_YAML = Path("configs/data_config.yml")
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


def test_project_run_tracking_timeout_is_configured(settings_obj: Settings):
    assert settings_obj.project.run_tracking.active_heartbeat_timeout_minutes == 10


def test_project_queue_resource_classes_are_configured(settings_obj: Settings):
    defaults = settings_obj.project.queue.resource_class_vram_mb
    assert defaults.light == 2000
    assert defaults.medium == 4000
    assert defaults.heavy == 6000
    assert settings_obj.project.queue.fixed_margin_pct == 0.20
    assert settings_obj.project.queue.paused is False
    assert settings_obj.project.queue.max_concurrent_runs_per_gpu == 1
