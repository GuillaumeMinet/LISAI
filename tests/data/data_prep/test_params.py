from pathlib import Path

import pytest

from lisai.config.models.training import DataSection


def test_data_section_maps_legacy_aliases_and_keeps_extra():
    cfg = DataSection.model_validate(
        {"dataset_name": "ds", "inp": "inp_single", "gt": "gt_avg", "paired": True, "unknown_flag": "x"},
    )

    assert cfg.input == "inp_single"
    assert cfg.target == "gt_avg"
    assert cfg.paired is True
    assert cfg.model_extra["unknown_flag"] == "x"


def test_resolved_allows_explicit_runtime_overrides():
    base = DataSection.model_validate(
        {"dataset_name": "ds", "input": "inp_single", "paired": False},
    )
    cfg = base.resolved(
        data_dir=Path("."),
        split="test",
        volumetric=True,
    )

    assert cfg.split == "test"
    assert cfg.volumetric is True
    assert cfg.data_dir == Path(".")


def test_resolved_data_format_priority():
    cfg = DataSection.model_validate(
        {"dataset_name": "ds", "input": "inp_single", "paired": False, "data_format": "single"},
    ).resolved(data_dir=Path("."), dataset_info={"data_format": "timelapse"})
    assert cfg.resolved_data_format == "timelapse"


def test_resolved_data_format_defaults_to_single_with_warning():
    cfg = DataSection.model_validate(
        {"dataset_name": "ds", "input": "inp_single", "paired": False},
    )
    with pytest.warns(UserWarning, match="single"):
        assert cfg.resolved_data_format == "single"


def test_validation_requires_target_for_paired():
    with pytest.raises(ValueError, match="target"):
        DataSection.model_validate({"dataset_name": "ds", "input": "inp_single", "paired": True})


def test_resolved_sets_norm_and_dataset_info():
    base = DataSection.model_validate({"dataset_name": "ds", "input": "inp_single"})
    cfg = base.resolved(
        data_dir=Path("."),
        norm_prm={"normalize_data": True},
        dataset_info={"data_format": "single"},
        model_norm_prm={"data_mean": 0.0, "data_std": 1.0},
    )

    assert cfg.norm_prm == {"normalize_data": True}
    assert cfg.dataset_info == {"data_format": "single"}
    assert cfg.model_norm_prm == {"data_mean": 0.0, "data_std": 1.0}


def test_validation_rejects_removed_masking_option():
    with pytest.raises(ValueError, match="masking"):
        DataSection.model_validate(
            {
                "dataset_name": "ds",
                "input": "inp_single",
                "masking": {"mask": "random"},
            },
        )
