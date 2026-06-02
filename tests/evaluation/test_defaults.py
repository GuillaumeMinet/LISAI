from __future__ import annotations

from pathlib import Path

import pytest

import lisai.evaluation.defaults as defaults_mod
from lisai.evaluation.defaults import (
    resolve_apply_options,
    resolve_evaluate_options,
    resolve_inference_config_path,
)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


@pytest.fixture
def inference_config_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    config_dir = tmp_path / "configs" / "inference"
    monkeypatch.setattr(defaults_mod, "config_dir", config_dir)
    return config_dir


def test_resolve_inference_config_path_defaults_to_configs_inference_defaults(inference_config_dir: Path):
    defaults_path = inference_config_dir / "defaults.yml"
    _write(defaults_path, "apply:\n  tiling_size: 256\n")

    resolved = resolve_inference_config_path(None)

    assert resolved == defaults_path.resolve()


def test_resolve_apply_options_merges_defaults_then_named_config_then_cli(inference_config_dir: Path):
    defaults_path = inference_config_dir / "defaults.yml"
    fast_path = inference_config_dir / "fast_upsamp.yml"
    _write(
        defaults_path,
        """
apply:
  tiling_size: 256
  denormalize_output: false
  fill_factor: 0.5
  color_code_prm:
    colormap: turbo
    saturation: 0.35
evaluate:
  split: test
""".strip()
        + "\n",
    )
    _write(
        fast_path,
        """
apply:
  tiling_size: 512
  crop_size: 128
  fill_factor: 0.75
""".strip()
        + "\n",
    )

    resolved = resolve_apply_options(config="fast_upsamp", save_inp=True)

    assert resolved["tiling_size"] == 512
    assert resolved["crop_size"] == 128
    assert resolved["fill_factor"] == pytest.approx(0.75)
    assert resolved["denormalize_output"] is False
    assert resolved["save_inp"] is True
    assert resolved["color_code_prm"]["colormap"] == "turbo"


def test_resolve_apply_options_preserves_legacy_downsamp_when_fill_factor_is_not_set(inference_config_dir: Path):
    defaults_path = inference_config_dir / "defaults.yml"
    _write(
        defaults_path,
        """
apply:
  downsamp: 2
  fill_factor: null
""".strip()
        + "\n",
    )

    resolved = resolve_apply_options()

    assert resolved["downsamp"] == 2
    assert resolved["fill_factor"] is None


def test_resolve_evaluate_options_requires_requested_section_in_named_config(inference_config_dir: Path):
    defaults_path = inference_config_dir / "defaults.yml"
    apply_only_path = inference_config_dir / "apply_only.yml"
    _write(defaults_path, "evaluate:\n  split: test\n")
    _write(apply_only_path, "apply:\n  tiling_size: 512\n")

    with pytest.raises(ValueError, match="does not define a 'evaluate' section"):
        resolve_evaluate_options(config="apply_only")


def test_inference_config_rejects_python_only_evaluate_hooks(inference_config_dir: Path):
    defaults_path = inference_config_dir / "defaults.yml"
    invalid_path = inference_config_dir / "invalid.yml"
    _write(defaults_path, "evaluate:\n  split: test\n")
    _write(invalid_path, "evaluate:\n  test_loader: bad\n")

    with pytest.raises(Exception, match="test_loader"):
        resolve_evaluate_options(config="invalid")
