from __future__ import annotations

from pathlib import Path

import pytest

from lisai.evaluation.defaults import (
    resolve_apply_options,
    resolve_evaluate_options,
    resolve_inference_config_path,
)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_resolve_inference_config_path_defaults_to_configs_inference_defaults(tmp_path: Path):
    defaults_path = tmp_path / "configs" / "inference" / "defaults.yml"
    _write(defaults_path, "apply:\n  tiling_size: 256\n")

    resolved = resolve_inference_config_path(None, cwd=tmp_path)

    assert resolved == defaults_path.resolve()


def test_resolve_apply_options_merges_defaults_then_named_config_then_cli(tmp_path: Path):
    defaults_path = tmp_path / "configs" / "inference" / "defaults.yml"
    fast_path = tmp_path / "configs" / "inference" / "fast_upsamp.yml"
    _write(
        defaults_path,
        """
apply:
  tiling_size: 256
  denormalize_output: false
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
""".strip()
        + "\n",
    )

    resolved = resolve_apply_options(config="fast_upsamp", cwd=tmp_path, save_inp=True)

    assert resolved["tiling_size"] == 512
    assert resolved["crop_size"] == 128
    assert resolved["denormalize_output"] is False
    assert resolved["save_inp"] is True
    assert resolved["color_code_prm"]["colormap"] == "turbo"


def test_resolve_evaluate_options_requires_requested_section_in_named_config(tmp_path: Path):
    defaults_path = tmp_path / "configs" / "inference" / "defaults.yml"
    apply_only_path = tmp_path / "configs" / "inference" / "apply_only.yml"
    _write(defaults_path, "evaluate:\n  split: test\n")
    _write(apply_only_path, "apply:\n  tiling_size: 512\n")

    with pytest.raises(ValueError, match="does not define a 'evaluate' section"):
        resolve_evaluate_options(config="apply_only", cwd=tmp_path)


def test_inference_config_rejects_python_only_evaluate_hooks(tmp_path: Path):
    defaults_path = tmp_path / "configs" / "inference" / "defaults.yml"
    invalid_path = tmp_path / "configs" / "inference" / "invalid.yml"
    _write(defaults_path, "evaluate:\n  split: test\n")
    _write(invalid_path, "evaluate:\n  test_loader: bad\n")

    with pytest.raises(Exception, match="test_loader"):
        resolve_evaluate_options(config="invalid", cwd=tmp_path)
