from pathlib import Path

import numpy as np
from tifffile import imwrite

from lisai.config.models.training import DataSection
from lisai.data.data_prep.dataset_io import (
    _timelapse_identifier,
    _timelapse_sampling_seed,
    load_image,
)


def _write_timelapse(path: Path, *, n_frames: int = 12) -> np.ndarray:
    path.parent.mkdir(parents=True, exist_ok=True)
    stack = np.arange(n_frames, dtype=np.float32)[:, None, None] * np.ones((n_frames, 4, 5), dtype=np.float32)
    imwrite(path, stack)
    return stack


def test_timelapse_shuffle_sampling_is_deterministic_per_file_and_seed(tmp_path: Path):
    inp_file = tmp_path / "input" / "train" / "c01_t34.tif"
    stack = _write_timelapse(inp_file, n_frames=12)

    cfg = DataSection.model_validate(
        {
            "dataset_name": "Gag_noisy_timelapses",
            "timelapse_prm": {"timelapse_max_frames": 5, "shuffle": True, "sampling_seed": 7},
        }
    ).resolved(data_dir=tmp_path)

    identifier = _timelapse_identifier(inp_file, config=cfg)
    seed = _timelapse_sampling_seed(base_seed=cfg.timelapse_prm.sampling_seed, identifier=identifier)
    expected_idx = np.random.default_rng(seed).choice(stack.shape[0], size=5, replace=False)

    out_1, _ = load_image(inp_file, config=cfg, data_format="timelapse")
    out_2, _ = load_image(inp_file, config=cfg, data_format="timelapse")

    expected = stack[expected_idx][:, None, :, :]

    assert np.array_equal(out_1, expected)
    assert np.array_equal(out_2, expected)


def test_timelapse_identifier_includes_dataset_name(tmp_path: Path):
    inp_file = tmp_path / "input" / "train" / "c01_t34.tif"
    _write_timelapse(inp_file, n_frames=12)

    cfg_a = DataSection.model_validate(
        {
            "dataset_name": "dataset_a",
            "timelapse_prm": {"timelapse_max_frames": 5, "shuffle": True, "sampling_seed": 3},
        }
    ).resolved(data_dir=tmp_path)
    cfg_b = DataSection.model_validate(
        {
            "dataset_name": "dataset_b",
            "timelapse_prm": {"timelapse_max_frames": 5, "shuffle": True, "sampling_seed": 3},
        }
    ).resolved(data_dir=tmp_path)

    out_a, _ = load_image(inp_file, config=cfg_a, data_format="timelapse")
    out_b, _ = load_image(inp_file, config=cfg_b, data_format="timelapse")

    assert not np.array_equal(out_a, out_b)
