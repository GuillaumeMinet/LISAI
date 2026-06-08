from pathlib import Path

import numpy as np
from tifffile import imwrite

from lisai.config.models.training import DataSection
from lisai.data.data_loaders.loaders import make_training_loaders
from lisai.data.data_loaders.split_manifest import make_unprepared_split_manifest


def _write_image(path: Path, shape: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imwrite(path, np.arange(shape[0] * shape[1], dtype=np.float32).reshape(shape))


def test_unprepared_split_manifest_uses_file_level_default_ratios(tmp_path: Path):
    for idx in range(5):
        _write_image(tmp_path / "dump" / f"img_{idx}.tif", (4 + idx, 5 + idx))

    cfg = DataSection.model_validate(
        {
            "dataset_name": "demo",
            "prep_before": False,
            "input": "dump",
            "filters": ["tif"],
            "patch_size": 2,
            "val_patch_size": 2,
        }
    ).resolved(data_dir=tmp_path)

    manifest = make_unprepared_split_manifest(cfg, data_format="single")

    assert [len(manifest["splits"][name]) for name in ("train", "val", "test")] == [3, 1, 1]
    assert manifest["paired"] is False
    assert manifest["input"] == "dump"


def test_unprepared_training_loader_handles_variable_image_sizes(tmp_path: Path):
    for idx, shape in enumerate([(4, 4), (5, 6), (6, 5), (7, 7), (8, 9)]):
        _write_image(tmp_path / "dump" / f"img_{idx}.tif", shape)

    cfg = DataSection.model_validate(
        {
            "dataset_name": "demo",
            "prep_before": False,
            "input": "dump",
            "filters": ["tif"],
            "patch_size": 2,
            "val_patch_size": 2,
            "batch_size": 2,
        }
    ).resolved(data_dir=tmp_path)

    train_loader, val_loader, _, patch_info, manifest = make_training_loaders(
        cfg,
        return_split_manifest=True,
    )

    train_x, train_y = train_loader.dataset.tensors
    val_x, val_y = val_loader.dataset.tensors

    assert train_x.shape[1:] == (1, 2, 2)
    assert val_x.shape[1:] == (1, 2, 2)
    assert train_y.isnan().all()
    assert val_y.isnan().all()
    assert patch_info["train_patch"][1:] == (1, 2, 2)
    assert [len(manifest["splits"][name]) for name in ("train", "val", "test")] == [3, 1, 1]
