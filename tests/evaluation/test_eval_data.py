from __future__ import annotations

from pathlib import Path

import numpy as np
from tifffile import imwrite

import lisai.evaluation.data as data_mod
from lisai.evaluation.saved_run import SavedTrainingRun
from lisai.models.params import UNetParams


class FakePaths:
    def __init__(self, root: Path):
        self.root = root

    def dataset_dir(self, *, dataset_name, data_subfolder):
        return self.root / dataset_name / data_subfolder

    def dataset_registry_path(self):
        return Path('/registry.yaml')



def _make_saved_run() -> SavedTrainingRun:
    return SavedTrainingRun(
        run_dir=Path('/runs/dataset_a/exp_a'),
        experiment_name='exp_a',
        dataset_name='dataset_a',
        data_subfolder='raw',
        data_cfg={
            'dataset_name': 'dataset_a',
            'canonical_load': True,
            'paired': False,
            'input': 'inp',
            'patch_size': 64,
        },
        model_architecture='unet',
        model_parameters=UNetParams(),
        data_norm_prm={'clip': 0},
        model_norm_prm={'data_mean': 1.0, 'data_std': 2.0},
        noise_model_name=None,
        checkpoint_methods=('state_dict',),
        patch_size=64,
        downsamp_factor=1,
        upsampling_factor=1,
        context_length=None,
        default_tiling_size=1024,
    )


def test_build_eval_source_resolves_data_and_applies_overrides(monkeypatch, tmp_path: Path):
    saved_run = _make_saved_run()
    data_root = tmp_path / 'data'
    data_dir = data_root / 'dataset_a' / 'override_subfolder'
    inp_dir = data_dir / 'inp' / 'val'
    gt_dir = data_dir / 'gt_folder' / 'val'
    inp_dir.mkdir(parents=True)
    gt_dir.mkdir(parents=True)
    imwrite(inp_dir / 'img_a.tif', np.ones((4, 5), dtype=np.float32) * 3)
    imwrite(gt_dir / 'img_a.tif', np.ones((4, 5), dtype=np.float32))

    monkeypatch.setattr(data_mod, 'Paths', lambda _settings: FakePaths(data_root))
    monkeypatch.setattr(data_mod, 'load_yaml', lambda path: {'dataset_a': {'data_format': 'single'}})

    source = data_mod.build_eval_source(
        saved_run,
        split='val',
        crop_size=32,
        eval_gt='gt_folder',
        data_prm_update={'subfolder': 'override_subfolder'},
    )

    assert isinstance(source, data_mod.EvalSampleSource)
    cfg = source.config
    assert cfg.data_dir == data_dir
    assert cfg.dataset_info == {'data_format': 'single'}
    assert cfg.target == 'gt_folder'
    assert cfg.initial_crop == 32
    assert cfg.split == 'val'
    assert cfg.model_norm_prm['data_mean_gt'] == 0
    assert cfg.model_norm_prm['data_std_gt'] == 1
    assert len(source.records) == 1

    sample = next(iter(source))
    assert sample.name == 'img_a'
    assert sample.x.shape == (1, 4, 5)
    assert sample.y is not None
    assert sample.y.shape == (1, 4, 5)



def test_resolve_eval_data_dir_returns_explicit_data_dir():
    saved_run = _make_saved_run()

    out = data_mod.resolve_eval_data_dir(saved_run, {'data_dir': '/custom/data'})

    assert out == Path('/custom/data')


def test_build_eval_source_streams_mixed_size_inputs(monkeypatch, tmp_path: Path):
    saved_run = _make_saved_run()
    data_dir = tmp_path / 'dataset'
    inp_dir = data_dir / 'inp' / 'test'
    inp_dir.mkdir(parents=True)
    imwrite(inp_dir / 'a_small.tif', np.ones((3, 4), dtype=np.float32))
    imwrite(inp_dir / 'b_large.tif', np.ones((5, 7), dtype=np.float32))

    monkeypatch.setattr(data_mod, 'load_yaml', lambda path: {'dataset_a': {'data_format': 'single'}})

    source = data_mod.build_eval_source(
        saved_run,
        split='test',
        data_prm_update={'data_dir': str(data_dir)},
    )

    shapes = []
    for sample in source:
        shapes.append(tuple(sample.x.shape))
        assert sample.y is None

    assert shapes == [(1, 3, 4), (1, 5, 7)]
