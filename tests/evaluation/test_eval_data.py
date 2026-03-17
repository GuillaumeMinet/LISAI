from __future__ import annotations

from pathlib import Path

import lisai.evaluation.data as data_mod
from lisai.evaluation.saved_run import SavedTrainingRun


class FakePaths:
    def dataset_dir(self, *, dataset_name, data_subfolder):
        return Path('/data') / dataset_name / data_subfolder

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
        model_parameters={},
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



def test_build_eval_loader_resolves_data_and_applies_overrides(monkeypatch):
    saved_run = _make_saved_run()
    captured = {}

    monkeypatch.setattr(data_mod, 'Paths', lambda _settings: FakePaths())
    monkeypatch.setattr(data_mod, 'load_yaml', lambda path: {'dataset_a': {'data_format': 'single'}})

    def fake_make_test_loader(*, config):
        captured['config'] = config
        return 'loader_obj'

    monkeypatch.setattr(data_mod, 'make_test_loader', fake_make_test_loader)

    loader = data_mod.build_eval_loader(
        saved_run,
        split='val',
        crop_size=32,
        eval_gt='gt_folder',
        data_prm_update={'subfolder': 'override_subfolder'},
    )

    assert loader == 'loader_obj'
    cfg = captured['config']
    assert cfg.data_dir == Path('/data/dataset_a/override_subfolder')
    assert cfg.dataset_info == {'data_format': 'single'}
    assert cfg.target == 'gt_folder'
    assert cfg.initial_crop == 32
    assert cfg.split == 'val'
    assert cfg.model_norm_prm['data_mean_gt'] == 0
    assert cfg.model_norm_prm['data_std_gt'] == 1



def test_resolve_eval_data_dir_returns_explicit_data_dir():
    saved_run = _make_saved_run()

    out = data_mod.resolve_eval_data_dir(saved_run, {'data_dir': '/custom/data'})

    assert out == Path('/custom/data')
