from lisai.infra.config.schema.experiment import ResolvedExperiment
from lisai.runtime.spec import RunSpec


def test_run_spec_should_load_and_extract_checkpoint_fields():
    cfg = ResolvedExperiment.model_validate(
        {
            "experiment": {
                "mode": "continue_training",
                "exp_name": "exp1",
                "origin_run_dir": "C:/tmp/origin_run",
            },
            "data": {
                "dataset_name": "ds",
                "patch_size": 64,
                "downsampling": {"downsamp_factor": 2},
            },
            "model": {
                "architecture": "lvae",
                "parameters": {"num_latents": 3},
            },
            "normalization": {"load_from_noise_model": True},
            "noise_model": {"name": "noise_A"},
            "load_model": {
                "enabled": True,
                "checkpoint": {
                    "method": "state_dict",
                    "selector": "last",
                    "epoch": 7,
                    "filename": "model_last_state_dict.pt",
                },
            },
        }
    )

    spec = RunSpec(cfg)
    model_spec = spec.model_spec()

    assert spec.should_load is True
    assert str(spec.origin_run_dir).replace("\\", "/").endswith("/origin_run")
    assert model_spec.architecture == "lvae"
    assert model_spec.patch_size == 64
    assert model_spec.downsamp_factor == 2
    assert model_spec.noise_model_name == "noise_A"
    assert model_spec.checkpoint_method == "state_dict"
    assert model_spec.checkpoint_selector == "last"
    assert model_spec.checkpoint_epoch == 7
    assert model_spec.checkpoint_filename == "model_last_state_dict.pt"


def test_run_spec_uses_val_patch_size_when_patch_size_missing():
    cfg = ResolvedExperiment.model_validate(
        {
            "experiment": {"mode": "train", "exp_name": "exp2"},
            "data": {"dataset_name": "ds", "val_patch_size": 128},
            "model": {"architecture": "unet", "parameters": {}},
        }
    )

    model_spec = RunSpec(cfg).model_spec()

    assert model_spec.patch_size == 128
    assert model_spec.downsamp_factor == 1
