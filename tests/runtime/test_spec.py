from pathlib import Path

from lisai.runtime.spec import InferenceSpec

def test_inference_spec_defaults_checkpoint_loading_fields():
    spec = InferenceSpec(
        run_dir=Path("/tmp/run_dir"),
        architecture="unet",
        parameters={},
        normalization={},
    )

    assert spec.run_dir == Path("/tmp/run_dir")
    assert spec.architecture == "unet"
    assert spec.parameters == {}
    assert spec.normalization == {}
    assert spec.model_norm_prm is None
    assert spec.noise_model_name is None
    assert spec.patch_size is None
    assert spec.downsamp_factor == 1
    assert spec.checkpoint_method == "state_dict"
    assert spec.checkpoint_selector == "best"
    assert spec.checkpoint_epoch is None
