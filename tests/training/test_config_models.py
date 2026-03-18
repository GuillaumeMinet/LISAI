import pytest
from pydantic import ValidationError

from lisai.config.models.training import ExperimentConfig


def test_char_edge_loss_requires_parameter_block():
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate({"loss_function": {"name": "CharEdge_loss"}})


def test_char_edge_loss_accepts_required_parameters():
    cfg = ExperimentConfig.model_validate(
        {"loss_function": {"name": "CharEdge_loss", "CharEdge_loss_prm": {"alpha": 0.05}}}
    )

    assert cfg.loss_function is not None
    assert cfg.loss_function.CharEdge_loss_prm is not None
    assert cfg.loss_function.CharEdge_loss_prm.alpha == pytest.approx(0.05)


def test_mse_upsampling_requires_parameter_block():
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate({"loss_function": {"name": "MSE_upsampling"}})


def test_mse_upsampling_requires_supported_factor():
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(
            {
                "loss_function": {
                    "name": "MSE_upsampling",
                    "MSE_upsampling_prm": {"upsampling_factor": 4},
                }
            }
        )


def test_standard_losses_reject_specialized_loss_params():
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(
            {"loss_function": {"name": "MSE", "CharEdge_loss_prm": {"alpha": 0.05}}}
        )


def test_normalization_section_is_typed():
    cfg = ExperimentConfig.model_validate(
        {
            "normalization": {
                "load_from_noise_model": True,
                "norm_prm": {
                    "clip": 0,
                    "normalize_data": True,
                    "avgObs": 1.5,
                    "stdObs": 0.2,
                },
            }
        }
    )

    assert cfg.normalization.load_from_noise_model is True
    assert cfg.normalization.norm_prm is not None
    assert cfg.normalization.norm_prm.avgObs == pytest.approx(1.5)
    assert cfg.normalization.norm_prm_dict() == {
        "clip": 0.0,
        "normalize_data": True,
        "avgObs": 1.5,
        "stdObs": 0.2,
    }
