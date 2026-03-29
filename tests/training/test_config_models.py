import pytest
from pydantic import ValidationError

from lisai.config.models import ResolvedExperiment
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



def test_model_section_resolves_architecture_specific_parameter_model():
    cfg = ExperimentConfig.model_validate(
        {
            "data": {
                "timelapse_prm": {"context_length": 3},
            },
            "model": {
                "architecture": "unet_rcan",
                "parameters": {
                    "upsampling_net": "rcan",
                    "upsampling_factor": 2,
                    "UNet_prm": {"in_channels": 3, "out_channels": 3},
                    "RCAN_prm": {"num_features": 32},
                },
            }
        }
    )

    assert cfg.model is not None
    assert cfg.model.architecture == "unet_rcan"
    assert cfg.model.parameters.upsampling_factor == 2
    assert cfg.model.parameters.UNet_prm.in_channels == 3
    assert cfg.model.parameters.RCAN_prm.num_features == 32



def test_lvae_model_requires_consistent_latent_layout():
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(
            {
                "model": {
                    "architecture": "lvae",
                    "parameters": {
                        "num_latents": 3,
                        "z_dims": [32, 32],
                    },
                }
            }
        )



def test_lvae_accepts_unet_style_norm_parameters():
    cfg = ExperimentConfig.model_validate(
        {
            "model": {
                "architecture": "lvae",
                "parameters": {
                    "norm": "group",
                    "gr_norm": 4,
                },
            }
        }
    )

    assert cfg.model is not None
    assert cfg.model.parameters.norm == "group"
    assert cfg.model.parameters.gr_norm == 4


def test_lvae_legacy_batchnorm_maps_to_norm():
    cfg = ExperimentConfig.model_validate(
        {
            "model": {
                "architecture": "lvae",
                "parameters": {
                    "batchnorm": False,
                },
            }
        }
    )

    assert cfg.model is not None
    assert cfg.model.parameters.norm is None
    assert cfg.model.parameters.batchnorm is False


def test_unet_remove_skip_connections_cannot_exceed_depth():
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(
            {
                "model": {
                    "architecture": "unet",
                    "parameters": {
                        "depth": 2,
                        "remove_skip_con": 3,
                    },
                }
            }
        )



def test_lvae_rejects_timelapse_context_longer_than_one():
    with pytest.raises(ValidationError, match="null or 1"):
        ExperimentConfig.model_validate(
            {
                "data": {
                    "timelapse_prm": {"context_length": 3},
                },
                "model": {
                    "architecture": "lvae",
                    "parameters": {"num_latents": 3, "z_dims": 32},
                },
            }
        )



def test_single_network_requires_input_channels_to_match_context_length():
    with pytest.raises(ValidationError, match="model.parameters.in_channels"):
        ExperimentConfig.model_validate(
            {
                "data": {
                    "timelapse_prm": {"context_length": 3},
                },
                "model": {
                    "architecture": "unet",
                    "parameters": {"in_channels": 1},
                },
            }
        )



def test_single_network_requires_out_channels_of_one():
    with pytest.raises(ValidationError, match="out_channels"):
        ExperimentConfig.model_validate(
            {
                "model": {
                    "architecture": "unet",
                    "parameters": {"out_channels": 2},
                },
            }
        )



def test_hybrid_network_requires_matching_multiple_downsampling_channels():
    cfg = ExperimentConfig.model_validate(
        {
            "data": {
                "downsampling": {
                    "downsamp_factor": 2,
                    "downsamp_method": "multiple",
                    "multiple_prm": {"fill_factor": 0.75, "random": False},
                },
            },
            "model": {
                "architecture": "unet_rcan",
                "parameters": {
                    "upsampling_net": "rcan",
                    "upsampling_factor": 2,
                    "UNet_prm": {"in_channels": 3, "out_channels": 3},
                    "RCAN_prm": {"out_channels": 1},
                },
            },
        }
    )

    assert cfg.model is not None
    assert cfg.model.parameters.UNet_prm.in_channels == 3
    assert cfg.model.parameters.UNet_prm.out_channels == 3
    assert cfg.model.parameters.RCAN_prm.out_channels == 1
def test_hybrid_network_requires_single_timelapse_null_context_to_use_one_input_channel():
    with pytest.raises(ValidationError, match="UNet_prm.in_channels"):
        ExperimentConfig.model_validate(
            {
                "data": {
                    "data_format": "timelapse",
                    "timelapse_prm": {"context_length": None},
                    "downsampling": {
                        "downsamp_factor": 2,
                        "downsamp_method": "random",
                    },
                },
                "model": {
                    "architecture": "unet_rcan",
                    "parameters": {
                        "upsampling_net": "rcan",
                        "upsampling_factor": 2,
                        "UNet_prm": {"in_channels": 3, "out_channels": 3},
                        "RCAN_prm": {"out_channels": 1},
                    },
                },
            }
        )


def test_training_scheduler_accepts_dict_with_kwargs():
    cfg = ExperimentConfig.model_validate(
        {
            "training": {
                "scheduler": {
                    "name": "ReduceLROnPlateau",
                    "patience": 7,
                    "factor": 0.5,
                },
            }
        }
    )

    scheduler = cfg.training.scheduler
    assert isinstance(scheduler, dict)
    assert scheduler["name"] == "ReduceLROnPlateau"
    assert scheduler["patience"] == 7
    assert scheduler["factor"] == pytest.approx(0.5)


def test_training_accepts_warmup_auto_stop_and_debug_stop_fields():
    cfg = ExperimentConfig.model_validate(
        {
            "training": {
                "debug_stop": True,
                "warmup": {
                    "enabled": True,
                    "steps": 123,
                    "start_factor": 0.2,
                },
                "auto_stop": {
                    "enabled": True,
                    "metrics": "loss",
                    "patience": 9,
                },
            }
        }
    )

    assert cfg.training.debug_stop is True
    assert cfg.training.warmup.enabled is True
    assert cfg.training.warmup.steps == 123
    assert cfg.training.warmup.start_factor == pytest.approx(0.2)
    assert cfg.training.auto_stop.enabled is True
    assert cfg.training.auto_stop.metrics == "loss"
    assert cfg.training.auto_stop.patience == 9


def test_training_accepts_deprecated_val_loss_patience_field():
    cfg = ExperimentConfig.model_validate(
        {
            "training": {
                "val_loss_patience": 30,
            }
        }
    )

    assert cfg.training.val_loss_patience == 30



def test_multiple_downsampling_is_incompatible_with_timelapse_context_window():
    with pytest.raises(ValidationError, match="incompatible"):
        ExperimentConfig.model_validate(
            {
                "data": {
                    "timelapse_prm": {"context_length": 3},
                    "downsampling": {
                        "downsamp_factor": 2,
                        "downsamp_method": "multiple",
                        "multiple_prm": {"fill_factor": 0.75, "random": False},
                    },
                },
                "model": {
                    "architecture": "unet",
                    "parameters": {"in_channels": 3},
                },
            }
        )



def test_multiple_downsampling_requires_matching_single_network_channels():
    with pytest.raises(ValidationError, match=r"int\(downsamp_factor\*\*2 \* fill_factor\)"):
        ExperimentConfig.model_validate(
            {
                "data": {
                    "downsampling": {
                        "downsamp_factor": 2,
                        "downsamp_method": "multiple",
                        "multiple_prm": {"fill_factor": 0.75, "random": False},
                    },
                },
                "model": {
                    "architecture": "unet",
                    "parameters": {"in_channels": 1},
                },
            }
        )



def test_hybrid_network_requires_rcan_output_channel_of_one():
    with pytest.raises(ValidationError, match="RCAN_prm.out_channels"):
        ExperimentConfig.model_validate(
            {
                "data": {
                    "downsampling": {
                        "downsamp_factor": 2,
                        "downsamp_method": "multiple",
                        "multiple_prm": {"fill_factor": 0.75, "random": False},
                    },
                },
                "model": {
                    "architecture": "unet_rcan",
                    "parameters": {
                        "UNet_prm": {"in_channels": 3, "out_channels": 3},
                        "RCAN_prm": {"out_channels": 2},
                    },
                },
            }
        )



def test_mismatched_downsampling_and_model_upsampling_emits_warning_for_paired_data():
    with pytest.warns(UserWarning, match="does not match the model effective upsampling factor"):
        ResolvedExperiment.model_validate(
            {
                "data": {
                    "dataset_name": "ds",
                    "paired": True,
                    "target": "gt",
                    "downsampling": {
                        "downsamp_factor": 2,
                        "downsamp_method": "blur",
                    },
                },
                "model": {
                    "architecture": "unet",
                    "parameters": {
                        "upsampling_factor": 3,
                        "upsampling_order": "after",
                    },
                },
            }
        )


def test_mismatched_downsampling_and_model_upsampling_is_error_for_unpaired_data():
    with pytest.raises(ValidationError, match=r"must match the model effective upsampling factor"):
        ExperimentConfig.model_validate(
            {
                "data": {
                    "downsampling": {
                        "downsamp_factor": 2,
                        "downsamp_method": "blur",
                    },
                },
                "model": {
                    "architecture": "unet",
                    "parameters": {
                        "upsampling_factor": 3,
                        "upsampling_order": "after",
                    },
                },
            }
        )


def test_timelapse_sampling_seed_must_be_non_negative():
    with pytest.raises(ValidationError, match="sampling_seed"):
        ExperimentConfig.model_validate(
            {
                "data": {
                    "timelapse_prm": {
                        "timelapse_max_frames": 5,
                        "shuffle": True,
                        "sampling_seed": -1,
                    }
                }
            }
        )



def test_lvae_rejects_multiple_downsampling_with_multi_channel_output():
    with pytest.raises(ValidationError, match=r"does not support multi-channel generated inputs"):
        ExperimentConfig.model_validate(
            {
                "data": {
                    "downsampling": {
                        "downsamp_factor": 2,
                        "downsamp_method": "multiple",
                        "multiple_prm": {"fill_factor": 0.75, "random": False},
                    },
                },
                "model": {
                    "architecture": "lvae",
                    "parameters": {"num_latents": 3, "z_dims": 32},
                },
            }
        )



def test_deterministic_multiple_downsampling_requires_supported_channel_count():
    with pytest.raises(ValidationError, match=r"Deterministic `multiple` downsampling is not implemented"):
        ExperimentConfig.model_validate(
            {
                "data": {
                    "downsampling": {
                        "downsamp_factor": 4,
                        "downsamp_method": "multiple",
                        "multiple_prm": {"fill_factor": 0.75, "random": False},
                    },
                },
                "model": {
                    "architecture": "unet",
                    "parameters": {"in_channels": 12},
                },
            }
        )
