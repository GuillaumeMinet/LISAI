import pytest

from lisai.infra.paths.checkpoint_naming import model_filename


def test_model_filename_state_dict_selectors():
    assert model_filename(load_method="state_dict", best_or_last="best") == "model_best_state_dict.pt"
    assert model_filename(load_method="state_dict", best_or_last="last") == "model_last_state_dict.pt"
    assert model_filename(load_method="state_dict", epoch_number=12) == "model_epoch_12_state_dict.pt"


def test_model_filename_full_model_selectors():
    assert model_filename(load_method="full_model", best_or_last="best") == "model_best.pt"
    assert model_filename(load_method="full_model", best_or_last="last") == "model_last.pt"
    assert model_filename(load_method="full_model", epoch_number=5) == "model_epoch_5.pt"


def test_model_filename_train_modes():
    assert (
        model_filename(load_method="state_dict", train_mode="continue_training")
        == "model_last_state_dict.pt"
    )
    assert model_filename(load_method="full_model", train_mode="retrain") == "model_best.pt"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"load_method": "state_dict"},
        {"load_method": "state_dict", "best_or_last": "invalid"},
        {"load_method": "state_dict", "train_mode": "train"},
        {"load_method": "unknown", "best_or_last": "best"},
    ],
)
def test_model_filename_rejects_invalid_inputs(kwargs):
    with pytest.raises(ValueError):
        model_filename(**kwargs)
