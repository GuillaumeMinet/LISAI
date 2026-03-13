import numpy as np
import torch

from lisai.data.data_prep.pipeline import (
    apply_normalization,
    calculate_dataset_normalization,
    make_tensor,
)


def test_apply_normalization_for_paired_dataset():
    inp = np.array([[[[1.0, 3.0], [5.0, 7.0]]]], dtype=np.float32)
    gt = np.array([[[[10.0, 20.0], [30.0, 40.0]]]], dtype=np.float32)
    norm = {
        "data_mean": 1.0,
        "data_std": 2.0,
        "data_mean_gt": 10.0,
        "data_std_gt": 5.0,
    }

    out = apply_normalization([(inp.copy(), gt.copy())], norm)
    out_inp, out_gt = out[0]

    assert np.allclose(out_inp, (inp - 1.0) / 2.0)
    assert np.allclose(out_gt, (gt - 10.0) / 5.0)


def test_make_tensor_fills_nan_gt_for_unpaired_dataset():
    inp = np.ones((2, 1, 4, 4), dtype=np.float32)

    out = make_tensor([(inp, None)])
    out_inp, out_gt = out[0]

    assert isinstance(out_inp, torch.Tensor)
    assert isinstance(out_gt, torch.Tensor)
    assert out_gt.shape == (2, 1, 1, 1)
    assert torch.isnan(out_gt).all()


def test_calculate_dataset_normalization_returns_expected_keys():
    inp_train = np.array([[[[0.0, 2.0], [4.0, 6.0]]]], dtype=np.float32)
    gt_train = np.array([[[[1.0, 3.0], [5.0, 7.0]]]], dtype=np.float32)
    inp_val = np.array([[[[2.0, 4.0], [6.0, 8.0]]]], dtype=np.float32)
    gt_val = np.array([[[[2.0, 4.0], [6.0, 8.0]]]], dtype=np.float32)

    norm = calculate_dataset_normalization((inp_train, gt_train), (inp_val, gt_val))

    assert set(norm.keys()) == {"data_mean", "data_std", "data_mean_gt", "data_std_gt"}
    assert norm["data_std"] > 0
    assert norm["data_std_gt"] > 0
