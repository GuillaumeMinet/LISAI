from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import lisai.training.trainers.standard as standard_mod


class _CudaLikeLoss:
    def detach(self):
        return self

    def item(self):
        return 2.5

    def __array__(self, dtype=None):
        raise TypeError("can't convert cuda:0 device type tensor to numpy")


class _FakeModel:
    def train(self):
        return None

    def to(self, device):
        return self

    def __call__(self, x, samp_pos=None):
        return x


def test_train_epoch_converts_loss_to_scalar_before_numpy_mean():
    fake_self = SimpleNamespace(
        model=_FakeModel(),
        device="cuda:0",
        train_loader=["batch_0"],
        pbar=False,
        update_console=False,
        _split_batch=lambda batch, warn_once=False: ([torch.tensor([1.0])], [torch.tensor([1.0])]),
        pos_encod=False,
        _prepare_batch=lambda x, y, samp_pos: (x, y, samp_pos),
        loss_function=lambda pred, y: _CudaLikeLoss(),
        _backward_virtual_batch=lambda raw_loss, num_virtual_batches: None,
        _optimizer_step=lambda: None,
        optimizer=SimpleNamespace(zero_grad=lambda: None),
        early_stop=False,
    )

    out = standard_mod.StandardTrainer.train_epoch(fake_self, epoch=0)

    assert out["loss"] == pytest.approx(2.5)
