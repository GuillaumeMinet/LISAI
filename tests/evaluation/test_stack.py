from __future__ import annotations

import numpy as np
import pytest

import lisai.evaluation.inference.stack as stack_mod


def test_predict_4d_stack_keeps_channel_axis_for_non_timelapse(monkeypatch: pytest.MonkeyPatch):
    captured = {}

    def _fake_infer_batch(*args, **_kwargs):
        x = args[1]
        captured["shape"] = tuple(x.shape)
        return {"prediction": np.zeros((1, 1, x.shape[-2], x.shape[-1]), dtype=np.float32)}

    monkeypatch.setattr(stack_mod, "infer_batch", _fake_infer_batch)

    img = np.ones((1, 2, 32, 32), dtype=np.float32)
    pred_stack, samples_stack = stack_mod.predict_4d_stack(
        model=None,
        img=img,
        timelapse=False,
        ch_out=None,
        device="cpu",
        is_lvae=False,
        tiling_size=None,
        lvae_num_samples=None,
        lvae_save_samples=False,
        upsamp=1,
        context_length=None,
        dark_frame_context_length=False,
        verbose=False,
    )

    assert captured["shape"] == (1, 2, 32, 32)
    assert pred_stack.shape == (1, 1, 32, 32)
    assert samples_stack is None
