from __future__ import annotations

from lisai.evaluation.run_evaluate import _build_evaluation_folder_name


def test_build_evaluation_folder_name_uses_requested_epoch_when_explicit():
    assert _build_evaluation_folder_name(
        best_or_last='best',
        requested_epoch=12,
        resolved_epoch=7,
        split='test',
    ) == 'evaluation_epoch_12'


def test_build_evaluation_folder_name_includes_selector_and_resolved_epoch():
    assert _build_evaluation_folder_name(
        best_or_last='last',
        requested_epoch=None,
        resolved_epoch=100,
        split='val',
    ) == 'evaluation_last_epoch_100_val'


def test_build_evaluation_folder_name_falls_back_when_epoch_is_unknown():
    assert _build_evaluation_folder_name(
        best_or_last='best',
        requested_epoch=None,
        resolved_epoch=None,
        split='test',
    ) == 'evaluation_best'
