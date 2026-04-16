from __future__ import annotations

import pytest
from pydantic import ValidationError

from lisai.config.models.inference import (
    InferenceConfig,
    InferenceDefaults,
    InferenceOverrides,
    ResolvedInferenceConfig,
)
from lisai.config.models.inference_defaults import (
    InferenceConfig as LegacyInferenceConfig,
    InferenceDefaults as LegacyInferenceDefaults,
    InferenceOverrides as LegacyInferenceOverrides,
    ResolvedInferenceConfig as LegacyResolvedInferenceConfig,
)


def test_legacy_inference_defaults_module_reexports_new_root_models():
    assert LegacyInferenceConfig is InferenceConfig
    assert LegacyInferenceDefaults is InferenceDefaults
    assert LegacyInferenceOverrides is InferenceOverrides
    assert LegacyResolvedInferenceConfig is ResolvedInferenceConfig


def test_backward_compatibility_aliases_point_to_clearer_names():
    assert InferenceConfig is InferenceOverrides
    assert InferenceDefaults is ResolvedInferenceConfig


def test_inference_models_accept_apply_fill_factor_when_null_or_valid():
    null_fill = InferenceOverrides.model_validate({"apply": {"fill_factor": None}})
    valid_fill = InferenceOverrides.model_validate({"apply": {"fill_factor": 0.5}})

    assert null_fill.apply is not None
    assert null_fill.apply.fill_factor is None
    assert valid_fill.apply is not None
    assert valid_fill.apply.fill_factor == pytest.approx(0.5)


def test_inference_models_reject_invalid_apply_fill_factor():
    with pytest.raises(ValidationError, match="fill_factor"):
        InferenceOverrides.model_validate({"apply": {"fill_factor": 0}})

    with pytest.raises(ValidationError, match="fill_factor"):
        InferenceOverrides.model_validate({"apply": {"fill_factor": 1.2}})
