from __future__ import annotations

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
