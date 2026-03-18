from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DataNormalizationParams(BaseModel):
    """Normalization coefficients consumed by the training data pipeline."""

    model_config = ConfigDict(extra="allow")

    clip: float | bool = Field(
        default=False,
        description="Whether to clip low values before normalization. Use true to clip negatives to 0, or provide a numeric threshold.",
    )
    normSig2Obs: bool = Field(
        default=False,
        description="Whether to remap signal statistics to observation statistics for multi-noise paired datasets.",
    )
    normalize_data: bool = Field(
        default=False,
        description="Whether to normalize the loaded data using the provided observation and signal statistics.",
    )
    avgObs: float | None = Field(
        default=None,
        description="Mean observation intensity used by the normalization pipeline.",
    )
    stdObs: float | None = Field(
        default=None,
        description="Standard deviation of the observation intensities used by the normalization pipeline.",
    )
    avgSig: float | None = Field(
        default=None,
        description="Mean signal intensity used when paired data normalization needs signal statistics.",
    )
    stdSig: float | None = Field(
        default=None,
        description="Standard deviation of the signal intensities used when paired data normalization needs signal statistics.",
    )

    @field_validator("stdObs", "stdSig")
    @classmethod
    def _validate_positive_std(cls, value: float | None):
        if value is not None and value <= 0:
            raise ValueError("Standard-deviation fields must be > 0 when provided.")
        return value

    def as_dict(self) -> dict:
        """Return a runtime-compatible normalization dict."""
        out = {}
        for field_name, field_info in self.__class__.model_fields.items():
            value = getattr(self, field_name)
            if value is None:
                continue
            if field_name in self.model_fields_set or value != field_info.default:
                out[field_name] = value
        return out


class NormalizationSection(BaseModel):
    """Training normalization settings resolved before data loading."""

    model_config = ConfigDict(extra="allow")

    load_from_noise_model: bool = Field(
        default=False,
        description="Whether Stage A normalization metadata should be loaded automatically from the configured noise model.",
    )
    norm_prm: DataNormalizationParams | None = Field(
        default=None,
        description="Explicit normalization parameters consumed by the data-loading pipeline.",
    )

    def norm_prm_dict(self) -> dict | None:
        """Return the runtime normalization dict used by training and evaluation helpers."""
        if self.norm_prm is None:
            return None
        return self.norm_prm.as_dict()
