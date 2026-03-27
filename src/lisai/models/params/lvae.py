from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .common import ActivationLVAE, NormType


class LVAEParams(BaseModel):
    """Typed constructor parameters for LadderVAE."""

    model_config = ConfigDict(extra="forbid")

    num_latents: int | None = Field(
        default=None,
        description="Optional number of latent levels. Required when z_dims is provided as a single integer.",
    )
    z_dims: int | list[int] = Field(
        default_factory=lambda: [32, 32, 32, 32],
        description="Latent dimensionality per level, either as one integer or an explicit list.",
    )
    color_ch: int = Field(default=1, description="Number of image channels handled by LadderVAE.")
    blocks_per_layer: int = Field(default=5, description="Number of residual blocks per hierarchical layer.")
    nonlin: ActivationLVAE = Field(default="elu", description="Nonlinearity used inside LadderVAE blocks.")
    merge_type: str = Field(default="residual", description="Top-down and bottom-up merge strategy.")
    norm: NormType = Field(default="group", description="Normalization type used inside LadderVAE residual blocks.")
    gr_norm: int = Field(default=8, description="Group count used when norm='group'.")
    batchnorm: bool | None = Field(
        default=None,
        description="Deprecated legacy toggle. If provided without `norm`, true maps to `norm='group'` and false maps to `norm=None`.",
    )
    stochastic_skip: bool = Field(default=True, description="Whether stochastic skip connections are enabled.")
    n_filters: int = Field(default=64, description="Feature-channel width used inside LadderVAE blocks.")
    dropout: float = Field(default=0.2, description="Dropout probability used inside LadderVAE blocks.")
    free_bits: float = Field(default=0.0, description="Free-bits threshold applied to the KL term.")
    learn_top_prior: bool = Field(default=True, description="Whether the top latent prior should be learned.")
    res_block_type: str = Field(default="bacdbacd", description="Residual-block implementation variant.")
    gated: bool = Field(default=True, description="Whether gated residual blocks are enabled.")
    no_initial_downscaling: bool = Field(default=True, description="Whether the initial convolution skips the first spatial downscaling step.")
    analytical_kl: bool = Field(default=True, description="Whether to use the analytical KL computation.")
    mode_pred: bool = Field(default=False, description="Whether LadderVAE runs in prediction mode instead of training mode.")
    use_uncond_mode_at: list[int] = Field(
        default_factory=list,
        description="Latent levels where unconditional mode should be used.",
    )

    @model_validator(mode="before")
    @classmethod
    def _map_legacy_batchnorm(cls, data):
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        if "norm" not in normalized and "batchnorm" in normalized:
            normalized["norm"] = "group" if normalized["batchnorm"] else None
        return normalized

    @field_validator("num_latents", "color_ch", "blocks_per_layer", "n_filters", "gr_norm")
    @classmethod
    def _validate_positive_optional_ints(cls, value: int | None):
        if value is not None and value <= 0:
            raise ValueError("LVAE size parameters must be > 0 when provided.")
        return value

    @field_validator("dropout")
    @classmethod
    def _validate_dropout(cls, value: float):
        if value < 0 or value >= 1:
            raise ValueError("`dropout` must be in the interval [0, 1).")
        return value

    @field_validator("free_bits")
    @classmethod
    def _validate_free_bits(cls, value: float):
        if value < 0:
            raise ValueError("`free_bits` must be >= 0.")
        return value

    @model_validator(mode="after")
    def _validate_latent_layout(self):
        resolved = self.resolved_z_dims()
        if self.batchnorm is not None and self.batchnorm != (self.norm is not None):
            raise ValueError("`batchnorm` conflicts with `norm`; remove `batchnorm` or make it consistent with whether normalization is enabled.")
        if self.num_latents is not None and len(resolved) != self.num_latents:
            raise ValueError("`num_latents` must match the number of resolved z_dims entries.")
        return self

    def resolved_z_dims(self) -> list[int]:
        if isinstance(self.z_dims, int):
            n_latents = self.num_latents if self.num_latents is not None else 1
            return [self.z_dims] * n_latents
        if any(dim <= 0 for dim in self.z_dims):
            raise ValueError("`z_dims` entries must all be > 0.")
        return list(self.z_dims)

    def effective_upsampling_factor(self) -> int:
        return 1
