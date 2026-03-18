from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

BasicLossName = Literal["MSE", "mse", "l2", "L2", "MAE", "mae", "l1", "L1"]
CharEdgeLossName = Literal["CharEdge_loss", "CharEdge"]
UpsamplingLossName = Literal["MSE_upsampling"]
LossName = BasicLossName | CharEdgeLossName | UpsamplingLossName


class CharEdgeLossParams(BaseModel):
    """Parameters for the CharEdge loss."""

    model_config = ConfigDict(extra="forbid")

    alpha: float = Field(
        description="Weight applied to the edge-loss term in the CharEdge objective.",
    )

    @field_validator("alpha")
    @classmethod
    def _validate_alpha(cls, value: float):
        if value < 0:
            raise ValueError("`alpha` must be >= 0.")
        return value


class MSEUpsamplingLossParams(BaseModel):
    """Parameters for the custom upsampling-aware MSE loss."""

    model_config = ConfigDict(extra="forbid")

    upsampling_factor: int = Field(
        description="Upsampling factor used by the custom loss. The current implementation only supports a factor of 2.",
    )
    alpha: float = Field(
        default=0.5,
        description="Weight applied to the custom sub-pixel consistency term, in the interval [0, 1).",
    )

    @field_validator("upsampling_factor")
    @classmethod
    def _validate_upsampling_factor(cls, value: int):
        if value != 2:
            raise ValueError("`upsampling_factor` must be 2 for the current `MSE_upsampling` implementation.")
        return value

    @field_validator("alpha")
    @classmethod
    def _validate_alpha(cls, value: float):
        if value < 0 or value >= 1:
            raise ValueError("`alpha` must be in the interval [0, 1).")
        return value


class LossFunctionConfig(BaseModel):
    """Typed training loss configuration passed to the trainer."""

    model_config = ConfigDict(extra="forbid")

    name: LossName = Field(
        description="Loss function name. Supported values are MSE/L2, MAE/L1, CharEdge_loss, CharEdge, and MSE_upsampling.",
    )
    CharEdge_loss_prm: CharEdgeLossParams | None = Field(
        default=None,
        description="Parameter block required when using CharEdge_loss or CharEdge.",
    )
    MSE_upsampling_prm: MSEUpsamplingLossParams | None = Field(
        default=None,
        description="Parameter block required when using MSE_upsampling.",
    )

    @model_validator(mode="after")
    def _validate_name_specific_params(self):
        if self.name in {"CharEdge_loss", "CharEdge"}:
            if self.CharEdge_loss_prm is None:
                raise ValueError("`CharEdge_loss_prm` is required when `loss_function.name` is `CharEdge_loss` or `CharEdge`.")
            if self.MSE_upsampling_prm is not None:
                raise ValueError("`MSE_upsampling_prm` is only allowed when `loss_function.name` is `MSE_upsampling`.")
            return self

        if self.name == "MSE_upsampling":
            if self.MSE_upsampling_prm is None:
                raise ValueError("`MSE_upsampling_prm` is required when `loss_function.name` is `MSE_upsampling`.")
            if self.CharEdge_loss_prm is not None:
                raise ValueError("`CharEdge_loss_prm` is only allowed when `loss_function.name` is `CharEdge_loss` or `CharEdge`.")
            return self

        if self.CharEdge_loss_prm is not None or self.MSE_upsampling_prm is not None:
            raise ValueError("Specialized loss parameter blocks are only allowed for `CharEdge_loss`/`CharEdge` and `MSE_upsampling`.")
        return self

    def as_kwargs(self) -> dict:
        """Return a runtime-compatible kwargs dict for `get_loss_function(**kwargs)`."""
        return self.model_dump(exclude_none=True)
