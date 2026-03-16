"""Compatibility shim for training runtime initialization."""

from lisai.training.runtime import TrainingRuntime, initialize, initialize_runtime

__all__ = ["TrainingRuntime", "initialize_runtime", "initialize"]
