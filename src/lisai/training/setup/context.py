from lisai.training.runtime import TrainingRuntime

# Backward-compatible alias while training setup migrates to TrainingRuntime.
TrainingContext = TrainingRuntime

__all__ = ["TrainingContext"]
