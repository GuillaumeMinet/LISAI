import logging

from lisai.models.loader import prepare_model_for_training
from lisai.runtime.spec import RunSpec


def build_model(spec: RunSpec, device, model_norm_prm, noise_model):
    """
    Instantiates the model using the typed RunSpec -> ModelSpec -> loader pipeline.
    """
    logger = logging.getLogger("lisai")

    model, state = prepare_model_for_training(
        spec=spec.model_spec(),
        device=device,
        model_norm_prm=model_norm_prm,
        noise_model = noise_model
    )

    logger.info(f"Model initialized: {type(model).__name__}")
    return model, state