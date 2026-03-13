

def normalize_inp(inp, clip, data_norm, model_norm):
    """
    Normalize input image before prediction.
    """
    if not isinstance(clip, bool):
        inp[inp < clip] = clip

    data_norm = data_norm or {}
    if data_norm.get("normalize_data"):
        inp = (inp - data_norm.get("avgObs")) / data_norm.get("stdObs")
    if model_norm is not None:
        inp = (inp - model_norm.get("data_mean")) / model_norm.get("data_std")
    return inp


def denormalize_pred(pred, data_norm, model_norm):
    """
    Denormalize model output.
    """
    data_norm = data_norm or {}
    if model_norm is not None:
        pred = pred * model_norm.get("data_std") + model_norm.get("data_mean")

    if data_norm.get("normalize_data"):
        pred = pred * data_norm.get("avgObs") + data_norm.get("stdObs")
    return pred

