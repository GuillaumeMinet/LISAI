def model_filename(
    *,
    load_method: str,
    best_or_last: str | None = None,
    train_mode: str | None = None,
    epoch_number: int | None = None,
) -> str:
    """
    Model filename policy.
    Args:
        load_method: 'state_dict' or 'full_model'
        best_or_last: 'best' or 'last' (mutually exclusive with train_mode, epoch_number)
        train_mode: 'continue_training' or 'retrain' (mutually exclusive with best_or_last, epoch_number)
        epoch_number: int (mutually exclusive with best_or_last, train_mode)
    """
    if load_method is None:
        raise ValueError("load_method must be provided")

    if train_mode is None and best_or_last is None and epoch_number is None:
        raise ValueError("One of train_mode, best_or_last, epoch_number must be provided")

    if train_mode is None:
        if epoch_number is not None:
            middle = f"epoch_{epoch_number}"
        else:
            if best_or_last not in {"best", "last"}:
                raise ValueError("best_or_last must be 'best' or 'last'")
            middle = best_or_last
    else:
        if train_mode == "continue_training":
            middle = "last"
        elif train_mode == "retrain":
            middle = "best"
        elif train_mode == "train":
            raise ValueError("train_mode cannot be 'train' when loading a model")
        else:
            raise ValueError(f"Unknown train_mode: {train_mode}")

    if load_method == "state_dict":
        return f"model_{middle}_state_dict.pt"
    if load_method == "full_model":
        return f"model_{middle}.pt"

    raise ValueError(f"Unknown load_method: {load_method}")
