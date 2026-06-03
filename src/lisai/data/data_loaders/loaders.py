import torch
from torch.utils.data import DataLoader, TensorDataset

from lisai.config.models.training import DataSection

from .pipeline import prep_data


def make_test_loader(config: DataSection):
    """
    Key-worded function that makes the loaders for training and validation.
    """

    list_datasets, _, _ = prep_data(
        config=config,
        for_training=False,
        model_norm_prm=config.model_norm_prm,
    )
    test_set = TensorDataset(*list_datasets[0])
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    return test_loader


def make_training_loaders(config: DataSection):
    """
    Key-worded function that makes the loaders for training and validation.
    """

    prep_before = config.prep_before
    assert prep_before is not None, "Data preparation missing argument `prep_before` (bool)"
    assert isinstance(prep_before, bool), "Expected prep_before to be a boolean"
    if prep_before:
        list_datasets, model_norm_prm, patch_info = prep_data(config=config, for_training=True)
        train_set = TensorDataset(*list_datasets[0])
        val_set = TensorDataset(*list_datasets[1])
    else:
        raise NotImplementedError()  # TODO

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False)

    return train_loader, val_loader, model_norm_prm, patch_info


def custom_collate_fn(batch):
    """
    Custom collate function that handles tuples of tensors and concatenates them using torch.cat.
    Ensures efficient concatenation with shared memory optimization.

    Args:
        batch (list): List of tuples, where each tuple contains pre-batched 4D tensors (input, ground-truth, time).

    Returns:
        tuple: Concatenated tensors for inputs, ground-truths, and times.
    """
    elem = batch[0]
    if isinstance(elem, tuple):
        transposed = zip(*batch)
        return [custom_collate_fn(samples) for samples in transposed]
    else:
        out = None
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            numel = sum(x.numel() for x in batch)
            storage = elem._typed_storage()._new_shared(numel, device=elem.device)
            out = elem.new(storage).resize_(numel // elem[0].numel(), *elem.size()[1:])

        x = torch.cat(batch, dim=0, out=out)
        return x
