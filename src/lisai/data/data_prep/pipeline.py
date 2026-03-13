import numpy as np
import torch

from lisai.infra.config.schema.experiment import DataSection

from .dataset_io import load_full_datasets
from .transforms import apply_additional_transforms, apply_inp_transformations


def prep_data(config: DataSection, *, for_training: bool, model_norm_prm=None):
    """
    Key-worded function that prepares data for training or evaluation,depending
    on arg:`for_trainining`(bool).
    Note that `data_format` is searched in order: in `config.dataset_info`,
    then in `config.data_format`, and finally assumed "single" if not found.
    For training, if dataset not already split in training/validation, an automatic
    split is done with proportion 0.85-0.15.

    Returns:
    --------
    list_datasets: list
        list of datasets: [(inp_train,gt_train),(inp_val,gt_val)]
        if training, or [(inp_test,gt_test)] if evaluation.

    model_norm: dict
        normalization parameters of the training dataset.
        None if evaluation.

    patch_info: dict
        info about #patches for train and validation splits
        None if evaluation.

    """

    assert isinstance(for_training, bool)

    data_format = config.resolved_data_format
    if config.data_dir is None:
        raise ValueError("`data_dir` must be provided (resolved upstream via lisai.infra.paths.Paths).")

    paired = config.paired
    if paired:
        assert config.target is not None, "paired dataset necessitates a ground-truth"

    # load dataset(s)
    list_datasets, patch_info = load_full_datasets(
        config=config,
        data_format=data_format,
        for_training=for_training,
    )

    if config.masking is not None or config.downsampling is not None:
        # NOTE: by default, we consider that upsampling-like ==> supervised training.
        # UNLESS we find in the masking/downsampling parameters a key "supervised_training"
        # set to False.
        # NOTE: If supervised_training and dataset not originally paired, we make it paired:
        # "new inp" = transformed(inp) and gt = "original inp" => "paired" will be updated.
        list_datasets, paired = apply_inp_transformations(
            list_datasets,
            config=config,
            for_training=for_training,
        )

    # additional transforms
    apply_additional_transforms(list_datasets, config.inp_transform, config.gt_transform)

    # get final normalization parameters NOTE: for training,
    # we normalize only if not already normalized in data_loading
    # for inference, will be normalized by model_norm_prm if found.
    if for_training:
        # norm_prm = config.norm_prm
        resolved_model_norm_prm = calculate_dataset_normalization(*list_datasets)

        # if norm_prm is None or not norm_prm.get("normalize_data",False):
        #     model_norm_prm = calculate_dataset_normalization(*list_datasets)
        # else:
        #     model_norm_prm = {"data_mean": 0,"data_std": 1,
        #                       "data_mean_gt": 0 if paired else None,
        #                       "data_std_gt": 1 if paired else None}
    else:
        resolved_model_norm_prm = model_norm_prm if model_norm_prm is not None else config.model_norm_prm

    list_datasets = apply_normalization(list_datasets, resolved_model_norm_prm)

    list_datasets = make_tensor(list_datasets)

    return list_datasets, resolved_model_norm_prm, patch_info


def make_tensor(list_datasets):
    """
    Transforms all dataset found in list_datasets
    to torch.Tensor. If not paired, gt is filled
    with NaN.
    """

    for i, dataset in enumerate(list_datasets):
        inp, gt = dataset
        inp = torch.from_numpy(inp).to(torch.float32)
        if gt is not None:
            gt = torch.from_numpy(gt).to(torch.float32)
        else:
            gt = torch.zeros(inp.shape[0], 1, 1, 1).fill_(float("nan"))
        list_datasets[i] = (inp, gt)

    return list_datasets


def apply_normalization(list_datasets, model_norm_prm=None):
    """
    Apply normalization parameters.
    """

    if model_norm_prm is None:
        return list_datasets

    for i, (inp, gt) in enumerate(list_datasets):
        data_mean = model_norm_prm.get("data_mean")
        data_std = model_norm_prm.get("data_std")
        inp = (inp - data_mean) / data_std

        if gt is not None:
            data_mean_gt = model_norm_prm.get("data_mean_gt")
            data_std_gt = model_norm_prm.get("data_std_gt")
            gt = (gt - data_mean_gt) / data_std_gt

        list_datasets[i] = (inp, gt)

    return list_datasets


def calculate_dataset_normalization(training_dataset, validation_dataset):
    """
    Calculate the dataset normalization parameters, returned as a dict.
    """
    inp_train, gt_train = training_dataset
    inp_val, gt_val = validation_dataset

    paired = True if gt_train is not None else False

    # calculate coefficients for weighed average
    c1 = (inp_train.shape[0]) / (inp_train.shape[0] + inp_val.shape[0])
    c2 = (inp_val.shape[0]) / (inp_train.shape[0] + inp_val.shape[0])

    data_mean = float(np.mean(inp_train) * c1 + np.mean(inp_val) * c2)
    data_std = float(np.std(inp_train) * c1 + np.std(inp_val) * c2)

    if gt_train is not None:
        data_mean_gt = float(np.mean(gt_train) * c1 + np.mean(gt_val) * c2)
        data_std_gt = float(np.std(gt_train) * c1 + np.std(gt_val) * c2)

    model_norm_prm = {
        "data_mean": data_mean,
        "data_std": data_std,
        "data_mean_gt": data_mean_gt if paired else None,
        "data_std_gt": data_std_gt if paired else None,
    }

    return model_norm_prm
