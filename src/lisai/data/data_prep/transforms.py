from copy import deepcopy

from lisai.data.utils import simple_transforms
from lisai.config.models.training import DataSection
from lisai.lib.upsamp.inp_generators import generate_downsamp_inp, generate_masked_inp


def apply_inp_transformations(
    list_datasets: list,
    *,
    config: DataSection,
    for_training: bool = True,
):
    """
    Apply masking or downsampling transformation for all
    dataset listed in arg:`list_datasets`.
    """
    if config.masking is not None:
        masking = True
        downsampling = False
        transform_prm = deepcopy(config.masking)
    elif config.downsampling is not None:
        masking = False
        downsampling = True
        transform_prm = deepcopy(config.downsampling.model_dump(exclude_none=True))
    else:
        raise ValueError("No input transformation provided (masking/downsampling).")

    if transform_prm.get("supervised_training", True):
        paired = True
    else:
        paired = False

    for i, dataset in enumerate(list_datasets):
        inp, gt = dataset
        if gt is None and paired:
            gt = inp.copy()
            if gt.shape[1] > 1 and not config.volumetric:
                idx = gt.shape[1] // 2
                gt = gt[:, idx : idx + 1, ...]

        if masking:
            if gt is not None and gt.shape[-2:] != inp.shape[-2:]:
                downsampled_inp = True
                downsamp_factor = gt.shape[-1] // inp.shape[-1]
            else:
                downsampled_inp = False
                downsamp_factor = None
            inp = generate_masked_inp(inp, transform_prm, downsampled_inp, downsamp_factor)

        elif downsampling:
            if not for_training:  # enforce deterministic downsampling for all inferences
                method = transform_prm.get("downsamp_method")
                if method == "random":
                    transform_prm["downsamp_method"] = "real"
                elif method == "multiple":
                    transform_prm.setdefault("multiple_prm", {})["random"] = False

            inp, _ = generate_downsamp_inp(inp, transform_prm)

        list_datasets[i] = (inp, gt)

    return list_datasets, paired


def apply_additional_transforms(list_datasets, inp_transform=None, gt_transform=None):
    for i, (inp, gt) in enumerate(list_datasets):
        if inp_transform is not None:
            inp = simple_transforms(inp, inp_transform)
        if gt is not None and gt_transform is not None:
            gt = simple_transforms(gt, gt_transform)

        list_datasets[i] = (inp, gt)
    return list_datasets
