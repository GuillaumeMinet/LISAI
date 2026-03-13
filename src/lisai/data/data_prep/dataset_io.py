import glob
import logging
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
from tifffile import imread

from lisai.data.utils import (
    augment_data,
    crop_center,
    extract_patches,
    make_pair_4d,
    select_patches,
)
from lisai.infra.config.schema.experiment import DataSection
from lisai.lib.upsamp.artificial_movement import apply_movement

logger = logging.getLogger("lisai.data_prep")


def load_full_datasets(
    config: DataSection,
    *,
    data_format: str,
    for_training: bool,
):
    """
    Loads all datasets: train and validation if arg:`for_training` is True,
    only the test dataset otherwise.
    """
    data_dir = config.data_dir
    input_name = config.input
    target_name = config.target
    paired = config.paired

    if for_training:
        if config.already_split:
            # train split
            inp_path = data_dir / input_name / "train"
            gt_path = data_dir / target_name / "train" if paired else None
            inp_train, gt_train = load_all_data(
                inp_path,
                data_dir_gt=gt_path,
                data_format=data_format,
                config=config,
                make_patches=True,
            )

            # val split
            inp_path = data_dir / input_name / "val"
            gt_path = data_dir / target_name / "val" if paired else None
            inp_val, gt_val = load_all_data(
                inp_path,
                data_dir_gt=gt_path,
                data_format=data_format,
                config=config,
                val_split=True,
                make_patches=True,
            )

        else:
            inp_path = data_dir / input_name
            gt_path = data_dir / target_name if target_name is not None else None
            inp_data, gt_data = load_all_data(
                inp_path,
                data_dir_gt=gt_path,
                data_format=data_format,
                config=config,
                make_patches=True,
            )

            inp_train = inp_data[: int(0.85 * inp_data.shape[0])]
            inp_val = inp_data[int(0.85 * inp_data.shape[0]) :]

            gt_train = gt_data[: int(0.85 * gt_data.shape[0])] if paired else None
            gt_val = gt_data[int(0.85 * gt_data.shape[0]) :] if paired else None

        # train dataset augmentation (optional)
        if config.augmentation:
            inp_train = augment_data(inp_train)
            if paired:
                gt_train = augment_data(gt_train)

        train_dataset = (inp_train, gt_train)
        val_dataset = (inp_val, gt_val)
        list_datasets = [train_dataset, val_dataset]

        patch_info = {
            "train_patch": inp_train.shape,
            "val_patch": inp_val.shape,
        }
        logger.info(f"Training patches: {inp_train.shape}, Validation patches: {inp_val.shape}")

    else:
        split = getattr(config, "split", "test")
        inp_path = data_dir / input_name / split
        gt_path = data_dir / target_name / split if target_name is not None else None
        inp_test, gt_test = load_all_data(
            inp_path,
            data_dir_gt=gt_path,
            data_format=data_format,
            config=config,
            make_patches=False,
        )
        test_dataset = (inp_test, gt_test)
        list_datasets = [test_dataset]
        patch_info = None

    return list_datasets, patch_info


def load_all_data(
    data_dir_inp: Path,
    *,
    config: DataSection,
    data_dir_gt: Optional[Path] = None,
    data_format: str = "single",
    val_split=False,
    make_patches=True,
):
    """
    Key-worded function that:
        - loads data found in `data_dir_inp`
        - loads data found in `data_dir_gt` if paired dataset
        - adds optional artificial movement
        - do all the necessary normalization, following `norm_prm`
        - extracts patches from each image (w/ optional patch selection)
        if arg:`make_patches` is True

    Returns array of patches ready to be trained:
            (inp_img,gt_img) if paired, (inp_img,None) otherwise

    Arguments:
        - data_dir_inp: Path
            path where to find the input data
        - data_dir_gt: Path (default = None)
            path where to find the gt data for paired dataset
        - data_format: str (default='single')
            'single','timelapse','mltpl_snr'
        - `config` contains data-format-specific info and normalization parameters
     """

    if data_format not in ["single", "timelapse", "mltpl_snr"]:
        raise ValueError(f"data_format {data_format} unknown.")

    if data_dir_gt is not None:
        paired = True
    else:
        paired = False

    filters = config.filters
    initial_crop = config.initial_crop
    mltpl_noise = config.mltpl_noise
    select_on_gt = config.select_on_gt
    norm_prm = config.norm_prm or {}
    clip = norm_prm.get("clip", False)
    normSig2Obs = norm_prm.get("normSig2Obs", False)
    normalize_data = norm_prm.get("normalize_data", False)

    if make_patches:
        patch_thresh = config.patch_thresh
        if val_split and config.val_patch_size is not None:
            patch_size = config.val_patch_size
        else:
            patch_size = config.patch_size
        if patch_size is None:
            raise ValueError("`patch_size` must be provided when `make_patches=True`.")

    # define normalization parameters
    if isinstance(clip, bool) and clip is True:
        clip = 0

    if not paired and normSig2Obs:
        normSig2Obs = False
        warnings.warn("`normSig2Obs` is True but unpaired dataset")
    if normalize_data or normSig2Obs:
        avgObs = norm_prm.get("avgObs")
        stdObs = norm_prm.get("stdObs")
        if paired:
            avgSig = norm_prm.get("avgSig")
            stdSig = norm_prm.get("stdSig")
    if normSig2Obs:
        if mltpl_noise:
            avgObs_per_noise = config.avgObs_per_noise
            stdObs_per_noise = config.stdObs_per_noise
        else:
            avgObs_per_noise = [avgObs]
            stdObs_per_noise = [stdObs]

    # get list of file path
    inp_files = []
    for image_filter in filters:
        inp_files += sorted(glob.glob(str(data_dir_inp) + f"/*{image_filter}"))
    if paired:
        gt_files = []
        for image_filter in filters:
            gt_files += sorted(glob.glob(str(data_dir_gt) + f"/*{image_filter}"))
        assert len(inp_files) == len(
            gt_files
        ), f"Found #{len(inp_files)} inp_files and #{len(gt_files)} gt_files"

    # loop over all files
    inp_data = []
    gt_data = [] if paired else None

    if make_patches:
        n_patch = 0
        if patch_thresh is not None:
            n_patch_removed = 0

    for i in range(len(inp_files)):
        inp_file = inp_files[i]
        gt_file = gt_files[i] if paired else None

        inp_img, gt_img = load_image(
            inp_file,
            gt_file=gt_file,
            data_format=data_format,
            config=config,
        )
        if inp_img is None:
            continue
        inp_img, gt_img = make_pair_4d(inp_img, gt_img)

        # artificial movement
        if config.artificial_movement is not None:
            prm = config.artificial_movement
            inp_img, gt_img = apply_movement((inp_img, gt_img), prm, volumetric=config.volumetric)

        # clip neg (must be done before sig2obs normalization)
        if not isinstance(clip, bool):
            inp_img[inp_img < clip] = clip
            if paired:
                gt_img[gt_img < clip] = clip

        # Sig2Obs normalization (for mltpl snr only)
        # if data_format == "mltpl_snr" and paired and normSig2Obs:
        #     for i in range(gt_img.shape[0]):
        #         normalized_frame = (gt_img[i] - avgSig) / stdSig
        #         gt_img[i] = normalized_frame * stdObs_per_noise[i] + avgObs_per_noise[i]

        # when inp already downsampled
        if paired and gt_img.shape[-2:] != inp_img.shape[-2:]:
            assert gt_img.shape[-1] % inp_img.shape[-1] == 0
            assert gt_img.shape[-1] // inp_img.shape[-1] == gt_img.shape[-2] // inp_img.shape[-2]
            downsamp_factor = gt_img.shape[-1] // inp_img.shape[-1]
        else:
            downsamp_factor = 1

        # initial crop
        if initial_crop is not None:
            if isinstance(initial_crop, int):
                crop_size = initial_crop // downsamp_factor
            else:
                crop_size = (initial_crop[0] // downsamp_factor, initial_crop[1] // downsamp_factor)
            inp_img = crop_center(inp_img, crop_size)
            if paired:
                gt_img = crop_center(gt_img, initial_crop)

        # opt. patch extraction and selection -> [patches,SNR,Time,patchsize,patchsize]
        if make_patches:
            _inp = extract_patches(inp_img, patch_size // downsamp_factor)
            _gt = extract_patches(gt_img, patch_size) if paired else None
            n_patch += _inp.shape[0] * _inp.shape[1]

            if patch_thresh is not None:
                for snr in range(_inp.shape[1]):  # selection done on each snr independently
                    _inp_snr = _inp[:, snr, ...]
                    _gt_snr = _gt[:, snr, ...] if paired else None
                    _inp_selected, _gt_selected, _n_removed = select_patches(
                        _inp_snr, _gt_snr, patch_thresh, select_on_gt=select_on_gt
                    )
                    n_patch_removed += _n_removed
                    inp_data.append(_inp_selected)
                    if paired:
                        gt_data.append(_gt_selected)
            else:
                inp_data.append(np.concatenate(_inp, axis=0))
                if paired:
                    gt_data.append(np.concatenate(_gt, axis=0))

        else:
            inp_data.append(inp_img)
            if paired:
                gt_data.append(gt_img)

    # patch selection logger info
    if make_patches and patch_thresh is not None:
        logger.info(f"{n_patch_removed}/{n_patch} removed patches, with threshold={patch_thresh}.")

    # transform list(s) into numpy array of all patches
    inp_data = np.concatenate(inp_data, axis=0)
    if paired:
        gt_data = np.concatenate(gt_data, axis=0)

    # full data normalization (optional)
    if normalize_data:
        inp_data = (inp_data - avgObs) / stdObs
        if paired:
            gt_data = (gt_data - avgSig) / stdSig

    return inp_data, gt_data


def load_image(
    inp_file: Path,
    *,
    config: DataSection,
    gt_file: Optional[Path] = None,
    data_format: str = "single",
):
    """
    Loads pair of images (inp,gt) - gt=None if not paired dataset.
    Images can be 2d, 3d or 4d, depending on the data_format:
        - 2d: single, or mltpl snr with 1 snr
        - 3d: timelapses
        - 4d: mltpl noise levels ([snr,1,h,w])

    Arguments:
        - inp_file: Path
        - gt_file: Path (default = None)
        - data_format: str (default = "single")
        - `config` contains specific info for the selected `data_format`

    Returns:
        - inp_img: np.array
        - gt_img: np.array or None

    """
    paired = True if gt_file is not None else False

    inp_img = imread(inp_file)
    gt_img = imread(gt_file) if paired else None

    if data_format == "single":
        return inp_img, gt_img

    elif data_format == "timelapse":
        assert len(inp_img.shape) == 3
        prm = config.timelapse_prm
        if prm is None:
            return inp_img, gt_img

        if prm.get("timelapse_max_frames", None) is not None:
            assert not paired, "timelapse max frames not implemented for paired dataset"
            nFrames = prm.get("timelapse_max_frames")
            if inp_img.shape[0] > nFrames:
                if prm.get("shuffle", False):
                    idx = np.arange(inp_img.shape[0])
                    np.random.shuffle(idx)
                    inp_img = inp_img[idx]
                inp_img = inp_img[:nFrames]

        if prm.get("context_length", None) is None or prm.get("context_length", None) == "None":
            inp_img = np.expand_dims(inp_img, axis=1)  # [time,1,h,w] => considered as [snr,1,h,w]
            return inp_img, None

        if prm.get("context_length", None) is not None:
            context_length = prm.get("context_length")
            if context_length == 1:
                return inp_img, gt_img

            if inp_img.shape[0] < context_length:
                name_file = Path(inp_file).name
                print(
                    f"Skipping {name_file} because #frames ({inp_img.shape[0]})<context_length ({context_length})"
                )
                return None, None

            side_frames = int((context_length - 1) / 2)
            inp_imgs = []
            gt_imgs = [] if paired else None
            for idx in range(side_frames, inp_img.shape[0] - side_frames):
                start = idx - side_frames
                stop = idx + side_frames + 1
                inp_imgs.append(inp_img[start:stop])
                if paired:
                    if config.volumetric:
                        gt_imgs.append(gt_img[start:stop])  # volumetric target
                    else:
                        gt_imgs.append(gt_img[idx : idx + 1])  # single frame target
            inp_imgs = np.stack(inp_imgs, axis=0)
            if paired:
                gt_imgs = np.stack(gt_imgs, axis=0)
            return inp_imgs, gt_imgs

    elif data_format == "mltpl_snr":
        prm = config.mltpl_snr_prm
        if prm is None or prm.get("snr_idx") is None:
            warnings.warn("`snr_idx` parameter not found.")
            return inp_img, gt_img

        snr = prm.get("snr_idx")
        mltplnoise = False
        if isinstance(snr, int):
            assert len(np.shape(inp_img)) == 3 and np.shape(inp_img)[0] > 1
            assert snr < np.shape(inp_img)[0]
        elif isinstance(snr, list):
            assert len(np.shape(inp_img)) == 3 and np.shape(inp_img)[0] >= len(snr)
            mltplnoise = True
        elif snr == "last":
            snr = np.shape(inp_img)[0] - 1  # NOTE: could be -1, but then not sure it can be used for sampling_strat_position
        elif snr == "random":
            snr = np.random.randint(low=0, high=np.shape(inp_img)[0])
        else:
            raise ValueError("Frame idx should be None, an integer,'last',or 'random'")

        inp_img = inp_img[snr]
        if paired and len(gt_img.shape) == 3:
            gt_img = gt_img[snr]

        if mltplnoise:
            if paired:
                if len(gt_img.shape) == 2:
                    gt_img = np.expand_dims(gt_img, axis=0)
                if gt_img.shape[0] == 1:
                    gt_img = np.repeat(gt_img, repeats=inp_img.shape[0], axis=0)
                else:
                    assert gt_img.shape[0] == inp_img.shape[0], "number of gt and inp frames don't correspond"
            # make 4d [snr,1,h,w]
            inp_img = np.expand_dims(inp_img, axis=1)
            if paired:
                gt_img = np.expand_dims(gt_img, axis=1)

        return inp_img, gt_img
