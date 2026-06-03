"""Evaluation data reconstruction helpers.

This module owns the evaluation-side data preparation logic that depends on a
`SavedTrainingRun`: resolving the dataset location, applying evaluation-only
overrides, and building a sample source for `run_evaluate`.
"""

from __future__ import annotations

import glob
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
from lisai.config import load_yaml, settings
from lisai.config.io import deep_merge
from lisai.config.models.training import DataSection
from lisai.data.data_loaders.dataset_io import load_image
from lisai.data.data_loaders.transforms import apply_additional_transforms, apply_inp_transformations
from lisai.data.utils import crop_center, make_pair_4d
from lisai.infra.paths import Paths
from lisai.lib.upsamp.artificial_movement import apply_movement

from .saved_run import SavedTrainingRun


@dataclass(frozen=True)
class EvalRecord:
    """File-backed description of one evaluation sample."""

    name: str
    inp_path: Path
    gt_path: Path | None
    split: str
    file_index: int
    sample_index: int = 0


@dataclass(frozen=True)
class EvalSample:
    """Prepared tensors and metadata for one evaluation sample."""

    name: str
    x: torch.Tensor
    y: torch.Tensor | None
    original_shape: tuple[int, int]
    record: EvalRecord


class EvalSampleSource:
    """Ordered iterable of prepared evaluation samples."""

    def __init__(self, *, records: Sequence[EvalRecord], config: DataSection):
        self.config = config
        self.records = tuple(records)

    @classmethod
    def from_config(cls, config: DataSection) -> "EvalSampleSource":
        file_records = cls.build_records(config)
        sample_records = cls.expand_records(file_records, config=config)
        return cls(records=sample_records, config=config)

    @staticmethod
    def build_records(config: DataSection) -> tuple[EvalRecord, ...]:
        """Resolve one file-level evaluation record per input/GT pair."""
        if config.data_dir is None:
            raise ValueError("`data_dir` must be provided for evaluation data loading.")
        if config.input is None:
            raise ValueError("`input` must be provided for evaluation data loading.")

        split = getattr(config, "split", "test")
        inp_dir = config.data_dir / config.input / split
        inp_files = _collect_split_files(inp_dir, config.filters)
        if not inp_files:
            raise FileNotFoundError(f"No input files found in {inp_dir} with filters={config.filters}.")

        gt_files: list[Path] | None = None
        if config.target is not None:
            gt_dir = config.data_dir / config.target / split
            gt_files = _collect_split_files(gt_dir, config.filters)
            if len(inp_files) != len(gt_files):
                raise ValueError(f"Found #{len(inp_files)} inp_files and #{len(gt_files)} gt_files")

        records = []
        for index, inp_path in enumerate(inp_files):
            gt_path = gt_files[index] if gt_files is not None else None
            records.append(
                EvalRecord(
                    name=inp_path.stem,
                    inp_path=inp_path,
                    gt_path=gt_path,
                    split=split,
                    file_index=index,
                )
            )
        return tuple(records)

    @staticmethod
    def expand_records(records: Sequence[EvalRecord], *, config: DataSection) -> tuple[EvalRecord, ...]:
        """Expand file-level records into one record per sample.

        Example: one timelapse file record that loads as shape [3, C, H, W]
        becomes three records with sample_index 0, 1, and 2.
        """
        expanded: list[EvalRecord] = []
        for record in records:
            n_samples = _count_eval_samples(record, config=config)
            if n_samples == 0:
                continue
            for sample_index in range(n_samples):
                name = record.name if n_samples == 1 else f"{record.name}_{sample_index}"
                expanded.append(replace(record, name=name, sample_index=sample_index))
        return tuple(expanded)

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[EvalSample]:
        for record in self.records:
            yield prepare_eval_sample(record, config=self.config)


def resolve_dataset_info(dataset_name: str | None) -> dict[str, Any] | None:
    """Load dataset-registry metadata for a dataset name when available."""
    if not dataset_name:
        return None

    paths = Paths(settings)
    try:
        registry = load_yaml(paths.dataset_registry_path())
    except FileNotFoundError:
        return None

    info = registry.get(dataset_name)
    if isinstance(info, Mapping):
        return dict(info)
    return None



def resolve_eval_data_dir(saved_run: SavedTrainingRun, data_cfg: Mapping[str, Any]) -> Path | None:
    """Resolve the dataset directory used for evaluation from saved config and overrides."""
    for key in ("data_dir", "full_data_path"):
        value = data_cfg.get(key)
        if value:
            return Path(value)

    if data_cfg.get("canonical_load") is False:
        return None

    dataset_name = data_cfg.get("dataset_name") or saved_run.dataset_name
    if not dataset_name:
        return None

    subfolder = data_cfg.get("subfolder")
    if subfolder is None:
        subfolder = saved_run.data_subfolder

    paths = Paths(settings)
    return paths.dataset_dir(dataset_name=dataset_name, data_subfolder=subfolder or "")


def _collect_split_files(data_dir: Path, filters: list[str]) -> list[Path]:
    files: list[Path] = []
    for image_filter in filters:
        files += [Path(path) for path in sorted(glob.glob(str(data_dir) + f"/*{image_filter}"))]
    return files


def _normalization_flags(config: DataSection) -> tuple[Any, bool, bool]:
    norm_prm = config.norm_prm or {}
    clip = norm_prm.get("clip", False)
    if isinstance(clip, bool) and clip is True:
        clip = 0
    normalize_data = norm_prm.get("normalize_data", False)
    norm_sig_to_obs = norm_prm.get("normSig2Obs", False)
    if config.target is None and norm_sig_to_obs:
        norm_sig_to_obs = False
    return clip, normalize_data, norm_sig_to_obs


def _load_eval_block(record: EvalRecord, *, config: DataSection):
    data_format = config.resolved_data_format
    inp_img, gt_img = load_image(
        record.inp_path,
        gt_file=record.gt_path,
        data_format=data_format,
        config=config,
    )
    if inp_img is None:
        return None, None

    inp_img, gt_img = make_pair_4d(inp_img, gt_img)

    if config.artificial_movement is not None:
        movement_prm = config.artificial_movement.model_dump(exclude_none=True)
        inp_img, gt_img = apply_movement((inp_img, gt_img), movement_prm, volumetric=config.volumetric)

    clip, normalize_data, _ = _normalization_flags(config)
    norm_prm = config.norm_prm or {}

    if not isinstance(clip, bool):
        inp_img[inp_img < clip] = clip
        if gt_img is not None:
            gt_img[gt_img < clip] = clip

    if gt_img is not None and gt_img.shape[-2:] != inp_img.shape[-2:]:
        if gt_img.shape[-1] % inp_img.shape[-1] != 0:
            raise ValueError("Ground-truth width must be an integer multiple of input width.")
        if gt_img.shape[-1] // inp_img.shape[-1] != gt_img.shape[-2] // inp_img.shape[-2]:
            raise ValueError("Ground-truth and input spatial scale factors do not match.")
        downsamp_factor = gt_img.shape[-1] // inp_img.shape[-1]
    else:
        downsamp_factor = 1

    initial_crop = config.initial_crop
    if initial_crop is not None:
        if isinstance(initial_crop, int):
            crop_size = initial_crop // downsamp_factor
        else:
            crop_size = (initial_crop[0] // downsamp_factor, initial_crop[1] // downsamp_factor)
        inp_img = crop_center(inp_img, crop_size)
        if gt_img is not None:
            gt_img = crop_center(gt_img, initial_crop)

    if normalize_data:
        inp_img = (inp_img - norm_prm.get("avgObs")) / norm_prm.get("stdObs")
        if gt_img is not None:
            gt_img = (gt_img - norm_prm.get("avgSig")) / norm_prm.get("stdSig")

    list_datasets = [(inp_img, gt_img)]

    if config.downsampling is not None:
        list_datasets, _ = apply_inp_transformations(
            list_datasets,
            config=config,
            for_training=False,
        )

    apply_additional_transforms(list_datasets, config.inp_transform, config.gt_transform)

    model_norm_prm = config.model_norm_prm
    if model_norm_prm is not None:
        inp_img, gt_img = list_datasets[0]
        inp_img = (inp_img - model_norm_prm.get("data_mean")) / model_norm_prm.get("data_std")
        if gt_img is not None:
            gt_mean = model_norm_prm.get("data_mean_gt")
            gt_std = model_norm_prm.get("data_std_gt")
            gt_img = (gt_img - gt_mean) / gt_std
        list_datasets[0] = (inp_img, gt_img)

    return list_datasets[0]


def _count_eval_samples(record: EvalRecord, *, config: DataSection) -> int:
    inp_img, gt_img = load_image(
        record.inp_path,
        gt_file=record.gt_path,
        data_format=config.resolved_data_format,
        config=config,
    )
    if inp_img is None:
        return 0

    inp_img, _ = make_pair_4d(inp_img, gt_img)
    return inp_img.shape[0]


def prepare_eval_sample(record: EvalRecord, *, config: DataSection) -> EvalSample:
    """Load and prepare one evaluation sample for model inference."""
    inp_img, gt_img = _load_eval_block(record, config=config)
    if inp_img is None:
        raise ValueError(f"Evaluation record could not be loaded: {record.inp_path}")

    sample_index = record.sample_index
    if sample_index >= inp_img.shape[0]:
        raise IndexError(f"Sample index {sample_index} out of range for {record.inp_path}.")
    if gt_img is not None and sample_index >= gt_img.shape[0]:
        raise IndexError(f"Sample index {sample_index} out of range for {record.gt_path}.")

    x_np = inp_img[sample_index]
    y_np = gt_img[sample_index] if gt_img is not None else None
    x = torch.from_numpy(np.asarray(x_np)).to(torch.float32)
    y = torch.from_numpy(np.asarray(y_np)).to(torch.float32) if y_np is not None else None

    return EvalSample(
        name=record.name,
        x=x,
        y=y,
        original_shape=tuple(x_np.shape[-2:]),
        record=record,
    )



def build_eval_source(
    saved_run: SavedTrainingRun,
    *,
    split: str = "test",
    crop_size: int | tuple[int, int] | None = None,
    eval_gt=None,
    data_prm_update: Mapping[str, Any] | None = None,
):
    """Build the evaluation sample source for a saved run and eval overrides."""

    # load data preparation config from the trained model
    data_cfg = dict(saved_run.data_cfg)
    model_norm_prm = dict(saved_run.model_norm_prm) if saved_run.model_norm_prm is not None else None

    # Update parameters from evaluation overrides.
    if eval_gt is not None and data_cfg.get("paired") is False:
        data_cfg["paired"] = True
        data_cfg["target"] = eval_gt
        if model_norm_prm is None:
            model_norm_prm = {}
        model_norm_prm["data_mean_gt"] = 0
        model_norm_prm["data_std_gt"] = 1

    if crop_size is not None:
        data_cfg["initial_crop"] = crop_size

    if data_prm_update is not None:
        data_cfg = deep_merge(data_cfg, dict(data_prm_update))

    data_dir = resolve_eval_data_dir(saved_run, data_cfg)
    if data_dir is None:
        raise ValueError(
            "Could not resolve `data_dir` for evaluation. "
            "Provide it through `data_prm_update={\'data_dir\': \'...path...\'}`."
        )

    dataset_info = resolve_dataset_info(data_cfg.get("dataset_name") or saved_run.dataset_name)

    # build data prep config with updated parameters
    prep_cfg = DataSection.model_validate(data_cfg).resolved(
        data_dir=Path(data_dir),
        norm_prm=saved_run.data_norm_prm,
        dataset_info=dataset_info,
        model_norm_prm=model_norm_prm,
        split=split,
    )

    return EvalSampleSource.from_config(prep_cfg)



__all__ = [
    "EvalRecord",
    "EvalSample",
    "EvalSampleSource",
    "build_eval_source",
    "prepare_eval_sample",
    "resolve_dataset_info",
    "resolve_eval_data_dir",
]
