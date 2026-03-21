from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from lisai.config import resolve_config, settings
from lisai.runs.scanner import DiscoveredRun
from lisai.runs.schema import TrainingSignature

from .schema import ResourceClass


@dataclass(frozen=True)
class SchedulingContext:
    config_path: Path
    dataset: str
    model_subfolder: str
    run_name: str
    training_signature: TrainingSignature


def load_scheduling_context(config_path: str | Path) -> SchedulingContext:
    resolved_path = Path(config_path).expanduser().resolve()
    cfg = resolve_config(resolved_path)
    return SchedulingContext(
        config_path=resolved_path,
        dataset=str(cfg.data.dataset_name).strip(),
        model_subfolder=str(cfg.routing.models_subfolder).strip(),
        run_name=str(cfg.experiment.exp_name).strip(),
        training_signature=build_training_signature_from_resolved_config(cfg),
    )


def build_training_signature_from_resolved_config(cfg) -> TrainingSignature:
    architecture = str(getattr(getattr(cfg, "model", None), "architecture", "")).strip()
    if not architecture:
        architecture = "unknown"

    data_batch = getattr(getattr(cfg, "data", None), "batch_size", None)
    training_batch = getattr(getattr(cfg, "training", None), "batch_size", None)
    batch_size = int(data_batch) if data_batch is not None else int(training_batch or 1)
    if batch_size <= 0:
        batch_size = 1

    data_cfg = getattr(cfg, "data", None)
    patch_size = None
    if data_cfg is not None:
        patch_size = getattr(data_cfg, "patch_size", None)
        if patch_size is None:
            patch_size = getattr(data_cfg, "val_patch_size", None)
        if patch_size is None and hasattr(data_cfg, "model_patch_size"):
            patch_size = getattr(data_cfg, "model_patch_size")
    if isinstance(patch_size, tuple):
        patch_size = list(patch_size)

    return TrainingSignature(
        architecture=architecture,
        batch_size=batch_size,
        patch_size=patch_size,
    )


def resource_class_defaults_mb() -> dict[ResourceClass, int]:
    configured = settings.project.queue.resource_class_vram_mb
    return {
        "light": int(configured.light),
        "medium": int(configured.medium),
        "heavy": int(configured.heavy),
    }


def training_signature_key(signature: TrainingSignature) -> tuple[str, int, tuple[int, ...] | int | None]:
    patch = signature.patch_size
    if isinstance(patch, list):
        patch_key: tuple[int, ...] | int | None = tuple(int(item) for item in patch)
    elif patch is None:
        patch_key = None
    else:
        patch_key = int(patch)
    return (signature.architecture, int(signature.batch_size), patch_key)


def signatures_match(left: TrainingSignature, right: TrainingSignature) -> bool:
    return training_signature_key(left) == training_signature_key(right)


def matching_peak_vram_mb(
    *,
    signature: TrainingSignature,
    runs: list[DiscoveredRun] | tuple[DiscoveredRun, ...],
) -> list[int]:
    peaks: list[int] = []
    for run in runs:
        metadata = run.metadata
        if metadata.status == "running" and not metadata.closed_cleanly:
            continue
        if metadata.training_signature is None:
            continue
        if metadata.runtime_stats is None or metadata.runtime_stats.peak_gpu_mem_mb is None:
            continue
        if not signatures_match(metadata.training_signature, signature):
            continue
        peaks.append(int(metadata.runtime_stats.peak_gpu_mem_mb))
    return peaks


def estimate_expected_vram_mb(
    *,
    signature: TrainingSignature | None,
    resource_class: ResourceClass,
    runs: list[DiscoveredRun] | tuple[DiscoveredRun, ...],
    resource_defaults_mb: Mapping[str, int] | None = None,
) -> tuple[int, str]:
    defaults = dict(resource_defaults_mb or resource_class_defaults_mb())

    if signature is not None:
        peaks = matching_peak_vram_mb(signature=signature, runs=runs)
        if peaks:
            return max(peaks), "history"

    return int(defaults[resource_class]), f"resource_class:{resource_class}"


__all__ = [
    "SchedulingContext",
    "build_training_signature_from_resolved_config",
    "estimate_expected_vram_mb",
    "load_scheduling_context",
    "matching_peak_vram_mb",
    "resource_class_defaults_mb",
    "signatures_match",
    "training_signature_key",
]
