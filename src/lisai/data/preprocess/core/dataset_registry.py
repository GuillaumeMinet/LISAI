from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from lisai.infra.config.yaml import load_yaml, save_yaml


class PipelineResultLike(Protocol):
    n_files: int
    n_frames: int | None
    snr_levels: int | list[int] | None


@dataclass
class DatasetRegistry:
    path: Path
    data: dict[str, Any] | None = None

    def __post_init__(self):
        self.path = Path(self.path)
        raw = load_yaml(self.path) if self.path.exists() else {}
        self.data = self._normalize_registry_data(raw)

    @staticmethod
    def _normalize_registry_data(raw: dict[str, Any]) -> dict[str, Any]:
        """
        Keep datasets at the top level.

        Backward compatibility:
        if a legacy registry contains a "datasets" key, merge its entries
        into the top-level map.
        """
        if not isinstance(raw, dict):
            return {}

        normalized = dict(raw)
        nested = normalized.pop("datasets", None)
        if isinstance(nested, dict):
            for dataset_name, dataset_value in nested.items():
                normalized.setdefault(dataset_name, dataset_value)
        return normalized

    def save(self) -> None:
        save_yaml(self.data or {}, self.path)

    def ensure_dataset(self, dataset_name: str) -> dict[str, Any]:
        datasets = self.data
        if dataset_name not in datasets:
            datasets[dataset_name] = {
                "format": None,
                "for_training": True,
                "size": {},
                "split": {},
                "structure": {},
            }
        return datasets[dataset_name]

    def update_after_preprocess(
        self,
        *,
        dataset_name: str,
        data_type: str,              # "recon" or "raw"
        data_format: str | None,     # e.g. "single", "timelapse", "mltpl_snr"
        structure: list[str],
        result: PipelineResultLike,
    ) -> None:
        ds = self.ensure_dataset(dataset_name)

        if data_format is not None:
            ds["format"] = data_format  # matches registry.yml "format"

        ds.setdefault("size", {})
        ds.setdefault("structure", {})

        size_entry: dict[str, Any] = {"n_files": result.n_files}
        if result.n_frames is not None:
            size_entry["n_frames"] = result.n_frames

        if result.snr_levels is not None:
            size_entry["snr_levels"] = result.snr_levels

        ds["size"][data_type] = size_entry
        ds["structure"][data_type] = structure
