from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import tifffile

from lisai.config import settings
from lisai.infra.fs.folders import ensure_folder

from .output_spec import OutputSpec

if TYPE_CHECKING:
    from lisai.infra.paths import Paths

class PreprocessSaver:
    """
    Handles all filesystem writing for preprocessing.

    Responsibilities:
      - Create preprocess folder structure based on OutputSpec
      - Apply canonical naming conventions (via infra)
      - Resolve output file paths using data.yml templates
      - Write numpy arrays to disk
    """

    def __init__(
        self,
        *,
        paths: Paths,
        dataset_name: str,
        data_type: str,
        fmt: str,
        output_spec: OutputSpec,
        overwrite_mode: str = "exist_ok",
    ):
        self.paths = paths
        self.spec = output_spec

        self.dataset_name = dataset_name
        self.data_type = data_type
        self.fmt = fmt
        self.overwrite_mode = overwrite_mode

        # make sure base_dir exists
        base_dir = self.paths.dataset_preprocess_dir(
            dataset_name=self.dataset_name,
            data_type=self.data_type,
        )
        ensure_folder(base_dir, mode=self.overwrite_mode)

        # create per-output dirs
        for out in self.spec.outputs:
            out_dir = base_dir / self.spec.folder_for(out.key)
            ensure_folder(out_dir, mode=self.overwrite_mode)

    def sample_id(self, idx: int) -> str:
        fmt = settings.NAMING.sample_id
        return fmt.format(id=idx)

    def save(
            self,
            *,
            key: str,
            array: np.ndarray,
            sample_id: str,
            **template_kwargs: Any,
        ) -> Path:
            """
            Save array image as a tiff (for now, might need to update for hdf5 raw files later)
            """

            out_path = self.paths.preprocessed_image_full_path(
                dataset_name=self.dataset_name,
                fmt=self.fmt,
                data_type=self.data_type,
                additional_subfolder=self.spec.folder_for(key),
                sample_id=sample_id,
                **template_kwargs,
                )

            ensure_folder(out_path.parent, mode="exist_ok")

            # Basic conventions: if stack, keep imagej metadata; else plain.
            if array.ndim == 3:
                tifffile.imwrite(out_path, array, imagej=True, metadata={"axes": "TYX"})
            else:
                tifffile.imwrite(out_path, array)

            return out_path
