# lisai/data/preprocess/pipelines/single.py
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict

import numpy as np
import tifffile

from lisai.infra.config import settings

from ..core import MAIN_OUTPUT_KEY, FolderSource, Item, OutputDecl, OutputSpec, Source
from ..transformations import crop_center_2d
from .base import BasePipeline, PipelineResult

if TYPE_CHECKING:
    from ..run_preprocess import PreprocessRun


@dataclass
class SingleReconConfig:
    """
    Preprocess reconstructed single images from dump/ into preprocess/recon/
    """
    dump_subfolder: str = ""             # inside dataset dump dir
    combine_subfolders: bool = False
    crop_size: int | None = None


class SingleReconPipeline(BasePipeline[SingleReconConfig]):
    Config = SingleReconConfig
    name = "single_recon"
    supported_data_types = {"recon"}
    supported_fmts = {"single"}

    def __init__(self, cfg: SingleReconConfig):
        super().__init__(cfg)

    def output_spec(self) -> OutputSpec:
        return OutputSpec(
            outputs=(OutputDecl(key=MAIN_OUTPUT_KEY, axes="YX", role="inp"),),
            save_at_root=True,
        )

    def build_source(self, *, run: PreprocessRun) -> Source:
        
        dump_root = run.paths.dataset_dump_dir(
            dataset_name = run.dataset_name,
            data_type = run.data_type,
            additional_subfolder =  self.cfg.dump_subfolder if self.cfg.dump_subfolder else "",
        )
        
        exts = tuple(settings.data.data_types[run.data_type])

        return FolderSource(root=dump_root, exts=exts, combine_subfolders=self.cfg.combine_subfolders)

    def process_item(self, *, item: Item) -> Dict[str, np.ndarray]:
        (p,) = item.paths
        img = tifffile.imread(p)

        if img.ndim != 2:
            raise ValueError(f"SingleReconPipeline expects 2D images, got {img.ndim}D for {p}")

        if self.cfg.crop_size is not None:
            img = crop_center_2d(img, self.cfg.crop_size)

        return {MAIN_OUTPUT_KEY: img}
    
    def template_kwargs(self, *, item: Item, outputs: dict[str, np.ndarray]) -> dict[str, object]:
        return {}
    
    def make_result(self,*, n_files: int, stats: dict[str, Any]) -> PipelineResult:
        result = PipelineResult(n_files=n_files)
        return result
