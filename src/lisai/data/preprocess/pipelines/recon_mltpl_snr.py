# lisai/data/preprocess/pipelines/recon_mltpl_snr.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import tifffile

from lisai.config import settings

from ..core import FolderSource, Item, OutputDecl, OutputSpec, Source
from ..run_preprocess import PreprocessRun
from ..transformations import compute_gt_avg, crop_center_stack, register_stack
from .base import BasePipeline, PipelineResult


@dataclass
class ReconMltplSnrConfig:
    """
    Preprocess reconstructed multiple-SNR stacks (T,Y,X).

    Refactored from legacy `recon_mltpl_snr` pipeline.

    Typical stack layouts:
      - first_low_inp=False:
            [snr0, snr1, snr2, ...]  -> inp_mltpl_snr is the full stack
      - first_low_inp=True:
            [low_inp, snr0, snr1, ...] -> inp_single is first frame, inp_mltpl_snr is remaining stack

    Outputs (depending on config):
      - inp_mltpl_snr (TYX)
      - inp_single (YX) (optional)
      - gt_snr0 (YX) (optional)
      - gt_avg (YX) (optional)

    Notes:
      - Registration uses pystackreg via `register_stack` (rigid-body) if enabled.
      - `gt_avg_n_frames` controls how many leading frames are averaged for gt_avg.
    """
    dump_subfolder: str = ""
    combine_subfolders: bool = False

    first_low_inp: bool = False
    registration: bool = True
    crop_size: int | None = None

    gt_types: List[str] | None = None   # e.g. ["snr0","avg"]; None means no GT

    gt_clip_neg: bool = False
    gt_avg_n_frames: int | None = None

    def __post_init__(self):
        if self.gt_types is None:
            return
        allowed = {"snr0", "avg"}
        unknown = sorted(set(self.gt_types) - allowed)
        if unknown:
            raise ValueError(f"Unknown gt_types={unknown}. Allowed: {sorted(allowed)}")


class ReconMltplSnrPipeline(BasePipeline[ReconMltplSnrConfig]):
    Config = ReconMltplSnrConfig
    name = "recon_mltpl_snr"
    supported_data_types = {"recon"}
    supported_fmts = {"mltpl_snr"}

    def output_spec(self) -> OutputSpec:
        # Dynamically build outputs based on config
        outs: list[OutputDecl] = [
            OutputDecl(key="inp_mltpl_snr", axes="TYX", role="inp"),
        ]
        if self.cfg.first_low_inp:
            outs.append(OutputDecl(key="inp_single", axes="YX", role="inp"))

        gt_types = self.cfg.gt_types or []
        if "snr0" in gt_types:
            outs.append(OutputDecl(key="gt_snr0", axes="YX", role="gt"))
        if "avg" in gt_types:
            outs.append(OutputDecl(key="gt_avg", axes="YX", role="gt"))

        return OutputSpec(outputs=tuple(outs), save_at_root=False)

    def build_source(self, *, run: PreprocessRun) -> Source:
        dump_root = run.paths.dataset_dump_dir(
            dataset_name=run.dataset_name,
            data_type=run.data_type,
            additional_subfolder=self.cfg.dump_subfolder if self.cfg.dump_subfolder else "",
        )
        exts = tuple(settings.data.data_types[run.data_type])
        return FolderSource(root=dump_root, exts=exts, combine_subfolders=self.cfg.combine_subfolders)

    def process_item(self, *, item: Item) -> Dict[str, np.ndarray]:
        (p,) = item.paths
        stack = tifffile.imread(p)

        if stack.ndim != 3 or stack.shape[0] < 2:
            raise ValueError(
                f"ReconMltplSnrPipeline expects 3D stack (T,Y,X) with T>1, got shape={getattr(stack,'shape',None)} for {p}"
            )

        # Optional registration and crop on the full stack
        if self.cfg.registration:
            ref_idx = 1 if self.cfg.first_low_inp else 0
            stack = register_stack(stack, reference_index=ref_idx)

        if self.cfg.crop_size is not None:
            stack = crop_center_stack(stack, self.cfg.crop_size)

        outputs: dict[str, np.ndarray] = {}

        if self.cfg.first_low_inp:
            outputs["inp_single"] = stack[0]
            outputs["inp_mltpl_snr"] = stack[1:]
            snr0_idx = 1
            gt_stack = stack[1:]
        else:
            outputs["inp_mltpl_snr"] = stack
            snr0_idx = 0
            gt_stack = stack

        gt_types = self.cfg.gt_types or []
        if "snr0" in gt_types:
            gt = stack[snr0_idx].copy()
            outputs["gt_snr0"] = gt
        if "avg" in gt_types:
            outputs["gt_avg"] = compute_gt_avg(gt_stack, n_frames=self.cfg.gt_avg_n_frames).copy()

        # GT post-processing
        for key in ("gt_snr0", "gt_avg"):
            if key not in outputs:
                continue
            arr = outputs[key]
            if self.cfg.gt_clip_neg:
                arr = arr.copy()
                arr[arr < 0] = 0
            outputs[key] = arr

        return outputs

    def template_kwargs(self, *, item: Item, outputs: dict[str, np.ndarray]) -> dict[str, object]:
        # No special naming needs for this pipeline in the refactor.
        return {}

    def init_stats(self) -> dict[str, Any]:
        return {"n_frames": 0, "snr_levels": set()}

    def update_stats(self, *, stats: dict[str, Any], item, outputs: dict[str, Any]) -> dict[str, Any]:
        # Track original stack length as SNR level count.
        # We can infer from inp_mltpl_snr output length (+1 if first_low_inp).
        mstack = outputs.get("inp_mltpl_snr")
        if mstack is not None and mstack.ndim == 3:
            t = int(mstack.shape[0])
            if self.cfg.first_low_inp:
                t += 1
            stats["n_frames"] += t
            stats["snr_levels"].add(t)
        return stats

    def make_result(self, *, n_files: int, stats: dict[str, Any]) -> PipelineResult:
        levels = sorted(int(x) for x in stats.get("snr_levels", set()))
        return PipelineResult(n_files=n_files, n_frames=int(stats.get("n_frames", 0)), snr_levels=levels or None)
