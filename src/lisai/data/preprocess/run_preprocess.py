from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from lisai.config import settings

from .core import DatasetRegistry, PreprocessConfig, PreprocessSaver
from .pipelines import PIPELINES_REGISTRY
from .pipelines.base import PipelineResult

if TYPE_CHECKING:
    from lisai.infra.paths import Paths

@dataclass
class PreprocessRun:
    """
    Entry point for dataset preprocessing.

    High-level workflow:
    1. Load user config and validate parameters by loading to a Pydantic Config.
    2. Validate pipeline compatibility.
    3. Create OutputSpec (via pipeline): 
            - an object that declares the structure of the generated dataset
            - folder creation, saving logic, and registry depends on it
            - each pipeline is responsible for parametrizing its own OutputSpec 
    4. Create Saver, which handles all the Saving, following OutputSpec.
    5. Build the Source:
            - an object that defines where the raw items come from
            - each pipeline is responsible for parametrizing its own Source 
    5. Iterate over Source items.
    5. Apply pipeline transforms and save outputs.
    6. Update dataset registry.

    See docs/preprocess.md for full architecture details.
    """
    dataset_name: str
    paths: Paths
    logger: logging.Logger
    registry: DatasetRegistry

    data_type: str                 # "raw" | "recon"
    fmt: str                       # "single" | "timelapse" | "mltpl_snr"
    pipeline_cfg: dict[str, Any]
    pipeline_name: str

    @classmethod
    def from_cfg(cls, cfg: dict, *, paths: Paths | None = None) -> PreprocessRun:

        # Validate + normalize config via pydantic model
        pcfg = PreprocessConfig.model_validate(cfg)

        if paths is None:
            from lisai.infra.paths import Paths
            paths = Paths(settings)
        logger = logging.getLogger(f"lisai.preprocess.{pcfg.dataset_name}")
        registry = DatasetRegistry(paths.dataset_registry_path())
        return cls(
            dataset_name=pcfg.dataset_name,
            pipeline_name=pcfg.pipeline,
            data_type=pcfg.data_type,
            fmt=pcfg.fmt,
            pipeline_cfg=pcfg.pipeline_cfg,
            paths=paths,
            registry=registry,
            logger=logger
        )
    
    def _build_pipeline(self):
        try:
            pipeline_cls = PIPELINES_REGISTRY[self.pipeline_name]
        except KeyError:
            raise KeyError(f"Unknown pipeline '{self.pipeline_name}'. Available: {sorted(PIPELINES_REGISTRY)}")

        supported_data_types = getattr(pipeline_cls, "supported_data_types", {getattr(pipeline_cls, "data_type", None)})
        supported_fmts = getattr(pipeline_cls, "supported_fmts", {getattr(pipeline_cls, "fmt", None)})

        # validate data_type and data_fmt
        if self.data_type not in supported_data_types:
            raise ValueError(
                f"Pipeline '{self.pipeline_name}' does not support data_type='{self.data_type}'. "
                f"Supported: {sorted(supported_data_types)}"
            )
        if self.fmt not in supported_fmts:
            raise ValueError(
                f"Pipeline '{self.pipeline_name}' does not support fmt='{self.fmt}'. "
                f"Supported: {sorted(supported_fmts)}"
            )
        
        cfg_obj = pipeline_cls.parse_cfg(self.pipeline_cfg)
        pipeline = pipeline_cls(cfg=cfg_obj)

        return pipeline


    def execute(self) -> PipelineResult:

        # build pipeline
        pipeline = self._build_pipeline()

        # build output_spec + saver
        spec = pipeline.output_spec()
        saver = PreprocessSaver(
            paths=self.paths,
            dataset_name=self.dataset_name,
            data_type=self.data_type,
            fmt=self.fmt,
            output_spec=spec,
        )

        # build source
        source = pipeline.build_source(run=self)

        # iterate over items, execute pipeline and save
        n_files = 0
        stats = pipeline.init_stats()
        for i, item in enumerate(source.iter_items()):
            sid = saver.sample_id(i)
            outputs = pipeline.process_item(item=item)

            for key, arr in outputs.items():
                saver.save(
                    key=key,
                    array=arr,
                    sample_id=sid,
                    **pipeline.template_kwargs(item=item, outputs=outputs),
                )

            stats = pipeline.update_stats(stats=stats, item=item, outputs=outputs)
            n_files += 1
        
        # make result
        result = pipeline.make_result(n_files=n_files, stats=stats)

        # registry update
        self.registry.update_after_preprocess(
            dataset_name=self.dataset_name,
            data_type=self.data_type,
            data_format=self.fmt,
            structure=spec.structure_keys(),
            result=result
        )
        self.registry.save()

        return result
