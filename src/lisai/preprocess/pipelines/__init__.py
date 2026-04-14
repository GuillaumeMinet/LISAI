from .recon_mltpl_snr import ReconMltplSnrPipeline
from .recon_timelapse_simple import ReconTimelapseSimplePipeline
from .single import SingleReconPipeline

PIPELINES_REGISTRY = {
    "single_recon": SingleReconPipeline,
    "recon_timelapse_simple": ReconTimelapseSimplePipeline,
    "recon_mltpl_snr": ReconMltplSnrPipeline,
}

__all__ = ["PIPELINES_REGISTRY"]
