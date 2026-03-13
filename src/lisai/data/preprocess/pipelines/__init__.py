from .recon_mltpl_snr import ReconMltplSnrPipeline
from .recon_timelapse_upsamp import ReconTimelapseUpsampPipeline
from .single import SingleReconPipeline

PIPELINES_REGISTRY = {
    "single_recon": SingleReconPipeline,
    "recon_timelapse_upsamp": ReconTimelapseUpsampPipeline,
    "recon_mltpl_snr": ReconMltplSnrPipeline,
}

__all__ = ["PIPELINES_REGISTRY"]
