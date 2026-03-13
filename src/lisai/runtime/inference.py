from __future__ import annotations

from pathlib import Path
from typing import Any

from lisai.infra.config import settings
from lisai.infra.paths import Paths
from lisai.runtime._old_runs_compatibility import (
    extract_data_prm,
    extract_model_architecture,
    extract_model_norm_prm,
    extract_model_prm,
    extract_noise_model_name,
    extract_patch_size_and_downsamp_factor,
    load_training_cfg_from_run,
    normalize_training_cfg_for_inference,
    preferred_load_method,
)
from lisai.runtime.spec import InferenceSpec


def build_inference_spec(
    *,
    model_folder: Path,
    best_or_last: str = "best",
    epoch_number: int | None = None,
) -> tuple[InferenceSpec, dict[str, Any], dict[str, Any]]:
    run_dir = Path(model_folder)
    training_cfg_raw = load_training_cfg_from_run(run_dir)
    training_cfg = normalize_training_cfg_for_inference(training_cfg_raw)

    architecture = extract_model_architecture(training_cfg)
    if not architecture:
        raise ValueError("Could not resolve model architecture from training config.")

    data_prm = extract_data_prm(training_cfg)
    patch_size, downsamp_factor = extract_patch_size_and_downsamp_factor(data_prm)

    spec = InferenceSpec(
        run_dir=run_dir,
        architecture=architecture,
        parameters=extract_model_prm(training_cfg),
        normalization=dict(training_cfg.get("normalization") or {}),
        model_norm_prm=extract_model_norm_prm(training_cfg, data_prm),
        noise_model_name=extract_noise_model_name(training_cfg_raw),
        patch_size=patch_size,
        downsamp_factor=downsamp_factor,
        checkpoint_method=preferred_load_method(training_cfg_raw),
        checkpoint_selector=best_or_last,
        checkpoint_epoch=epoch_number,
    )
    return spec, training_cfg, training_cfg_raw


def iter_inference_checkpoint_candidates(spec: InferenceSpec):
    paths = Paths(settings)

    methods: list[str] = []
    for method in (spec.checkpoint_method, "state_dict", "full_model"):
        if method and method not in methods:
            methods.append(method)

    for method in methods:
        kwargs: dict[str, Any] = {"run_dir": spec.run_dir, "load_method": method}
        if spec.checkpoint_epoch is not None:
            kwargs["epoch_number"] = spec.checkpoint_epoch
        else:
            kwargs["best_or_last"] = spec.checkpoint_selector
        yield method, paths.checkpoint_path(**kwargs)

