"""High-level entrypoint for dataset-based evaluation of a saved model.

This module orchestrates the full evaluation flow: resolve the saved run,
initialize the inference runtime, rebuild the evaluation loader when needed,
run predictions, compute metrics, and persist outputs.
"""

import warnings
from pathlib import Path
from typing import Any

import torch

from lisai.evaluation.defaults import UNSET, UnsetType, resolve_evaluate_options
from lisai.evaluation.data import build_eval_loader
from lisai.evaluation.inference.stack import infer_batch
from lisai.evaluation.io import (
    create_save_folder,
    ensure_save_folder,
    save_metrics_json,
    save_outputs,
)
from lisai.evaluation.metrics import compute as metrics
from lisai.evaluation.runtime import initialize_runtime
from lisai.evaluation.saved_run import load_saved_run, resolve_run_dir


def run_evaluate(dataset_name:str,
             model_name:str,
             model_subfolder:str="",
             best_or_last: str | UnsetType = UNSET,
             epoch_number: int | None | UnsetType = UNSET,
             test_loader: Any | None | UnsetType = UNSET,
             tiling_size: int | None | UnsetType = UNSET,
             crop_size: int | tuple[int, int] | None | UnsetType = UNSET,
             metrics_list: list[str] | None | UnsetType = UNSET,
             lvae_num_samples: int | None | UnsetType = UNSET,
             results: dict | None | UnsetType = UNSET,
             save_folder: Path | str | None | UnsetType = UNSET,
             overwrite: bool | UnsetType = UNSET,
             eval_gt: str | None | UnsetType = UNSET,
             data_prm_update: dict | None | UnsetType = UNSET,
             ch_out: int | None | UnsetType = UNSET,
             split: str | UnsetType = UNSET,
             limit_n_imgs: int | None | UnsetType = UNSET,
             config: str | Path | None = None
             ):
    """Evaluate a saved run on a dataset split and optionally compute metrics.

    Any omitted optional argument is resolved from `configs/inference/defaults.yml`
    or from the named config passed via `config`.
    """
    options = resolve_evaluate_options(
        config=config,
        best_or_last=best_or_last,
        epoch_number=epoch_number,
        test_loader=test_loader,
        tiling_size=tiling_size,
        crop_size=crop_size,
        metrics_list=metrics_list,
        lvae_num_samples=lvae_num_samples,
        results=results,
        save_folder=save_folder,
        overwrite=overwrite,
        eval_gt=eval_gt,
        data_prm_update=data_prm_update,
        ch_out=ch_out,
        split=split,
        limit_n_imgs=limit_n_imgs,
    )
    run_dir = resolve_run_dir(dataset_name=dataset_name, subfolder=model_subfolder, exp_name=model_name)
    saved_run = load_saved_run(run_dir)

    if options["save_folder"] is None:
        if options["epoch_number"] is not None:
            save_name = f"evaluation_epoch{options['epoch_number']}"
        else:
            save_name = f"evaluation_{options['best_or_last']}"
        if options["split"] != "test":
            save_name = f"{save_name}_{options['split']}"
        save_folder = create_save_folder(path=run_dir / save_name,
                                         overwrite=options["overwrite"], parent_exists_check=True)
    else:
        save_folder = ensure_save_folder(Path(options["save_folder"]))

    if save_folder is None:
        raise FileNotFoundError("Model folder not found.")

    runtime = initialize_runtime(
        saved_run=saved_run,
        best_or_last=options["best_or_last"],
        epoch_number=options["epoch_number"],
        tiling_size=options["tiling_size"],
    )
    if saved_run.is_lvae:
        assert options["lvae_num_samples"] is not None, (
            "for LVAE prediction, number of samples needs to be specified"
        )

    upsamp = saved_run.upsampling_factor
    print(f"Found upsampling factor to be: {upsamp}\n")
    tiling_size = runtime.tiling_size

    test_loader = options["test_loader"]
    if test_loader is None:
        test_loader = build_eval_loader(
            saved_run,
            split=options["split"],
            crop_size=options["crop_size"],
            eval_gt=options["eval_gt"],
            data_prm_update=options["data_prm_update"],
        )
    results = options["results"]

    for batch_id, (x, y) in enumerate(test_loader):
        print(f"Image {batch_id} / {len(test_loader)}")

        if torch.isnan(y).all().item():
            y = None

        x = x.to(runtime.device)
        print(f"Input shape: {x.shape}")

        outputs = infer_batch(
            runtime.model,
            x,
            is_lvae=saved_run.is_lvae,
            tiling_size=tiling_size,
            num_samples=options["lvae_num_samples"],
            upsamp=upsamp,
            ch_out=options["ch_out"],
        )
        tosave = {
            "inp": x.cpu().detach().numpy(),
            "gt": y.cpu().detach().numpy() if y is not None else None,
            "pred": outputs.get("prediction"),
            "samples": outputs.get("samples"),
        }
        save_outputs(tosave, save_folder, img_name=f"img_{batch_id}")

        if options["metrics_list"] is not None and y is None:
            warnings.warn("no ground-truth provided, cannot calculate metrics")
        elif options["metrics_list"] is not None:
            if x.shape[-2:] == y.shape[-2:]:
                inp = x.cpu().detach().numpy()
            else:
                inp = None

            gt = y.cpu().detach().numpy()
            results = metrics.calculate_metrics(
                img_name=f"img_{batch_id}",
                metrics=options["metrics_list"],
                results=results,
                pred=outputs.get("prediction"),
                gt=gt,
                inp=inp,
            )
        if options["limit_n_imgs"] is not None and batch_id >= options["limit_n_imgs"] - 1:
            print("Stopping eval because reached limit_n_imgs")
            break

    if options["metrics_list"] is not None and results is not None:
        save_metrics_json(save_folder, results)
