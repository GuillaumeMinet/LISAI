"""User entry point for model evaluation.

Edit values in USER PARAMETERS, then run:
    python src/scripts/evaluate_model.py
"""

from lisai.evaluation import run_evaluate


def main():
    # =========================
    # USER PARAMETERS
    # =========================
    dataset_name = "Vim_fixed_mltplSNR_30nm"
    model_name = "CHANGE_ME"
    model_subfolder = "HDN"

    # checkpoint selection
    best_or_last = "best"  # "best" or "last"
    epoch_number = None  # set int to force one checkpoint epoch

    # evaluation behavior
    split = "test"
    metrics_list = ["psnr", "ssim", "ra_psnr"]  # or None
    lvae_num_samples = None  # required for LVAE models
    tiling_size = None
    crop_size = None
    ch_out = None
    eval_gt = None
    data_prm_update = None  # e.g. {"data_dir": r"..."}
    limit_n_imgs = None

    # io
    save_folder = None  # None -> default evaluation folder in run dir
    overwrite = False

    run_evaluate(
        dataset_name=dataset_name,
        model_name=model_name,
        model_subfolder=model_subfolder,
        best_or_last=best_or_last,
        epoch_number=epoch_number,
        tiling_size=tiling_size,
        crop_size=crop_size,
        metrics_list=metrics_list,
        lvae_num_samples=lvae_num_samples,
        save_folder=save_folder,
        overwrite=overwrite,
        eval_gt=eval_gt,
        data_prm_update=data_prm_update,
        ch_out=ch_out,
        split=split,
        limit_n_imgs=limit_n_imgs,
    )


if __name__ == "__main__":
    main()
