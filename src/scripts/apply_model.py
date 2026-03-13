"""User entry point for applying a trained model on image file(s).

Edit values in USER PARAMETERS, then run:
    python src/scripts/apply_model.py
"""

from pathlib import Path

from lisai.evaluation import run_apply_model


def main():
    # =========================
    # USER PARAMETERS
    # =========================
    model_dataset = "Vim_fixed_mltplSNR_30nm"
    model_subfolder = "HDN"
    model_name = "CHANGE_ME"

    # input path can be a file or a folder
    data_path = r"CHANGE_ME"

    # checkpoint selection
    best_or_last = "best"  # "best" or "last"
    epoch_number = None

    # inference behavior
    filters = ["tif", "tiff"]
    skip_if_contain = None
    crop_size = None
    tiling_size = None
    stack_selection_idx = None
    timelapse_max = None
    lvae_num_samples = None  # required for LVAE models
    lvae_save_samples = True
    denormalize_output = True
    inp_masking = None
    downsamp = None
    dark_frame_context_length = False

    # io
    save_folder = "default"  # "default" or explicit path
    in_place = False
    save_inp = False

    # optional volumetric color coding
    apply_color_code = False
    color_code_prm = {
        "colormap": "turbo",
        "saturation": 0.35,
        "add_colorbar": True,
        "zstep": 0.4,
    }

    run_apply_model(
        model_dataset=model_dataset,
        model_subfolder=model_subfolder,
        model_name=model_name,
        data_path=Path(data_path),
        save_folder=save_folder,
        in_place=in_place,
        epoch_number=epoch_number,
        best_or_last=best_or_last,
        filters=filters,
        skip_if_contain=skip_if_contain,
        crop_size=crop_size,
        tiling_size=tiling_size,
        stack_selection_idx=stack_selection_idx,
        timelapse_max=timelapse_max,
        lvae_num_samples=lvae_num_samples,
        lvae_save_samples=lvae_save_samples,
        denormalize_output=denormalize_output,
        inp_masking=inp_masking,
        save_inp=save_inp,
        downsamp=downsamp,
        apply_color_code=apply_color_code,
        color_code_prm=color_code_prm,
        dark_frame_context_length=dark_frame_context_length,
    )


if __name__ == "__main__":
    main()
