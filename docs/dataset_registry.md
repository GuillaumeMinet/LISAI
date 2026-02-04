## Dataset registry

The dataset registry lists all datasets that are available for training or inference. It is automatically updated during preprocessing and should not be edited during training.

It should be in the data folder.

Each entry specifies:
- dataset format (timelapse, mltpl_snr, single)
- task type (denoising, upsampling)
- dataset size and statistics
- train/val/test split strategy
- available input/ground-truth channels

## Example

Vim_live_timelapse_Monalisa1_35nm:
    format: timelapse
    for_training: True

    size:
      recon:
        n_files: 42
        n_frames: 890

    split:
      recon:
        method: random
        val: 6
        test: 6

    structure:
      recon: [] # empty because no inp/gt pairs



