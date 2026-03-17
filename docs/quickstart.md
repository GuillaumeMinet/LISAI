# LISAI Quick Start

This guide covers the shortest path from clone to a first training or evaluation run.

## Environment

Create one Conda environment from the repo root:

```powershell
conda env create -f environment.cpu.yml
conda activate lisai-cpu
```

or:

```powershell
conda env create -f environment.cuda.yml
conda activate lisai-cuda
```

Install the package in editable mode:

```powershell
pip install -e . --no-deps
```

## Smoke Test

Check that the CLI is available:

```powershell
lisai train --help
```

## Training

Run a config from `configs/experiments` by name:

```powershell
lisai train hdn_training
```

You can also pass an explicit file path:

```powershell
lisai train configs/experiments/hdn_training.yml
```

Training resolves the config, creates a run directory, saves `config_train.yaml`, and writes checkpoints and logs under the run folder.

## Evaluation

Use the CLI for evaluation workflows:

```powershell
lisai evaluate Gag/Upsamp/my_model --split val --metrics psnr,ssim
lisai apply Gag/Upsamp/my_model /data/images --tiling-size 512
```

Both commands accept `--config <name>` to load settings from `configs/inference/<name>.yml`,
and any CLI argument overrides the config value.

## Preprocess

Run preprocessing from a YAML config:

```powershell
python src/scripts/preprocess/preprocess.py path/to/preprocess.yml
```

## Where To Look Next

- [`docs/architecture.md`](architecture.md): current module boundaries
- [`docs/data_organization.md`](data_organization.md): path and run layout overview
- [`docs/preprocess.md`](preprocess.md): preprocess flow and concepts
