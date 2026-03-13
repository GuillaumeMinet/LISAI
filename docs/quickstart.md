# LISAI Quick Start

This guide gives a clean, from-scratch setup to run LISAI on a new machine.

## 1) Prerequisites

- Conda installed (Miniconda or Anaconda)
- Git installed
- For GPU setup: NVIDIA driver compatible with CUDA 12.1

## 2) Clone the repository

```powershell
git clone <your-repo-url>
cd LISAI
```

## 3) Create the environment

Choose **one** of the two options.

### Option A: CPU

```powershell
conda env create -f environment.cpu.yml
conda activate lisai-cpu
```

### Option B: CUDA (GPU)

```powershell
conda env create -f environment.cuda.yml
conda activate lisai-cuda
```

If the env already exists and you want to sync it with the YAML:

```powershell
conda env update -f environment.cpu.yml --prune
# or: conda env update -f environment.cuda.yml --prune
```

## 4) Install LISAI as an editable package

From the repo root:

```powershell
pip install -e . --no-deps
```

Why this step:

- Registers `lisai` as an importable package
- Uses your local source tree (`src/lisai`) directly, so code edits are picked up immediately
- `--no-deps` avoids pip re-resolving dependencies already managed by conda

## 5) First-run infrastructure config

On first run, LISAI may prompt for a data root path and create:

- `configs/local_config.yml`

If you prefer, create it manually before running:

```yaml
infrastructure:
  data_root: "D:/your/data/root"
```

## 6) Smoke test

```powershell
python -c "import lisai; print('lisai import OK')"
python -c "from lisai.training.run_training import run_training; print('training import OK')"
```

## 7) Run training

```powershell
python src/scripts/train.py --config configs/experiments/hdn_training.yml
```

## 8) VS Code (optional)

- Open command palette -> `Python: Select Interpreter`
- Select the interpreter from `lisai-cpu` or `lisai-cuda`

## 9) Typical daily workflow

```powershell
conda activate lisai-cpu    # or lisai-cuda
pip install -e . --no-deps  # only needed again if packaging metadata changed
python src/scripts/train.py --config <your-config.yml>
```
