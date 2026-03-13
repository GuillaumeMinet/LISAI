# LISAI Architecture

This document describes the production architecture of the repository, focused on `src/lisai/**` and its configuration contract (`configs/**`).

## Scope and Entry Points

- Primary training CLI entry point: `src/scripts/train.py` (`main`)
- Training orchestration entry point: `src/lisai/training/run_training.py` (`run_training`)
- Main architecture code: `src/lisai/**`
- Configuration sources:
  - `configs/project/project.yml`
  - `configs/data/data.yml`
  - experiment config passed to `--config` (for example `configs/experiments/hdn_training.yml`)

## High-Level Runtime Flow

1. CLI parses `--config` and calls `run_training(config_path)`  
   Source: `src/scripts/train.py` (`main`)
2. Configs are merged and normalized into a typed object (`ResolvedExperiment`)  
   Source: `src/lisai/infra/config/resolver.py` (`resolve_config`)
3. System context is initialized (paths, logging, run directory, tensorboard, device, callbacks)  
   Source: `src/lisai/training/setup/system.py` (`initialize`)
4. Data loaders are prepared from resolved dataset paths and registry metadata  
   Source: `src/lisai/training/setup/data.py` (`prepare_data`)
5. Model is constructed (and optionally restored from checkpoint) through runtime specs  
   Sources:
   - `src/lisai/training/setup/model.py` (`build_model`)
   - `src/lisai/runtime/spec.py` (`RunSpec.model_spec`)
   - `src/lisai/models/loader.py` (`prepare_model_for_training`)
6. Trainer implementation is selected by architecture and executes the training loop  
   Sources:
   - `src/lisai/training/trainers/factory.py` (`get_trainer`)
   - `src/lisai/training/trainers/base.py` (`BaseTrainer.train`)

## Layered Architecture

### 1) Configuration and Path Infrastructure

This layer provides typed config resolution and canonical filesystem path construction.

- YAML I/O:
  - `src/lisai/infra/config/yaml.py` (`load_yaml`, `save_yaml`)
- Merge and resolve:
  - `src/lisai/infra/config/merge.py` (`deep_merge`)
  - `src/lisai/infra/config/resolver.py` (`resolve_config`, `prune_config_for_saving`)
- Typed schema:
  - `src/lisai/infra/config/schema/experiment.py` (`ResolvedExperiment` and section models)
- Settings bootstrap and template context:
  - `src/lisai/infra/config/settings.py` (`Settings`, module singleton `settings`)
- Canonical path API:
  - `src/lisai/infra/paths/paths.py` (`Paths`)

Key architectural behavior:

- `resolve_config` merges project + data + experiment configs, normalizes mode aliases, resolves `load_model`, and returns a typed `ResolvedExperiment`.
- `Settings` reads global configs and exposes templated path resolution, while `Paths` is the only intended API for run/data/checkpoint paths.
- Run layout (checkpoints, artifacts, retrain origin files) is centralized in `Paths`.

### 2) Runtime Spec Layer

`RunSpec` and `ModelSpec` provide a stable interface between resolved config and model-loading logic.

- `src/lisai/runtime/spec.py`:
  - `RunSpec` exposes mode/dataset/origin information.
  - `RunSpec.model_spec()` derives model-relevant fields (including checkpoint selector and patch/downsampling-derived shape values).
  - `ModelSpec` carries architecture, parameters, mode, checkpoint info, and LVAE-related fields.

This layer keeps setup code from depending on raw nested config dict details.

### 3) Training Setup and Orchestration

The setup package is a composition boundary between infrastructure and trainers.

- Exposed setup API: `src/lisai/training/setup/__init__.py`
- Context initialization: `src/lisai/training/setup/system.py` (`initialize`)
- Run directory policy: `src/lisai/training/setup/run_dir.py` (`prepare_run_dir`)
- Data setup: `src/lisai/training/setup/data.py` (`prepare_data`)
- Model setup: `src/lisai/training/setup/model.py` (`build_model`)
- Orchestration: `src/lisai/training/run_training.py` (`run_training`)

Key behavior:

- `prepare_run_dir` implements mode semantics:
  - `train`: create new run dir
  - `continue_training`: reuse origin run dir
  - `retrain`: create new run dir + copy origin artifacts into `retrain_origin`
- `initialize` configures logging, optional TensorBoard writer, selected callbacks, and device fallback (`cuda` -> `cpu` when unavailable).

### 4) Model Construction and Loading

Model creation is registry-driven and separated from trainer logic.

- Registry: `src/lisai/models/registry.py` (`MODEL_REGISTRY`, `get_model_class`)
- Loader pipeline: `src/lisai/models/loader.py`
  - `prepare_model_for_training`
  - `init_model`
  - `_origin_checkpoint_path`
  - `load_noise_model` (LVAE path)

Key behavior:

- Architecture string maps to import path/class dynamically.
- For `continue_training` and `retrain`, checkpoints are resolved from canonical run layout.
- `full_model` checkpoint loading is supported, but state-dict loading remains the general path.
- LVAE has additional dependency flow: noise model + normalization strategy + derived image shape.

### 5) Trainer Layer

Trainer interfaces and implementations encapsulate training loops, optimizer/scheduler state, callbacks, and checkpoint writing.

- Factory: `src/lisai/training/trainers/factory.py` (`get_trainer`)
- Base abstraction: `src/lisai/training/trainers/base.py` (`BaseTrainer`)
- Standard trainer: `src/lisai/training/trainers/standard.py` (`StandardTrainer`)
- LVAE trainer: `src/lisai/training/trainers/lvae.py` (`LVAETrainer`)
- Checkpoint service: `src/lisai/training/checkpointing/manager.py` (`CheckpointManager`)

Key behavior:

- `BaseTrainer.train` owns epoch-level lifecycle:
  - train epoch
  - validate
  - update state dict
  - step scheduler
  - save checkpoints/loss file
  - run callbacks
- `StandardTrainer` supports configurable loss (`lisai.training.losses.get_loss_function`) with MSE fallback.
- `LVAETrainer` uses `lisai.lib.hdn.forwardpass` helpers and tracks KL/reconstruction losses.

### 6) Data Pipeline Layer

Data setup is composed from preprocess utilities and loader construction.

- Loader creation entry point:
  - `src/lisai/data/data_prep/make_loaders.py` (`make_training_loaders`)
- Setup integration:
  - `src/lisai/training/setup/data.py` (`prepare_data`)
- Supporting utilities:
  - `src/lisai/data/utils/**`
  - `src/lisai/data/preprocess/**`

Key behavior:

- `prepare_data` resolves dataset directory using `Paths.dataset_dir`.
- It optionally reads dataset metadata from the dataset registry file.
- It passes normalized config parameters into loader creation and returns both loaders and normalization/patch metadata for model setup.

## Core Architectural Contracts

1. Config contract is typed at runtime  
   Symbol: `ResolvedExperiment` in `src/lisai/infra/config/schema/experiment.py`
2. Path contract is centralized in `Paths`  
   Symbol: `Paths` in `src/lisai/infra/paths/paths.py`
3. Training mode semantics are centralized in config resolution + run-dir setup  
   Symbols:
   - `resolve_config` in `src/lisai/infra/config/resolver.py`
   - `prepare_run_dir` in `src/lisai/training/setup/run_dir.py`
4. Model lifecycle depends on `ModelSpec` rather than raw config dicts  
   Symbols:
   - `RunSpec.model_spec` in `src/lisai/runtime/spec.py`
   - `prepare_model_for_training` in `src/lisai/models/loader.py`
5. Trainer selection is architecture-based and explicit  
   Symbol: `get_trainer` in `src/lisai/training/trainers/factory.py`

## Directory Responsibilities (Production-Relevant)

- `src/lisai/infra/**`: config loading, schema, path templates, logging
- `src/lisai/runtime/**`: typed runtime specification objects
- `src/lisai/training/**`: setup pipeline, trainer abstractions, callbacks, checkpointing
- `src/lisai/models/**`: model registry and instantiation/loading logic
- `src/lisai/data/**`: loader creation, transform/preprocess utilities
- `src/lisai/evaluation/**`: inference/evaluation pipeline and metrics helpers
- `src/scripts/train.py`: top-level training CLI

## Extension Points

- Add a new model architecture:
  1. Implement model class under `src/lisai/models/`
  2. Register in `MODEL_REGISTRY` (`src/lisai/models/registry.py`)
  3. If training behavior differs, add a trainer and update `get_trainer`
- Add a new callback:
  1. Implement under `src/lisai/training/callbacks/`
  2. Instantiate in `initialize` (`src/lisai/training/setup/system.py`)
- Add a new run artifact or path template:
  1. Update project config templates/layout
  2. Expose accessor in `Paths`

## Non-Goals / Out-of-Scope for Architecture Baseline

- `graphs/**` analysis content and notebooks are not part of production architecture.
- Legacy caches (`__pycache__`) and notebook artifacts are excluded from architectural contracts.
