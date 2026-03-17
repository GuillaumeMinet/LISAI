# LISAI Architecture

This page describes the current production architecture under `src/lisai/**`.

## Main Flows

### Training

1. `lisai train ...` resolves a training config through [`src/lisai/training/cli.py`](../src/lisai/training/cli.py).
2. [`resolve_config`](../src/lisai/config/io/resolver.py) merges and validates configs into `ResolvedExperiment`.
3. [`initialize_runtime`](../src/lisai/training/runtime.py) creates `TrainingRuntime`.
4. [`prepare_data`](../src/lisai/training/setup/data.py) returns `PreparedTrainingData`.
5. [`build_model`](../src/lisai/training/setup/model.py) returns the model and optional restored state.
6. [`get_trainer`](../src/lisai/training/trainers/factory.py) selects the trainer and runs the loop.

### Evaluation

1. [`run_apply_model`](../src/lisai/evaluation/run_apply_model.py) and [`run_evaluate`](../src/lisai/evaluation/run_evaluate.py) are the entrypoints.
2. [`load_saved_run`](../src/lisai/evaluation/saved_run.py) turns a saved `config_train.yaml` into `SavedTrainingRun`.
3. [`initialize_runtime`](../src/lisai/evaluation/runtime.py) loads the model checkpoint into `InferenceRuntime`.
4. [`build_eval_loader`](../src/lisai/evaluation/data.py) rebuilds a test loader when dataset-based evaluation is needed.
5. [`src/lisai/evaluation/inference/**`](../src/lisai/evaluation/inference) performs inference, and [`src/lisai/evaluation/io.py`](../src/lisai/evaluation/io.py) saves outputs.

### Preprocess

1. [`src/scripts/preprocess/preprocess.py`](../src/scripts/preprocess/preprocess.py) is the entry script.
2. [`PreprocessRun`](../src/lisai/data/preprocess/run_preprocess.py) orchestrates preprocessing.
3. Pipelines under [`src/lisai/data/preprocess/pipelines/**`](../src/lisai/data/preprocess/pipelines) define the actual dataset transformation logic.
4. The dataset registry is updated after a successful run.

## Core Boundaries

- `ResolvedExperiment` in [`src/lisai/config/models/training/root.py`](../src/lisai/config/models/training/root.py)
  The typed training config contract.
- `TrainingRuntime` in [`src/lisai/training/runtime.py`](../src/lisai/training/runtime.py)
  Live training infrastructure only.
- `PreparedTrainingData` in [`src/lisai/training/setup/data.py`](../src/lisai/training/setup/data.py)
  Typed output of training data setup.
- `TrainingModelSpec` in [`src/lisai/training/setup/model.py`](../src/lisai/training/setup/model.py)
  Training-local model setup contract.
- `SavedTrainingRun` in [`src/lisai/evaluation/saved_run.py`](../src/lisai/evaluation/saved_run.py)
  Evaluation-side view of a saved training run.
- `InferenceRuntime` in [`src/lisai/evaluation/runtime.py`](../src/lisai/evaluation/runtime.py)
  Live evaluation resources only.
- `Paths` in [`src/lisai/infra/paths/paths.py`](../src/lisai/infra/paths/paths.py)
  Canonical filesystem/path API.

## Directory Responsibilities

- `src/lisai/config/**`: config loading, validation, settings, schema
- `src/lisai/infra/paths/**`: canonical path resolution
- `src/lisai/training/**`: training orchestration, setup, trainers, checkpointing
- `src/lisai/models/**`: model registry and load/build logic
- `src/lisai/data/**`: data loading and preprocessing
- `src/lisai/evaluation/**`: saved-run loading, evaluation runtime, inference, metrics, output IO
- `src/scripts/**`: convenience entry scripts

## Rule of Thumb

- Config objects describe what should happen.
- Runtime objects carry what is live in the current process.
- Path resolution goes through `Paths`, not hardcoded strings.
