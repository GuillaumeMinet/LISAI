# Preprocess

This page describes the current preprocessing flow used by LISAI.

## Entry Point

Run preprocessing with:

```powershell
python src/scripts/preprocess/preprocess.py path/to/config.yml
```

The script loads the YAML config, builds a [`PreprocessRun`](../src/lisai/data/preprocess/run_preprocess.py), and executes it.

## Flow

1. Load the preprocess config.
2. Build `PreprocessRun` with project paths.
3. Instantiate the requested pipeline.
4. Build the source, output spec, and saver.
5. Iterate raw items, transform them, write outputs, and update the dataset registry.

## Main Concepts

### `PreprocessRun`

The orchestration layer. It owns runtime configuration, path resolution, pipeline creation, execution, and registry updates.

### Pipeline

A pipeline defines how a dataset is discovered and transformed. It declares its supported formats, creates a source, processes items, and exposes an output spec.

### Source

The source discovers raw input items and yields them to the pipeline.

### Output Spec

The output spec declares the structure of the generated dataset: output keys, axes, and folder layout.

### Saver

The saver owns file naming, folder creation, and writing processed outputs to disk.

## Outputs

Processed datasets are written under the preprocess area resolved by [`Paths.dataset_preprocess_dir(...)`](../src/lisai/infra/paths/paths.py).

## Registry

A successful preprocess run updates the dataset registry so the produced dataset can be discovered later by loading and evaluation code.
