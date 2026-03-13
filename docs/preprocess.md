# Preprocess Module

## Overview

The `preprocess` module converts raw dataset files into a structured,
standardized format ready for training or evaluation.

------------------------------------------------------------------------

## High-Level Architecture

    User Config (yaml)
            │
            ▼
    PreprocessRun
            │
            ├── Validate pipeline compatibility (data_type, fmt)
            │
            ├── Pipeline
            │      ├── OutputSpec (declares dataset structure)
            │      └── Source (discovers raw items)
            │
            ├── PreprocessSaver (handles file writing)
            │
            └── Processing Loop
                   ├── iterate Source items
                   ├── pipeline transforms arrays
                   ├── saver writes outputs
                   └── registry updated

------------------------------------------------------------------------

## Workflow

Preprocessing follows these steps:

1.  **Load user configuration**
    -   `dataset_name`
    -   `pipeline`
    -   `data_type` (`raw` or `recon`)
    -   `fmt` (`single`, `timelapse`, `mltpl_snr`, ...)
    -   `pipeline_cfg`
2.  **Instantiate `PreprocessRun`**
    -   Resolves paths
    -   Stores runtime context
    -   Validates pipeline compatibility
3.  **Instantiate pipeline**
    -   Ensures it supports requested `data_type` and `fmt`
4.  **Create `OutputSpec`**
    -   Declares the structure of the processed dataset
    -   Defines output keys
    -   Controls folder layout
    -   Drives registry structure
5.  **Create `PreprocessSaver`**
    -   Handles folder creation
    -   Applies naming conventions
    -   Writes arrays to disk
6.  **Build `Source`**
    -   Discovers raw items
    -   Yields `Item` objects
7.  **Processing loop**
    -   Pipeline loads raw arrays
    -   Applies transformations
    -   Returns output arrays
    -   Saver writes them to disk
8.  **Update dataset registry**
    -   Stores dataset structure
    -   Stores metadata (e.g. number of files)

------------------------------------------------------------------------

## Core Concepts

### PreprocessRun

Orhcestation layer.
Responsibilities: - Own runtime configuration - Instantiate and validate
pipeline - Create Saver - Execute processing loop - Update registry

------------------------------------------------------------------------

### Pipeline

Defines: - Supported `data_type` - Supported `fmt` - How to build its
`Source` - How to process each `Item` - What outputs it produces
(`OutputSpec`)

Pipelines: - Load arrays - Apply transformations - Return output arrays
Pipelines do not: - Write files - Manage paths - Update the registry

------------------------------------------------------------------------

### OutputSpec

`OutputSpec` declares the structure of the generated dataset.

It defines: - Output keys (e.g., `"main"`, `"inp"`, `"gt"`) - Axes
semantics (`"YX"`, `"TYX"`) - Whether outputs are saved at root or in
subfolders

Example:

``` python
OutputSpec(
    outputs=(
        OutputDecl(key="inp", axes="YX", role="inp"),
        OutputDecl(key="gt", axes="YX", role="gt"),
    )
)
```

This creates:

    preprocess/<data_type>/inp/
    preprocess/<data_type>/gt/

If `save_at_root=True` and key is `"main"`:

    preprocess/<data_type>/

------------------------------------------------------------------------

### Source

A `Source` discovers raw input items and yields `Item` objects.

Example:

``` python
Item(
    key="c01",
    paths=(Path(".../file.tif"),)
)
```

Different pipelines may use different source types.

------------------------------------------------------------------------

### Transformations

Transformations are small, reusable functions that operate on numpy
arrays.

They: - Take arrays as input - Return arrays - Have no filesystem or
registry logic

Examples: - `crop_center_2d` - `remove_first_frame` -
`bleach_correct_simple_ratio` - `compute_gt_avg`

Pipelines compose these as needed.

------------------------------------------------------------------------

### PreprocessSaver

Responsible for: - Naming outputs - Creating folders - Writing files -
Applying filename templates from `data.yml`

Saving logic is centralized here.

------------------------------------------------------------------------

### Dataset Registry

After preprocessing completes, the registry is updated with: - Dataset
name - Data type - Format - Structure (subfolders) - Metadata (file
counts, etc.)

The registry is the authoritative description of processed datasets.

------------------------------------------------------------------------

## Design Principles

-   Pipelines declare structure, not file layout.
-   Saving logic is centralized.
-   Transforms are pure functions.
-   Runtime context lives in one place (`PreprocessRun`).
-   Registry updates are automatic.
-   No hidden behavior.

------------------------------------------------------------------------

## Minimal Example

``` python
cfg = load_yaml("config.yml")
run = PreprocessRun.from_cfg(cfg)
run.execute()
```

This produces:

    <dataset_root>/
      preprocess/
        <data_type>/
          ...

and updates the dataset registry accordingly.
