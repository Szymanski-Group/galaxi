
<p align="center">
  <img src="assets/logo.png" alt="GALAXI Logo" width="700">
</p>

GALAXI is a Python package designed to automate the detection of crystalline phases in X-ray diffraction (XRD) patterns using deep learning. The package supports generating training data, training new phase-specific models, and applying custom models to experimental patterns. To analyze patterns using our pretrained library of 64,594 structures, use the public [GALAXI web interface](https://galaxi-xrd.com).

## Table of Contents

- [Installation](#installation)
- [Verify Installation](#verify-installation)
- [Quick Start](#quick-start)
- [Workflow Configuration](#workflow-configuration)
- [Visualization](#visualization)
- [Pretrained Catalog Example](#pretrained-catalog-example)

## Installation

```bash
# Clone the repository
git clone https://github.com/Szymanski-Group/galaxi.git
cd galaxi

# Install it using pip
pip install .
```

## Verify Installation

```bash
# Test the installation
python -c "import galaxi; print('GALAXI successfully installed!')"
```

## One-Time Data Setup

Training-data generation needs two downloaded assets. Both install into a user data directory
(`~/.local/share/galaxi/` by default) and are only needed once per machine:

```bash
galaxi-setup-cod           # COD reference structures (~5.5GB)
galaxi-setup-bg-profiles   # pre-simulated background/negative patterns (~670MB)
```

Check either at any time with `--verify-only`. Neither is required just to *evaluate* patterns
with pretrained models — see [`examples/`](examples/) for that path.

## Quick Start

`StreamlinedWorkflow` is GALAXI's primary interface: one config object drives pattern generation, training,
and evaluation. Put your target phases' CIFs in `References/` first (your own files, or query them with
`galaxi.CODQuery`), and run the one-time data setup above.

```python
from galaxi.workflows.streamlined_workflow import StreamlinedWorkflow, create_default_config

config = create_default_config()  # writes workflow_config.json; edit paths/phases as needed
workflow = StreamlinedWorkflow(config=config)

phases = ["Co3O4_1538531"]  # target phase(s), matching CIF filenames in References/

workflow.step_1_generate_training_data(phases=phases)      # generate + save training patterns
workflow.step_2_train_models(phases=phases)                # train one detection model per phase
workflow.step_3_evaluate_experimental_patterns()           # evaluate on real experimental patterns
```

Or run it from the CLI in one shot:

```bash
galaxi-workflow --create-config
galaxi-workflow --config workflow_config.json
```

`StreamlinedWorkflow` also has `step_4_generate_comprehensive_test_data`/`step_5_evaluate_models`, which
build a synthetic test set spanning individual artifact types (strain, texture, displacement, impurities)
and score the trained models against it.

The [`tutorials/`](tutorials/) notebooks cover the two pieces underneath this workflow:
`01_basic_pattern_generation` walks through the physical simulation and each artifact it can apply, and
`02_model_training` trains a single detection model end to end and plots its learning curves and ROC.

## Workflow Configuration

`workflow_config.json` (from `create_default_config()`) has these top-level sections:

- `directories` — references/CIFs, COD, experimental patterns, output paths
- `training_data_generation` — pattern counts, positive/negative mixture ratios, peak perturbations
- `test_data_generation` — comprehensive test-set generation settings
- `shared_xrd_generation_config` — physical simulation ranges (angle, shift, strain, texture, noise, ...)
- `model_config` — CNN architecture, training hyperparameters, output options
- `evaluation` — probability threshold, tolerance
- `performance` — CPU/COD-scan limits
- `ensemble` — ensembling toggle

Every key is commented in the file `create_default_config()` writes. Edit that JSON directly, or build
and modify the same dict in Python and pass it to `StreamlinedWorkflow(config=...)`.

## Visualization

GALAXI provides visualization tools for pattern analysis:

```python
from galaxi.visualization import XRDVisualizer

visualizer = XRDVisualizer(figsize=(10, 6))
visualizer.plot_multiple_files(
    file_paths=["pattern1.xy", "pattern2.xy"],  # raw (2θ, intensity) files
    labels=["Sample A", "Sample B"],
    save=True,
    filename="comparison"
)
```

## Pretrained Catalog Example

The [`examples/pretrained_catalog/`](examples/pretrained_catalog/) directory has a 365-phase chemical space with pretrained models and real experimental patterns, so you can try GALAXI without running the pipeline yourself. See
[`examples/pretrained_catalog/README.md`](examples/pretrained_catalog/README.md) — it contains reference CIFs, pretrained models
(configs/history/reports tracked in git, weights fetched separately), and experimental patterns grouped by
artifact (pristine, ball milling, sample displacement, impurity, uniform peak shift, texture).
