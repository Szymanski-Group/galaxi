# Pretrained Catalog Example

A worked example chemical space (365 reference phases, plus the Li4Ti5O12 entry added later, see below) for trying
GALAXI's evaluation and DARA-refinement workflows without first running the query → generate → train pipeline
yourself.

## Layout

```
examples/pretrained_catalog/
├── reference_cifs/           # 365 reference structures (.cif), one per phase
├── pretrained_models/        # 365 trained detection models, one folder per phase
│   └── models_<phase>/
│       ├── detection_model_<phase>.pth              # model weights (not tracked in git, see below)
│       ├── detection_model_<phase>_config.json       # ModelConfig used to build/train the model
│       ├── training_history_<phase>.json             # per-epoch loss/accuracy/AUC history
│       ├── classification_report_<phase>.json        # held-out test-set metrics
│       └── peak_list.json                            # reference peak positions used during generation
└── experimental_patterns/    # real experimental XRD patterns (.xy), grouped by artifact
    ├── pristine/                   # pristine patterns, one target phase per file
    ├── ball_milling/               # ball-milled samples
    ├── sample_displacement/        # sample-displacement sweep (z-offset series)
    ├── impurity/                   # impurity / target-fraction sweeps (multi-phase mixtures)
    ├── uniform_peak_shift/         # uniform peak-shift sweep
    └── texture/                    # preferred-orientation sweep: no_bm, 1min, 5min, 20min ball-milling times
```

`reference_cifs/` and `pretrained_models/` correspond 1:1 by phase name — e.g. `reference_cifs/LiC_225.cif` is the
structure that `pretrained_models/models_LiC_225/` was trained to detect.

## Model weights

The `.pth` weight files under `pretrained_models/` are excluded from git (see `.gitignore`, `*.pth`) since 365
individual files is too many for a single figshare upload, and ~640MB total is more than should live in git history.
Everything else (configs, training history, classification reports, peak lists) is tracked normally so the folder
structure and metadata are browsable directly on GitHub.

## The Li4Ti5O12 entry

`reference_cifs/Li4Ti5O12_227.cif` (COD 1001098, Li1.333Ti1.667O4, Fd-3m, a = 8.357 Å) and
`pretrained_models/models_Li4Ti5O12_227/` were added after the rest of the catalog. The ten pristine patterns
measured on commercial lithium titanate are this phase; they were previously named after Li4Ti7O16, the closest
spinel already in the catalog. Refining the single-phase pattern against each gives Rwp 34.50% for Li4Ti5O12 and
41.87% for Li4Ti7O16.

Both titanates are kept, so this directory now holds 366 references and 366 models. The 365-phase catalog used in
the manuscript is this set without `Li4Ti5O12_227`. The two structures have a calculated-profile similarity of
0.983, above the 0.90 threshold used to group XRD-indistinguishable phases for scoring, so they are scored as one
group either way.

Because this entry postdates the figshare deposit, its weights are not in `pretrained_model_weights.tar.gz`; the
single ~1.8MB checkpoint is committed directly instead, as the MoO3 texture model is.

The weights themselves are bundled into one tarball, `pretrained_model_weights.tar.gz`, hosted on figshare
(https://doi.org/10.6084/m9.figshare.33360183). They are only fetched when a user actually needs to run inference,
not as part of cloning the repo. To fetch and unpack them into place:

```bash
python examples/pretrained_catalog/fetch_model_weights.py
```

This downloads the archive to the repo root, verifies its sha256 checksum, and extracts it — dropping each
`detection_model_<phase>.pth` into its existing `examples/pretrained_catalog/pretrained_models/models_<phase>/` folder alongside the
config/history/report files already there. Pass `--archive <path>` instead of `--url` if you already downloaded the
tarball by hand, or `--skip-checksum` to bypass verification.

## Usage

**Important**: these models were trained on a 5-105 deg 2θ grid (7001 points), not the library's generic
10-80 deg default. `detection_model_<phase>_config.json` records this per-model, and `PhaseDetectionModel.load_model()`
reads it automatically -- but `ModelEvaluator` shares one preprocessing config across many models, so pass it
explicitly via `xrd_config` as shown below. Getting this wrong doesn't raise an error: it silently resamples every
pattern onto the wrong grid and collapses every prediction toward zero, which looks exactly like "the model doesn't
work" rather than a preprocessing mismatch.

Evaluate the pretrained models against the experimental patterns:

```python
from galaxi.evaluation.model_evaluator import ModelEvaluator
from galaxi.core.config import XRDGenerationConfig

evaluator = ModelEvaluator(
    models_dir="examples/pretrained_catalog/pretrained_models",
    output_dir="evaluation_results",
    xrd_config=XRDGenerationConfig(min_angle=5.0, max_angle=105.0, num_points=7001),
)

# Ground truth is parsed automatically from filenames
exp_results = evaluator.evaluate_experimental_patterns("examples/pretrained_catalog/experimental_patterns/pristine")
```

Or load a single model directly:

```python
from galaxi import PhaseDetectionModel
from galaxi.core.pattern_utils import regularize_input
import numpy as np

model = PhaseDetectionModel(target_phase="Li3PO4_31")
model.load_model("examples/pretrained_catalog/pretrained_models/models_Li3PO4_31/detection_model_Li3PO4_31.pth")

# Real .xy patterns have a 2-line non-numeric header ("test" / "Wavelength = ...").
fname = "examples/pretrained_catalog/experimental_patterns/pristine/Li3PO4_31.xy"
data = np.loadtxt(fname, skiprows=2)
pattern = np.stack([data[:, 0], data[:, 1]], axis=1)  # (N, 2): [two_theta, intensity]

intensity = regularize_input(
    fname, pattern,
    min_angle=model.config.min_angle, max_angle=model.config.max_angle,
    target_length=model.config.input_size, use_mask=model.config.use_mask,
    model_config=model.config,
)
probability = model.predict(intensity)
```
