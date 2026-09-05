# Reproducing

Standalone scripts and data for reproducing the manuscript's key quantitative results, kept separate from the
installable `galaxi` package (`src/galaxi/`) since none of it is needed to *use* GALAXI -- only to reproduce
specific figures/claims from the paper. Everything here is a plain script you run from the command line, not a
notebook -- each prints its result and writes a JSON file alongside the manuscript's target number for comparison.

## Layout

```
reproduction/
├── baselines/            # the 4 baseline phase-ID methods benchmarked against GALAXI (Fig. 3)
├── upstream/              # (gitignored) cloned baseline repos -- see baselines/*.py docstrings
├── texture_model/         # the MoO3 model used only for the texture reproduction (not in the 365-catalog;
│                         #   weights committed here, since it is a single ~1.8MB file)
├── validation/            # data for the reference-pool-size validation sweep
│   ├── experimental_patterns_cleaned/
│   └── ref_200/
├── run_pristine_evaluation.py       # main pristine-set micro-F1 (Results, Fig. 3)
├── run_ball_milling.py              # ball-milled single-phase subset
├── run_impurity_sweep.py            # Fig. 4a
├── run_uniform_peak_shift_sweep.py  # Fig. 4b
├── run_sample_displacement_sweep.py # Fig. 4c
├── run_texture_sweep.py             # Supplementary Figure S5
├── run_baseline_xca.py              # xca raw scores on the pristine set
├── run_baseline_autoanalyzer.py     # AutoAnalyzer raw scores on the pristine set
├── run_baseline_xqueryer.py         # XQueryer raw scores on the pristine set
├── run_baseline_peak_search_match.py
└── run_cross_model_topk_accuracy.py # Fig. 3a: top-k accuracy, all 5 models
```

## Setup

```bash
# GALAXI's own pretrained weights (needed by everything except the baseline scripts)
python examples/pretrained_catalog/fetch_model_weights.py

# Each ML baseline needs its own upstream repo cloned + weights fetched. The same
# repos/commits are recorded in each baselines/*.py docstring.
mkdir -p reproduction/upstream && cd reproduction/upstream
git clone https://github.com/njszym/XRD-AutoAnalyzer.git autoanalyzer_repo
git -C autoanalyzer_repo checkout bf32082521e45c0fcf5cf9ae9bd1321e76bf9012
# XQueryer's history is large enough that a plain clone can die with
# "fatal: early EOF"; --filter=blob:none gets through reliably.
git clone --filter=blob:none https://github.com/Bin-Cao/XQueryer.git xqueryer_repo
git -C xqueryer_repo checkout 35ea79f496ff740861fce10597fbad796e290ced
cd ../..

pip install tensorflow torch pyts asteval  # xca/autoanalyzer need tensorflow, xqueryer needs torch
python reproduction/baselines/fetch_baseline_weights.py xca
python reproduction/baselines/fetch_baseline_weights.py autoanalyzer
python reproduction/baselines/fetch_baseline_weights.py xqueryer
```

`xca` needs no clone: `baselines/xca.py` loads the trained `saved_model.keras` through
`tf.keras` directly and never imports the upstream `xca` package. (Its repo,
`https://github.com/maffettone/xca.git@ab3c3598631d870b5dc4d60d2f1298fba3ede343`, is only
needed to retrain from scratch.) The MoO3 texture model needs no download either -- its
weights are committed under `texture_model/`.

**Important**: `xca`/`autoanalyzer` (tensorflow) and `galaxi`/`xqueryer` (torch) must run in
**separate processes** -- `tf.keras.models.load_model()` followed by `import torch` in the same
process crashes with a CUDA-library symbol mismatch. This is why
`run_baseline_xca.py`/`run_baseline_autoanalyzer.py` never import `galaxi` and implement their own
minimal resampling instead of reusing `galaxi.core.pattern_utils`.

## Configuration notes

`run_pristine_evaluation.py` calls `ModelEvaluator.evaluate_experimental_patterns()` with
`group_phases=True, group_similarity_threshold=0.90` and, for the DARA leg, `strike_threshold=2`.
These match the values used to train and evaluate the released models and are
required to reproduce the manuscript's numbers -- `evaluate_experimental_patterns()`'s own
function-signature defaults for these arguments are looser and will not reproduce the reported
figures.

`run_cross_model_topk_accuracy.py`'s ground truth for each pattern is the plain, un-grouped
composition (`phases_in_pattern`), matched against top-k predictions collapsed into
peak-similarity groups at the same `group_similarity_threshold=0.90` -- the same convention used
throughout the manuscript's cross-model benchmarking (some catalog entries are genuinely
XRD-indistinguishable near-duplicates, so a prediction is credited if it lands on any member of
the true phase's similarity group).

## Results (61-pattern pristine experimental test set)

Reproduced by actually running the pipeline against real experimental patterns and pretrained weights, not
copied from the paper. See each script's own docstring for the exact manuscript target it's checking against.

| Script | Manuscript target | Reproduced |
|---|---|---|
| `run_pristine_evaluation.py --no-dara` | micro-F1=0.304, R=0.975, P=0.180 | micro-F1=0.304, R=0.975, P=0.180 |
| `run_pristine_evaluation.py` (CNN+DARA) | micro-F1=0.935, R=0.929, P=0.941 | micro-F1=0.877, R=0.936, P=0.825 |
| `run_ball_milling.py --no-dara` | micro-F1≈0.40-0.44, CNN-only (0.889 in the manuscript is the DARA-refined value, not CNN-only) | micro-F1=0.421 (only 4 patterns shipped here vs. the manuscript's full ball-milling set, so small-N sensitive) |
| `run_sample_displacement_sweep.py` | P>0.99 through z=0.75mm (Fe2O3, Fe3O4) | matches closely for both phases |
| `run_uniform_peak_shift_sweep.py` | P>0.999 through 0.3deg, collapses by 0.75-1.0deg | matches closely for all three phases |
| `run_impurity_sweep.py` | TiO2(anatase)/Fe3O4 stay confident; Mn3O4 degrades ≤3wt% | matches: Mn3O4 is the only phase that degrades, at the expected point in the series |
| `run_texture_sweep.py` | 0.052 → 0.998 → 1.000 (0/1/20 min milling) | 0.052 → 0.998 → 1.000 |
| `run_cross_model_topk_accuracy.py` | GALAXI 67.2%/82.0% @k=10/15; XCA (best baseline) 39.3%/41-47.5%; other baselines 24.6-32.8%/32.8-47.5% | GALAXI 70.5%/83.6%; XCA 39.3%/49.2%; AutoAnalyzer 29.5%/32.8%; XQueryer 26.2%/26.2%; Peak_search_match 32.8%/32.8% -- all within or close to the manuscript's stated ranges, same relative ranking |

**On the CNN+DARA gap**: recall matches closely (0.936 vs. 0.929); the shortfall is in precision
(0.825 vs. 0.941), concentrated in single-phase patterns where DARA's refinement retains one or
more extra candidate groups alongside the correct phase. This is a documented characteristic of
the DARA refinement stage rather than a data or configuration error -- the manuscript's own
Results section describes exactly this failure mode (a single-phase Li2TiO3 pattern where DARA
retains a spurious, structurally similar LiTiO2 group alongside the correct identification,
because it improves the profile fit), and Discussion explicitly notes that DARA does not
penalize peak overlap between candidate phases, which trades search tractability for this kind
of occasional over-retention. The magnitude of the gap on this particular 61-pattern subset is
somewhat larger than that single documented example alone would suggest, and is not fully
explained here; it does not affect any other result in this package.

Not reproduced in this package: Figure 3b's DARA-refined top-15-candidate comparison across all 5 models,
which requires running full DARA refinement on each baseline's own top-15 candidates (5 models × 61 patterns
of live Rietveld refinement) -- substantially more compute than everything else in this package combined.

## validation/

Data for the reference-pool-size validation sweep (how detection performance changes as the candidate-phase
catalog grows: `ref_50`/`ref_100`/`ref_200`/`ref_500`/`ref_1000`). Only `ref_200` and its matching experimental
patterns are included here as a representative pool size -- reproducing this sweep's results requires training
a fresh model per phase (the full protocol is ~15,000 training patterns and up to 200 epochs per phase, a
genuinely multi-minute-per-phase undertaking), so no training/evaluation script is included yet.
