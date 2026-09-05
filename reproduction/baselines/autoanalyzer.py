"""Inference adapter for the XRD-AutoAnalyzer baseline (Fig. 3a/b).

Requires the upstream `autoXRD` package (not vendored here -- this is the
PI's own prior tool, https://github.com/njszym/XRD-AutoAnalyzer):
    pip install "tensorflow" "git+https://github.com/njszym/XRD-AutoAnalyzer.git@bf32082521e45c0fcf5cf9ae9bd1321e76bf9012"

The model itself (`Model.h5`, ~72MB) is not committed to git -- fetch it
with `galaxi.baselines.fetch_baseline_weights.fetch("autoanalyzer")` first,
or point `load_model` at your own retrained file. Class ordering is
implicit: it must match `sorted(os.listdir(reference_dir))` at training
time, so `reference_dir` must be the exact same reference-CIF directory
(and untouched since) used to train the model being loaded.

This wraps only the tool's single first-pass CNN classification (no
residual-subtraction/iterative-refinement loop, no BGMN dependency) --
i.e. what `run_CNN.py`'s first iteration alone would report, matching the
"raw, no-DARA baseline" convention the other three baselines here use. A
real DARA-refinement pass over the resulting candidates, if wanted, should
go through `galaxi.evaluation.model_evaluator.run_dara_refinement`
(the same helper the CNN and Peak_search_match baselines use), not this
tool's own BGMN-dependent residual-subtraction loop.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple


def load_model(model_path: str | Path, reference_dir: str | Path):
    """Load the trained CNN and the reference-phase list whose sorted order
    defines the model's output class indices.

    Requires `tensorflow` and the upstream `autoXRD` package to be installed.
    """
    import tensorflow as tf
    from tensorflow.keras.utils import custom_object_scope
    from autoXRD.spectrum_analysis import CustomDropout

    with custom_object_scope({"CustomDropout": CustomDropout}):
        model = tf.keras.models.load_model(str(model_path), compile=False)

    reference_phases = [f for f in sorted(os.listdir(reference_dir)) if f[0] != "."]
    if len(reference_phases) != model.output_shape[-1]:
        raise ValueError(
            f"Model has {model.output_shape[-1]} outputs but reference_dir has "
            f"{len(reference_phases)} files -- reference_dir must be the exact "
            f"directory (and file set) used to train this model."
        )
    return model, reference_phases


def predict(
    model,
    reference_phases: List[str],
    spectra_dir: str | Path,
    spectrum_fname: str,
    model_path: str | Path,
    reference_dir: str | Path,
    min_angle: float = 10.0,
    max_angle: float = 80.0,
    min_conf: float = 40.0,
) -> Dict[str, object]:
    """Run the tool's single-pass MC-dropout CNN classification on one pattern.

    Returns {"full_scores": {phase: mean_softmax_probability, ...},
             "predicted_phases": [(phase, consensus_score), ...]} -- the
    latter using the tool's own MC-dropout consensus rule (fraction of
    dropout passes whose argmax agree, gated at `min_conf`).
    """
    import numpy as np
    from autoXRD.spectrum_analysis import KerasDropoutPrediction, SpectrumAnalyzer

    sa = SpectrumAnalyzer(
        spectra_dir=str(spectra_dir),
        spectrum_fname=spectrum_fname,
        max_phases=1,
        cutoff_intensity=5,
        min_conf=min_conf,
        wavelen="CuKa",
        reference_dir=str(reference_dir),
        min_angle=min_angle,
        max_angle=max_angle,
        model_path=str(model_path),
        is_pdf=False,
    )
    spectrum = sa.formatted_spectrum

    kdp = KerasDropoutPrediction(model)
    prediction, num_phases, certainties, num_outputs = kdp.predict(spectrum, min_conf=min_conf)
    if num_outputs != len(reference_phases):
        raise ValueError("Model output count does not match reference_phases length.")

    full_scores = {reference_phases[i][:-4]: float(prediction[i]) for i in range(len(prediction))}

    order = np.argsort(prediction)[::-1]
    predicted_phases: List[Tuple[str, float]] = [
        (reference_phases[order[i]][:-4], float(certainties[i])) for i in range(num_phases)
    ]

    return {"full_scores": full_scores, "predicted_phases": predicted_phases}
