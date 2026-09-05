"""Inference adapter for the xca baseline (Fig. 3a/b).

Requires the upstream `xca` package (not vendored here):
    pip install "tensorflow" "git+https://github.com/maffettone/xca.git@ab3c3598631d870b5dc4d60d2f1298fba3ede343"

The model itself (`saved_model.keras`, ~17MB) and its `phase_mapping.json`
(index -> reference-phase-name mapping) are not committed to git -- fetch
them with `galaxi.baselines.fetch_baseline_weights.fetch("xca")` first, or
point `load_model` at your own retrained files.

The ensemble CNN was trained from scratch (no pretrained xca weights exist
publicly) on cctbx-simulated patterns from GALAXI's own reference catalog,
using xca's own data-synthesis/model-building code as-is (only patched for
TF/Keras API compatibility, not logic). See the manuscript's Methods for
the training procedure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import json
import numpy as np


def load_model(saved_model_path: str | Path, phase_mapping_path: str | Path):
    """Load the trained ensemble model and its class-index -> phase-name mapping.

    Requires `tensorflow` to be installed.
    """
    import tensorflow as tf

    model = tf.keras.models.load_model(str(saved_model_path))
    mapping = json.loads(Path(phase_mapping_path).read_text())
    idx_to_stem = {int(k): v for k, v in mapping.items()}
    n_classes = len(idx_to_stem)
    if model.output_shape[-1] != n_classes:
        raise ValueError(
            f"Model has {model.output_shape[-1]} outputs but phase_mapping.json has "
            f"{n_classes} phases -- mapping does not match this model."
        )
    return model, idx_to_stem


def predict(model, idx_to_stem: Dict[int, str], intensity: np.ndarray) -> Dict[str, float]:
    """Predict phase probabilities for one already-resampled intensity trace.

    `intensity` must be a 1D array on the same grid the model was trained on
    (10-80 deg 2θ, 3501 points, Cu-Ka1) -- resample with
    `galaxi.core.pattern_utils.resample_pattern` if needed.

    Returns a full {phase_stem: probability} dict over every catalog phase.
    """
    intensity = np.asarray(intensity, dtype=np.float32)
    vmax = intensity.max()
    if vmax <= 0:
        raise ValueError("Pattern has non-positive max intensity.")

    # Matches xca.ml.tf.data_proc.test_preprocess: max-normalize, then map [0, 1] -> [-1, 1].
    x = intensity / vmax * 2 - 1
    x = x.reshape(1, -1, 1)

    probs = model({"X": x}, training=False).numpy()[0]
    return {idx_to_stem[i]: float(probs[i]) for i in range(len(probs))}


def top_candidates(
    full_scores: Dict[str, float], top_k: int = 5, confidence_floor: float = 0.10
) -> list[Tuple[str, float]]:
    """Discretize `predict`'s full score dict into a ranked candidate list:
    rank 1 always kept, ranks 2..top_k kept only while score >= confidence_floor.
    """
    ranked = sorted(full_scores.items(), key=lambda kv: -kv[1])
    candidates = ranked[:1]
    for name, score in ranked[1:top_k]:
        if score < confidence_floor:
            break
        candidates.append((name, score))
    return candidates
