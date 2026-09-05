"""Inference adapter for the XQueryer baseline (Fig. 3a/b).

Requires the upstream `XQueryer` package source on your PYTHONPATH (not
vendored here): clone https://github.com/Bin-Cao/XQueryer.git at commit
35ea79f496ff740861fce10597fbad796e290ced and pass its `src/` directory as
`repo_src` below.

The checkpoint (~3.7GB, inference-only -- optimizer state stripped from the
~11GB training checkpoint) and the CGCNN atom-embedding table used for the
composition channel are not committed to git -- fetch them with
`galaxi.baselines.fetch_baseline_weights.fetch("xqueryer")` first.

XQueryer's Xmodel takes BOTH the XRD pattern and an elemental-composition
channel (cross-attention over CGCNN atom embeddings). At inference time on
real unknown patterns there is no independent per-pattern composition
measurement without leaking the answer, so this adapter -- like the
manuscript's benchmark run -- feeds a FIXED, pattern-independent
composition prior: the mean-pooled CGCNN embedding of your reference
catalog's full elemental universe (pass `elements`, e.g. `["Li", "Fe",
"O"]`). This is deliberately not per-pattern ground truth (same constant
vector for every pattern) but also not a null vector.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

# Periodic table atomic numbers for the elements likely to appear in a
# typical inorganic reference catalog. Extend as needed for your own set.
ATOMIC_NUMBERS = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9,
    "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16,
    "Cl": 17, "K": 19, "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24,
    "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30, "Zr": 40,
    "Nb": 41, "Mo": 42, "Ba": 56, "La": 57, "Ce": 58, "W": 74,
}


def load_model(
    checkpoint_path: str | Path,
    labels_path: str | Path,
    repo_src: str | Path,
    device: str = "cpu",
):
    """Load the trained Xmodel and its class-index -> phase-name catalog.

    `repo_src` is the `src/` directory of a checkout of the upstream
    XQueryer repository (see module docstring for the exact commit).
    Requires `torch` to be installed.
    """
    import torch

    repo_src = str(repo_src)
    if repo_src not in sys.path:
        sys.path.insert(0, repo_src)
    from model.XQueryer import Xmodel  # noqa: PLC0415 (deliberately lazy/optional import)

    with open(labels_path) as f:
        labels = json.load(f)
    catalog = [entry["name"] for entry in sorted(labels, key=lambda e: e["idx"])]

    model = Xmodel(embed_dim=3500, num_classes=len(catalog))
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, catalog


def build_element_vector(cgcnn_emb_path: str | Path, elements: Sequence[str]):
    """Mean-pooled CGCNN embedding for a set of elements (the fixed,
    pattern-independent composition prior fed to Xmodel at inference)."""
    import torch

    with open(cgcnn_emb_path) as f:
        cgcnn_emb = json.load(f)

    codes = [ATOMIC_NUMBERS[el] for el in elements] or [0]
    values = [cgcnn_emb[str(c)] for c in codes]
    return torch.mean(torch.tensor(values, dtype=torch.float32), dim=0)


def predict(
    model,
    catalog: List[str],
    intensity: np.ndarray,
    element_vector,
    device: str = "cpu",
) -> Dict[str, float]:
    """Predict phase probabilities for one pattern resampled onto a
    10-80 deg, 3501-point 2θ grid, max-normalized to [0, 100].
    """
    import torch

    x = torch.tensor(intensity, dtype=torch.float32).reshape(1, -1).to(device)
    elem = element_vector.reshape(1, -1).to(device)

    with torch.no_grad():
        logits = model(x, elem)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    return {catalog[i]: float(probs[i]) for i in range(len(probs))}
