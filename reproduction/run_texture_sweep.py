#!/usr/bin/env python3
"""Reproduce Supplementary Figure S5: raw CNN probability for MoO3 as
preferred orientation (texture) is progressively reduced by milling.

Manuscript target: "For the fully textured sample (0 min milling), the CNN
assigns the correct phase a probability of only 0.052. This increases to
0.998 after 1 minute of milling and reaches 1.000 after 20 minutes."

MoO3 isn't part of the 365-phase pretrained catalog -- its own model
is fetched separately, see texture_model/fetch_moo3_weights.py.

Note: an earlier candidate checkpoint (MoO3_14, space group P21/c) gave
near-zero probability across the whole milling series regardless of
texture -- that's the monoclinic beta-MoO3 polymorph, not the layered
orthorhombic alpha-MoO3 (space group Pbnm, #62) that "2D MoO3 nanosheets"
actually refers to. MoO3_62 (used below) reproduces the manuscript's
numbers almost exactly; wrong-polymorph mismatches like this fail
silently the same way a wrong angular range does.

Usage:
    python reproduction/texture_model/fetch_moo3_weights.py   # once
    python reproduction/run_texture_sweep.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _probability_utils import predict_pattern

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(__file__).resolve().parent / "texture_model"
SWEEP_DIR = REPO_ROOT / "examples" / "pretrained_catalog" / "experimental_patterns" / "texture"

MILLING_TIMES = [("no_bm", 0), ("1min", 1), ("5min", 5), ("20min", 20)]
TARGETS = {0: 0.052, 1: 0.998, 20: 1.000}


def main() -> None:
    from galaxi.detection.detection_model import PhaseDetectionModel

    model = PhaseDetectionModel(target_phase="MoO3_62", use_gpu=False)
    model.load_model(str(MODEL_DIR / "detection_model_MoO3_62.pth"))

    results = {}
    for suffix, minutes in MILLING_TIMES:
        fname = SWEEP_DIR / f"MoO3_{suffix}.xy"
        p = predict_pattern(model, fname)
        results[minutes] = p
        target = f" (manuscript: {TARGETS[minutes]})" if minutes in TARGETS else ""
        print(f"{minutes:>2} min milling  ->  P = {p:.5f}{target}")

    with open("texture_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
