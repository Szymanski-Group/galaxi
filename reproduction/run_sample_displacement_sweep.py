#!/usr/bin/env python3
"""Reproduce Figure 4c: raw CNN probability for the correct phase as a
function of sample-displacement magnitude (z), for Fe2O3 and Fe3O4.

Manuscript target: "the raw CNN probability for both phases remains above
0.99 up to a displacement of 0.75 mm, extending beyond the +/-0.5 mm
sample-displacement range used during training."

Usage:
    python reproduction/run_sample_displacement_sweep.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _probability_utils import load_model, predict_pattern

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "examples" / "pretrained_catalog" / "pretrained_models"
SWEEP_DIR = REPO_ROOT / "examples" / "pretrained_catalog" / "experimental_patterns" / "sample_displacement"

# (target phase name in filenames, model name)
PHASES = [("Fe2O3", "Fe2O3_167_1"), ("Fe3O4", "Fe3O4_227")]
MAGNITUDES = [0, 0.1, 0.3, 0.5, 0.75, 1.0, 1.5]


def main() -> None:
    results = {}
    for filename_prefix, model_name in PHASES:
        model = load_model(MODELS_DIR, model_name)
        probs = {}
        for z in MAGNITUDES:
            fname = SWEEP_DIR / f"{filename_prefix}_z_{z}.xy"
            probs[z] = predict_pattern(model, fname)
        results[model_name] = probs
        print(f"{model_name}:")
        for z, p in probs.items():
            print(f"  z={z:>4} mm  ->  P = {p:.5f}")

    with open("sample_displacement_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nManuscript target: probability remains above 0.99 up to z=0.75mm for both phases")


if __name__ == "__main__":
    main()
