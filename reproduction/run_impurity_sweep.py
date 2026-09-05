#!/usr/bin/env python3
"""Reproduce Figure 4a: raw CNN probability for a minority ("impurity")
phase as its weight fraction is diluted, for three phase pairs.

Manuscript target: patterns are numbered 1-7, diluted from 20 down to 1 wt%
(the exact per-index wt% mapping isn't in the manuscript text extracted
here, so results are reported against the file index, 1=most concentrated
.. 7=most dilute, consistent with the qualitative claims):
  "TiO2 (anatase) and Fe3O4 remain confidently detected... across the
   entire 20-to-1 wt% range, while Mn3O4 is the only phase whose
   probability degrades as its weight fraction drops to 3 wt% or below."

Usage:
    python reproduction/run_impurity_sweep.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _probability_utils import load_model, predict_pattern

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "examples" / "pretrained_catalog" / "pretrained_models"
SWEEP_DIR = REPO_ROOT / "examples" / "pretrained_catalog" / "experimental_patterns" / "impurity"

# (filename prefix, minority/"impurity" phase model name)
PAIRS = [
    ("TiO2_136_TiO2_141", "TiO2_141"),
    ("MnO2_136_Mn3O4_141", "Mn3O4_141"),
    ("Fe2O3_167_Fe3O4_227", "Fe3O4_227"),
]
INDICES = range(1, 8)


def main() -> None:
    results = {}
    for filename_prefix, model_name in PAIRS:
        model = load_model(MODELS_DIR, model_name)
        probs = {}
        for i in INDICES:
            fname = SWEEP_DIR / f"{filename_prefix}_{i}.xy"
            probs[i] = predict_pattern(model, fname)
        results[model_name] = probs
        print(f"{model_name} (impurity in {filename_prefix}):")
        for i, p in probs.items():
            print(f"  index={i} (more dilute as index increases)  ->  P = {p:.5f}")

    with open("impurity_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nManuscript target: TiO2(anatase)/Fe3O4 stay confident across the full range;")
    print("Mn3O4 degrades toward the dilute end (<=3 wt%, roughly the higher-index files)")


if __name__ == "__main__":
    main()
