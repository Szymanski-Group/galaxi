#!/usr/bin/env python3
"""Reproduce Figure 4b: raw CNN probability for the correct phase as a
function of a uniform 2θ offset, for Fe2O3, LiMn2O4, and TiO2.

Manuscript target: "Raw CNN probability for the correct phase remains
above 0.999 through a 0.3 deg shift for all three phases tested (Fe2O3,
LiMn2O4, TiO2)... falls sharply by 0.5 deg... collapses to near zero by
0.75-1.0 deg."

Usage:
    python reproduction/run_uniform_peak_shift_sweep.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _probability_utils import load_model, predict_pattern

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "examples" / "pretrained_catalog" / "pretrained_models"
SWEEP_DIR = REPO_ROOT / "examples" / "pretrained_catalog" / "experimental_patterns" / "uniform_peak_shift"

PHASES = [("Fe2O3_167", "Fe2O3_167_1"), ("LiMn2O4_227", "LiMn2O4_227"), ("TiO2_136", "TiO2_136")]
SHIFTS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1]


def main() -> None:
    results = {}
    for filename_prefix, model_name in PHASES:
        model = load_model(MODELS_DIR, model_name)
        probs = {}
        for shift in SHIFTS:
            shift_str = str(int(shift)) if shift == int(shift) else str(shift)
            fname = SWEEP_DIR / f"{filename_prefix}_shift_{shift_str}.xy"
            probs[shift] = predict_pattern(model, fname)
        results[model_name] = probs
        print(f"{model_name}:")
        for shift, p in probs.items():
            print(f"  shift={shift:>5} deg  ->  P = {p:.5f}")

    with open("uniform_peak_shift_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nManuscript target: P > 0.999 through 0.3deg, falls sharply by 0.5deg, ~0 by 0.75-1.0deg")


if __name__ == "__main__":
    main()
