#!/usr/bin/env python3
"""Run the xca baseline over the pristine 61-pattern test set, saving raw
full-catalog scores per pattern to a JSON file.

Deliberately does NOT import galaxi (or anything that imports torch) --
tensorflow's bundled CUDA libraries conflict with torch's in the same
process (confirmed: tf.keras.models.load_model() followed by `import torch`
crashes with a cusparse/nvJitLink symbol mismatch). Run this as its own
process; combine its output with other baselines/galaxi in a separate
scoring step.

Usage:
    python reproduction/run_baseline_xca.py [--limit N] [--output xca_predictions.json]
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
sys_path_extra = str(THIS_DIR)
import sys
if sys_path_extra not in sys.path:
    sys.path.insert(0, sys_path_extra)
from baselines import xca

PRISTINE_DIR = REPO_ROOT / "examples" / "pretrained_catalog" / "experimental_patterns" / "pristine"
WEIGHTS_DIR = THIS_DIR / "baselines" / "weights"
N_POINTS = 3501


def resample(two_theta: np.ndarray, intensity: np.ndarray, target_min: float, target_max: float, n_points: int) -> np.ndarray:
    """Minimal local resample (avoids importing galaxi.core.pattern_utils)."""
    order = np.argsort(two_theta)
    two_theta, intensity = two_theta[order], intensity[order]
    f = CubicSpline(two_theta, intensity)
    grid = np.linspace(target_min, target_max, n_points)
    return f(grid)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default="xca_predictions.json")
    args = parser.parse_args()

    model, idx_to_stem = xca.load_model(WEIGHTS_DIR / "xca_saved_model.keras", WEIGHTS_DIR / "xca_phase_mapping.json")

    pattern_files = sorted(PRISTINE_DIR.glob("*.xy"))
    if args.limit:
        pattern_files = pattern_files[: args.limit]

    results = {}
    for i, fname in enumerate(pattern_files, start=1):
        data = np.loadtxt(fname, skiprows=2)
        intensity = resample(data[:, 0], data[:, 1], 10.0, 80.0, N_POINTS)
        scores = xca.predict(model, idx_to_stem, intensity)
        results[fname.name] = scores
        print(f"[{i}/{len(pattern_files)}] {fname.name}: top-1 = {max(scores, key=scores.get)}")

    with open(args.output, "w") as f:
        json.dump(results, f)
    print(f"\nWrote {len(results)} patterns' scores to {args.output}")


if __name__ == "__main__":
    main()
