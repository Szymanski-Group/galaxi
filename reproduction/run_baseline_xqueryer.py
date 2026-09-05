#!/usr/bin/env python3
"""Run the XQueryer baseline over the pristine 61-pattern test set, saving
raw full-catalog scores per pattern to a JSON file.

Uses torch (not tensorflow), so unlike run_baseline_xca.py/
run_baseline_autoanalyzer.py this CAN safely import galaxi in the same
process if needed -- kept structurally consistent with the other two
anyway (no galaxi import) since it doesn't need anything from it.

Usage:
    python reproduction/upstream/xqueryer_repo  # must exist first, see baselines/xqueryer.py docstring
    python reproduction/baselines/fetch_baseline_weights.py xqueryer  # once
    python reproduction/run_baseline_xqueryer.py [--limit N] [--output xqueryer_predictions.json]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
from baselines import xqueryer

PRISTINE_DIR = REPO_ROOT / "examples" / "pretrained_catalog" / "experimental_patterns" / "pristine"
WEIGHTS_DIR = THIS_DIR / "baselines" / "weights"
REPO_SRC = THIS_DIR / "upstream" / "xqueryer_repo" / "src"

MIN_ANGLE, MAX_ANGLE, NPOINTS = 10.0, 80.0, 3501
# Elements spanned by the pretrained reference catalog (Li-Fe-Mn-Ti-P-C-O).
CHEM_SPACE_ELEMENTS = ["Li", "C", "O", "P", "Ti", "Mn", "Fe"]


def resample_and_normalize(two_theta: np.ndarray, intensity: np.ndarray) -> np.ndarray:
    order = np.argsort(two_theta)
    two_theta, intensity = np.unique(two_theta[order], return_index=False), intensity[order]
    f = interp1d(two_theta, intensity[: len(two_theta)], kind="slinear", fill_value="extrapolate")
    grid = np.linspace(MIN_ANGLE, MAX_ANGLE, NPOINTS)
    resampled = np.clip(f(grid), 0, None)
    m = resampled.max()
    return 100.0 * resampled / m if m > 0 else resampled


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default="xqueryer_predictions.json")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    model, catalog = xqueryer.load_model(
        WEIGHTS_DIR / "xqueryer_model_best_inference_only.pth",
        WEIGHTS_DIR / "xqueryer_labels.json",
        repo_src=REPO_SRC,
        device=args.device,
    )
    element_vector = xqueryer.build_element_vector(
        WEIGHTS_DIR / "xqueryer_CGCNN_atom_emb.json", elements=CHEM_SPACE_ELEMENTS
    )

    pattern_files = sorted(PRISTINE_DIR.glob("*.xy"))
    if args.limit:
        pattern_files = pattern_files[: args.limit]

    results = {}
    for i, fname in enumerate(pattern_files, start=1):
        data = np.loadtxt(fname, skiprows=2)
        intensity = resample_and_normalize(data[:, 0], data[:, 1])
        scores = xqueryer.predict(model, catalog, intensity, element_vector, device=args.device)
        results[fname.name] = scores
        print(f"[{i}/{len(pattern_files)}] {fname.name}: top-1 = {max(scores, key=scores.get)}")

    with open(args.output, "w") as f:
        json.dump(results, f)
    print(f"\nWrote {len(results)} patterns' scores to {args.output}")


if __name__ == "__main__":
    main()
