#!/usr/bin/env python3
"""Reproduce GALAXI's performance on the ball-milled single-phase test set.

Manuscript target (Results, "Pristine samples" / ball-milling paragraph): micro-F1 = 0.889.
This is the DARA-refined number, not CNN-only -- "threshold-based" describes the CNN screening
cutoff (probability_threshold=0.5) that gates which candidates get passed into DARA refinement,
it does not mean DARA is skipped, matching how the pristine set's own no-DARA/DARA-refined
figures (0.304 / 0.935) are reported.

This script runs with --no-dara by default, whose correct comparison target is ~0.40-0.44, not
0.889; drop --no-dara to target the DARA-refined figure. Only 4 patterns ship in
examples/pretrained_catalog/experimental_patterns/ball_milling/, so results here are small-N sensitive to a
handful of false positives.

Thin wrapper around run_pristine_evaluation.py's shared evaluation logic,
pointed at examples/pretrained_catalog/experimental_patterns/ball_milling/ instead of
the pristine set.

Usage:
    python reproduction/run_ball_milling.py
"""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BALL_MILLING_DIR = REPO_ROOT / "examples" / "pretrained_catalog" / "experimental_patterns" / "ball_milling"

if __name__ == "__main__":
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parent / "run_pristine_evaluation.py"),
            "--no-dara",
            "--patterns-dir", str(BALL_MILLING_DIR),
            "--output", "ball_milling_results.json",
            "--eval-out-dir", "reproduction_ball_milling_eval_out",
        ],
    )
    print("\nThis --no-dara run's target: micro-F1 ~ 0.40-0.44 (CNN-only; see script docstring).")
    print("Manuscript's headline micro-F1 = 0.889 is the DARA-refined number -- drop --no-dara to target it.")
    sys.exit(result.returncode)
