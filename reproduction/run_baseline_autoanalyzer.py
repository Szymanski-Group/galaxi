#!/usr/bin/env python3
"""Run the XRD-AutoAnalyzer baseline over the pristine 61-pattern test set,
saving raw full-catalog scores per pattern to a JSON file.

Deliberately does NOT import galaxi -- see run_baseline_xca.py's docstring
for why (tensorflow/torch CUDA-library conflict in one process).

Usage:
    python reproduction/run_baseline_autoanalyzer.py [--limit N] [--output autoanalyzer_predictions.json]
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
AUTOANALYZER_REPO = THIS_DIR / "upstream" / "autoanalyzer_repo"
for _p in (THIS_DIR, AUTOANALYZER_REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from baselines import autoanalyzer

PRISTINE_DIR = REPO_ROOT / "examples" / "pretrained_catalog" / "experimental_patterns" / "pristine"
REFERENCE_DIR = REPO_ROOT / "examples" / "pretrained_catalog" / "reference_cifs"
WEIGHTS_DIR = THIS_DIR / "baselines" / "weights"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default="autoanalyzer_predictions.json")
    args = parser.parse_args()

    model_path = WEIGHTS_DIR / "autoanalyzer_Model.h5"
    model, reference_phases = autoanalyzer.load_model(model_path, REFERENCE_DIR)

    pattern_files = sorted(PRISTINE_DIR.glob("*.xy"))
    if args.limit:
        pattern_files = pattern_files[: args.limit]

    # SpectrumAnalyzer reads its .xy file directly with a bare np.loadtxt --
    # it does not skip the 2-line non-numeric header ("test" / "Wavelength = ...").
    scratch = Path(tempfile.mkdtemp(prefix="galaxi_autoanalyzer_"))
    for fname in pattern_files:
        lines = fname.read_text().splitlines(keepends=True)
        (scratch / fname.name).write_text("".join(lines[2:]))

    results = {}
    for i, fname in enumerate(pattern_files, start=1):
        result = autoanalyzer.predict(
            model, reference_phases,
            spectra_dir=scratch, spectrum_fname=fname.name,
            model_path=model_path, reference_dir=REFERENCE_DIR,
        )
        results[fname.name] = result["full_scores"]
        top1 = result["predicted_phases"][0][0] if result["predicted_phases"] else None
        print(f"[{i}/{len(pattern_files)}] {fname.name}: top-1 = {top1}")

    with open(args.output, "w") as f:
        json.dump(results, f)
    print(f"\nWrote {len(results)} patterns' scores to {args.output}")


if __name__ == "__main__":
    main()
