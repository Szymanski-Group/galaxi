#!/usr/bin/env python3
"""Run the Peak_search_match baseline over the pristine 61-pattern test
set, saving ranked-candidate scores per pattern to a JSON file (same
{filename: {phase: score}} shape as the other three run_baseline_*.py
scripts, for uniform downstream scoring).

Usage:
    python reproduction/run_baseline_peak_search_match.py [--limit N] [--output psm_predictions.json]
"""
import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
from baselines import match_phases

PRISTINE_DIR = REPO_ROOT / "examples" / "pretrained_catalog" / "experimental_patterns" / "pristine"
REFERENCE_DIR = REPO_ROOT / "examples" / "pretrained_catalog" / "reference_cifs"


def strip_header(src: Path, dst_dir: Path, n_lines: int = 2) -> Path:
    lines = src.read_text().splitlines(keepends=True)
    dst = dst_dir / src.name
    dst.write_text("".join(lines[n_lines:]))
    return dst


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-matches", type=int, default=30, help="How many ranked candidates to keep per pattern")
    parser.add_argument("--output", default="peak_search_match_predictions.json")
    args = parser.parse_args()

    references = sorted(REFERENCE_DIR.glob("*.cif"))
    pattern_files = sorted(PRISTINE_DIR.glob("*.xy"))
    if args.limit:
        pattern_files = pattern_files[: args.limit]

    scratch = Path(tempfile.mkdtemp(prefix="galaxi_psm_"))
    cache_dir = scratch / "cache"
    cache_dir.mkdir()

    results = {}
    for i, fname in enumerate(pattern_files, start=1):
        cleaned = strip_header(fname, scratch)
        ranked = match_phases(cleaned, references, top_matches=args.top_matches, cache_dir=cache_dir)
        # Normalize FoM to a pseudo-probability in [0, 1] via min-max over this pattern's own ranking,
        # so it's comparable in shape to the other baselines' {phase: score} dicts.
        foms = [r.fom for r in ranked]
        lo, hi = (min(foms), max(foms)) if foms else (0.0, 1.0)
        span = hi - lo if hi > lo else 1.0
        scores = {r.reference_name: (r.fom - lo) / span for r in ranked}
        results[fname.name] = scores
        print(f"[{i}/{len(pattern_files)}] {fname.name}: top-1 = {ranked[0].reference_name if ranked else None}")

    with open(args.output, "w") as f:
        json.dump(results, f)
    print(f"\nWrote {len(results)} patterns' scores to {args.output}")
    shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    main()
