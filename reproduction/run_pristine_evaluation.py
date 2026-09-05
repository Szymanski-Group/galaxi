#!/usr/bin/env python3
"""Reproduce GALAXI's native (threshold-based) performance on the pristine
61-pattern experimental test set.

Manuscript targets (Results, "Pristine samples"):
  CNN screening only (probability_threshold=0.5): micro-F1=0.304, recall=0.975, precision=0.180
  CNN + DARA:                                     micro-F1=0.935, recall=0.929, precision=0.941
  CNN + DARA by true phase count (Supplementary Figure S2):
    1 phase: 0.963   2 phases: 0.941   3 phases: 0.941   4 phases: 0.875

Usage:
    python reproduction/run_pristine_evaluation.py [--no-dara] [--output results_pristine.json] [--limit N]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PRISTINE_DIR = REPO_ROOT / "examples" / "pretrained_catalog" / "experimental_patterns" / "pristine"
REFERENCE_DIR = REPO_ROOT / "examples" / "pretrained_catalog" / "reference_cifs"
MODELS_DIR = REPO_ROOT / "examples" / "pretrained_catalog" / "pretrained_models"


def micro_f1_from_counts(tp: int, fp: int, fn: int):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return f1, precision, recall


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-dara", action="store_true", help="CNN screening only, skip DARA refinement")
    parser.add_argument("--output", default="pristine_evaluation_results.json")
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N patterns (for quick testing)")
    parser.add_argument("--patterns-dir", default=None, help="Override pattern directory (default: pretrained-catalog pristine set)")
    parser.add_argument("--eval-out-dir", default="reproduction_pristine_eval_out", help="Raw ModelEvaluator output directory (each run overwrites its own)")
    args = parser.parse_args()

    from galaxi.evaluation.model_evaluator import ModelEvaluator
    from galaxi.core.config import XRDGenerationConfig, DEFAULT_MODEL_CONFIG

    pattern_dir = Path(args.patterns_dir) if args.patterns_dir else PRISTINE_DIR
    if args.limit is not None:
        import shutil
        import tempfile
        scratch = Path(tempfile.mkdtemp(prefix="galaxi_pristine_"))
        for i, f in enumerate(sorted(pattern_dir.glob("*.xy"))):
            if i >= args.limit:
                break
            shutil.copy(f, scratch / f.name)
        pattern_dir = scratch

    out_dir = Path(args.eval_out_dir)
    evaluator = ModelEvaluator(
        models_dir=str(MODELS_DIR),
        ref_dir=str(REFERENCE_DIR),
        output_dir=str(out_dir),
        xrd_config=XRDGenerationConfig(min_angle=5.0, max_angle=105.0, num_points=7001),
        model_config=DEFAULT_MODEL_CONFIG,
    )
    # strike_threshold=2: how many "no_improvement" attempts a phase gets before DARA
    # permanently removes it from the search. group_similarity_threshold=0.90: the
    # peak-similarity threshold used to group XRD-indistinguishable catalog phases.
    # Both match the values used to train/evaluate the released models; see
    # "Configuration notes" in reproduction/README.md.
    dara_dict = None if args.no_dara else {"use_dara": True, "strike_threshold": 2}
    evaluator.evaluate_experimental_patterns(
        str(pattern_dir), dara_dict=dara_dict,
        group_phases=True, group_similarity_threshold=0.90,
    )

    with open(out_dir / "experimental_evaluation.json") as f:
        exp_results = json.load(f)

    overall_tp = overall_fp = overall_fn = 0
    by_count = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "n_patterns": 0})
    per_pattern = []

    for pattern_name, pattern_results in exp_results.items():
        metrics = pattern_results.get("phase_metrics")
        if not metrics:
            continue
        tp, fp, fn = metrics["true_positives"], metrics["false_positives"], metrics["false_negatives"]
        n_phases = len(metrics["true_formulas"])

        overall_tp += tp
        overall_fp += fp
        overall_fn += fn
        by_count[n_phases]["tp"] += tp
        by_count[n_phases]["fp"] += fp
        by_count[n_phases]["fn"] += fn
        by_count[n_phases]["n_patterns"] += 1

        per_pattern.append({
            "file": pattern_results.get("filename", pattern_name),
            "n_phases": n_phases,
            "f1": metrics["f1_score"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
        })

    overall_f1, overall_p, overall_r = micro_f1_from_counts(overall_tp, overall_fp, overall_fn)

    by_count_results = {}
    for n, counts in sorted(by_count.items()):
        f1, p, r = micro_f1_from_counts(counts["tp"], counts["fp"], counts["fn"])
        by_count_results[n] = {"micro_f1": f1, "precision": p, "recall": r, "n_patterns": counts["n_patterns"]}

    results = {
        "mode": "cnn_only" if args.no_dara else "cnn_dara",
        "n_patterns": len(per_pattern),
        "overall": {"micro_f1": overall_f1, "precision": overall_p, "recall": overall_r},
        "by_phase_count": by_count_results,
        "per_pattern": per_pattern,
    }

    print(json.dumps({k: v for k, v in results.items() if k != "per_pattern"}, indent=2))
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results written to {args.output}")

    if args.patterns_dir is None:
        print("\nManuscript targets for comparison (pristine 61-pattern test set):")
        if args.no_dara:
            print("  CNN-only: micro-F1=0.304, recall=0.975, precision=0.180")
        else:
            print("  CNN+DARA overall: micro-F1=0.935, recall=0.929, precision=0.941")
            print("  CNN+DARA by phase count: 1=0.963  2=0.941  3=0.941  4=0.875")


if __name__ == "__main__":
    main()
