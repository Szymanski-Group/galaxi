#!/usr/bin/env python3
"""Reproduce Figure 3a: top-k accuracy for GALAXI vs. the four baselines on
the pristine 61-pattern test set. A prediction is counted correct at a
given k if every true phase group in the pattern appears among that model's
top-k ranked candidates -- after first collapsing top-k names into structural
groups via peak-profile similarity (the same catalog GALAXI's own CNN
screening and DARA's internal grouping use), since the 365-phase catalog
contains genuine XRD-indistinguishable near-duplicates (different formulas,
identical peak profiles) as well as literal duplicate-CIF entries (same
formula+space-group, disambiguated only by a "_1"/"_2" suffix).

Manuscript target: GALAXI reaches 67.2% at k~10 and 82.0% at k~15; XCA
(strongest baseline) reaches 39.3% at k~10; the remaining baselines reach
24.6-32.6% at k~10, and baselines overall range 32.8-47.5% at k~15.

Requires each baseline's raw predictions already generated:
    python reproduction/run_baseline_xca.py --output xca_predictions.json
    python reproduction/run_baseline_autoanalyzer.py --output autoanalyzer_predictions.json
    python reproduction/run_baseline_xqueryer.py --output xqueryer_predictions.json
    python reproduction/run_baseline_peak_search_match.py --output peak_search_match_predictions.json
And GALAXI's own raw CNN screening scores (via run_pristine_evaluation.py --no-dara,
reading its --eval-out-dir's experimental_evaluation.json for the full per-phase
probabilities, not just the F1 summary).

Usage:
    python reproduction/run_cross_model_topk_accuracy.py \\
        --galaxi-eval-dir reproduction_pristine_cnn_only_out \\
        --xca xca_predictions.json --autoanalyzer autoanalyzer_predictions.json \\
        --xqueryer xqueryer_predictions.json --psm peak_search_match_predictions.json
"""
import argparse
import json
from pathlib import Path

from galaxi.core.pattern_utils import build_phase_groups_from_peaks, group_phases_func

REPO_ROOT = Path(__file__).resolve().parent.parent


def clean_name(x: str) -> str:
    x = x.replace("_mp", "")
    parts = x.split("_")
    if len(parts) >= 2 and parts[1].isdigit():
        return f"{parts[0]}_{parts[1]}"
    return parts[0]


def true_phases_for_pattern(pattern_results: dict) -> list:
    """Ground truth for one pattern, as the manuscript scores it.

    phase_metrics["true_formulas"] expands each labelled phase into the group
    of catalog entries that are XRD-indistinguishable from it, and that is the
    ground truth the published Figure 3a uses -- the manuscript's sweep reads
    the same expansion out of experimental_evaluation_summary.csv's "True
    phases" column. Scoring the bare phases_in_pattern label instead is more
    permissive, because a pattern then counts as a hit as soon as any one
    member of the group is retrieved: on the 61-pattern pristine set that
    reads 70.5% at k=10 and 83.6% at k=15, against the published 67.2% and
    82.0%. The whole difference sits on the twelve LiMn2O4-bearing patterns,
    whose group holds several lithium-manganese spinels.

    Falls back to phases_in_pattern for evaluation outputs that predate
    phase_metrics.

    NOTE the expansion is a property of the evaluation run that produced the
    file, not of this script: an evaluation produced before the October 2026
    phase-grouping fix carries coarser groups and will score lower here.
    """
    metrics = pattern_results.get("phase_metrics") or {}
    true_formulas = metrics.get("true_formulas")
    if not true_formulas:
        return [clean_name(x) for x in pattern_results["phases_in_pattern"]]
    return [
        clean_name(member)
        for group in true_formulas
        for member in (group if isinstance(group, list) else [group])
    ]


def load_galaxi_scores(eval_dir: Path):
    with open(Path(eval_dir) / "experimental_evaluation.json") as f:
        exp_results = json.load(f)
    scores_by_pattern = {}
    true_phases_by_pattern = {}
    for pattern_key, pattern_results in exp_results.items():
        filename = pattern_results.get("filename", pattern_key)
        scores_by_pattern[filename] = pattern_results["probabilities"]
        true_phases_by_pattern[filename] = true_phases_for_pattern(pattern_results)
    return scores_by_pattern, true_phases_by_pattern


def load_baseline_scores(path: Path):
    with open(path) as f:
        raw = json.load(f)
    # Keep raw (uncleaned) reference names -- the phase-similarity catalog is
    # keyed by the literal models_dir directory names (e.g. "NiO_225_1"), and
    # clean_name is applied only when converting matched groups to units below.
    return raw


def pred_units_for_topk(scores: dict, k: int, catalog) -> set:
    ranked_raw = [name for name, _ in sorted(scores.items(), key=lambda kv: -kv[1])[:k]]
    raw_groups = group_phases_func(ranked_raw, catalog) if catalog is not None else [[n] for n in ranked_raw]
    return set(frozenset(clean_name(m) for m in g) for g in raw_groups if g)


def true_units_for_pred(true_phases_clean: list, pred_units: set) -> set:
    phase_to_group = {}
    for grp in pred_units:
        for m in grp:
            phase_to_group[m] = grp
    return set(phase_to_group.get(c, frozenset([c])) for c in true_phases_clean)


def top_k_hit(scores: dict, true_phases: list, k: int, catalog) -> bool:
    if not scores:
        return False
    pred_units = pred_units_for_topk(scores, k, catalog)
    true_units = true_units_for_pred(true_phases, pred_units)
    return true_units.issubset(pred_units)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--galaxi-eval-dir", required=True)
    parser.add_argument("--xca", default=None)
    parser.add_argument("--autoanalyzer", default=None)
    parser.add_argument("--xqueryer", default=None)
    parser.add_argument("--psm", default=None)
    parser.add_argument("--max-k", type=int, default=15)
    parser.add_argument("--models-dir", default=str(REPO_ROOT / "examples" / "pretrained_catalog" / "pretrained_models"),
                         help="Reference catalog dir (models_<phase>/peak_list.json per phase) used to "
                              "build the peak-similarity grouping catalog.")
    parser.add_argument("--group-threshold", type=float, default=0.90,
                         help="group_similarity_threshold for build_phase_groups_from_peaks (g90 = 0.90, "
                              "the current production convention). Ignored if --no-grouping.")
    parser.add_argument("--no-grouping", action="store_true",
                         help="Disable peak-similarity grouping (each candidate its own singleton, still "
                              "clean_name-matched) -- matches production's 'none' grouping variant.")
    parser.add_argument("--output", default="topk_accuracy_results.json")
    args = parser.parse_args()

    galaxi_scores, true_phases_by_pattern = load_galaxi_scores(args.galaxi_eval_dir)

    baselines = {"GALAXI": galaxi_scores}
    for name, path in [("XCA", args.xca), ("AutoAnalyzer", args.autoanalyzer),
                        ("XQueryer", args.xqueryer), ("Peak_search_match", args.psm)]:
        if path:
            baselines[name] = load_baseline_scores(Path(path))

    if args.no_grouping:
        catalog = None
    else:
        print(f"Building phase-similarity catalog from {args.models_dir} (threshold={args.group_threshold})...")
        catalog = build_phase_groups_from_peaks(Path(args.models_dir), group_similarity_threshold=args.group_threshold)

    patterns = sorted(true_phases_by_pattern.keys())
    results = {}
    for model_name, scores_by_pattern in baselines.items():
        accuracies = {}
        for k in range(1, args.max_k + 1):
            hits = 0
            n_evaluated = 0
            for pattern in patterns:
                if pattern not in scores_by_pattern:
                    continue
                n_evaluated += 1
                if top_k_hit(scores_by_pattern[pattern], true_phases_by_pattern[pattern], k, catalog):
                    hits += 1
            accuracies[k] = hits / n_evaluated if n_evaluated else 0.0
        results[model_name] = accuracies
        print(f"{model_name}:")
        for k in (1, 5, 10, 15):
            if k in accuracies:
                print(f"  k={k:>2}: {accuracies[k]*100:.1f}%")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull per-k results written to {args.output}")
    print("\nManuscript target: GALAXI 67.2% @k~10, 82.0% @k~15;")
    print("XCA (strongest baseline) 39.3% @k~10; others 24.6-32.8% @k~10, baselines 32.8-47.5% @k~15")


if __name__ == "__main__":
    main()
