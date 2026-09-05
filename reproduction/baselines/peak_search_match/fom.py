from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from .io import read_peak_csv
from .models import Peak, d_from_two_theta

from pathlib import Path


def select_strongest_peaks(peaks: Sequence[Peak], limit: int) -> List[Peak]:
    """Return the strongest peaks up to a limit, preserving 2θ order."""
    strongest = sorted(peaks, key=lambda p: p.intensity, reverse=True)[:limit]
    strongest = [p for p in strongest if not math.isnan(p.two_theta)]
    return sorted(strongest, key=lambda p: p.two_theta)


def find_best_match_d(
    obs_d: float,
    ref_ds: Sequence[float],
    used: set[int],
    tol_d: float,
) -> Tuple[int | None, float | None]:
    """Find the closest unused reference peak in d-space within tolerance."""
    best_idx = None
    best_delta = None
    for idx, ref_d in enumerate(ref_ds):
        if idx in used:
            continue
        delta = abs(obs_d - ref_d)
        if delta <= tol_d and (best_delta is None or delta < best_delta):
            best_idx = idx
            best_delta = delta
    return best_idx, best_delta


@dataclass
class FoMResult:
    reference_name: str
    fom: float
    matched: int
    unmatched: int
    avg_abs_delta: float
    avg_weighted_delta: float
    match_fraction: float
    shift_deg: float


def compute_smith_snyder_fom(
    exp_peaks: Sequence[Peak],
    ref_peaks: Sequence[Peak],
    *,
    top_n: int = 20,
    top_n_ref: int = 50,
    min_matches: int = 6,
    strong_hits_required: int = 2,
    strong_ref_top_k: int = 5,
    base_tolerance_d_abs: float = 0.0005,
    rel_tolerance_d_frac: float = 0.0020,
    zero_shift_range_deg: float = 0.1,
    zero_shift_step_deg: float = 0.01,
) -> FoMResult:
    """Smith/Snyder-style figure of merit used in the original script."""
    exp_subset = select_strongest_peaks(exp_peaks, top_n)
    ref_subset = select_strongest_peaks(ref_peaks, top_n_ref)

    exp_two_theta = [p.two_theta for p in exp_subset]
    ref_two_theta = [p.two_theta for p in ref_subset]
    ref_intensity = [p.intensity for p in ref_subset]
    ref_d = [d_from_two_theta(tt) for tt in ref_two_theta]
    exp_intensity = [p.intensity for p in exp_subset]

    if exp_intensity:
        max_int = max(exp_intensity)
        exp_weights = [i / max_int if max_int > 0 else 0.0 for i in exp_intensity]
    else:
        exp_weights = []

    top3_indices_by_intensity = {
        exp_subset.index(p)
        for p in sorted(exp_subset, key=lambda p: p.intensity, reverse=True)[:3]
    }
    top_ref_indices_by_intensity = {
        ref_subset.index(p)
        for p in sorted(ref_subset, key=lambda p: p.intensity, reverse=True)[:strong_ref_top_k]
    }

    best_result: FoMResult | None = None
    shift = -zero_shift_range_deg

    while shift <= zero_shift_range_deg + 1e-9:
        used_ref = set()
        abs_deltas: List[float] = []
        weighted_deltas: List[float] = []
        weight_sum = 0.0
        matched_exp_indices: List[int] = []
        matched_ref_indices: List[int] = []

        for exp_idx, (obs_tt, w) in enumerate(zip(exp_two_theta, exp_weights)):
            shifted_tt = obs_tt + shift
            shifted_d = d_from_two_theta(shifted_tt)
            tol_d = max(base_tolerance_d_abs, rel_tolerance_d_frac * shifted_d)
            idx, delta_d = find_best_match_d(shifted_d, ref_d, used_ref, tol_d)
            if idx is not None and delta_d is not None:
                used_ref.add(idx)
                ref_tt = ref_two_theta[idx]
                delta_tt = abs(shifted_tt - ref_tt)
                abs_deltas.append(delta_tt)
                theta = math.radians(shifted_tt / 2)
                weighted_deltas.append(delta_tt * math.tan(theta) * (0.5 + 0.5 * w))
                weight_sum += (0.5 + 0.5 * w)
                matched_exp_indices.append(exp_idx)
                matched_ref_indices.append(idx)

        matched = len(abs_deltas)
        unmatched = len(exp_two_theta) - matched

        if matched < min_matches:
            shift += zero_shift_step_deg
            continue

        strong_hits = sum(1 for idx in matched_exp_indices if idx in top3_indices_by_intensity)
        if strong_hits < strong_hits_required:
            shift += zero_shift_step_deg
            continue

        avg_abs_delta = sum(abs_deltas) / matched
        avg_weighted_delta = (
            sum(weighted_deltas) / weight_sum if weight_sum > 0 else float("inf")
        )
        match_fraction_exp = matched / len(exp_two_theta) if exp_two_theta else 0.0
        match_fraction_ref = matched / len(ref_two_theta) if ref_two_theta else 0.0

        strong_ref_hits = sum(1 for idx in matched_ref_indices if idx in top_ref_indices_by_intensity)
        strong_ref_norm = strong_ref_hits / max(1, len(top_ref_indices_by_intensity))
        strong_exp_norm = strong_hits / max(1, len(top3_indices_by_intensity))

        quality = (match_fraction_exp * match_fraction_ref) ** 2
        quality *= (0.5 + 0.5 * strong_ref_norm)
        quality *= (0.5 + 0.5 * strong_exp_norm)
        fom = quality / (avg_weighted_delta + 1e-6) if avg_weighted_delta > 0 else 0.0

        candidate = FoMResult(
            reference_name="",
            fom=fom,
            matched=matched,
            unmatched=unmatched,
            avg_abs_delta=avg_abs_delta,
            avg_weighted_delta=avg_weighted_delta,
            match_fraction=match_fraction_exp,
            shift_deg=shift,
        )

        if best_result is None or candidate.fom > best_result.fom:
            best_result = candidate

        shift += zero_shift_step_deg

    if best_result is None:
        return FoMResult(
            reference_name="",
            fom=0.0,
            matched=0,
            unmatched=len(exp_two_theta),
            avg_abs_delta=float("inf"),
            avg_weighted_delta=float("inf"),
            match_fraction=0.0,
            shift_deg=0.0,
        )

    return best_result


def rank_references(
    exp_peaks: Sequence[Peak],
    reference_peaks: Dict[str, Sequence[Peak]],
    *,
    top_n_matches: int = 5,
    fallback_kwargs: dict | None = None,
    **fom_kwargs,
) -> List[FoMResult]:
    """Compute FoM against many references and return top hits."""
    rankings: List[FoMResult] = []

    for ref_name, ref_peaks in reference_peaks.items():
        primary = compute_smith_snyder_fom(exp_peaks, ref_peaks, **fom_kwargs)
        best = primary
        if fallback_kwargs:
            fallback = compute_smith_snyder_fom(exp_peaks, ref_peaks, **fallback_kwargs)
            if fallback.fom > best.fom:
                best = fallback
        best.reference_name = ref_name
        rankings.append(best)

    rankings.sort(key=lambda r: r.fom, reverse=True)
    return rankings[: min(top_n_matches, len(rankings))]


def rank_reference_files(
    exp_file: str | Path,
    ref_files: Sequence[str | Path],
    *,
    top_n_matches: int = 5,
    fallback_kwargs: dict | None = None,
    **fom_kwargs,
) -> List[FoMResult]:
    """Small helper when everything is already on disk as CSV peak lists."""
    exp_peaks = read_peak_csv(Path(exp_file))
    ref_map = {Path(p).stem: read_peak_csv(Path(p)) for p in ref_files}
    return rank_references(
        exp_peaks,
        ref_map,
        top_n_matches=top_n_matches,
        fallback_kwargs=fallback_kwargs,
        **fom_kwargs,
    )
