from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from .fom import FoMResult, rank_references
from .io import read_peak_csv, write_peak_csv
from .models import Peak
from .patterns import pattern_from_cif
from .peaks import extract_peaks_from_xy

DEFAULT_FOM_KWARGS = dict(
    top_n=20,
    top_n_ref=50,
    min_matches=6,
    strong_hits_required=2,
    base_tolerance_d_abs=0.0005,
    rel_tolerance_d_frac=0.0020,
    zero_shift_range_deg=0.1,
    zero_shift_step_deg=0.01,
)

DEFAULT_FALLBACK_KWARGS = dict(
    top_n=25,
    top_n_ref=80,
    min_matches=4,
    strong_hits_required=1,
    base_tolerance_d_abs=0.0010,
    rel_tolerance_d_frac=0.0030,
    zero_shift_range_deg=0.3,
    zero_shift_step_deg=0.02,
)


def match_phases(
    experimental_pattern: str | Path,
    references: str | Path | Sequence[str | Path],
    *,
    top_matches: int = 5,
    theta_range: tuple[float, float] = (10, 80),
    cache_dir: str | Path | None = None,
    write_rankings_to: str | Path | None = None,
    save_extracted_peaks: str | Path | None = None,
    plot_peaks_to: str | Path | None = None,
    peak_kwargs: dict | None = None,
    fom_kwargs: dict | None = None,
    fallback_kwargs: dict | None = DEFAULT_FALLBACK_KWARGS,
) -> List[FoMResult]:
    """
    End-to-end search/match helper.

    Pass an experimental pattern (XY or d/I CSV) and reference structures
    (directory of CIF or CSV, or a list of files). Returns ranked FoM results.
    """
    peak_kwargs = peak_kwargs or {}
    fom_kwargs = {**DEFAULT_FOM_KWARGS, **(fom_kwargs or {})}

    exp_peaks = _load_experimental_peaks(
        experimental_pattern,
        cache_dir=cache_dir,
        save_as=save_extracted_peaks,
        plot_path=plot_peaks_to,
        peak_kwargs=peak_kwargs,
    )

    ref_map = _load_reference_peaks(references, theta_range=theta_range, cache_dir=cache_dir)

    results = rank_references(
        exp_peaks,
        ref_map,
        top_n_matches=top_matches,
        fallback_kwargs=fallback_kwargs,
        **fom_kwargs,
    )

    if write_rankings_to:
        _write_rankings(Path(write_rankings_to), Path(experimental_pattern).stem, results)

    return results


def _load_experimental_peaks(
    pattern: str | Path,
    *,
    cache_dir: str | Path | None,
    save_as: str | Path | None,
    plot_path: str | Path | None,
    peak_kwargs: dict,
) -> List[Peak]:
    path = Path(pattern)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        peaks = read_peak_csv(path)
    elif suffix == ".xy":
        peaks = extract_peaks_from_xy(
            path,
            save_csv_to=_maybe_output_path(path, cache_dir, save_as, ".csv"),
            plot_path=_maybe_output_path(path, cache_dir, plot_path, ".png"),
            **peak_kwargs,
        )
    else:
        raise ValueError(f"Unsupported experimental pattern type: {path.suffix}")

    if save_as and suffix == ".csv":
        write_peak_csv(peaks, Path(save_as))

    return peaks


def _load_reference_peaks(
    references: str | Path | Sequence[str | Path],
    *,
    theta_range: tuple[float, float],
    cache_dir: str | Path | None,
) -> Dict[str, List[Peak]]:
    ref_map: Dict[str, List[Peak]] = {}
    sources = _normalize_sources(references)

    for ref in sources:
        suffix = ref.suffix.lower()
        if suffix == ".csv":
            ref_map[ref.stem] = read_peak_csv(ref)
        elif suffix == ".cif":
            peaks = pattern_from_cif(ref, theta_range=theta_range)
            out_path = _maybe_output_path(ref, cache_dir, None, ".csv")
            if out_path:
                write_peak_csv(peaks, out_path)
            ref_map[ref.stem] = peaks

    return ref_map


def _normalize_sources(sources: str | Path | Sequence[str | Path]) -> List[Path]:
    if isinstance(sources, (str, Path)):
        sources = [sources]

    paths: List[Path] = []
    for src in sources:
        path = Path(src)
        if path.is_dir():
            paths.extend(sorted(path.glob("*.csv")))
            paths.extend(sorted(path.glob("*.cif")))
        else:
            paths.append(path)
    return paths


def _maybe_output_path(
    source: Path,
    cache_dir: str | Path | None,
    explicit: str | Path | None,
    ext: str,
) -> Path | None:
    if explicit is not None:
        return Path(explicit)
    if cache_dir is not None:
        return Path(cache_dir) / f"{source.stem}{ext}"
    return None


def _write_rankings(output_dir: Path, phase_name: str, results: Sequence[FoMResult]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{phase_name}.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "rank",
                "reference",
                "F_N",
                "matched_peaks",
                "unmatched_peaks",
                "avg_abs_delta_2theta_deg",
                "avg_weighted_delta_deg",
                "match_fraction",
                "best_shift_deg",
            ]
        )
        for idx, res in enumerate(results, start=1):
            writer.writerow(
                [
                    idx,
                    res.reference_name,
                    f"{res.fom:.4f}",
                    res.matched,
                    res.unmatched,
                    f"{res.avg_abs_delta:.4f}",
                    f"{res.avg_weighted_delta:.5f}",
                    f"{res.match_fraction:.3f}",
                    f"{res.shift_deg:.3f}",
                ]
            )
