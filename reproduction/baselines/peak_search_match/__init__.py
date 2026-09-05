"""
Lightweight search-match helpers for powder XRD patterns.
"""

from .api import match_phases
from .fom import FoMResult, compute_smith_snyder_fom, rank_references, rank_reference_files
from .patterns import generate_library, pattern_from_cif
from .peaks import extract_peaks_from_xy, extract_peaks, load_xy_file, plot_pattern_with_peaks
from .models import Peak, WAVELENGTH

__all__ = [
    "match_phases",
    "FoMResult",
    "compute_smith_snyder_fom",
    "rank_references",
    "rank_reference_files",
    "generate_library",
    "pattern_from_cif",
    "extract_peaks_from_xy",
    "extract_peaks",
    "load_xy_file",
    "plot_pattern_with_peaks",
    "Peak",
    "WAVELENGTH",
]
