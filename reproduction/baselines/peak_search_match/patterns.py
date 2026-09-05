from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core import Structure

from .io import write_peak_csv
from .models import Peak, WAVELENGTH

TwoThetaRange = Tuple[float, float]


def pattern_from_cif(cif_file: Path | str, theta_range: TwoThetaRange = (10, 80)) -> List[Peak]:
    """Generate a d/I list from a CIF file using pymatgen."""
    cif_path = Path(cif_file)
    structure = Structure.from_file(str(cif_path))
    calculator = XRDCalculator()
    pattern = calculator.get_pattern(structure, two_theta_range=theta_range)
    two_thetas = pattern.x
    intensities = pattern.y
    d_spacings = WAVELENGTH / (2 * np.sin(np.radians(two_thetas / 2)))
    return [Peak(float(d), float(i)) for d, i in zip(d_spacings, intensities)]


def generate_library(
    cif_sources: Sequence[Path | str],
    output_dir: Path | str,
    theta_range: TwoThetaRange = (10, 80),
) -> List[Path]:
    """
    Build a d/I CSV library from CIF files.

    Returns a list of written CSV paths.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    for cif_file in _iter_cif_files(cif_sources):
        peaks = pattern_from_cif(cif_file, theta_range=theta_range)
        out_file = output_path / f"{Path(cif_file).stem}.csv"
        write_peak_csv(peaks, out_file)
        written.append(out_file)

    return written


def _iter_cif_files(sources: Iterable[Path | str]) -> Iterable[Path]:
    for src in sources:
        path = Path(src)
        if path.is_dir():
            yield from sorted(path.glob("*.cif"))
        elif path.suffix.lower() == ".cif":
            yield path
