from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List

from .models import Peak


def read_peak_csv(path: Path) -> List[Peak]:
    """Load peaks from a CSV with headers 'd-spacing (Angstrom),Intensity'."""
    peaks: List[Peak] = []
    with Path(path).open() as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        for row in reader:
            if len(row) < 2:
                continue
            try:
                d_val = float(row[0])
                i_val = float(row[1])
            except ValueError:
                continue
            peaks.append(Peak(d_val, i_val))
    return peaks


def write_peak_csv(peaks: Iterable[Peak], path: Path) -> None:
    """Write peaks to CSV using the standard d/I header."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["d-spacing (Angstrom)", "Intensity"])
        for peak in peaks:
            writer.writerow([f"{peak.d_spacing:.6f}", f"{peak.intensity:.2f}"])
