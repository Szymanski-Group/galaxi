from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

from .io import write_peak_csv
from .models import Peak, WAVELENGTH


def load_xy_file(xy_file: Path | str) -> Tuple[np.ndarray, np.ndarray]:
    """Load an XY pattern (2θ, intensity)."""
    data = np.loadtxt(xy_file)
    return data[:, 0], data[:, 1]


def extract_peaks(
    two_theta: np.ndarray,
    intensity: np.ndarray,
    *,
    wavelength: float = WAVELENGTH,
    min_prominence: float | None = None,
    min_height: float | None = None,
    min_distance: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Peak pick using scipy.signal.find_peaks (same defaults as the script)."""
    if min_prominence is None:
        min_prominence = 0.05 * np.max(intensity)
    if min_height is None:
        min_height = 0.02 * np.max(intensity)

    peak_indices, _ = find_peaks(
        intensity,
        prominence=min_prominence,
        height=min_height,
        distance=min_distance,
    )

    peak_two_theta = two_theta[peak_indices]
    peak_intensities = intensity[peak_indices]

    theta_radians = np.radians(peak_two_theta / 2)
    d_spacings = wavelength / (2 * np.sin(theta_radians))

    peak_intensities_normalized = (peak_intensities / np.max(peak_intensities)) * 100
    return d_spacings, peak_intensities_normalized, peak_indices, peak_two_theta


def extract_peaks_from_xy(
    xy_file: Path | str,
    *,
    save_csv_to: Path | None = None,
    plot_path: Path | None = None,
    **peak_kwargs,
) -> List[Peak]:
    """
    Convenience wrapper: read XY, pick peaks, optionally write CSV/plot.

    Returns a list of Peak objects sorted by 2θ.
    """
    two_theta, intensity = load_xy_file(xy_file)
    d_spacings, peak_intensities, peak_indices, peak_two_theta = extract_peaks(
        two_theta,
        intensity,
        **peak_kwargs,
    )
    peaks = [
        Peak(float(d), float(i)) for d, i in sorted(zip(d_spacings, peak_intensities))
    ]

    if save_csv_to is not None:
        write_peak_csv(peaks, save_csv_to)

    if plot_path is not None:
        plot_pattern_with_peaks(two_theta, intensity, peak_indices, peak_two_theta, plot_path)

    return peaks


def plot_pattern_with_peaks(
    two_theta: np.ndarray,
    intensity: np.ndarray,
    peak_indices: np.ndarray,
    peak_two_theta: np.ndarray,
    output_path: Path | str,
) -> None:
    """Plot pattern with peaks highlighted."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(two_theta, intensity, "b-", linewidth=1, label="XRD Pattern")
    peak_intensities = intensity[peak_indices]
    plt.vlines(
        peak_two_theta,
        0,
        peak_intensities,
        colors="r",
        linestyles="--",
        linewidth=1.3,
        alpha=0.7,
        label="Detected Peaks",
    )
    plt.plot(
        peak_two_theta,
        peak_intensities,
        "r^",
        markersize=7,
        markerfacecolor="red",
        markeredgecolor="darkred",
        markeredgewidth=1,
    )
    plt.xlabel("2θ (degrees)")
    plt.ylabel("Intensity (a.u.)")
    plt.title("XRD Pattern with Peak Detection")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3, linestyle=":")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
