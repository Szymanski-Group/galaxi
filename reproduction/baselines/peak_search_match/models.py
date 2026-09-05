from __future__ import annotations

import math
from dataclasses import dataclass

# Cu K-alpha wavelength in Angstrom (used across the original scripts)
WAVELENGTH = 1.54184


@dataclass
class Peak:
    """Simple peak container."""

    d_spacing: float
    intensity: float

    @property
    def two_theta(self) -> float:
        """Convert d-spacing to 2θ (degrees)."""
        argument = WAVELENGTH / (2 * self.d_spacing)
        if argument > 1.0:
            return float("nan")
        theta = math.asin(argument)
        return math.degrees(2 * theta)


def d_from_two_theta(two_theta_deg: float) -> float:
    """Convert 2θ (deg) to d-spacing (Angstrom)."""
    theta_rad = math.radians(two_theta_deg / 2)
    denom = 2 * math.sin(theta_rad)
    return WAVELENGTH / denom if denom > 0 else float("inf")
