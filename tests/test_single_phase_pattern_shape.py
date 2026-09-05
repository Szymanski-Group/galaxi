"""generate_single_phase_pattern() must return a single, unbatched pattern.

The underlying generator is batched, so this convenience wrapper drops the batch
axis: two_theta and intensity come back as 1-D arrays of equal length, ready to
plot directly, and the dilute-phase background is drawn per point rather than as
a single broadcast value.
"""

from pathlib import Path

import numpy as np
import pytest

from galaxi.core.config import XRDGenerationConfig
from galaxi.core.pattern_generator import UnifiedPatternGenerator

REPO_ROOT = Path(__file__).resolve().parent.parent
CIF_DIR = REPO_ROOT / "tutorials" / "data" / "cif_files"


@pytest.fixture(scope="module")
def generator():
    if not CIF_DIR.is_dir() or not any(CIF_DIR.glob("*.cif")):
        pytest.skip(f"tutorial CIFs not available at {CIF_DIR}")
    return UnifiedPatternGenerator(
        reference_dir=str(CIF_DIR),
        config=XRDGenerationConfig(min_angle=10.0, max_angle=80.0, num_points=4501),
    )


def _a_cif() -> str:
    return str(sorted(CIF_DIR.glob("*.cif"))[0])


@pytest.mark.parametrize("target_concentration", [1.0, 0.6])
def test_two_theta_and_intensity_have_matching_shape(generator, target_concentration):
    pattern = generator.generate_single_phase_pattern(
        cif_file=_a_cif(), target_concentration=target_concentration
    )
    two_theta = np.asarray(pattern["two_theta"])
    intensity = np.asarray(pattern["intensity"])

    assert intensity.ndim == 1, f"expected an unbatched pattern, got shape {intensity.shape}"
    assert two_theta.shape == intensity.shape


def test_dilute_background_is_per_point_not_a_broadcast_scalar(generator):
    """With target_concentration < 1 the added background must vary point to
    point rather than being one value repeated across the pattern."""
    pattern = generator.generate_single_phase_pattern(
        cif_file=_a_cif(), target_concentration=0.5
    )
    intensity = np.asarray(pattern["intensity"], dtype=float)

    assert intensity.size > 1000
    assert np.isfinite(intensity).all()
    # A single broadcast constant would leave the baseline perfectly flat.
    baseline = np.sort(intensity)[: intensity.size // 2]
    assert np.std(baseline) > 0, "background appears constant across the pattern"
