"""XRDVisualizer must read the .xy files this package ships.

The experimental patterns under examples/pretrained_catalog/ carry a two-line
non-numeric header ("test" / "Wavelength = ..."), while GALAXI-generated
patterns have none. The visualizer detects this per-file, via
pattern_utils.count_header_lines, so both conventions load correctly.
"""

from pathlib import Path

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from galaxi.visualization import XRDVisualizer

REPO_ROOT = Path(__file__).resolve().parent.parent
PRISTINE_DIR = REPO_ROOT / "examples" / "pretrained_catalog" / "experimental_patterns" / "pristine"


def _a_shipped_pattern() -> Path:
    if not PRISTINE_DIR.is_dir():
        pytest.skip(f"shipped experimental patterns not found at {PRISTINE_DIR}")
    files = sorted(PRISTINE_DIR.glob("*.xy"))
    if not files:
        pytest.skip("no .xy patterns shipped")
    return files[0]


def test_loads_shipped_pattern_with_header():
    path = _a_shipped_pattern()
    two_theta, intensity = XRDVisualizer()._load_pattern_data(path)

    assert len(two_theta) == len(intensity) > 0
    # A real 2theta axis, not the header being misread as data.
    assert np.all(np.diff(two_theta) > 0), "2theta axis is not monotonically increasing"
    assert two_theta[0] < two_theta[-1]


def test_loads_headerless_pattern(tmp_path):
    """The header-detecting loader must not eat real rows from files that have
    no header -- the failure mode a hardcoded skiprows=2 would introduce."""
    src = _a_shipped_pattern()
    expected = np.loadtxt(src, skiprows=2)

    stripped = tmp_path / "no_header.xy"
    np.savetxt(stripped, expected)

    two_theta, intensity = XRDVisualizer()._load_pattern_data(stripped)

    assert len(two_theta) == len(expected)
    np.testing.assert_allclose(two_theta, expected[:, 0], rtol=1e-6)


def test_plot_multiple_files_on_shipped_patterns(tmp_path):
    """The README's own visualization example, against shipped data."""
    if not PRISTINE_DIR.is_dir():
        pytest.skip("shipped experimental patterns not found")
    files = sorted(PRISTINE_DIR.glob("*.xy"))[:2]
    if len(files) < 2:
        pytest.skip("need two shipped patterns")

    out = tmp_path / "comparison"
    XRDVisualizer(figsize=(10, 6)).plot_multiple_files(
        file_paths=[str(f) for f in files],
        labels=["Sample A", "Sample B"],
        save=True,
        filename=str(out),
    )
    assert out.with_suffix(".png").exists(), "no figure was written"
