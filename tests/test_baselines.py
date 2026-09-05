"""Tests for the baseline phase-ID methods in reproduction/baselines/ (Fig.
3a/b of the manuscript). Lives outside the installed galaxi package (see
reproduction/README.md), so tests add reproduction/ to sys.path themselves,
mirroring how a user of reproduction/ would.
"""

import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
REPRODUCTION_DIR = REPO_ROOT / "reproduction"
if str(REPRODUCTION_DIR) not in sys.path:
    sys.path.insert(0, str(REPRODUCTION_DIR))

from baselines import match_phases  # noqa: E402

EXPERIMENTAL_PATTERN = (
    REPO_ROOT / "examples" / "pretrained_catalog" / "experimental_patterns" / "sample_displacement" / "Fe3O4_z_1.0.xy"
)
REFERENCE_CIF_DIR = REPO_ROOT / "examples" / "pretrained_catalog" / "reference_cifs"


def _strip_two_line_header(src: Path) -> str:
    lines = src.read_text().splitlines(keepends=True)
    with tempfile.NamedTemporaryFile(suffix=".xy", mode="w", delete=False) as f:
        f.writelines(lines[2:])
        return f.name


@pytest.mark.skipif(
    not EXPERIMENTAL_PATTERN.exists() or not REFERENCE_CIF_DIR.exists(),
    reason="pretrained-catalog experimental patterns/reference CIFs not present",
)
def test_match_phases_ranks_correct_fe_oxide_phase_highly():
    cleaned = _strip_two_line_header(EXPERIMENTAL_PATTERN)
    references = sorted(REFERENCE_CIF_DIR.glob("*.cif"))[:20]

    results = match_phases(cleaned, references, top_matches=5)

    assert len(results) > 0
    top_names = [r.reference_name for r in results]
    assert any(name.startswith("Fe") for name in top_names), (
        f"Expected an Fe-containing phase near the top for an Fe3O4 pattern, got {top_names}"
    )


def test_baseline_adapters_import_without_optional_heavy_deps():
    """xca/autoanalyzer/xqueryer each depend on an external ML framework and
    upstream repo, imported lazily inside functions -- importing the module
    itself must not require tensorflow/torch/the upstream repos."""
    import baselines.autoanalyzer  # noqa: F401
    import baselines.fetch_baseline_weights  # noqa: F401
    import baselines.xca  # noqa: F401
    import baselines.xqueryer  # noqa: F401
