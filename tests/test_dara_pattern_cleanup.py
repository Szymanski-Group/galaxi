"""Leading zero-intensity runs are stripped before refinement.

Some experimental patterns begin scanning well before the first real peak,
leaving a long run of exactly-zero intensity. BGMN wastes computation, or fails,
trying to fit that flat region. DARA reads the pattern from disk rather than
from an in-memory array, so the cleaned pattern has to be written to a real file
and that path handed over.
"""

import numpy as np
import pytest

from galaxi.evaluation.model_evaluator import _strip_leading_zero_intensity, run_dara_refinement
from galaxi.core.config import DaraConfig


def _write_pattern(path, two_theta, intensity):
    data = np.stack([two_theta, intensity], axis=1)
    np.savetxt(path, data, header="2theta intensity")


def test_strips_leading_zero_run(tmp_path):
    pattern_path = tmp_path / "pattern.xy"
    two_theta = np.linspace(0.0, 10.0, 11)
    intensity = np.array([0, 0, 0, 0, 5, 10, 3, 0, 8, 0, 1], dtype=float)
    _write_pattern(pattern_path, two_theta, intensity)

    cleaned_path, cleanup_dir = _strip_leading_zero_intensity(pattern_path)

    assert cleanup_dir is not None
    assert cleaned_path != pattern_path
    cleaned = np.loadtxt(cleaned_path, skiprows=1)
    assert cleaned[0, 1] == 5  # first nonzero intensity
    assert len(cleaned) == 7  # original 11 points minus the 4 leading zeros
    # interior/trailing zeros must be preserved -- only the leading run is stripped
    assert 0 in cleaned[:, 1]

    cleanup_dir.cleanup()
    assert not cleaned_path.exists()


def test_no_leading_zeros_returns_original_path_unchanged(tmp_path):
    pattern_path = tmp_path / "pattern.xy"
    two_theta = np.linspace(1.0, 5.0, 5)
    intensity = np.array([3, 0, 8, 0, 1], dtype=float)
    _write_pattern(pattern_path, two_theta, intensity)

    result_path, cleanup_dir = _strip_leading_zero_intensity(pattern_path)

    assert result_path == pattern_path
    assert cleanup_dir is None


def test_all_zero_pattern_returns_original_path_unchanged(tmp_path):
    pattern_path = tmp_path / "pattern.xy"
    two_theta = np.linspace(1.0, 5.0, 5)
    intensity = np.zeros(5)
    _write_pattern(pattern_path, two_theta, intensity)

    result_path, cleanup_dir = _strip_leading_zero_intensity(pattern_path)

    assert result_path == pattern_path
    assert cleanup_dir is None


def test_run_dara_refinement_uses_cleaned_pattern_and_cleans_up(tmp_path, monkeypatch):
    pattern_path = tmp_path / "pattern.xy"
    two_theta = np.linspace(0.0, 10.0, 11)
    intensity = np.array([0, 0, 0, 0, 5, 10, 3, 0, 8, 0, 1], dtype=float)
    _write_pattern(pattern_path, two_theta, intensity)

    captured = {}

    class FakeSearchTree:
        def get_search_results(self):
            return []

    def fake_search_phases(**kwargs):
        captured["pattern_path"] = kwargs["pattern_path"]
        assert kwargs["pattern_path"].exists()
        return FakeSearchTree()

    import galaxi.evaluation.model_evaluator as model_evaluator_module

    monkeypatch.setattr(model_evaluator_module, "search_phases", fake_search_phases)

    result = run_dara_refinement(
        pattern_path=pattern_path,
        sorted_candidates=[],
        pinned_phases=[],
        dara_config=DaraConfig(),
    )

    assert result == (None, None, None, None)
    assert captured["pattern_path"] != pattern_path
    assert captured["pattern_path"].name == pattern_path.name
    assert not captured["pattern_path"].exists(), "temp pattern file must be cleaned up"
