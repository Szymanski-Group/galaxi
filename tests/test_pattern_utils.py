"""Tests for core pattern preprocessing utilities."""

import numpy as np
import pytest

from galaxi.core.pattern_utils import (
    convert_q_to_two_theta,
    convert_two_theta_to_q,
    normalize_pattern,
    resample_pattern,
)


def test_two_theta_q_round_trip():
    two_theta = np.linspace(10.0, 80.0, 50)
    q = convert_two_theta_to_q(two_theta)
    round_tripped = convert_q_to_two_theta(q)
    np.testing.assert_allclose(round_tripped, two_theta, atol=1e-6)


def test_normalize_pattern_max_method_scales_to_100():
    pattern = np.array([0.0, 5.0, 10.0, 2.5])
    normalized = normalize_pattern(pattern, method="max")
    assert normalized.max() == pytest.approx(100.0)
    assert normalized.min() == pytest.approx(0.0)


def test_normalize_pattern_shifts_negative_baseline_to_zero():
    pattern = np.array([-5.0, 0.0, 5.0])
    normalized = normalize_pattern(pattern, method="max")
    assert normalized.min() == pytest.approx(0.0)


def test_resample_pattern_returns_requested_number_of_points():
    intensity = np.sin(np.linspace(0, 4 * np.pi, 200)) + 2
    resampled = resample_pattern(intensity, target_num_points=500)
    assert len(resampled) == 500


def test_resample_pattern_preserves_endpoints_roughly():
    two_theta = np.linspace(10.0, 80.0, 200)
    intensity = np.cos(two_theta / 10.0) + 2
    resampled = resample_pattern((two_theta, intensity), target_num_points=1000)
    assert len(resampled) == 1000
    # Endpoints should match the source pattern's endpoint values closely.
    assert resampled[0] == pytest.approx(intensity[0], abs=0.1)
    assert resampled[-1] == pytest.approx(intensity[-1], abs=0.1)
