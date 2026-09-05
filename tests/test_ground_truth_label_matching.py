"""Comprehensive test-set labels must survive the nested directory layout.

comprehensive_test_generator keys its ground truth on the path relative to the
test directory (positive/single_phase/strain/strain_0000.xy), so the pattern
loader reports filenames in the same form. Flat directories -- which is what
evaluate_experimental_patterns() reads -- still yield bare basenames.

A ground truth that matches no pattern at all is a wiring error, and must raise
rather than silently label every pattern class 0.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from galaxi.evaluation.model_evaluator import ModelEvaluator

PHASE = "Li2MnF5_15_1530560"


def _write_pattern(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    two_theta = np.linspace(10.0, 80.0, 64)
    np.savetxt(path, np.stack([two_theta, np.abs(np.sin(two_theta))], axis=1))


@pytest.fixture
def nested_test_set(tmp_path):
    """A miniature of the real comprehensive layout: nested, both classes."""
    rel_paths = {
        "positive/single_phase/strain/strain_0000.xy": True,
        "positive/multi_phase/texture/texture_0000.xy": True,
        "negative/negative_2_phase/mix_0000.xy": False,
        "negative/negative_unphysical_perturbations/perturb_0000.xy": False,
    }
    ground_truth = {}
    for rel, present in rel_paths.items():
        _write_pattern(tmp_path / rel)
        ground_truth[rel] = {
            "target_present": present,
            "target_phase": PHASE,
            "pattern_type": "single_phase" if present else "negative",
        }
    (tmp_path / "comprehensive_ground_truth.json").write_text(json.dumps(ground_truth))
    return tmp_path, ground_truth


def test_loader_returns_paths_relative_to_search_root(nested_test_set):
    test_dir, ground_truth = nested_test_set
    evaluator = ModelEvaluator(models_dir=str(test_dir), output_dir=str(test_dir / "out"))

    _, filenames = evaluator.load_patterns_from_directory(str(test_dir))

    assert set(filenames) == set(ground_truth), (
        "loaded filenames must use the same convention as the ground-truth keys"
    )


def test_both_classes_are_recovered(nested_test_set):
    test_dir, ground_truth = nested_test_set
    evaluator = ModelEvaluator(models_dir=str(test_dir), output_dir=str(test_dir / "out"))

    _, filenames = evaluator.load_patterns_from_directory(str(test_dir))
    labels = evaluator.extract_labels_from_ground_truth(ground_truth, filenames, PHASE)

    assert sorted(labels) == [0, 0, 1, 1], f"expected both classes, got {labels}"
    for name, label in zip(filenames, labels):
        assert label == int(ground_truth[name]["target_present"])


def test_flat_directory_filenames_are_unchanged(tmp_path):
    """evaluate_experimental_patterns' directories are flat, and its filenames
    carry the ground truth -- relative paths must equal bare basenames there."""
    for name in ("Fe2O3_167.xy", "Fe3O4_227_MnCO3_167.xy"):
        _write_pattern(tmp_path / name)

    evaluator = ModelEvaluator(models_dir=str(tmp_path), output_dir=str(tmp_path / "out"))
    _, filenames = evaluator.load_patterns_from_directory(str(tmp_path))

    assert sorted(filenames) == ["Fe2O3_167.xy", "Fe3O4_227_MnCO3_167.xy"]


def test_completely_unmatched_ground_truth_raises(nested_test_set):
    """A ground truth matching no pattern must raise, not label everything 0."""
    test_dir, _ = nested_test_set
    evaluator = ModelEvaluator(models_dir=str(test_dir), output_dir=str(test_dir / "out"))
    _, filenames = evaluator.load_patterns_from_directory(str(test_dir))

    bogus = {"some/other/convention/pattern_0000.xy": {"target_present": True,
                                                       "target_phase": PHASE}}

    with pytest.raises(ValueError, match="matched none of the"):
        evaluator.extract_labels_from_ground_truth(bogus, filenames, PHASE)
