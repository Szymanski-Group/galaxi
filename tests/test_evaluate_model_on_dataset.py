"""evaluate_model_on_dataset() must feed the model real input, and must not
report metrics assembled out of swallowed exceptions.

The pattern loader returns (N, 2) [2theta, intensity] arrays, so every pattern is
regularized onto the model's own angular range and point count before inference,
including when the test set already happens to have that many points.

Individual unreadable patterns are scored as negatives, but a run in which every
prediction failed must raise rather than report metrics describing the error
path.
"""

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from galaxi.evaluation.model_evaluator import ModelEvaluator

REPO_ROOT = Path(__file__).resolve().parent.parent
CATALOG = REPO_ROOT / "examples" / "pretrained_catalog" / "pretrained_models"


def _a_pretrained_phase():
    if not CATALOG.is_dir():
        pytest.skip("pretrained catalog not present")
    for weights in sorted(CATALOG.glob("models_*/detection_model_*.pth")):
        return weights.parent.name.removeprefix("models_")
    pytest.skip("pretrained weights not fetched (see fetch_model_weights.py)")


def _write_xy(path: Path, n_points: int, min_angle: float, max_angle: float, seed: int):
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    two_theta = np.linspace(min_angle, max_angle, n_points)
    intensity = np.abs(rng.normal(size=n_points)) + 0.1
    np.savetxt(path, np.stack([two_theta, intensity], axis=1))


def test_predictions_are_real_not_swallowed_failures(tmp_path):
    """Inference must run on every pattern rather than falling back to a default."""
    phase = _a_pretrained_phase()

    evaluator = ModelEvaluator(models_dir=str(CATALOG), output_dir=str(tmp_path / "out"))
    model = evaluator.import_model_from_path(evaluator.available_models[phase], phase)
    cfg = model.config

    # Nested layout, matching the comprehensive test sets, at exactly the
    # model's own point count.
    rel_paths = {
        "positive/single_phase/strain/strain_0000.xy": True,
        "negative/negative_2_phase/mix_0000.xy": False,
    }
    ground_truth = {}
    for i, (rel, present) in enumerate(rel_paths.items()):
        _write_xy(tmp_path / rel, cfg.input_size, cfg.min_angle, cfg.max_angle, seed=i)
        ground_truth[rel] = {"target_present": present, "target_phase": phase}

    gt_file = tmp_path / "comprehensive_ground_truth.json"
    gt_file.write_text(json.dumps(ground_truth))

    metrics = evaluator.evaluate_model_on_dataset(phase, str(tmp_path), str(gt_file))

    # If every predict() had raised, the handler would have produced
    # probabilities of exactly 0.0 for every pattern and an AUC of 0.5.
    assert metrics.auc_score is None or not np.isnan(metrics.auc_score)
    cm = np.asarray(metrics.confusion_matrix)
    assert cm.sum() == len(rel_paths), "not every pattern was scored"


def test_all_predictions_failing_raises(tmp_path, monkeypatch):
    """Metrics must never describe the exception handler."""
    phase = _a_pretrained_phase()

    evaluator = ModelEvaluator(models_dir=str(CATALOG), output_dir=str(tmp_path / "out"))
    model = evaluator.import_model_from_path(evaluator.available_models[phase], phase)
    cfg = model.config

    rel = "positive/single_phase/strain/strain_0000.xy"
    _write_xy(tmp_path / rel, cfg.input_size, cfg.min_angle, cfg.max_angle, seed=0)
    gt_file = tmp_path / "comprehensive_ground_truth.json"
    gt_file.write_text(json.dumps({rel: {"target_present": True, "target_phase": phase}}))

    class _Exploding:
        config = cfg

        def predict(self, *_args, **_kwargs):
            raise RuntimeError("simulated inference failure")

    monkeypatch.setattr(
        ModelEvaluator, "import_model_from_path",
        lambda self, path, name: _Exploding(),
    )

    with pytest.raises(RuntimeError, match="Every one of the"):
        evaluator.evaluate_model_on_dataset(phase, str(tmp_path), str(gt_file))
