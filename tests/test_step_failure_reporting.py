"""A step that accomplished nothing must not report success.

step_1/step_2/step_4 catch per-phase exceptions and continue, which is right for
a many-phase run where one bad CIF should not abort everything. Their summary
lines must still distinguish "N of N succeeded" from "0 of N succeeded", name
the phases that failed, and -- for training -- raise when every attempted phase
failed, since that is systemic rather than partial failure.
"""

import pytest

from galaxi.workflows.streamlined_workflow import StreamlinedWorkflow


def report(succeeded, attempted, failures, capsys):
    StreamlinedWorkflow._report_step_outcome("Models", succeeded, attempted, failures)
    return capsys.readouterr().out


def test_total_failure_is_not_reported_as_success(capsys):
    out = report([], ["a", "b"], [("a", "boom"), ("b", "boom")], capsys)

    assert "✓" not in out, "printed a success glyph for zero successes"
    assert "✗" in out
    assert "all 2 phase(s) failed" in out


def test_failures_are_named_not_just_counted(capsys):
    out = report(["a"], ["a", "b"], [("b", "RuntimeError: kaboom")], capsys)

    assert "b" in out and "kaboom" in out, "failed phase and cause must appear in the summary"


def test_full_success_reports_success_unchanged(capsys):
    out = report(["a", "b"], ["a", "b"], [], capsys)

    assert "✓ Models available for 2 phases total" in out
    assert "✗" not in out
    assert "failed" not in out


def test_partial_success_reports_both(capsys):
    out = report(["a"], ["a", "b"], [("b", "boom")], capsys)

    assert "✓ Models available for 1 phases total" in out
    assert "(1 failed)" in out


class _Workflow:
    """Exercise step_2's raise/no-raise decision without a real training run."""

    def __init__(self, existing, missing):
        self._existing = existing
        self._missing = missing

    def _check_existing_models(self, phases):
        return list(self._existing), list(self._missing)


def _run_step_2(existing, missing, monkeypatch, tmp_path):
    wf = object.__new__(StreamlinedWorkflow)
    wf.__dict__.update(
        seed=0,
        device="cpu",
        output_dir=tmp_path,
        references_dir=str(tmp_path),
        model_config=None,
        config={},
        cache_training_data=False,
        use_ensemble=False,
        generation_param_tuning=False,
        results={},
    )
    monkeypatch.setattr(
        StreamlinedWorkflow, "_check_existing_models",
        lambda self, phases: (list(existing), list(missing)),
    )
    monkeypatch.setattr(StreamlinedWorkflow, "set_random_seeds", lambda self, seed: None)
    monkeypatch.setattr(StreamlinedWorkflow, "handle_model_output_dir", lambda self, phase: None)
    return wf


def test_raises_when_every_attempted_phase_fails(monkeypatch, tmp_path):
    """Zero successes out of a non-empty attempt list is systemic, not partial."""
    wf = _run_step_2(existing=[], missing=["a", "b"], monkeypatch=monkeypatch, tmp_path=tmp_path)

    with pytest.raises(RuntimeError) as excinfo:
        wf.step_2_train_models(["a", "b"])

    assert "All 2 phase(s) failed to train" in str(excinfo.value)


def test_does_not_raise_when_nothing_needed_training(monkeypatch, tmp_path):
    """A fully-resumed run legitimately trains zero new models."""
    wf = _run_step_2(existing=["a"], missing=[], monkeypatch=monkeypatch, tmp_path=tmp_path)

    assert wf.step_2_train_models(["a"]) == ["a"]
