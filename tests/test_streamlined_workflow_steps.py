"""StreamlinedWorkflow's step numbering and internal call signatures.

The common three-step path -- generate training data, train models, evaluate on
experimental patterns -- is step_1/step_2/step_3, with synthetic test-set
generation and evaluation as step_4/step_5.

These tests pin that numbering and check that run_complete_workflow() calls each
step with the arguments it actually accepts, since a signature mismatch there
would only surface partway through a full workflow run.
"""

import inspect

from galaxi.workflows.streamlined_workflow import StreamlinedWorkflow, create_default_config


def test_renamed_step_methods_exist_with_expected_signatures():
    assert hasattr(StreamlinedWorkflow, "step_1_generate_training_data")
    assert hasattr(StreamlinedWorkflow, "step_2_train_models")
    assert hasattr(StreamlinedWorkflow, "step_3_evaluate_experimental_patterns")
    assert hasattr(StreamlinedWorkflow, "step_4_generate_comprehensive_test_data")
    assert hasattr(StreamlinedWorkflow, "step_5_evaluate_models")

    sig = inspect.signature(StreamlinedWorkflow.step_3_evaluate_experimental_patterns)
    assert list(sig.parameters) == ["self"], "step_3 should take no arguments besides self"


def test_old_step_names_no_longer_exist():
    assert not hasattr(StreamlinedWorkflow, "step_3_generate_comprehensive_test_data")
    assert not hasattr(StreamlinedWorkflow, "step_4_evaluate_models")
    assert not hasattr(StreamlinedWorkflow, "step_5_evaluate_experimental_patterns")


def test_run_complete_workflow_calls_steps_with_correct_arity(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = create_default_config()
    workflow = StreamlinedWorkflow(config=config)

    calls = []
    monkeypatch.setattr(workflow, "step_1_generate_training_data", lambda: ["PhaseA"])
    monkeypatch.setattr(workflow, "step_2_train_models", lambda phases: ["PhaseA"])
    monkeypatch.setattr(workflow, "step_3_evaluate_experimental_patterns", lambda: calls.append(("step_3", None)))
    monkeypatch.setattr(workflow, "step_4_generate_comprehensive_test_data", lambda phases: calls.append(("step_4", phases)))
    monkeypatch.setattr(workflow, "step_5_evaluate_models", lambda phases: calls.append(("step_5", phases)))
    monkeypatch.setattr(workflow, "_save_workflow_results", lambda: None)

    # step_3 takes no arguments, unlike step_4/step_5 which take the phase
    # list; run_complete_workflow() must call each with the right shape.
    workflow.run_complete_workflow()

    assert calls == [
        ("step_3", None),
        ("step_4", ["PhaseA"]),
        ("step_5", ["PhaseA"]),
    ]
