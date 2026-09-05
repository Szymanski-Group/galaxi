"""ModelEvaluator's config defaults.

`xrd_config` sets the angular grid every pattern is resampled onto, and
`model_config` carries the SNIP/gain-mapping preprocessing hyperparameters.
Both affect correctness without changing any shape, so omitting either must
produce a usable default and say so, rather than failing partway through a run.
"""

from galaxi.evaluation.model_evaluator import ModelEvaluator


def test_model_evaluator_defaults_model_config_when_not_passed(tmp_path, capsys):
    evaluator = ModelEvaluator(models_dir=str(tmp_path), output_dir=str(tmp_path / "out"))

    assert evaluator.model_config is not None
    assert hasattr(evaluator.model_config, "snip_iter")

    captured = capsys.readouterr()
    assert "model_config" in captured.out


def test_model_evaluator_defaults_xrd_config_when_not_passed(tmp_path, capsys):
    evaluator = ModelEvaluator(models_dir=str(tmp_path), output_dir=str(tmp_path / "out"))

    assert evaluator.xrd_config is not None
    assert hasattr(evaluator.xrd_config, "min_angle")

    captured = capsys.readouterr()
    assert "xrd_config" in captured.out


def test_model_evaluator_respects_explicit_model_config(tmp_path):
    from galaxi.core.config import ModelConfig

    custom = ModelConfig(snip_iter=99)
    evaluator = ModelEvaluator(
        models_dir=str(tmp_path), output_dir=str(tmp_path / "out"), model_config=custom
    )

    assert evaluator.model_config is custom
    assert evaluator.model_config.snip_iter == 99
