"""ModelConfig and DetectionCNN defaults must match the pretrained catalog.

The architecture defaults are declared in more than one place -- ModelConfig's
dataclass fields, DetectionCNN's constructor, and the README example -- and all
of them have to agree with the configs saved alongside the models in
examples/pretrained_catalog/, since those models are loaded with these defaults.
"""

import json
from pathlib import Path

import pytest

from galaxi.core.config import ModelConfig
from galaxi.detection.detection_model import DetectionCNN

REPO_ROOT = Path(__file__).resolve().parent.parent
PRETRAINED_CATALOG_CONFIGS = sorted(
    (REPO_ROOT / "examples" / "pretrained_catalog" / "pretrained_models").glob("*/*_config.json")
)


def test_model_config_defaults_match_frozen_architecture():
    config = ModelConfig()
    assert config.conv_channels == [16, 16, 16]
    assert config.conv_kernels == [27, 15, 11]
    assert config.pool_size == [2, 2, 1]
    assert config.fc_sizes == [16]
    assert config.dropout_rate == 0.0
    assert config.optimizer == "rmsprop"
    assert config.num_epochs == 200
    assert config.early_stopping_patience == 5
    assert config.input_size == 7001
    assert config.min_angle == 5.0
    assert config.max_angle == 105.0
    assert config.use_mask is True


def test_detection_cnn_defaults_accept_model_config_defaults_input():
    """DetectionCNN's own raw constructor defaults should be usable with
    an input shaped for ModelConfig's default input_size (they describe
    the same frozen architecture, so a forward pass at that size must not
    raise)."""
    import torch

    config = ModelConfig()
    model = DetectionCNN()
    model.eval()
    x = torch.randn(2, 1, config.input_size)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2,)


@pytest.mark.skipif(
    not PRETRAINED_CATALOG_CONFIGS, reason="pretrained-catalog model configs not present"
)
def test_model_config_defaults_match_shipped_pretrained_catalog_configs():
    """Cross-check ModelConfig's defaults directly against a real shipped
    model config, so this test fails loudly if the two ever diverge again.
    """
    config = ModelConfig()
    shipped = json.loads(PRETRAINED_CATALOG_CONFIGS[0].read_text())

    assert config.conv_channels == shipped["conv_channels"]
    assert config.conv_kernels == shipped["conv_kernels"]
    assert config.pool_size == shipped["pool_size"]
    assert config.fc_sizes == shipped["fc_sizes"]
    assert config.dropout_rate == shipped["dropout_rate"]
    assert config.optimizer == shipped["optimizer"]
    assert config.early_stopping_patience == shipped["early_stopping_patience"]
    assert config.input_size == shipped["input_size"]
    assert config.min_angle == shipped["min_angle"]
    assert config.max_angle == shipped["max_angle"]
    assert config.use_mask == shipped["use_mask"]


def test_load_model_reads_min_angle_max_angle_from_saved_config():
    """load_model() must carry the saved angular range onto self.config.

    Callers read model.config.min_angle/max_angle to build the grid they
    resample onto, so if those still held defaults after loading, every pattern
    would be regularized onto the wrong grid and predictions would collapse
    toward zero without raising."""
    if not PRETRAINED_CATALOG_CONFIGS:
        pytest.skip("pretrained-catalog model configs not present")

    from galaxi.detection.detection_model import PhaseDetectionModel

    config_path = PRETRAINED_CATALOG_CONFIGS[0]
    model_path = str(config_path).replace("_config.json", ".pth")
    if not Path(model_path).exists():
        pytest.skip("pretrained-catalog model weights not fetched")

    shipped = json.loads(config_path.read_text())
    model = PhaseDetectionModel(target_phase=shipped["target_phase"], use_gpu=False)
    model.load_model(model_path)

    assert model.config.min_angle == shipped["min_angle"]
    assert model.config.max_angle == shipped["max_angle"]
    assert model.config.input_size == shipped["input_size"]
