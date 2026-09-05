"""Configuration handling and file-loading safety.

Covers the guarantees that a misconfiguration or an untrusted input file must
fail loudly rather than quietly doing something else:

- every activation named in ModelConfig is implemented, and an unknown name
  raises instead of falling back to ReLU;
- mismatched conv_channels/conv_kernels/pool_size/dilation_size lengths raise
  rather than being truncated to the shortest;
- a ModelConfig shared between instances is copied, so one model's in-place
  config changes cannot leak into another;
- model weights load with `weights_only=True`;
- tar extraction rejects members whose paths escape the destination.
"""

import tarfile

import pytest
import torch
import torch.nn as nn

from galaxi.core.config import DEFAULT_MODEL_CONFIG, ModelConfig
from galaxi.core.model_base import BaseModel
from galaxi.detection.detection_model import DetectionCNN, PhaseDetectionModel


def _cnn_kwargs(**overrides):
    config = ModelConfig()
    kwargs = dict(
        input_size=config.input_size,
        conv_channels=config.conv_channels,
        conv_kernels=config.conv_kernels,
        pool_size=config.pool_size,
        fc_sizes=config.fc_sizes,
        dropout_rate=config.dropout_rate,
        activation=config.activation,
        use_batch_norm=config.use_batch_norm,
        use_mask=config.use_mask,
    )
    kwargs.update(overrides)
    return kwargs


@pytest.mark.parametrize("activation", ["relu", "leaky_relu", "elu", "gelu", "swish", "tanh", "sigmoid"])
def test_all_readme_advertised_activations_are_implemented(activation):
    model = DetectionCNN(**_cnn_kwargs(activation=activation))
    x = torch.randn(2, 1, ModelConfig().input_size)
    out = model(x)
    assert out.shape == (2,)


def test_unknown_activation_raises_instead_of_silently_using_relu():
    with pytest.raises(ValueError, match="Unknown activation"):
        DetectionCNN(**_cnn_kwargs(activation="not_a_real_activation"))


def test_mismatched_architecture_array_lengths_raise():
    with pytest.raises(ValueError, match="same length"):
        DetectionCNN(**_cnn_kwargs(conv_kernels=[27, 15]))


def test_non_positive_input_size_raises():
    with pytest.raises(ValueError, match="positive"):
        DetectionCNN(**_cnn_kwargs(input_size=0))


def test_default_model_config_singleton_is_not_aliased_across_instances():
    original_input_size = DEFAULT_MODEL_CONFIG.input_size

    model_a = PhaseDetectionModel(target_phase="Test", reference_dir=None, config=None)
    model_a.config.input_size = original_input_size + 999

    model_b = PhaseDetectionModel(target_phase="Test2", reference_dir=None, config=None)

    assert model_b.config.input_size == original_input_size
    assert DEFAULT_MODEL_CONFIG.input_size == original_input_size


def test_base_model_save_load_round_trips_via_safe_state_dict(tmp_path):
    class DummyModel(BaseModel):
        def train(self, **kwargs):
            pass

        def predict(self, pattern):
            pass

    m1 = DummyModel.__new__(DummyModel)
    m1.model = nn.Linear(4, 2)
    m1.config = None

    model_path = str(tmp_path / "dummy.pt")
    m1.save_model(model_path)

    m2 = DummyModel.__new__(DummyModel)
    m2.model = nn.Linear(4, 2)
    m2.load_model(model_path)

    for p1, p2 in zip(m1.model.parameters(), m2.model.parameters()):
        assert torch.allclose(p1, p2)


def test_base_model_load_model_refuses_unconstructed_model(tmp_path):
    class DummyModel(BaseModel):
        def train(self, **kwargs):
            pass

        def predict(self, pattern):
            pass

    m = DummyModel.__new__(DummyModel)
    m.model = nn.Linear(4, 2)
    m.config = None
    model_path = str(tmp_path / "dummy.pt")
    m.save_model(model_path)

    m2 = DummyModel.__new__(DummyModel)
    m2.model = None
    with pytest.raises(ValueError, match="must be constructed"):
        m2.load_model(model_path)


def test_tar_extraction_rejects_path_traversal_members(tmp_path):
    from galaxi.scripts.setup_cod import _safe_tar_members

    tar_path = tmp_path / "archive.tar.gz"
    good_src = tmp_path / "good.cif"
    good_src.write_bytes(b"legit cif content")
    evil_src = tmp_path / "evil.txt"
    evil_src.write_bytes(b"pwned")

    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(good_src, arcname="FilteredCIFs/good.cif")
        tar.add(evil_src, arcname="../../../etc/pwned.txt")

    extract_dir = tmp_path / "extracted"
    extract_dir.mkdir()

    with tarfile.open(tar_path, "r:gz") as tar:
        members = list(_safe_tar_members(tar, extract_dir))
        names = [m.name for m in members]
        assert "FilteredCIFs/good.cif" in names
        assert not any("pwned" in n for n in names)
