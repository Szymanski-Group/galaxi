"""Smoke tests for DetectionCNN's forward pass, with and without a mask
channel, at the frozen config's shapes."""

import torch

from galaxi.core.config import ModelConfig
from galaxi.detection.detection_model import DetectionCNN


def _build_model():
    config = ModelConfig()
    model = DetectionCNN(
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
    model.eval()
    return model, config


def test_forward_pass_without_mask_channel():
    model, config = _build_model()
    x = torch.randn(4, 1, config.input_size)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (4,)
    assert torch.all((out >= 0) & (out <= 1))


def test_forward_pass_with_mask_channel():
    model, config = _build_model()
    x = torch.randn(4, 2, config.input_size)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (4,)


def test_forward_pass_return_logit():
    model, config = _build_model()
    x = torch.randn(4, 1, config.input_size)
    with torch.no_grad():
        out = model(x, return_logit=True)
    assert out.shape == (4,)
