"""DetectionCNN must accept any conv kernel size when use_mask=True.

The signal path runs conv -> maxpool while the mask path runs maxpool only, and
Conv1d is built with padding=(kernel_size-1)//2 -- length-preserving for odd
kernels, one sample shorter per layer for even ones. The forward pass re-aligns
the two so SpatialAttentionMask can combine them either way.

Both the pretrained catalog's architecture and whatever create_default_config()
currently ships are covered here.
"""

import pytest

torch = pytest.importorskip("torch")

from galaxi.core.config import ModelConfig
from galaxi.detection.detection_model import DetectionCNN


def _forward(input_size, conv_kernels, pool_size, use_mask):
    model = DetectionCNN(
        input_size=input_size,
        conv_channels=[8, 16, 32],
        conv_kernels=conv_kernels,
        pool_size=pool_size,
        fc_sizes=[16],
        use_mask=use_mask,
    )
    model.eval()
    channels = 2 if use_mask else 1
    x = torch.rand(4, channels, input_size)
    with torch.no_grad():
        return model(x)


@pytest.mark.parametrize("conv_kernels", [[16, 12, 8], [4, 4, 4], [16, 15, 8]])
def test_even_kernels_forward_with_mask(conv_kernels):
    out = _forward(4501, conv_kernels, [2, 2, 2], use_mask=True)
    assert out.shape == (4,)
    assert torch.isfinite(out).all()


def test_released_architecture_still_works():
    """The frozen (27, 15, 11) / 7001 architecture behind every shipped model."""
    out = _forward(7001, [27, 15, 11], [2, 2, 1], use_mask=True)
    assert out.shape == (4,)
    assert torch.isfinite(out).all()


def test_default_config_architecture_forward():
    """Whatever create_default_config() ships must actually run."""
    cfg = ModelConfig()
    out = _forward(cfg.input_size, cfg.conv_kernels, cfg.pool_size, use_mask=cfg.use_mask)
    assert out.shape == (4,)
    assert torch.isfinite(out).all()
