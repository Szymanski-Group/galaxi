"""Shared helper for the raw-CNN-probability artifact sweeps (impurity,
uniform peak shift, sample displacement, texture): load a pretrained
detection model and score one experimental .xy pattern against it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def load_model(models_dir: Path, phase: str, use_gpu: bool = False):
    from galaxi.detection.detection_model import PhaseDetectionModel

    model = PhaseDetectionModel(target_phase=phase, use_gpu=use_gpu)
    model.load_model(str(models_dir / f"models_{phase}" / f"detection_model_{phase}.pth"))
    return model


def predict_pattern(model, pattern_path: Path, header_lines: int = 2) -> float:
    from galaxi.core.pattern_utils import regularize_input

    data = np.loadtxt(pattern_path, skiprows=header_lines)
    pattern = np.stack([data[:, 0], data[:, 1]], axis=1)

    intensity = regularize_input(
        str(pattern_path), pattern,
        min_angle=model.config.min_angle, max_angle=model.config.max_angle,
        target_length=model.config.input_size, use_mask=model.config.use_mask,
        model_config=model.config,
    )
    return float(model.predict(intensity))
