"""
Core galaxi functionality with unified interfaces.
"""

from .config import (
    XRDGenerationConfig,
    ModelConfig,
    DEFAULT_XRD_CONFIG,
    CONSERVATIVE_XRD_CONFIG,
    REALISTIC_XRD_CONFIG,
    DEFAULT_MODEL_CONFIG
)

from .structures import StructureManager

from .pattern_generator import UnifiedPatternGenerator

from .model_base import BaseModel

from .pattern_utils import resample_pattern, get_model_metadata, save_model_with_metadata, normalize_pattern

__all__ = [
    # Configuration
    'XRDGenerationConfig',
    'ModelConfig',
    'DEFAULT_XRD_CONFIG',
    'CONSERVATIVE_XRD_CONFIG',
    'REALISTIC_XRD_CONFIG',
    'DEFAULT_MODEL_CONFIG',

    # Core components
    'StructureManager',
    'UnifiedPatternGenerator',
    'BaseModel',

    # Pattern utilities
    'resample_pattern',
    'get_model_metadata',
    'save_model_with_metadata',
    'normalize_pattern'
]
