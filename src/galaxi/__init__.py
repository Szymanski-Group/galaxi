# GALAXI: General Automated Learning and Analysis for Xrd-based Phase Detection

# Core phase detection interface
from .core import (
    UnifiedPatternGenerator,
    XRDGenerationConfig,
    ModelConfig,
    DEFAULT_XRD_CONFIG,
    CONSERVATIVE_XRD_CONFIG,
    REALISTIC_XRD_CONFIG
)

# Phase detection functionality
from .detection import PhaseDetectionModel

# Test pattern generation and model evaluation
from .pattern_generation.comprehensive_test_generator import ComprehensiveTestGenerator
from .evaluation import ModelEvaluator, EvaluationMetrics

# Visualization
from .visualization import XRDVisualizer

# COD functionality
from .cod_query import CODQuery, query_chemical_system

# Alias for backwards compatibility
PatternGenerator = UnifiedPatternGenerator

__all__ = [
    # Core interface for phase detection
    "UnifiedPatternGenerator",
    "XRDGenerationConfig",
    "ModelConfig",
    "DEFAULT_XRD_CONFIG",
    "CONSERVATIVE_XRD_CONFIG",
    "REALISTIC_XRD_CONFIG",

    # Phase detection functionality
    "PhaseDetectionModel",

    # Test pattern generation and model evaluation
    "ComprehensiveTestGenerator",
    "ModelEvaluator",
    "EvaluationMetrics",

    # Visualization
    "XRDVisualizer",

    # Pattern generator alias
    "PatternGenerator",

    # COD functionality
    "CODQuery",
    "query_chemical_system"
]
