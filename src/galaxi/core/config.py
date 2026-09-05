"""
Centralized configuration for XRD pattern generation and model parameters.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


@dataclass
class XRDGenerationConfig:
    """Configuration for realistic XRD pattern generation."""
    min_angle: float = 10.0
    max_angle: float = 80.0
    num_points: int = 4501
    convert_to_q: bool = False

    # Peak shift and displacement
    uniform_shift_range: Tuple[float, float] = (-0.25, 0.25)
    sample_displacement: Tuple[float, float] = (-0.2, 0.2)
    goniometer_radius: float = 240.0

    # Crystallite and strain effects
    crystallite_size_log_scale: bool = False
    crystallite_size_range: Tuple[float, float] = (1.0, 100.0)
    microstrain_range: Tuple[float, float] = (0.0, 0.003)
    lattice_strain_range: Tuple[float, float] = (0.0, 0.04)

    # Instrumental effects
    instrumental_broadening: Dict[str, float] = None
    pseudo_voigt_eta_range: Tuple[float, float] = (0.3, 0.8)

    # Temperature and thermal effects
    temperature_range: Tuple[float, float] = (250, 350)
    atomic_displacement_range: Tuple[float, float] = (0.005, 0.02)

    # Texture and preferred orientation
    texture_range: Tuple[float, float] = (0.4, 1.6)
    weights_low_index: float = 0.7
    low_index_orientation: List[Tuple[int, int, int]] = None
    high_index_orientation: List[Tuple[int, int, int]] = None

    # Background and noise (as percentages)
    background_level: Tuple[float, float] = (0.5, 2.0)
    noise_level: Tuple[float, float] = (0, 0.5)

    # Diffuse scattering
    diffuse_scattering_intensity: Tuple[float, float] = (0.0, 25.0)
    diffuse_scattering_b_factor: Tuple[float, float] = (0.5, 3.0)

    # Amorphous contributions
    amorphous_intensity: Tuple[float, float] = (0.0, 25.0)
    amorphous_neighbor_distance: Tuple[float, float] = (2.0, 4.0)
    amorphous_disorder: Tuple[float, float] = (0.2, 0.8)

    # Impurity peaks
    enable_impurities: bool = True
    impurity_num_peaks_range: Tuple[int, int] = (1, 10)
    impurity_intensity_range: Tuple[float, float] = (0.5, 70.0)
    impurity_width_range: Tuple[float, float] = (0.05, 0.3)
    impurity_eta_range: Tuple[float, float] = (0.2, 0.9)

    def __post_init__(self):
        """Set default values for None fields."""
        if self.instrumental_broadening is None:
            self.instrumental_broadening = {'u': 0.01, 'v': -0.005, 'w': 0.002}

        if self.low_index_orientation is None:
            self.low_index_orientation = [
                (1, 0, 0), (0, 1, 0), (0, 0, 1),      # Primary axes
                (1, 1, 0), (1, 0, 1), (0, 1, 1),      # Face diagonals
                (1, 1, 1), (-1, 1, 1), (1, -1, 1), (1, 1, -1)  # Body diagonals
            ]
        if self.high_index_orientation is None:
            self.high_index_orientation = [
                (2, 1, 0), (1, 2, 0), (2, 0, 1), (0, 2, 1), (1, 0, 2), (0, 1, 2),
                (1, 1, 2), (1, 2, 1), (2, 1, 1), (3, 1, 1), (1, 3, 1), (1, 1, 3),
                (3, 2, 1), (3, 1, 2), (1, 3, 2), (2, 3, 1), (1, 2, 3), (2, 1, 3),
                (3, 2, 2), (2, 3, 2), (2, 2, 3), (3, 3, 2), (3, 2, 3), (2, 3, 3)
            ]


    def to_dict(self) -> Dict:
        """Convert to dictionary for compatibility with existing code."""
        return {
            'min_angle': self.min_angle,
            'max_angle': self.max_angle,
            'num_points': self.num_points,
            'convert_to_q': self.convert_to_q,
            'uniform_shift_range': self.uniform_shift_range,
            'sample_displacement': self.sample_displacement,
            'goniometer_radius': self.goniometer_radius,
            'crystallite_size_log_scale': self.crystallite_size_log_scale,
            'crystallite_size_range': self.crystallite_size_range,
            'microstrain_range': self.microstrain_range,
            'instrumental_broadening': self.instrumental_broadening,
            'pseudo_voigt_eta_range': self.pseudo_voigt_eta_range,
            'temperature_range': self.temperature_range,
            'atomic_displacement_range': self.atomic_displacement_range,
            'texture_range': self.texture_range,
            'weights_low_index': self.weights_low_index,
            'low_index_orientation': self.low_index_orientation,
            'high_index_orientation': self.high_index_orientation,
            'lattice_strain_range': self.lattice_strain_range,
            'background_level': self.background_level,
            'noise_level': self.noise_level,
            'diffuse_scattering_intensity': self.diffuse_scattering_intensity,
            'diffuse_scattering_b_factor': self.diffuse_scattering_b_factor,
            'amorphous_intensity': self.amorphous_intensity,
            'amorphous_neighbor_distance': self.amorphous_neighbor_distance,
            'amorphous_disorder': self.amorphous_disorder,
            # RealisticXRDGenerator.DEFAULT_PARAMS calls this 'impurity_enable' --
            # without this mapping, enable_impurities is silently dropped and the
            # simulator always falls back to its own default (impurities off),
            # regardless of what this config says.
            'impurity_enable': self.enable_impurities,
            'impurity_num_peaks_range': self.impurity_num_peaks_range,
            'impurity_intensity_range': self.impurity_intensity_range,
            'impurity_width_range': self.impurity_width_range,
            'impurity_eta_range': self.impurity_eta_range
        }


@dataclass
class ModelConfig:
    """Configuration for model training."""

    # Training parameters -- these defaults match the frozen architecture
    # actually used to train the models shipped in examples/pretrained_catalog/ (see
    # e.g. examples/pretrained_catalog/pretrained_models/*/detection_model_*_config.json,
    # identical across every phase-specific model).
    optimizer: str = "rmsprop"
    num_epochs: int = 200
    learning_rate: float = 0.001
    batch_size: int = 32
    early_stopping_patience: int = 5
    test_fraction: float = 0.2
    dropout_rate: float = 0.0

    # Input data configuration. These must match the grid a model was trained
    # on -- 7001 points over 5-105 deg for the pretrained catalog. A mismatch
    # does not raise: patterns are resampled onto the wrong grid and every
    # prediction collapses toward zero, which looks like a model that does not
    # work rather than a preprocessing error.
    input_size: int = 7001  # Number of points in input patterns
    min_angle: float = 5.0
    max_angle: float = 105.0

    # Preprocessing
    smoothing_window_length: int = 21
    snip_iter: int = 24
    noise_sensitivity: float = 6.0
    gate_sharpness: float = 6.0
    magnification_power: float = 0.3

    # Neural network architecture
    conv_channels: List[int] = None
    conv_kernels: List[int] = None
    dilation_size: List[int] = None
    pool_size: List[int] = None
    fc_sizes: List[int] = None
    activation: str = "relu"
    use_batch_norm: bool = True
    use_configurable: bool = False
    model_config: str = "default"  # "default", "deep", "lightweight", "custom"
    target_phase: str = None

    # Mask
    use_mask: bool = True
    mask_start: int = None
    mask_end: int = None

    # Training options
    save_workflow_config: bool = True  # Save the full workflow config for model repeatability
    save_models: bool = True
    save_plots: bool = True
    seed: Optional[int] = 42

    def __post_init__(self):
        """Set default values for None fields."""
        if self.conv_channels is None:
            self.conv_channels = [16, 16, 16]
        if self.conv_kernels is None:
            self.conv_kernels = [27, 15, 11]
        if self.pool_size is None:
            self.pool_size = [2, 2, 1]
        if self.fc_sizes is None:
            self.fc_sizes = [16]

    def get_metadata(self) -> Dict:
        """Get model metadata for saving with trained models."""
        return {
            'input_size': self.input_size,
            'num_points': self.input_size,
            'min_angle': self.min_angle,
            'max_angle': self.max_angle,
            'conv_channels': self.conv_channels,
            'fc_sizes': self.fc_sizes,
            'activation': self.activation,
            'dropout_rate': self.dropout_rate
        }

    @classmethod
    def from_xrd_config(cls, xrd_config: 'XRDGenerationConfig', **kwargs) -> 'ModelConfig':
        """Create ModelConfig from XRDGenerationConfig, ensuring consistency."""
        return cls(
            input_size=xrd_config.num_points,
            min_angle=xrd_config.min_angle,
            max_angle=xrd_config.max_angle,
            **kwargs
        )


# Predefined configuration presets
DEFAULT_XRD_CONFIG = XRDGenerationConfig()

CONSERVATIVE_XRD_CONFIG = XRDGenerationConfig(
    uniform_shift_range=(-0.1, 0.1),
    crystallite_size_range=(10.0, 50.0),
    microstrain_range=(0.0, 0.001),
    lattice_strain_range=(0.0, 0.01),
    texture_range=(0.8, 1.2),
    background_level=(0.5, 1.0),
    noise_level=(1.0, 3.0),  # 1-3% noise
    impurity_intensity_range=(0.0, 10.0)
)

REALISTIC_XRD_CONFIG = XRDGenerationConfig(
    uniform_shift_range=(-0.5, 0.5),
    crystallite_size_range=(1.0, 200.0),
    microstrain_range=(0.0, 0.005),
    lattice_strain_range=(0.0, 0.08),
    texture_range=(0.2, 2.0),
    temperature_range=(100, 800),
    background_level=(0.5, 20.0),
    noise_level=(1.0, 10.0),  # 1-10% noise
    diffuse_scattering_intensity=(5.0, 50.0),
    amorphous_intensity=(5.0, 75.0),
    impurity_intensity_range=(0.0, 50.0)
)

DEFAULT_MODEL_CONFIG = ModelConfig()

@ dataclass
class DaraConfig:
    """Configuration for DARA search and refinement."""
    use_dara: bool = False
    save_refined_plot: bool = True
    downsized_length: int = 3000
    dara_batch_size: int = 5
    max_dara_iteration: int = 10
    enable_angular_cut: bool = True
    score_coefficients: Dict[str, float] = None
    # Maximum distance (1 - similarity) at which DARA merges two candidate
    # phases into one group; 0.2 corresponds to a similarity of 0.8. Larger
    # values merge aggressively and can collapse structurally unrelated phases
    # into a single group.
    maximum_grouping_distance: float = 0.2
    # Which phase-grouping similarity metric DARA uses: "legacy_jaccard"
    # (an asymmetric greedy-matched pseudo-Jaccard metric, kept only for
    # historical comparison) or "gaussian_cosine" (a symmetric,
    # density-normalized alternative -- see dara.search.phase_grouping's
    # module docstring). Defaults to "gaussian_cosine" so DARA's own
    # grouping uses the exact same similarity+clustering code as CNN-side
    # grouping (`pattern_utils.build_phase_groups_from_peaks`'s default
    # `grouping_backend="gaussian_cosine"`) -- callers that pin
    # maximum_grouping_distance for the old legacy_jaccard metric should set
    # grouping_metric="legacy_jaccard" explicitly to keep prior behavior.
    grouping_metric: str = "gaussian_cosine"
    phase_params: Dict[str, any] = None
    refinement_params: Dict[str, any] = None
    show_search_tree: bool = False
    strike_threshold: int = 1
    rpb_threshold: float = 2.0
    overfitting_threshold: float = 2.0
    false_peak_threshold: float = 0.03
    strain_threshold: float = 0.02
    early_stopping: bool = False