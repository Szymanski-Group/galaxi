#!/usr/bin/env python3
"""
Streamlined Phase Detection Workflow

This script provides a complete, automated workflow for:
1. Training phase detection models
2. Generating comprehensive test data
3. Evaluating model performance
4. Testing on experimental patterns

Usage:
    galaxi-workflow --config workflow_config.json

Or with command line arguments:
    galaxi-workflow --references References/ --exp-patterns Exp-Patterns/ --num-patterns 100
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
import warnings
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union
import h5py
import numpy as np
import os
import torch
import zarr
import time

from galaxi.core.config import DaraConfig
from galaxi.log_config import configure_cli_logging
from galaxi.core.pattern_utils import preprocess_xrd_pattern
from galaxi.paths import get_default_bg_profiles_path, get_default_cod_dir
from galaxi.pattern_generation.realistic_xrd import PeakList

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen")

from galaxi import (
    UnifiedPatternGenerator, PhaseDetectionModel,
    ComprehensiveTestGenerator, ModelEvaluator,
    XRDGenerationConfig, ModelConfig
)
from galaxi.pattern_generation.realistic_xrd import RealisticXRDGenerator


class StreamlinedWorkflow:
    """Complete streamlined workflow for phase detection model training and evaluation."""

    def __init__(self, config: Dict):
        """
        Initialize the workflow with configuration.

        Args:
            config: Configuration dictionary with all workflow parameters
        """

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.seed = config.get('seed', 42)
        ensemble_config = config.get("ensemble", {})
        self.use_ensemble = ensemble_config.get("use_ensemble", False)
        self.ensemble_index = ensemble_config.get("ensemble_index", None)

        self.config = config
        self.results = {}

        # Extract key paths from nested structure or fallback to flat structure for backward compatibility
        if 'directories' in config:
            self.references_dir = config['directories']['references_dir']
            self.bg_profiles_path = config['directories'].get('bg_profiles_path')
            self.exp_patterns_dir = config['directories'].get('exp_patterns_dir')
            self.cod_dir = config['directories'].get('cod_dir') or get_default_cod_dir()
            self.output_dir = Path(config['directories'].get('output_dir', 'workflow_results'))
            self.pinned_phases_dir = config['directories'].get('pinned_phases_dir', None)
        else:
            # Backward compatibility with flat structure
            self.references_dir = config['references_dir']
            self.exp_patterns_dir = config.get('exp_patterns_dir')
            self.cod_dir = config.get('cod_dir') or get_default_cod_dir()
            self.output_dir = Path(config.get('output_dir', 'workflow_results'))

        # Create output directory structure
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / 'training_data').mkdir(exist_ok=True)
        (self.output_dir / 'models').mkdir(exist_ok=True)
        (self.output_dir / 'test_data').mkdir(exist_ok=True)
        (self.output_dir / 'evaluation').mkdir(exist_ok=True)

        # Parameter tuning options
        # Enable generation parameter tuning, external dataset using original config is automatically enabled
        self.generation_param_tuning =config.get('model_tuning', {}).get('generation_param_tuning', False)
        # Enable model hyperparameter tuning
        self.model_param_tuning = config.get('model_tuning', {}).get('model_param_tuning', False)
        if self.generation_param_tuning and self.model_param_tuning:
            raise ValueError("Both 'generation_param_tuning' and 'model_param_tuning' cannot be True at the same time.")
        self.hyperparameter_dict = config.get('model_tuning', {}).get('hyperparameter_dict', {})

        self.generation_output_dir = None  # Path to specific phase during training data generation. To be set during data generation
        self.model_output_dir = None  # Path to specific model during training. To be set during model training

        # Extract training data generation parameters
        self.training_data_config = config.get('training_data_generation', {})

        self.generation_batch_size = self.training_data_config.get('generation_batch_size', 256)

        self.cache_training_data = self.training_data_config.get('generation_settings', {}).get('cache_training_data', False)
        self.cached_patterns = {"positive": [], "negative": []}
        self.cached_labels = []

        self.total_num = self.training_data_config.get('total_num', 5000)

        # Phase mixture controls. phase_mixture_controls.num_N_phase_patterns
        # -- the schema create_default_config() writes -- are relative weights
        # across the four phase counts, not fractions of total_num.
        # Normalizing to fractions that sum to 1 here means every downstream
        # `total_num * fraction * frac_N_phase` computation keeps working
        # unchanged, whether num_N_phase_patterns is given as raw pattern
        # counts (the default config's convention) or already-normalized
        # fractions.
        phase_mixture = self.training_data_config.get('phase_mixture_controls', {})
        _phase_weights = [
            phase_mixture.get('num_1_phase_patterns', 1000),
            phase_mixture.get('num_2_phase_patterns', 1500),
            phase_mixture.get('num_3_phase_patterns', 1500),
            phase_mixture.get('num_4_phase_patterns', 1000),
        ]
        _phase_weight_total = sum(_phase_weights) or 1
        self.frac_1_phase, self.frac_2_phase, self.frac_3_phase, self.frac_4_phase = (
            w / _phase_weight_total for w in _phase_weights
        )

        # Positive/negative controls
        pos_neg_controls = self.training_data_config.get('positive_negative_controls', {})
        self.positive_fraction = pos_neg_controls.get('positive_fraction', 0.5)
        self.negative_fraction = pos_neg_controls.get('negative_fraction', 0.5)

        positive_types = pos_neg_controls.get('positive_single_phase_types', {})
        _clean_raw = positive_types.get('clean_fraction', 0.5)
        _augmented_raw = positive_types.get('augmented_fraction', 0.5)
        _positive_types_total = (_clean_raw + _augmented_raw) or 1
        self.clean_fraction = _clean_raw / _positive_types_total
        self.augmented_fraction = _augmented_raw / _positive_types_total

        negative_types = pos_neg_controls.get('negative_types', {})
        _background_raw = negative_types.get('background_only_fraction', 0.4)
        _perturbation_raw = negative_types.get('peak_perturbation_fraction', 0.6)
        _negative_types_total = (_background_raw + _perturbation_raw) or 1
        self.background_only_fraction = _background_raw / _negative_types_total
        self.peak_perturbation_fraction = _perturbation_raw / _negative_types_total

        # Peak perturbation controls
        self.peak_perturbation_config = self.training_data_config.get('peak_perturbation_controls', {})

        # Peak augmentation controls
        self.peak_augmentation_config = self.training_data_config.get('peak_augmentation_controls', {})

        # Extract training data caching option
        self.cache_training_data = self.training_data_config.get('generation_settings', {}).get('cache_training_data', False)

        # Extract test data generation parameters
        self.test_data_config = config.get('test_data_generation', {})
        self.num_patterns_per_artifact = self.test_data_config.get('num_patterns_per_artifact', 100)
        self.num_patterns_per_multiphase = self.test_data_config.get('num_patterns_per_multiphase', 100)

        # Extract evaluation parameters
        evaluation_config = config.get('evaluation', {})
        self.probability_threshold = evaluation_config.get('probability_threshold', 0.5)
        self.top_k_config = evaluation_config.get('top_k', None)
        self.group_phases = evaluation_config.get('group_phases', False)
        self.group_similarity_threshold = evaluation_config.get('group_similarity_threshold', 0.90)
        # dara settings
        self.dara_dict = evaluation_config.get('dara', {})

        # Create configurations from JSON config
        self.xrd_config = self._create_xrd_config_from_dict(config)
        self.model_config = self._create_model_config_from_dict(config)

        print(f"Streamlined Workflow initialized")
        print(f"References: {self.references_dir}")
        print(f"Output: {self.output_dir}")
        print(f"XRD config: From JSON config file")
        print(f"Model config: From JSON config file")
        print(f"")
        print(f"Training Data Generation:")
        print(f"  Positive/Negative ratio: {self.positive_fraction:.1%} / {self.negative_fraction:.1%}")
        print(f"  Negative types: {self.background_only_fraction:.1%} background, {self.peak_perturbation_fraction:.1%} perturbations")
        if self.peak_perturbation_config.get('enable_peak_perturbations', True):
            print(f"  Peak perturbations: enabled")
            perturbation_types = self.peak_perturbation_config.get('perturbation_types', ['removal', 'shift', 'intensity_change'])
            print(f"    Types: {', '.join(perturbation_types)}")
        else:
            print(f"  Peak perturbations: disabled")
        print(f"")
        print(f"Test Data Generation:")
        print(f"  Patterns per artifact: {self.num_patterns_per_artifact}")
        print(f"  Patterns per multi-phase combination: {self.num_patterns_per_multiphase}")
        print(f"")
        print(f"Evaluation:")
        if self.probability_threshold:
            print(f"  Probability threshold: {self.probability_threshold}")
        elif self.top_k_config and self.top_k_config.get('use_top_k', False):
            print(f"  Probability top k: top {self.top_k_config.get('k_value', None)}")


    def set_random_seeds(self, seed: Optional[int] = 42):
        """
        Set all random seeds once at the top level.

        Args:
            seed: Random seed for reproducibility.
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            # Set Python hash seed for additional determinism
            os.environ['PYTHONHASHSEED'] = str(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _check_existing_training_data(self, phases: List[str]) -> tuple[List[str], List[str]]:
        """Check which phases already have training data generated."""
        existing_phases = []
        missing_phases = []

        for phase in phases:
            training_dir = self.output_dir / 'training_data' / f'training_patterns_{phase}'
            ground_truth_file = training_dir / 'detection_ground_truth.json'

            self.handle_model_output_dir(phase)

            # Check if model exist. If yes, then skip training
            if self._model_exists(phase):
                existing_phases.append(phase)
                print(f"✓ Model exists for {phase}, skipping training data check")
            # Check if training data directory exists and has ground truth file
            elif training_dir.exists() and ground_truth_file.exists():
                # Additional check: verify positive and negative directories exist with files
                pos_dir = training_dir / 'positive'
                neg_dir = training_dir / 'negative'

                if (pos_dir.exists() and neg_dir.exists() and
                    len(list(pos_dir.glob('*.xy'))) > 0 and
                    len(list(neg_dir.glob('*.xy'))) > 0):
                    existing_phases.append(phase)
                    print(f"✓ Training data exists for {phase}")
                else:
                    missing_phases.append(phase)
                    print(f"⚠ Incomplete training data for {phase}")
            else:
                missing_phases.append(phase)
                print(f"✗ No training data for {phase}")

        return existing_phases, missing_phases

    def _check_existing_models(self, phases: List[str]) -> tuple[List[str], List[str]]:
        """Check which phases already have trained models."""
        existing_models = []
        missing_models = []

        for phase in phases:
            model_dir = self.output_dir / 'models' / f'models_{phase}'
            model_file = model_dir / f'detection_model_{phase}.pth'

            if model_dir.exists() and model_file.exists():
                existing_models.append(phase)
                print(f"✓ Model exists for {phase}")
            else:
                missing_models.append(phase)
                print(f"✗ No model for {phase}")

        return existing_models, missing_models

    def _check_existing_test_data(self, phases: List[str]) -> tuple[List[str], List[str]]:
        """Check which phases already have test data generated."""
        existing_test = []
        missing_test = []

        for phase in phases:
            test_dir = self.output_dir / 'test_data' / f'test_{phase}'
            ground_truth_file = test_dir / 'comprehensive_ground_truth.json'

            if test_dir.exists() and ground_truth_file.exists():
                existing_test.append(phase)
                print(f"✓ Test data exists for {phase}")
            else:
                missing_test.append(phase)
                print(f"✗ No test data for {phase}")

        return existing_test, missing_test

    def _create_xrd_config_from_dict(self, config: Dict) -> XRDGenerationConfig:
        """Create XRDGenerationConfig from nested dictionary configuration."""
        if 'shared_xrd_generation_config' not in config:
            return XRDGenerationConfig()  # Use default

        xrd_config = config['shared_xrd_generation_config']

        # Extract nested parameters
        basic = xrd_config.get('basic_parameters', {})
        peak_pos = xrd_config.get('peak_position_effects', {})
        crystal_strain = xrd_config.get('crystallite_size_and_strain', {})
        instrumental = xrd_config.get('instrumental_effects', {})
        temperature = xrd_config.get('temperature_effects', {})
        texture = xrd_config.get('texture_and_orientation', {})
        bg_noise = xrd_config.get('background_and_noise', {})
        diffuse = xrd_config.get('diffuse_scattering', {})
        amorphous = xrd_config.get('amorphous_contributions', {})
        impurity = xrd_config.get('impurity_peaks', {})

        return XRDGenerationConfig(
            # Basic parameters
            min_angle=basic.get('min_angle', 10.0),
            max_angle=basic.get('max_angle', 80.0),
            num_points=basic.get('num_points', 4501),
            convert_to_q=basic.get('convert_to_q', False),

            # Peak position effects
            uniform_shift_range=tuple(peak_pos.get('uniform_shift_range', [-0.25, 0.25])),
            sample_displacement=tuple(peak_pos.get('sample_displacement', [-0.2, 0.2])),
            goniometer_radius=peak_pos.get('goniometer_radius', 240.0),

            # Crystallite size and strain
            crystallite_size_log_scale=crystal_strain.get('crystallite_size_log_scale', False),
            crystallite_size_range=tuple(crystal_strain.get('crystallite_size_range', [5.0, 100.0])),
            microstrain_range=tuple(crystal_strain.get('microstrain_range', [0.0, 0.003])),
            lattice_strain_range=tuple(crystal_strain.get('lattice_strain_range', [0.0, 0.01])),

            # Instrumental effects
            instrumental_broadening=instrumental.get('instrumental_broadening', {'u': 0.01, 'v': -0.005, 'w': 0.002}),
            pseudo_voigt_eta_range=tuple(instrumental.get('pseudo_voigt_eta_range', [0.3, 0.8])),

            # Temperature effects
            temperature_range=tuple(temperature.get('temperature_range', [200, 300])),
            atomic_displacement_range=tuple(temperature.get('atomic_displacement_range', [0.005, 0.02])),

            # Texture and orientation
            texture_range=tuple(texture.get('texture_range', [0.5, 1.5])),
            weights_low_index=texture.get('weights_low_index', 0.7),
            low_index_orientation=texture.get('low_index',
                                  [
                            (1, 0, 0), (0, 1, 0), (0, 0, 1),      # Primary axes
                            (1, 1, 0), (1, 0, 1), (0, 1, 1),      # Face diagonals
                            (1, 1, 1), (-1, 1, 1), (1, -1, 1), (1, 1, -1)  # Body diagonals
                        ]),
            high_index_orientation=texture.get('high_index',
                                   [(2, 1, 0), (1, 2, 0), (2, 0, 1), (0, 2, 1), (1, 0, 2), (0, 1, 2),
                            (1, 1, 2), (1, 2, 1), (2, 1, 1), (3, 1, 1), (1, 3, 1), (1, 1, 3),
                            (3, 2, 1), (3, 1, 2), (1, 3, 2), (2, 3, 1), (1, 2, 3), (2, 1, 3),
                            (3, 2, 2), (2, 3, 2), (2, 2, 3), (3, 3, 2), (3, 2, 3), (2, 3, 3)
                        ]),

            # Background and noise
            background_level=tuple(bg_noise.get('background_level', [0.5, 5.0])),
            noise_level=tuple(bg_noise.get('noise_level', [0.1, 0.5])),

            # Diffuse scattering
            diffuse_scattering_intensity=tuple(diffuse.get('diffuse_scattering_intensity', [5.0, 25.0])),
            diffuse_scattering_b_factor=tuple(diffuse.get('diffuse_scattering_b_factor', [0.5, 3.0])),

            # Amorphous contributions
            amorphous_intensity=tuple(amorphous.get('amorphous_intensity', [5.0, 50.0])),
            amorphous_neighbor_distance=tuple(amorphous.get('amorphous_neighbor_distance', [2.0, 4.0])),
            amorphous_disorder=tuple(amorphous.get('amorphous_disorder', [0.2, 0.8])),

            # Impurity peaks
            enable_impurities=impurity.get('enable_impurities', True),
            impurity_num_peaks_range=tuple(impurity.get('impurity_num_peaks_range', [1, 10])),
            impurity_intensity_range=tuple(impurity.get('impurity_intensity_range', [0.0, 25.0])),
            impurity_width_range=tuple(impurity.get('impurity_width_range', [0.05, 0.3])),
            impurity_eta_range=tuple(impurity.get('impurity_eta_range', [0.2, 0.9]))
        )

    def _create_model_config_from_dict(self, config: Dict) -> ModelConfig:
        """Create ModelConfig from nested dictionary configuration."""
        if 'model_config' not in config:
            return ModelConfig()  # Use default

        model_config = config['model_config']

        # Extract nested parameters
        prep = model_config.get('preprocessing', {})
        arch = model_config.get('architecture', {})
        mask = model_config.get('mask', {})
        training = model_config.get('training', {})
        output = model_config.get('output', {})

        # Input size will be derived from XRD config automatically via ModelConfig.from_xrd_config
        # or use default from architecture config

        # Create ModelConfig from XRD config to ensure input_size consistency
        base_model_config = ModelConfig.from_xrd_config(
            self.xrd_config,

            # Preprocessing
            smoothing_window_length=prep.get('smoothing_window_length', 21),
            snip_iter=prep.get('snip_iter', 24),
            noise_sensitivity=prep.get('noise_sensitivity', 6.0),
            gate_sharpness=prep.get('gate_sharpness', 6.0),
            magnification_power=prep.get('magnification_power', 0.3),

            # Architecture
            conv_channels=arch.get('conv_channels', [32, 64, 128]),
            conv_kernels=arch.get('conv_kernels', [16, 12, 8]),
            dilation_size=arch.get('dilation_size', [1, 1, 1]),
            pool_size=arch.get('pool_size', [2, 2, 2]),
            fc_sizes=arch.get('fc_sizes', [1024, 512, 128]),
            activation=arch.get('activation', 'relu'),
            use_batch_norm=arch.get('use_batch_norm', True),
            dropout_rate=arch.get('dropout_rate', 0.4),
            model_config=arch.get('model_config', 'detection'),
            use_configurable=arch.get('use_configurable', False),

            # Mask
            use_mask=mask.get('use_mask', True),           # Use masking for model input
            mask_start=mask.get('mask_start', 20),            # Mask start angle
            mask_end=mask.get('mask_end', 60),              # Mask end angle

            # Training
            optimizer=training.get('optimizer', 'adam'),
            num_epochs=training.get('num_epochs', 10),
            learning_rate=training.get('learning_rate', 0.001),
            batch_size=training.get('batch_size', 32),
            early_stopping_patience=training.get('early_stopping_patience', 10),
            test_fraction=training.get('test_fraction', 0.2),

            # Output
            save_models=output.get('save_models', True),
            save_plots=output.get('save_plots', True)
        )

        # Store val_size separately since it's not part of ModelConfig but needed for training
        base_model_config.val_size = training.get('val_size', 0.2)
        # Save workflow config option separately since it's not part of ModelConfig but needed for repeatability
        base_model_config.save_workflow_config = output.get('save_workflow_config', True)

        return base_model_config

    def check_workflow_status(self):
        """Check the current status of the workflow without running anything."""

        print("="*80)
        print("WORKFLOW STATUS CHECK")
        print("="*80)

        # Get available phases
        generator = UnifiedPatternGenerator(reference_dir=self.references_dir)
        available_phases = generator.get_available_phases()

        print(f"Total phases available: {len(available_phases)}")

        # Check training data
        existing_training, missing_training = self._check_existing_training_data(available_phases)
        print(f"\nTraining Data: {len(existing_training)}/{len(available_phases)} complete")

        # Check models
        existing_models, missing_models = self._check_existing_models(existing_training)
        print(f"Models: {len(existing_models)}/{len(existing_training)} complete")

        # Check test data
        existing_test, missing_test = self._check_existing_test_data(existing_models)
        print(f"Test Data: {len(existing_test)}/{len(existing_models)} complete")

        print(f"\nNext steps needed:")
        if missing_training:
            print(f"  - Generate training data for: {missing_training}")
        if missing_models:
            print(f"  - Train models for: {missing_models}")
        if missing_test:
            print(f"  - Generate test data for: {missing_test}")

        return {
            'available_phases': available_phases,
            'existing_training': existing_training,
            'existing_models': existing_models,
            'existing_test': existing_test
        }

    def step_0_generate_external_test_set(self, phases: Optional[List[str]] = None) -> List[str]:
        """
        Step 0: Generate external test sets for all phases.
        Raises an error if generation_param_tuning is False.

        Args:
            phases (List[str], optional): List of phase names (subfolders under reference_dir). If None, all subfolders are used.

        Returns:
            List[str]: List of successfully processed phases.
        """
        if not getattr(self, "generation_param_tuning", False):
            raise RuntimeError(
                "[Step 0] External test set generation requires 'generation_param_tuning=True'. "
                "Please enable it in the workflow configuration."
            )

        self.set_random_seeds(self.seed)

        # Initialize generator
        generator = UnifiedPatternGenerator(
            reference_dir=self.references_dir,
            config=self.xrd_config
        )

        # Get available phases
        available_phases = generator.get_available_phases()

        # Determine which phases to handle
        if phases:
            selected_phases = [p for p in phases if p in available_phases]
            missing_from_ref = [p for p in phases if p not in available_phases]
            if missing_from_ref:
                print(f"Warning: the following requested phases are not found in reference directory: {missing_from_ref}")
        else:
            selected_phases = available_phases
            phases = available_phases

        print("\n" + "="*80)
        print("STEP 0: GENERATING EXTERNAL TEST SET")
        print("="*80)
        print(f"Found {len(available_phases)} reference phases in total: {available_phases}")
        print(f"Selected {len(selected_phases)} phases to process: {selected_phases}")

        # Load COD structures once for reuse across phases
        detection_params = self.training_data_config.get('generation_settings', {})

        successfully_generated = []

        for phase in selected_phases:
            external_test_dir = Path(self.output_dir) / "external_test_set" / phase
            if self._pattern_exists(external_test_dir, phase):
                continue

            # Skip if test set already exists
            negatives_folder = external_test_dir / 'negative'
            positives_folder = external_test_dir / 'positive'
            if external_test_dir.exists() and any(negatives_folder.glob("*.xy")) and any(positives_folder.glob("*.xy")):
                print(f"  - Phase {phase}: already exists, skipping.")
                successfully_generated.append(phase)
                continue

            print(f"  - Phase {phase}: generating external test set...")
            external_test_dir.mkdir(parents=True, exist_ok=True)

            # Compute number of patterns for external test set
            phase_mixture_controls = self.training_data_config.get("phase_mixture_controls", {})
            total_patterns = (
                phase_mixture_controls.get("num_1_phase_patterns", 0) +
                phase_mixture_controls.get("num_2_phase_patterns", 0) +
                phase_mixture_controls.get("num_3_phase_patterns", 0) +
                phase_mixture_controls.get("num_4_phase_patterns", 0)
            )

            positive_fraction = self.training_data_config.get("positive_negative_controls", {}).get("positive_fraction", 0.5)
            negative_fraction = self.training_data_config.get("positive_negative_controls", {}).get("negative_fraction", 0.5)

            # Use test_fraction if defined in model config
            test_fraction = getattr(self.model_config, "test_fraction", 0.2)
            num_positive = int(total_patterns * positive_fraction * test_fraction)
            num_negative = int(total_patterns * negative_fraction * test_fraction)

            # Generate external test set
            try:
                self._generate_customizable_training_patterns(
                    generator=generator,
                    target_phase_name=phase,
                    detection_params=detection_params,
                    output_dir=str(external_test_dir),
                    is_external=True
                )
                print(f"✓ External test set generated for {phase}")
                successfully_generated.append(phase)
            except Exception as e:
                print(f"✗ Error generating external test set for {phase}: {e}")
                continue

        print(f"[Step 0] External test set generation complete. ({len(successfully_generated)} phases processed)")
        return successfully_generated

    def _pattern_exists(self, generation_output_dir, phase: str) -> bool:
        if (Path(generation_output_dir) / "detection_ground_truth.json").exists():
            print(f"Training pattern for {phase} already exists, skip training pattern generation")
            return True
        else:
            return False

    def _require_bg_profiles(self) -> None:
        """Fail fast, with an actionable message, if the background-profile
        library is missing.

        Called at the top of step 1 rather than at the point of use: the pool is
        not read until every COD structure has been loaded, so checking early
        avoids several minutes of work before reporting a missing file.
        """
        if not self.bg_profiles_path:
            raise ValueError(
                "Training data generation requires config['directories']['bg_profiles_path'] "
                "to be set to a real background-profile HDF5 file; it was not provided. "
                "Install the default library with `galaxi-setup-bg-profiles`."
            )
        if not os.path.exists(self.bg_profiles_path):
            raise FileNotFoundError(
                f"Background-profile library not found at {self.bg_profiles_path!r}.\n"
                "Training-data generation samples this pool of pre-simulated "
                "single-phase patterns to build the negatives and the non-target "
                "phases of multi-phase patterns, so it is required for step 1.\n"
                "Install it with:  galaxi-setup-bg-profiles\n"
                "(or point config['directories']['bg_profiles_path'] at an "
                "existing copy, or set $GALAXI_BG_PROFILES)."
            )

    @staticmethod
    def _report_step_outcome(label, succeeded, attempted, failures) -> None:
        """Print a step's summary line, reporting failures alongside successes.

        A step that produced nothing is reported with a failure marker and the
        per-phase causes, rather than a success line reading "0 phases total".
        """
        if failures:
            print(f"\n  {len(failures)} of {len(attempted)} phase(s) failed:")
            for phase, err in failures:
                print(f"    ✗ {phase}: {err}")

        if succeeded:
            suffix = f" ({len(failures)} failed)" if failures else ""
            print(f"\n✓ {label} available for {len(succeeded)} phases total{suffix}")
        else:
            print(f"\n✗ {label} available for 0 phases: all {len(attempted)} phase(s) failed")

    def step_1_generate_training_data(self, phases: Optional[List[str]] = None) -> List[str]:
        """Step 1: Generate training data for all reference phases (with resume capability)."""

        print("\n" + "="*80)
        print("STEP 1: GENERATING TRAINING DATA")
        print("="*80)

        self._require_bg_profiles()

        self.set_random_seeds(self.seed)

        # Use the XRD configuration from config file or custom one
        training_config = self.xrd_config

        # Initialize generator
        generator = UnifiedPatternGenerator(
            reference_dir=self.references_dir,
            config=training_config
        )

        # Get available phases
        available_phases = generator.get_available_phases()

        # Determine which phases to handle
        if phases:
            selected_phases = [p for p in phases if p in available_phases]
            missing_from_ref = [p for p in phases if p not in available_phases]
            if missing_from_ref:
                print(f"Warning: the following requested phases are not found in reference directory: {missing_from_ref}")
        else:
            selected_phases = available_phases
            phases = available_phases

        print(f"Found {len(available_phases)} reference phases in total: {available_phases}")
        print(f"Selected {len(selected_phases)} phases to process: {selected_phases}")

        # Check existing training data based on both training data and trained models
        existing_phases, missing_phases = self._check_existing_training_data(selected_phases)

        print("-"*80)
        print(f"Training data status:")
        print(f"  Already completed: {len(existing_phases)} phases")
        print(f"  Need to generate: {len(missing_phases)} phases")

        if missing_phases:
            print(f"  Missing phases: {missing_phases}")

        # Generate training data only for missing phases
        successfully_trained = existing_phases.copy()
        failures: List[tuple] = []

        if missing_phases:
            # Load COD structures once for all phases to avoid repeated loading
            print("-"*80)
            print(f"Loading COD structures once for all phases...")
            detection_params = self.training_data_config.get('generation_settings', {})

        for phase in missing_phases:
            print("-"*80)
            print(f"Generating training data for {phase}...")

            try:
                # Get generation settings from training data config
                detection_params = self.training_data_config.get('generation_settings', {})

                # self.generation_output_dir starts as None (set in __init__) and is
                # otherwise only ever assigned inside handle_model_output_dir(), which
                # step_2_train_models() calls after generation already happened. Set it
                # here so it points at a real per-phase directory before it's used
                # below (matches step_2_train_models()'s own independently-computed
                # `training_dir`, so training reads from where generation wrote).
                if self.generation_param_tuning:
                    folder_name = self.hyperparameter_dict_to_name(self.hyperparameter_dict)
                    self.generation_output_dir = self.output_dir / 'training_data' / f'training_patterns_{phase}' / folder_name
                else:
                    self.generation_output_dir = self.output_dir / 'training_data' / f'training_patterns_{phase}'

                # Generate patterns using the new customizable approach with pre-loaded COD structures
                self._generate_customizable_training_patterns(
                    generator=generator,
                    target_phase_name=phase,
                    detection_params=detection_params,
                    output_dir=str(self.generation_output_dir),
                )

                print(f"✓ Training data generated for {phase}")
                successfully_trained.append(phase)

            except Exception as e:
                print(f"✗ Error generating training data for {phase}: {e}")
                traceback.print_exc()  # This prints the full traceback
                failures.append((phase, f"{type(e).__name__}: {e}"))
                continue

        self.results['trained_phases'] = successfully_trained
        self._report_step_outcome(
            "Training data", successfully_trained, missing_phases, failures
        )

        return successfully_trained

    def _model_exists(self, phase: str) -> bool:
        if (self.model_output_dir / f"detection_model_{phase}.pth").exists():
            print(f"{self.model_output_dir} already exists, skipping training")
            return True
        else:
            return False

    def handle_model_output_dir(self, phase: str):
        # generation_output_dir is only set when step 1 ran in this process, so
        # derive it here when training resumes against data generated earlier.
        if self.generation_param_tuning and self.generation_output_dir is None:
            folder_name = self.hyperparameter_dict_to_name(self.hyperparameter_dict)
            self.generation_output_dir = self.output_dir / 'training_data' / f'training_patterns_{phase}' / folder_name

        # 2. Existing logic for ensemble/tuning
        if self.use_ensemble and not self.model_param_tuning and not self.generation_param_tuning:
            self.model_output_dir = self.output_dir / 'models' / f'models_{phase}' / f'models_ensemble_{self.ensemble_index}'
            if self._model_exists(phase):
                return

        if self.model_param_tuning:
            folder_name = self.hyperparameter_dict_to_name(self.hyperparameter_dict)
            if self.use_ensemble:
                self.model_output_dir = self.output_dir / 'models' / f'models_{phase}' / folder_name / f'models_ensemble_{self.ensemble_index}'
            else:
                self.model_output_dir = self.output_dir / 'models' / f'models_{phase}' / folder_name

            if self._model_exists(phase):
                return

        if self.generation_param_tuning:
            # Now self.generation_output_dir is guaranteed to not be None
            if self.use_ensemble:
                self.model_output_dir = self.output_dir / 'models' / f'models_{phase}' / self.generation_output_dir.name / f'models_ensemble_{self.ensemble_index}'
            else:
                self.model_output_dir = self.output_dir / 'models' / f'models_{phase}' / self.generation_output_dir.name

            if self._model_exists(phase):
                return

        # Fallback for standard training
        if not self.use_ensemble and not self.model_param_tuning and not self.generation_param_tuning:
            self.model_output_dir = self.output_dir / 'models' / f'models_{phase}'

        self.model_output_dir.mkdir(parents=True, exist_ok=True)

    def step_2_train_models(self, phases: List[str]) -> List[str]:
        """Step 2: Train detection models for all phases (with resume capability)."""

        print("\n" + "="*80)
        print("STEP 2: TRAINING DETECTION MODELS")
        print("="*80)

        self.set_random_seeds(self.seed)

        # Check existing models
        existing_models, missing_models = self._check_existing_models(phases)

        print(f"\nModel training status:")
        print(f"  Already trained: {len(existing_models)} models")
        print(f"  Need to train: {len(missing_models)} models")

        if missing_models:
            print(f"  Missing models: {missing_models}")

        print(f"Using device: {self.device}")

        successfully_trained = existing_models.copy()
        failures: List[tuple] = []

        for phase in missing_models:
            print("-"*80)
            print(f"Training model for {phase}...")
            training_dir = self.output_dir / 'training_data' / f'training_patterns_{phase}'

            self.handle_model_output_dir(phase)

            try:
                workflow_config = self.config if getattr(self.model_config, 'save_workflow_config', False) else None
                external_generation_output_dir = Path(self.output_dir) / "external_test_set" / phase

                model = PhaseDetectionModel(phase, config=self.model_config, reference_dir=self.references_dir)
                if self.cache_training_data:
                    model.cache_training_data = True
                    model.cached_patterns = self.cached_patterns
                    model.cached_labels = self.cached_labels
                results = model.train_detection(
                    generation_output_dir=training_dir,
                    model_output_dir=str(self.model_output_dir),
                    test_size=self.model_config.test_fraction,
                    val_size=self.model_config.val_size,
                    workflow_config=workflow_config,
                    use_external_test_set=True if self.generation_param_tuning else False,
                    external_generation_output_dir=external_generation_output_dir if self.generation_param_tuning else None,
                )

                # print training results for this phase
                print("-"*80)
                print(f"✓ Model trained for {phase}")
                if not self.use_ensemble:
                    print(f"  Test accuracy: {results['test']['test_accuracy']:.4f}")
                    print(f"  Test AUC: {results['test']['test_auc']:.4f}")
                    print(f"  Test log loss: {results['test']['test_log_loss']:.4f}")

                if phase not in successfully_trained:
                    successfully_trained.append(phase)

            except Exception as e:
                print(f"✗ Error training model for {phase}: {e}")
                import traceback
                traceback.print_exc()
                failures.append((phase, f"{type(e).__name__}: {e}"))
                continue

        self.results['successfully_trained'] = successfully_trained
        self._report_step_outcome("Models", successfully_trained, missing_models, failures)

        # Zero successes out of a non-empty attempt list is systemic rather
        # than partial failure -- a broken architecture config, an unreadable
        # training directory, a bad device -- so surface it here instead of
        # returning an empty list. A fully-resumed run legitimately trains zero
        # new models, which is why this is gated on missing_models.
        if missing_models and not successfully_trained:
            detail = "; ".join(f"{phase}: {err}" for phase, err in failures)
            raise RuntimeError(
                f"All {len(missing_models)} phase(s) failed to train. {detail}"
            )

        return successfully_trained

    def step_4_generate_comprehensive_test_data(self, phases: List[str]):
        """Step 4: Generate comprehensive test data with categorization (with resume capability)."""

        print("\n" + "="*80)
        print("STEP 4: GENERATING COMPREHENSIVE TEST DATA")
        print("="*80)

        self.set_random_seeds(self.seed)

        # Check existing test data
        existing_test, missing_test = self._check_existing_test_data(phases)

        print(f"\nTest data status:")
        print(f"  Already generated: {len(existing_test)} phases")
        print(f"  Need to generate: {len(missing_test)} phases")

        if missing_test:
            print(f"  Missing test data: {missing_test}")

        generated = list(existing_test)
        failures: List[tuple] = []

        # Initialize comprehensive test generator
        print(f"\nInitializing ComprehensiveTestGenerator...")
        print(f"  Reference directory: {self.references_dir}")
        print(f"  COD directory: {self.cod_dir}")

        # Get fraction ranges from config
        fraction_ranges = self.test_data_config.get('fraction_ranges', None)

        test_generator = ComprehensiveTestGenerator(
            reference_dir=self.references_dir,
            cod_dir=self.cod_dir,
            fraction_ranges=fraction_ranges,
            xrd_config=self.xrd_config
        )

        print(f"  Loaded {len(test_generator.reference_phases)} reference phases")
        print(f"  Loaded {len(test_generator.cod_phases)} COD phases")

        for phase in missing_test:
            print("\n" + "="*60)
            print(f"GENERATING TEST DATA FOR {phase}")
            print("="*60)

            output_dir = self.output_dir / 'test_data' / f'test_{phase}'
            print(f"Output directory: {output_dir}")

            try:
                import time
                start_time = time.time()

                print(f"\nStarting comprehensive test data generation...")
                print(f"  Target phase: {phase}")
                print(f"  Patterns per artifact: {self.num_patterns_per_artifact}")
                print(f"  Patterns per multi-phase combination: {self.num_patterns_per_multiphase}")

                test_generator.save_comprehensive_test_data(
                    target_phase=phase,
                    output_dir=str(output_dir),
                    num_patterns_per_artifact=self.num_patterns_per_artifact,
                    num_patterns_per_multiphase=self.num_patterns_per_multiphase
                )

                elapsed_time = time.time() - start_time
                print(f"\n✓ Comprehensive test data generated for {phase} (took {elapsed_time:.1f}s)")
                generated.append(phase)

            except Exception as e:
                elapsed_time = time.time() - start_time
                print(f"\n✗ Error generating test data for {phase} after {elapsed_time:.1f}s:")
                error_msg = str(e) if str(e) else f"{type(e).__name__}: {e.__class__.__module__}"
                print(f"    Error: {error_msg}")
                import traceback
                traceback.print_exc()
                failures.append((phase, error_msg))
                continue

        self._report_step_outcome("Test data", generated, missing_test, failures)

    def step_5_evaluate_models(self, phases: List[str]):
        """Step 5: Evaluate models on test data."""

        print("\n" + "="*80)
        print("STEP 5: EVALUATING MODELS ON TEST DATA")
        print("="*80)

        self.set_random_seeds(self.seed)

        # Initialize evaluator
        evaluator = ModelEvaluator(
            models_dir=str(self.output_dir / 'models'),
            output_dir=str(self.output_dir / 'evaluation'),
            xrd_config=self.xrd_config
        )

        # Define test datasets
        test_datasets = {}

        for phase in phases:
            test_base_dir = self.output_dir / 'test_data' / f'test_{phase}'

            if test_base_dir.exists():
                test_datasets[f"{phase}_comprehensive"] = {
                    'test_dir': str(test_base_dir),
                    'ground_truth_file': str(test_base_dir / 'comprehensive_ground_truth.json')
                }

        # Run evaluation
        try:
            results = evaluator.run_comprehensive_evaluation(test_datasets)
            self.results['test_evaluation'] = results
            print("\n✓ Model evaluation on test data completed")

        except Exception as e:
            print(f"\n✗ Error during model evaluation: {e}")

    def step_3_evaluate_experimental_patterns(self):
        """Step 3: Evaluate models on experimental patterns."""

        if not self.exp_patterns_dir or not Path(self.exp_patterns_dir).exists():
            print("\n⚠ Experimental patterns directory not provided or doesn't exist")
            print("Skipping experimental evaluation")
            return

        print("\n" + "="*80)
        print("STEP 3: EVALUATING MODELS ON EXPERIMENTAL PATTERNS")
        print("="*80)

        self.set_random_seeds(self.seed)

        models_dir = self.output_dir / 'models'

        # Getting all ref phases. Useful when evaluating experimental patterns!
        phases = [p.stem for p in Path(self.references_dir).glob("*.cif")]

        if self.generation_param_tuning or self.model_param_tuning:
            folder_name = self.hyperparameter_dict_to_name(self.hyperparameter_dict)
            output_dir = self.output_dir / 'evaluation' / folder_name
        else:
            output_dir = self.output_dir / 'evaluation'

        # Initialize evaluator
        evaluator = ModelEvaluator(
            phases=phases,
            ref_dir=self.references_dir,
            models_dir=str(models_dir),
            output_dir=str(output_dir),
            xrd_config=self.xrd_config,
            model_config=self.model_config,
            hyperparameter_dict=self.hyperparameter_dict, # If hyperparameter_dict is not empty then param_tuning is on
            use_ensemble=self.use_ensemble # If use_ensemble, then use average score as prediction
        )

        try:
            probability_threshold = self.probability_threshold
            top_k_config = self.top_k_config

            exp_results, dara_results = evaluator.evaluate_experimental_patterns(
                exp_patterns_dir=self.exp_patterns_dir,
                group_phases=self.group_phases,
                group_similarity_threshold=self.group_similarity_threshold,
                probability_threshold=probability_threshold,
                top_k_config=top_k_config,
                dara_dict=self.dara_dict if self.dara_dict else None,
                pinned_phases_dir=self.pinned_phases_dir
            )
            self.results['experimental_evaluation'] = exp_results
            self.results['dara_evaluation'] = dara_results

            print(f"\n✓ Experimental pattern evaluation completed")

        except Exception as e:
            print(f"✗ Error during experimental evaluation: {e}")
            import traceback
            traceback.print_exc()


    def run_complete_workflow(self):
        """Run the complete workflow from start to finish."""

        print("="*80)
        print("STREAMLINED PHASE DETECTION WORKFLOW")
        print("="*80)
        print(f"Configuration:")
        for key, value in self.config.items():
            print(f"  {key}: {value}")

        try:
            # Step 1: Generate training data
            trained_phases = self.step_1_generate_training_data()

            if not trained_phases:
                print("\n✗ No training data available. Workflow cannot continue.")
                return False

            # Step 2: Train models
            successfully_trained = self.step_2_train_models(trained_phases)

            if not successfully_trained:
                print("\n✗ No models trained successfully. Workflow cannot continue.")
                return False

            # Step 3: Evaluate on experimental patterns
            self.step_3_evaluate_experimental_patterns()

            # Step 4: Generate comprehensive test data
            self.step_4_generate_comprehensive_test_data(successfully_trained)

            # Step 5: Evaluate on test data
            self.step_5_evaluate_models(successfully_trained)

            # Save final results
            self._save_workflow_results()

            print("\n" + "="*80)
            print("WORKFLOW COMPLETED SUCCESSFULLY!")
            print("="*80)
            print(f"Results saved to: {self.output_dir}")
            print(f"Successfully processed {len(successfully_trained)} phases:")
            for phase in successfully_trained:
                print(f"  ✓ {phase}")

            return True

        except Exception as e:
            print(f"\n✗ Workflow failed: {e}")
            raise

    def hyperparameter_dict_to_name(self, hyperparameter_dict):
        if hyperparameter_dict is not None:
            return "_".join(f"{k}_{v}" for k, v in hyperparameter_dict.items())
        else:
            # Fallback to time based random naming
            import time
            ts = int(time.time_ns())
            return f"time_{ts}"

    def _save_workflow_results(self):
        """Save complete workflow results."""
        results_file = self.output_dir / 'workflow_results.json'

        # Convert results to JSON-serializable format
        json_results = {}
        for key, value in self.results.items():
            if key in ['trained_phases', 'successfully_trained']:
                json_results[key] = value
            elif key == 'test_evaluation':
                json_results[key] = {
                    phase: {
                        dataset: metrics.to_dict() if hasattr(metrics, 'to_dict') else str(metrics)
                        for dataset, metrics in datasets.items()
                    }
                    for phase, datasets in value.items()
                }
            else:
                json_results[key] = str(value)

        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)

        print(f"\n✓ Complete workflow results saved to {results_file}")

    def _save_metadata(self, output_dir: str, metadata: dict):
        """Save metadata to a JSON file."""
        metadata_file = os.path.join(output_dir, f"detection_ground_truth.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def _load_peak_lists_from_h5(self, target_phase_name, h5_path):
        """
        Load peak lists from h5 file, filtering by min/max angle.
        Returns:
            bg_names (List[str]): List of phase names
            bg_peak_lists (List[Dict]): List of peak_list dictionaries
        """
        import h5py
        import numpy as np
        import random

        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"Impurity peak list file not found: {h5_path}")

        # 1. Configuration
        max_cod_structures = self.training_data_config.get('generation_settings', {}).get('max_cod_structures', 500)
        target_safe_name = str(target_phase_name).replace("/", "_")

        # Get the current experiment's angular range
        min_angle = self.xrd_config.min_angle
        max_angle = self.xrd_config.max_angle

        loaded_data = []

        with h5py.File(h5_path, 'r') as f:
            # 2. Get Candidates (Filter out the target phase)
            all_phases = list(f.keys())
            candidates = [k for k in all_phases if k != target_safe_name]

            # 3. Random Sampling
            num_to_sample = min(len(candidates), max_cod_structures)

            if num_to_sample == 0:
                print("Warning: No background phases found in H5 library.")
                return [], []

            sampled_keys = random.sample(candidates, num_to_sample)

            # 4. Load & Filter Data
            for phase_name in sampled_keys:
                phase_grp = f[phase_name]
                peak_list_dict = {}

                has_valid_data = False

                for source in ['ka1', 'ka2']:
                    if source in phase_grp:
                        src_grp = phase_grp[source]

                        # Load raw arrays
                        x = src_grp['x'][:]       # Shape (N,)
                        y = src_grp['y'][:]       # Shape (N,)
                        hkls = src_grp['hkls'][:]    # Shape (N, 3)
                        dhkls = src_grp['dhkls'][:] # Shape (N,)

                        scale_factor = src_grp.attrs.get("max_original_intensity", 1.0)
                        y = y * scale_factor  # Rescale intensities back to original range

                        # Create a boolean mask for peaks inside the range
                        mask = (x >= min_angle) & (x <= max_angle)

                        # Apply mask to all arrays
                        filtered_x = x[mask]
                        filtered_y = y[mask]
                        filtered_hkls = hkls[mask]
                        filtered_d = dhkls[mask]

                        # Only add if we actually have peaks in this range?
                        # (Optional: keep empty lists if you want to explicitly say "no peaks here")
                        if len(filtered_x) > 0:
                            pl = PeakList(
                                x=filtered_x,
                                y=filtered_y,
                                hkls=filtered_hkls,
                                d_hkls=filtered_d
                            )
                            peak_list_dict[source] = pl
                            has_valid_data = True

                # Only add this phase if it has at least one valid pattern in range
                if has_valid_data:
                    loaded_data.append((phase_name, peak_list_dict))

        # 5. Unpack and Return
        if not loaded_data:
            print(f"Warning: No background phases found within range {min_angle}-{max_angle}°.")
            return [], []

        bg_names, bg_peak_lists = zip(*loaded_data)

        return list(bg_names), list(bg_peak_lists)

    def _generate_customizable_training_patterns(self, generator, target_phase_name: str,
                                                 detection_params: dict, output_dir: str,
                                                 is_external: bool=False):
        """
        Generates training data by simulating the target phase on-the-fly via PyTorch,
        and sampling backgrounds instantly from a pre-loaded, single-dataset HDF5 RAM pool.
        """
        if is_external:
            total_num = int(self.total_num * self.model_config.test_fraction)
            print(f"Generating customizable external test patterns using device {self.device}:")
        else:
            total_num = self.total_num
            print(f"Using device {self.device}:")

        # Create output directories
        os.makedirs(os.path.join(output_dir, 'positive'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'negative'), exist_ok=True)

        realistic_generator = RealisticXRDGenerator(params=generator.config.to_dict())
        target_structure = generator.structure_manager.get_structure(target_phase_name)

        # ==================== PRE-LOAD HDF5 Profile Library into RAM ====================
        self._require_bg_profiles()
        print("Pre-loading unified background library into RAM...")
        time_load_start = time.time()

        with h5py.File(self.bg_profiles_path, 'r') as h5_f:
            bg_pool_ram = h5_f['patterns'][:]
            bg_names_ram = np.array([name.decode('utf-8') for name in h5_f['phase_names'][:]])
            bg_two_theta = h5_f['two_theta'][:]

        valid_mask = np.array([target_phase_name not in name for name in bg_names_ram])

        # Apply the mask
        bg_pool_ram = bg_pool_ram[valid_mask]
        bg_names_ram = bg_names_ram[valid_mask]

        if len(bg_pool_ram) < 10:
            raise ValueError("Not enough background profiles in HDF5 library!")

        # The HDF5 background library is pre-generated at a fixed grid. If the configured
        # (min_angle, max_angle, num_points) differs -- e.g. when sweeping num_points, or training
        # on a narrower angular window than the library -- interpolate every background profile
        # onto the configured grid using its real 2θ coordinates (not just index position), so
        # a narrower target window correctly pulls the matching physical sub-range of the background
        # rather than stretching/squeezing the whole curve into the new point count.
        bg_min, bg_max, bg_len = float(bg_two_theta[0]), float(bg_two_theta[-1]), len(bg_two_theta)
        target_min, target_max, target_len = self.xrd_config.min_angle, self.xrd_config.max_angle, self.xrd_config.num_points
        if bg_len != target_len or abs(bg_min - target_min) > 1e-6 or abs(bg_max - target_max) > 1e-6:
            if target_min < bg_min - 1e-6 or target_max > bg_max + 1e-6:
                raise NotImplementedError(
                    f"Configured angular range [{target_min}, {target_max}] extends beyond the background "
                    f"library's covered range [{bg_min}, {bg_max}]; cannot interpolate outside measured data."
                )
            print(f"Interpolating background library from [{bg_min}, {bg_max}] ({bg_len} pts) to "
                  f"[{target_min}, {target_max}] ({target_len} pts)...")
            target_two_theta = np.linspace(target_min, target_max, target_len)
            idx = np.clip(np.searchsorted(bg_two_theta, target_two_theta) - 1, 0, bg_len - 2)
            x0, x1 = bg_two_theta[idx], bg_two_theta[idx + 1]
            weight = (target_two_theta - x0) / (x1 - x0)
            y0, y1 = bg_pool_ram[:, idx], bg_pool_ram[:, idx + 1]
            bg_pool_ram = (y0 + weight * (y1 - y0)).astype(np.float32)

        print(f"Loaded {len(bg_pool_ram)} valid background profiles into RAM in {time.time() - time_load_start:.3f}s.")

        # ==================== Handling positive patterns ====================
        num_1_phase = int(total_num * self.positive_fraction * self.frac_1_phase)
        num_2_phase = int(total_num * self.positive_fraction * self.frac_2_phase)
        num_3_phase = int(total_num * self.positive_fraction * self.frac_3_phase)
        num_4_phase = int(total_num * self.positive_fraction * self.frac_4_phase)
        phase_mixtures = [
            (1, num_1_phase),
            (2, num_2_phase),
            (3, num_3_phase),
            (4, num_4_phase)
        ]

        file_idx = 0
        all_metadata = []

        for n_phases, target_count in phase_mixtures:
            if target_count <= 0:
                continue

            if n_phases == 1:
                num_clean_patterns = int(self.clean_fraction * target_count)
                num_augmented_patterns = int(self.augmented_fraction * target_count)

                # ---------- Clean single phases ----------
                print(f"- Generating {num_clean_patterns} clean single-phase patterns...")
                time_start = time.time()
                for i in range(0, num_clean_patterns, self.generation_batch_size):
                    target_peak_list = realistic_generator.get_peak_list(target_structure)
                    batch_size_actual = min(self.generation_batch_size, num_clean_patterns - i)
                    two_theta, patterns = realistic_generator.generate_realistic_pattern(
                        peak_list=target_peak_list,
                        batch_size=batch_size_actual,
                        apply_all_effects=True,
                        enable_impurity=True
                    )
                    patterns_cpu = patterns.cpu().numpy()
                    self._save_patterns(output_dir, patterns_cpu, file_idx, category="positive", is_external=is_external)
                    file_idx += batch_size_actual

                    # NOTE: 'fraction' below (here and everywhere else in this file) is a
                    # synthetic mixing coefficient, not a weight/mole fraction: each phase's
                    # pattern is independently max-normalized before these coefficients are
                    # applied, so they weight each phase's strongest reflection rather than a
                    # physically calibrated wt%/mol% scale factor.
                    metadata = {'file_name': f'positive_00001~{file_idx-1:05d}.xy', 'type': f'clean_single_phase',
                                'target_phase': {'phase_name': target_phase_name, 'fraction': 1.0},
                                'bg_phase': {'phase_name': None, 'fraction': 0.0}}
                    all_metadata.append(metadata)
                print(f'  ✓ Generated {num_clean_patterns} clean single-phase patterns, took {time.time() - time_start:.2f}s')

                # ---------- Augmented single phases ----------
                print(f"- Generating {num_augmented_patterns} augmented single-phase patterns...")
                time_start = time.time()
                peak_augmentation_config = self.peak_augmentation_config
                realistic_generator.enable_peak_augmentations(peak_augmentation_config, enable=True)
                for i in range(0, num_augmented_patterns, self.generation_batch_size):
                    target_peak_list = realistic_generator.get_peak_list(target_structure)
                    batch_size_actual = min(self.generation_batch_size, num_augmented_patterns - i)
                    two_theta, patterns, augmentation_info = realistic_generator.generate_realistic_pattern(
                        peak_list=target_peak_list,
                        batch_size=batch_size_actual,
                        apply_all_effects=True,
                        enable_impurity=True
                    )
                    patterns_cpu = patterns.cpu().numpy()
                    self._save_patterns(output_dir, patterns_cpu, file_idx, category="positive", is_external=is_external)
                    for b in range(batch_size_actual):
                        metadata = {'file_name': f'positive_{file_idx + b:05d}.xy', 'type': f'augmented_single_phase',
                                    'target_phase': {'phase_name': target_phase_name, 'fraction': 1.0, 'augmentation': augmentation_info[b]},
                                    'bg_phase': {'phase_name': None, 'fraction': 0.0}}
                        all_metadata.append(metadata)
                    file_idx += batch_size_actual
                realistic_generator.enable_peak_augmentations(enable=False)
                print(f'  ✓ Generated {num_augmented_patterns} augmented single-phase patterns, took {time.time() - time_start:.2f}s')

            # ---------- Mixtures ----------
            else:
                print(f"- Generating {target_count} positive patterns with {n_phases} phases...")
                time_start = time.time()
                for i in range(0, target_count, self.generation_batch_size):
                    target_peak_list = realistic_generator.get_peak_list(target_structure)
                    batch_size_actual = min(self.generation_batch_size, target_count - i)

                    # Target fractions
                    target_fraction_range = detection_params.get('target_fraction_range', [0.05, 0.95])
                    low_fraction_bias = detection_params.get('low_fraction_bias', 0)
                    u = torch.rand(batch_size_actual, device=self.device)
                    exponent = 1 / (1 - low_fraction_bias)

                    minf, maxf = target_fraction_range
                    target_fractions = minf + (maxf - minf) * (u ** exponent)
                    remaining_fraction = 1.0 - target_fractions
                    target_fractions = target_fractions.view(-1, 1)

                    # Generate target batch on the fly
                    two_theta, target_patterns = realistic_generator.generate_realistic_pattern(
                        peak_list=target_peak_list,
                        batch_size=batch_size_actual,
                        apply_all_effects=True,
                        enable_impurity=True
                    )

                    # Background fraction sampling (Casted to .float()!)
                    num_bg_phases = n_phases - 1
                    if num_bg_phases >= 2:
                        bg_fractions = torch.from_numpy(
                            np.random.dirichlet(np.ones(num_bg_phases), size=batch_size_actual)
                        ).float().to(self.device) * remaining_fraction.view(-1, 1)
                    else:
                        bg_fractions = remaining_fraction.view(-1, 1)

                    # -------- Instant RAM Background Sampling --------
                    total_bg_needed = batch_size_actual * num_bg_phases

                    sampled_indices = np.random.choice(len(bg_pool_ram), size=total_bg_needed, replace=True)

                    # Slice from RAM, cast to float32, and push to device
                    bg_patterns_flat = torch.tensor(
                        bg_pool_ram[sampled_indices].astype(np.float32),
                        device=self.device
                    )

                    # Reshape
                    bg_patterns = bg_patterns_flat.view(batch_size_actual, num_bg_phases, -1)
                    bg_names_batch = bg_names_ram[sampled_indices].reshape(batch_size_actual, num_bg_phases)

                    # Weighted mixture
                    patterns = (
                        target_patterns * target_fractions +
                        (bg_patterns * bg_fractions.unsqueeze(-1)).sum(dim=1)
                    )

                    # Save
                    patterns_cpu = patterns.cpu().numpy()
                    self._save_patterns(output_dir, patterns_cpu, file_idx, category="positive", is_external=is_external)

                    # Metadata
                    for b in range(batch_size_actual):
                        meta_bg_phase_names = []
                        meta_bg_fractions = []
                        for j in range(num_bg_phases):
                            meta_bg_phase_names.append(str(bg_names_batch[b, j]))
                            meta_bg_fractions.append(float(bg_fractions[b, j].item()))

                        metadata = {
                            'file_name': f'positive_{file_idx + b:05d}.xy',
                            'type': f'positive_{n_phases}_phase_mixture',
                            'target_phase': {'phase_name': target_phase_name, 'fraction': float(target_fractions[b].item())},
                            'bg_phase': {'phase_name': meta_bg_phase_names, 'fraction': meta_bg_fractions}
                        }
                        all_metadata.append(metadata)
                    file_idx += batch_size_actual
                print(f"  ✓ Generated {target_count} positive patterns with {n_phases} phases, took {time.time() - time_start:.2f}s.")

        # ==================== Handling negative patterns ====================
        # background_only_fraction and peak_perturbation_fraction split
        # self.negative_fraction between the two negative types -- both need
        # applying here, or the two counts are each sized as if they alone
        # consumed the entire negative fraction and their sum overshoots
        # total_num * negative_fraction.
        num_perturbation_negatives = int(total_num * self.negative_fraction * self.peak_perturbation_fraction)
        file_idx = 0

        # ---------- Negative background ----------
        num_1_phase = int(total_num * self.negative_fraction * self.background_only_fraction * self.frac_1_phase)
        num_2_phase = int(total_num * self.negative_fraction * self.background_only_fraction * self.frac_2_phase)
        num_3_phase = int(total_num * self.negative_fraction * self.background_only_fraction * self.frac_3_phase)
        num_4_phase = int(total_num * self.negative_fraction * self.background_only_fraction * self.frac_4_phase)
        phase_mixtures = [
            (1, num_1_phase),
            (2, num_2_phase),
            (3, num_3_phase),
            (4, num_4_phase)
        ]

        for n_phases, target_count in phase_mixtures:
            if target_count == 0:
                continue
            print(f"- Generating {target_count} negative patterns with {n_phases} phases...")
            time_start = time.time()

            for i in range(0, target_count, self.generation_batch_size):
                batch_size_actual = min(self.generation_batch_size, target_count - i)

                # Background fraction sampling (Casted to .float()!)
                num_bg_phases = n_phases
                if num_bg_phases >= 2:
                    bg_fractions = torch.from_numpy(
                        np.random.dirichlet(np.ones(num_bg_phases), size=batch_size_actual)
                    ).float().to(self.device)
                else:
                    bg_fractions = torch.ones(batch_size_actual, 1, device=self.device)

                # -------- Instant RAM Background Sampling --------
                total_bg_needed = batch_size_actual * num_bg_phases

                sampled_indices = np.random.choice(len(bg_pool_ram), size=total_bg_needed, replace=True)

                # Slice from RAM, cast to float32, and push to device
                bg_patterns_flat = torch.tensor(
                    bg_pool_ram[sampled_indices].astype(np.float32),
                    device=self.device
                )

                # Reshape
                bg_patterns = bg_patterns_flat.view(batch_size_actual, num_bg_phases, -1)
                bg_names_batch = bg_names_ram[sampled_indices].reshape(batch_size_actual, num_bg_phases)

                # Weighted mixture
                patterns = (bg_patterns * bg_fractions.unsqueeze(-1)).sum(dim=1)

                # Save
                patterns_cpu = patterns.cpu().numpy()
                self._save_patterns(output_dir, patterns_cpu, file_idx, category="negative", is_external=is_external)

                # Metadata
                for b in range(batch_size_actual):
                    meta_bg_phase_names = []
                    meta_bg_fractions = []
                    for j in range(num_bg_phases):
                        meta_bg_phase_names.append(str(bg_names_batch[b, j]))
                        meta_bg_fractions.append(float(bg_fractions[b, j].item()))

                    metadata = {
                        'file_name': f'negative_{file_idx + b:05d}.xy',
                        'type': f'negative_{n_phases}_phase_mixture',
                        'target_phase': {'phase_name': None, 'fraction': 0.0},
                        'bg_phase': {'phase_name': meta_bg_phase_names, 'fraction': meta_bg_fractions}
                    }
                    all_metadata.append(metadata)
                file_idx += batch_size_actual
            print(f"  ✓ Generated {target_count} negative patterns with {n_phases} phases in {time.time() - time_start:.2f}s")

        # ---------- Negative perturbation ----------
        print(f"- Generating {num_perturbation_negatives} negative patterns with peak perturbations...")
        time_start = time.time()
        realistic_generator.enable_peak_perturbations(self.peak_perturbation_config, enable=True)
        for i in range(0, num_perturbation_negatives, self.generation_batch_size):
            target_peak_list = realistic_generator.get_peak_list(target_structure)
            batch_size_actual = min(self.generation_batch_size, num_perturbation_negatives - i)
            two_theta, patterns, applied_perturbations = realistic_generator.generate_realistic_pattern(
                peak_list=target_peak_list,
                batch_size=batch_size_actual,
                apply_all_effects=True,
                enable_impurity=False
            )
            patterns_cpu = patterns.cpu().numpy()
            self._save_patterns(output_dir, patterns_cpu, file_idx, category="negative", is_external=is_external)

            for b in range(batch_size_actual):
                metadata = {'file_name': f'negative_{file_idx + b:05d}.xy', 'type': f'negative_perturbation',
                            'target_phase': {'phase_name': None, 'fraction': 0.0, 'perturbations': applied_perturbations[b]},
                            'bg_phase': {'phase_name': None, 'fraction': 0.0}}
                all_metadata.append(metadata)
            file_idx += batch_size_actual
        realistic_generator.enable_peak_perturbations(enable=False)
        print(f'  ✓ Generated {num_perturbation_negatives} negative patterns with peak perturbations, took {time.time() - time_start:.2f}s.')

        if not self.cache_training_data:
            self._save_metadata(output_dir, all_metadata)

    def _save_patterns(self, output_dir: str, intensities: np.ndarray, start_idx: int,
                    category: str, is_external: bool = False):
        label_dir = Path(output_dir) / category
        label_dir.mkdir(parents=True, exist_ok=True)

        B = intensities.shape[0]
        L = intensities.shape[1]

        # Default is 'npy', not 'xy': this method's own output for use_mask=True
        # is a preprocessed (2, L) intensity+mask array, not a genuine 2θ/
        # intensity pattern -- writing it as ".xy" collides with the ".xy"
        # extension's conventional meaning (angle, intensity) used by
        # UnifiedPatternGenerator's public output and by PatternsDataset's own
        # "2 rows = (intensity, mask)" assumption when it later reads these
        # files back for training. .npy carries no such convention and avoids
        # the ambiguity entirely.
        file_format = self.training_data_config.get('generation_settings', {}).get('file_format', 'npy').lower()
        use_mask = self.model_config.use_mask

        # Generate mask if needed
        if use_mask:
            # 1. Generate Random Angles (B, 1)
            mask_start = self.model_config.mask_start
            mask_end = self.model_config.mask_end

            min_angle = self.xrd_config.min_angle
            max_angle = self.xrd_config.max_angle

            if mask_start >= mask_end:
                raise ValueError(f"Invalid mask range: mask_start ({mask_start}) must be less than mask_end ({mask_end})")
            if mask_start < self.xrd_config.min_angle or mask_end > self.xrd_config.max_angle:
                raise ValueError(f"Mask angles must be within XRD angle range: [{self.xrd_config.min_angle}, {self.xrd_config.max_angle}]")

            # Random floats for start and end angles
            start_angles = np.random.uniform(min_angle, mask_start, size=(B, 1))
            end_angles = np.random.uniform(mask_end, max_angle, size=(B, 1))

            # 2. Convert Angles to Indices (Vectorized)
            span = max_angle - min_angle
            start_indices = ((start_angles - min_angle) / span * L).astype(int)
            end_indices = ((end_angles - min_angle) / span * L).astype(int)
            start_indices = np.clip(start_indices, 0, L)           # (B, 1)
            end_indices = np.clip(end_indices, 0, L)            # (B, 1)

            # 3. Create Mask via Broadcasting
            positions = np.arange(L)[None, :]       # (1, L)

            # Logic: 1.0 if index is between start and end, else 0.0
            mask = (positions >= start_indices) & (positions < end_indices)     # (B, L)
            mask = mask.astype(np.float32)

            # 4. Apply Mask & Format Input
            intensities = np.array([preprocess_xrd_pattern(p, m, self.model_config) for p, m in zip(intensities, mask)], dtype=np.float32)

            # Zero out invisible edges
            intensities = intensities * mask

            intensities = np.concatenate([
                np.expand_dims(intensities, axis=1),
                np.expand_dims(mask, axis=1)
            ], axis=1) # (B, 2, L)

        else:
            intensities = np.array([preprocess_xrd_pattern(p, None, self.model_config) for p in intensities], dtype=np.float32)



        # File caching mode
        # If is_external, then always saving to disk to prevent messing up with training data cache
        if self.cache_training_data and not is_external:
            # Save in memory
            self.cached_patterns[category].extend(intensities)

            label_value = 1 if category.lower() == "positive" else 0
            self.cached_labels.extend([label_value] * len(intensities))

            return

        # Determine total patterns for this category
        if is_external:
            if category.lower() == "positive":
                total_num = int(self.total_num * self.model_config.test_fraction * self.positive_fraction)
            else:
                total_num = int(self.total_num * self.model_config.test_fraction * (1 - self.positive_fraction))
        else:
            if category.lower() == "positive":
                total_num = int(self.total_num * self.positive_fraction)
            else:
                total_num = int(self.total_num * (1 - self.positive_fraction))

        # XY / NPY
        if file_format in ("xy", "npy"):
            for i, p in enumerate(intensities):
                idx = start_idx + i
                filename = f"{category}_{idx:05d}.{file_format}"
                path = label_dir / filename
                arr = np.asarray(p)
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)
                elif arr.ndim == 2 and arr.shape[0] == 1:
                    arr = arr.T
                if file_format == "xy":
                    np.savetxt(path, arr, fmt="%.6f")
                elif file_format == "npy":
                    np.save(path, arr)
            return

        # ZARR
        elif file_format == "zarr":
            zarr_path = label_dir / "patterns.zarr"
            group = zarr.open_group(zarr_path, mode="a")

            # Detect Zarr version
            zarr_ver = tuple(map(int, zarr.__version__.split(".")[:2]))

            # With use_mask=True, `intensities` is (B, 2, L) -- (intensity, mask)
            # per sample -- not (B, L), so the on-disk array needs a matching
            # 3-D shape. Using (total_num, L) here (rank 2) for both cases used
            # to raise a broadcast error on every masked batch write.
            array_shape = (total_num, 2, L) if use_mask else (total_num, L)
            array_chunks = (1, 2, L) if use_mask else (1, L)

            if zarr_ver[0] == 2:
                # Zarr 2.x
                patterns_arr = group.require_dataset(
                    name="patterns",
                    shape=array_shape,
                    chunks=array_chunks,
                    dtype="float32",
                    compressor=None
                )
            else:
                # Zarr 3.x
                patterns_arr = group.require_array(
                    name="patterns",
                    shape=array_shape,
                    chunks=array_chunks,
                    dtype="float32",
                    compressors=None
                )

            end_idx = start_idx + B
            if end_idx > total_num:
                raise RuntimeError(f"Attempt to write beyond allocated size: {end_idx} > {total_num}")

            patterns_arr[start_idx:end_idx] = intensities.astype("float32")
            return

        else:
            raise RuntimeError(f"Unknown file_format: {file_format}")

def evaluate_experimental_pattern_for_website(
        pattern_path: Union[str, Path],
        candidate_cif_paths: List[Union[str, Path]],
        dara_config: DaraConfig,
        pinned_cif_paths: Optional[List[Union[str, Path]]] = None
    ):
        """
        Evaluates an experimental pattern using a single DARA refinement batch.

        Args:
            pattern_path: Path to the XRD pattern file.
            candidate_cif_paths: List of paths to candidate CIF files.
            dara_config: DaraConfig object containing refinement parameters.
            pinned_cif_paths: Optional list of CIF paths that should be always included in refinement.

        Returns:
            The search results object from refined_results.get_search_results() or None if no result.
        """
        from dara import search_phases
        from galaxi.core.config import DaraConfig

        pattern_path = Path(pattern_path)
        candidate_cifs = [Path(p) for p in candidate_cif_paths]
        pinned_cifs = [Path(p) for p in pinned_cif_paths] if pinned_cif_paths else []

        # Combine candidates and pinned CIFs into a single batch
        all_phases = candidate_cifs + pinned_cifs

        # Remove duplicates while preserving order
        unique_phases = []
        seen = set()
        for p in all_phases:
            if p not in seen:
                unique_phases.append(p)
                seen.add(p)

        if not unique_phases:
            print("No phases provided for refinement.")
            return None

        print(f"\nDARA Refinement (Single Batch): Refining {len(unique_phases)} phases: {[p.stem for p in unique_phases]}")

        refined_results = search_phases(
            pattern_path=pattern_path,
            downsized_length=dara_config.downsized_length,
            phases=unique_phases,
            pinned_phases=pinned_cifs,
            max_phases=4,
            wavelength="Cu",
            instrument_profile="Aeris-fds-Pixcel1d-Medipix3",
            express_mode=True,
            enable_angular_cut=dara_config.enable_angular_cut,
            maximum_grouping_distance=dara_config.maximum_grouping_distance,
            grouping_metric=dara_config.grouping_metric,
            phase_params=dara_config.phase_params or {},
            refinement_params=dara_config.refinement_params or {},
            return_search_tree=True,
            record_peak_matcher_scores=True,
            score_coefficients=dara_config.score_coefficients,
            strike_threshold=dara_config.strike_threshold,
            rpb_threshold=dara_config.rpb_threshold,
            overfitting_threshold=dara_config.overfitting_threshold,
            false_peak_threshold=dara_config.false_peak_threshold,
            strain_threshold=dara_config.strain_threshold,
            early_stopping=dara_config.early_stopping,
        )

        if dara_config.show_search_tree and refined_results:
            print(f'Search tree:')
            refined_results.show(stdout=True, idhidden=True)

        return refined_results.get_search_results() if refined_results else None

def load_config(config_file: str) -> Dict:
    """Load configuration from JSON file."""
    with open(config_file, 'r') as f:
        return json.load(f)


def create_default_config(output_file: str = "workflow_config.json"):
    """Create a default configuration file with comprehensive parameters using the modern organized structure."""
    default_config = {
        "ensemble": {
            "use_ensemble": False,
            "ensemble_index": None
        },

        "directories": {
            "references_dir": "References",
            # HDF5 library of pre-simulated single-phase patterns, sampled to
            # build the "other phases" in multi-phase positives and the
            # negatives. Required for step_1_generate_training_data; install it
            # with `galaxi-setup-bg-profiles`. An absolute path in the user data
            # directory, so it resolves from any working directory.
            "bg_profiles_path": get_default_bg_profiles_path(),
            "exp_patterns_dir": "Exp-Patterns",
            "cod_dir": get_default_cod_dir(),
            "output_dir": "workflow_results"
        },

        "training_data_generation": {
            "total_num": 5000,
            "phase_mixture_controls": {
                "num_1_phase_patterns": 1000,
                "num_2_phase_patterns": 500,
                "num_3_phase_patterns": 200,
                "num_4_phase_patterns": 100
            },
            "positive_negative_controls": {
                "positive_fraction": 0.5,
                "negative_fraction": 0.5,
                "positive_single_phase_types": {
                    "clean_fraction": 0.5,
                    "augmented_fraction": 0.5
                },
                "negative_types": {
                    "background_only_fraction": 0.5,
                    "peak_perturbation_fraction": 0.5
                }
            },
            "peak_perturbation_controls": {
                "enable_peak_perturbations": True,
                "perturbation_types": ["removal", "shift", "intensity_change"],
                "perturbation_strength": {
                    "removal_fraction_range": [0.2, 0.6],
                    "shift_std_range": [0.3, 1.5],
                    "intensity_factor_range": [0.1, 10.0]
                }
            },
            "generation_settings": {
                "target_fraction_range": [0.05, 0.95],
                "max_cod_structures": 5000,
                # .npy, not .xy: this pipeline's own saved patterns are
                # preprocessed (intensity, mask) arrays, not genuine
                # (2θ, intensity) pairs -- see _save_patterns()'s comment
                # on why ".xy" specifically is the wrong extension for them.
                "file_format": "npy",
                "save_ground_truth": True
            }
        },

        "test_data_generation": {
            "num_patterns_per_artifact": 100,
            "num_patterns_per_multiphase": 100,
            "artifact_types": [
                "strain", "texture", "background", "peak_shift",
                "crystallite_size", "temperature", "noise", "all_artifacts"
            ],
            "fraction_ranges": [
                [0.1, 0.4],
                [0.4, 0.7],
                [0.7, 1.0]
            ]
        },

        "shared_xrd_generation_config": {
            "basic_parameters": {
                "min_angle": 10.0,
                "max_angle": 80.0,
                "num_points": 4501,
                "convert_to_q": False
            },
            "peak_position_effects": {
                "uniform_shift_range": [-0.25, 0.25],
                "sample_displacement": [-0.2, 0.2],
                "goniometer_radius": 240.0
            },
            "crystallite_size_and_strain": {
                "crystallite_size_range": [5.0, 100.0],
                "microstrain_range": [0.0, 0.003],
                "lattice_strain_range": [0.0, 0.01]
            },
            "instrumental_effects": {
                "instrumental_broadening": {"u": 0.01, "v": -0.005, "w": 0.002},
                "pseudo_voigt_eta_range": [0.3, 0.8]
            },
            "temperature_effects": {
                "temperature_range": [200, 300],
                "atomic_displacement_range": [0.005, 0.02]
            },
            "texture_and_orientation": {
                "texture_range": [0.5, 1.5],
                "weights_low_index": 0.7,
                "low_index": [
                        (1, 0, 0), (0, 1, 0), (0, 0, 1),      # Primary axes
                        (1, 1, 0), (1, 0, 1), (0, 1, 1),      # Face diagonals
                        (1, 1, 1), (-1, 1, 1), (1, -1, 1), (1, 1, -1)  # Body diagonals
                    ],
                "high_index": [(2, 1, 0), (1, 2, 0), (2, 0, 1), (0, 2, 1), (1, 0, 2), (0, 1, 2),
                        (1, 1, 2), (1, 2, 1), (2, 1, 1), (3, 1, 1), (1, 3, 1), (1, 1, 3),
                        (3, 2, 1), (3, 1, 2), (1, 3, 2), (2, 3, 1), (1, 2, 3), (2, 1, 3),
                        (3, 2, 2), (2, 3, 2), (2, 2, 3), (3, 3, 2), (3, 2, 3), (2, 3, 3)
                    ]
            },
            "background_and_noise": {
                "background_level": [0.5, 5.0],
                "noise_level": [0.1, 0.5]
            },
            "diffuse_scattering": {
                "diffuse_scattering_intensity": [5.0, 25.0],
                "diffuse_scattering_b_factor": [0.5, 3.0]
            },
            "amorphous_contributions": {
                "amorphous_intensity": [5.0, 50.0],
                "amorphous_neighbor_distance": [2.0, 4.0],
                "amorphous_disorder": [0.2, 0.8]
            },
            "impurity_peaks": {
                "enable_impurities": True,
                "impurity_num_peaks_range": [1, 10],
                "impurity_intensity_range": [0.0, 25.0],
                "impurity_width_range": [0.05, 0.3],
                "impurity_eta_range": [0.2, 0.9]
            }
        },

        "model_config": {
            "architecture": {
                "conv_channels": [32, 64, 128],
                "conv_kernels": [16, 12, 8],
                "pool_size": [2, 2, 2],
                "fc_sizes": [1024, 512, 128],
                "activation": "relu",
                "use_batch_norm": True,
                "dropout_rate": 0.4,
                "input_size": 4501,
                "model_config": "detection",
                "use_configurable": False
            },
            "training": {
                "num_epochs": 10,
                "learning_rate": 0.001,
                "batch_size": 32,
                "test_fraction": 0.2,
                "val_size": 0.2
            },
            "output": {
                "save_models": True,
                "save_plots": True
            }
        },

        "evaluation": {
            "probability_threshold": 0.5,
            "tolerance": 0.1,
            "save_evaluation_plots": True
        },

        "performance": {
            "max_phases_cod": 1000,
            "num_cpu": None
        }
    }

    with open(output_file, 'w') as f:
        json.dump(default_config, f, indent=2)

    print(f"Default configuration saved to {output_file}")
    return default_config


def create_example_custom_configs():
    """Create example configuration files showing advanced JSON configs."""

    print("Creating example configurations...")

    example_code = '''
# Example: High-quality training configuration using JSON-only approach
from galaxi.workflows.streamlined_workflow import StreamlinedWorkflow
import json

# Advanced workflow config with high-quality settings
advanced_config = {
    "directories": {
        "references_dir": "References",
        "output_dir": "high_quality_results",
        "cod_dir": "cod_sample"
    },

    "training_data_generation": {
        "phase_mixture_controls": {
            "num_1_phase_patterns": 2500,    # More training data
            "num_2_phase_patterns": 2500,
            "num_3_phase_patterns": 2500,
            "num_4_phase_patterns": 2500
        },
        "positive_negative_controls": {
            "positive_fraction": 0.6,        # More positive examples
            "negative_fraction": 0.4
        },
        "generation_settings": {
            "target_fraction_range": [0.05, 0.95],
            "max_cod_structures": 10000      # More COD variety
        }
    },

    "shared_xrd_generation_config": {
        "basic_parameters": {
            "min_angle": 5.0,               # Wider angular range
            "max_angle": 90.0,
            "num_points": 8001              # Higher resolution
        },
        "crystallite_size_and_strain": {
            "crystallite_size_range": [2.0, 200.0],  # Broader size range
            "microstrain_range": [0.0, 0.008],       # More strain variation
            "lattice_strain_range": [0.0, 0.03]      # Higher strain effects
        },
        "texture_and_orientation": {
            "texture_range": [0.3, 2.0]     # Stronger texture effects
        },
        "temperature_effects": {
            "temperature_range": [100, 800] # Wider temperature range
        },
        "background_and_noise": {
            "background_level": [0.1, 15.0], # More background variation
            "noise_level": [0.05, 1.5]       # More noise variation
        },
        "diffuse_scattering": {
            "diffuse_scattering_intensity": [0.0, 30.0]  # More diffuse scattering
        },
        "amorphous_contributions": {
            "amorphous_intensity": [0.0, 60.0]  # More amorphous content
        },
        "impurity_peaks": {
            "enable_impurities": True,
            "impurity_intensity_range": [0.0, 15.0]  # More impurities
        }
    },

    "model_config": {
        "architecture": {
            "use_pre_norm": True,
            "conv_channels": [64, 128, 256, 512],    # Deeper CNN architecture
            "conv_kernels": [24, 20, 16, 12],        # Larger kernels
            "fc_sizes": [2048, 1024, 512, 128],      # Larger fully connected layers
            "dropout_rate": 0.3,                     # Less dropout
            "activation": "leaky_relu"               # Different activation
        },
        "training": {
            "num_epochs": 25,                        # More training epochs
            "learning_rate": 0.0005,                 # Lower learning rate
            "batch_size": 64                         # Larger batch size
        }
    },

    "test_data_generation": {
        "num_patterns_per_artifact": 500,
        "num_patterns_per_multiphase": 500
    },

    "evaluation": {
        "probability_threshold": 0.5
    }
}

# Save the advanced config
with open("advanced_workflow_config.json", 'w') as f:
    json.dump(advanced_config, f, indent=2)

# Initialize workflow with JSON-only configuration
workflow = StreamlinedWorkflow(config=advanced_config)

# Run the workflow
workflow.run_complete_workflow()
'''

    with open("example_advanced_workflow.py", 'w') as f:
        f.write(example_code)

    print("✓ Created example_advanced_workflow.py")
    print("This file shows how to use advanced JSON configurations.")
    return example_code


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description="Streamlined Phase Detection Workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with configuration file
  galaxi-workflow --config workflow_config.json

  # Basic usage with command line arguments
  galaxi-workflow --references References/ --exp-patterns Exp-Patterns/ --num-patterns 100

  # Generate default configuration
  galaxi-workflow --create-config

  # Create example files showing custom XRD and Model configurations
  galaxi-workflow --create-examples

  # Check current status without running
  galaxi-workflow --config workflow_config.json --check-status

Advanced Usage with JSON Configuration:
  # Edit your workflow_config.json file to customize:
  # - Training data generation (phase mixtures, positive/negative ratios)
  # - XRD generation parameters (noise, background, impurities)
  # - Model architecture and training parameters
  # - Test data generation settings
  # - Evaluation parameters

  # Then run with:
  galaxi-workflow --config your_custom_config.json
        """
    )

    parser.add_argument('--config', type=str,
                       help='Path to JSON configuration file')
    parser.add_argument('--references', type=str, default='References',
                       help='Directory containing reference CIF files')
    parser.add_argument('--exp-patterns', type=str, default='Exp-Patterns',
                       help='Directory containing experimental patterns')
    parser.add_argument('--output', type=str, default='workflow_results',
                       help='Output directory for results')
    parser.add_argument('--num-patterns', type=int, default=100,
                       help='Number of test patterns per category')
    parser.add_argument('--training-patterns', type=int, default=5000,
                       help='Number of training patterns per phase')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--create-config', action='store_true',
                       help='Create default configuration file and exit')
    parser.add_argument('--create-examples', action='store_true',
                       help='Create example files showing custom configurations and exit')
    parser.add_argument('--check-status', action='store_true',
                       help='Check workflow status without running')

    args = parser.parse_args()

    configure_cli_logging()

    if args.create_config:
        create_default_config()
        return 0

    if args.create_examples:
        create_example_custom_configs()
        return 0

    # Load or create configuration
    if args.config and Path(args.config).exists():
        config = load_config(args.config)
        print(f"Loaded configuration from {args.config}")
    else:
        config = {
            'references_dir': args.references,
            'exp_patterns_dir': args.exp_patterns,
            'output_dir': args.output,
            'num_patterns_per_artifact': args.num_patterns,
            'num_patterns_per_multiphase': args.num_patterns,
            'num_training_patterns': args.training_patterns,
            'training_epochs': args.epochs
        }
        print("Using command-line configuration")

    # Initialize workflow with JSON config only
    workflow = StreamlinedWorkflow(config)

    if args.check_status:
        # Just check status and exit
        workflow.check_workflow_status()
        return 0

    # Run workflow. Exit non-zero when it could not complete, so this is usable
    # from a script or CI.
    return 0 if workflow.run_complete_workflow() else 1


if __name__ == "__main__":
    sys.exit(main())
