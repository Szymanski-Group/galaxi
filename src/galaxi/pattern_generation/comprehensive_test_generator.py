"""
Comprehensive test pattern generator for phase detection model evaluation.

This module generates organized test datasets with:
1. Single-phase patterns with individual artifacts
2. Multi-phase patterns categorized by target phase fractions
3. Negative examples with unphysical perturbations
"""

import os
import json
import numpy as np
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from ..core.pattern_generator import UnifiedPatternGenerator, _make_generator, _generate_one_pattern
from ..core.config import XRDGenerationConfig
from ..paths import get_default_cod_dir
from .realistic_xrd import RealisticXRDGenerator
from pymatgen.core import Structure


class ComprehensiveTestGenerator:
    """Enhanced test pattern generator for comprehensive model evaluation."""

    def __init__(self,
                 reference_dir: str,
                 cod_dir: Optional[str] = None,
                 fraction_ranges: Optional[List[List[float]]] = None,
                 xrd_config: Optional[XRDGenerationConfig] = None):
        """
        Initialize the comprehensive test generator.

        Args:
            reference_dir: Directory containing reference CIF files
            cod_dir: Directory containing COD background phases (default:
                the `data/cod` directory of the installed galaxi package;
                see `galaxi.scripts.setup_cod`)
            fraction_ranges: Custom fraction ranges for test data categorization
            xrd_config: XRD generation configuration (if None, uses default)
        """
        self.reference_dir = Path(reference_dir)
        self.cod_dir = Path(cod_dir if cod_dir is not None else get_default_cod_dir())

        # Use provided XRD config or default
        if xrd_config is not None:
            self.xrd_config = xrd_config
            self.base_generator = UnifiedPatternGenerator(reference_dir=str(reference_dir), config=xrd_config)
        else:
            self.xrd_config = XRDGenerationConfig()  # Default config
            self.base_generator = UnifiedPatternGenerator(reference_dir=str(reference_dir))

        # Set custom fraction ranges or use default
        if fraction_ranges:
            self.fraction_ranges = [(float(r[0]), float(r[1])) for r in fraction_ranges]
        else:
            # Default fraction ranges
            self.fraction_ranges = [
                (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
                (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9)
            ]

        # Load reference phases
        self.reference_phases = self._load_reference_phases()

        # Load COD phases (sample for efficiency)
        self.cod_phases = self._load_cod_phases(max_phases=1000)

    def _load_reference_phases(self) -> Dict[str, Structure]:
        """Load reference phase structures."""
        phases = {}
        for cif_file in self.reference_dir.glob("*.cif"):
            try:
                structure = Structure.from_file(str(cif_file))
                phase_name = cif_file.stem  # Use full filename without extension
                phases[phase_name] = structure
            except Exception as e:
                print(f"Warning: Could not load {cif_file}: {e}")
        return phases

    def _load_cod_phases(self, max_phases: int = 1000) -> List[Structure]:
        """Load sample of COD phase structures."""
        cod_files = list(self.cod_dir.glob("*.cif"))
        selected_files = random.sample(cod_files, min(len(cod_files), max_phases))

        phases = []
        error_count = 0
        for cif_file in selected_files:
            try:
                structure = Structure.from_file(str(cif_file))
                phases.append(structure)
            except Exception:
                error_count += 1
                continue  # Skip problematic files

        print(f"Loaded {len(phases)} COD background phases"
              + (f" ({error_count} skipped due to errors)" if error_count else ""))
        return phases

    def generate_artifact_configs(self) -> Dict[str, XRDGenerationConfig]:
        """Generate configurations for individual artifacts."""

        base_config = XRDGenerationConfig(
            min_angle=self.xrd_config.min_angle,
            max_angle=self.xrd_config.max_angle,
            num_points=self.xrd_config.num_points,
            convert_to_q=self.xrd_config.convert_to_q,
            uniform_shift_range=(0.0, 0.0),
            crystallite_size_range=(50.0, 50.0),
            microstrain_range=(0.0, 0.0),
            lattice_strain_range=(0.0, 0.0),
            texture_range=(1.0, 1.0),
            temperature_range=(298, 298),
            background_level=(1.0, 1.0),
            noise_level=(0.1, 0.1),
            diffuse_scattering_intensity=(0.0, 0.0),
            amorphous_intensity=(0.0, 0.0),
            impurity_intensity_range=(0.0, 0.0),
            enable_impurities=False
        )

        artifacts = {
            'strain': XRDGenerationConfig(
                min_angle=base_config.min_angle,
                max_angle=base_config.max_angle,
                microstrain_range=(0.002, 0.008),
                lattice_strain_range=(0.005, 0.02),
                enable_impurities=False
            ),
            'texture': XRDGenerationConfig(
                min_angle=base_config.min_angle,
                max_angle=base_config.max_angle,
                texture_range=(0.3, 2.0),
                enable_impurities=False
            ),
            'background': XRDGenerationConfig(
                min_angle=base_config.min_angle,
                max_angle=base_config.max_angle,
                background_level=(2.0, 8.0),
                amorphous_intensity=(10.0, 40.0),
                enable_impurities=False
            ),
            'peak_shift': XRDGenerationConfig(
                min_angle=base_config.min_angle,
                max_angle=base_config.max_angle,
                uniform_shift_range=(-0.15, 0.15),
                enable_impurities=False
            ),
            'crystallite_size': XRDGenerationConfig(
                min_angle=base_config.min_angle,
                max_angle=base_config.max_angle,
                crystallite_size_range=(3.0, 80.0),
                enable_impurities=False
            ),
            'temperature': XRDGenerationConfig(
                min_angle=base_config.min_angle,
                max_angle=base_config.max_angle,
                temperature_range=(200, 400),
                enable_impurities=False
            ),
            'noise': XRDGenerationConfig(
                min_angle=base_config.min_angle,
                max_angle=base_config.max_angle,
                noise_level=(0.3, 1.0),
                enable_impurities=False
            ),
            'all_artifacts': XRDGenerationConfig(
                min_angle=10.0,
                max_angle=80.0,
                uniform_shift_range=(-0.1, 0.1),
                crystallite_size_range=(5.0, 100.0),
                microstrain_range=(0.001, 0.005),
                lattice_strain_range=(0.002, 0.015),
                texture_range=(0.4, 1.8),
                temperature_range=(220, 350),
                background_level=(1.5, 6.0),
                noise_level=(0.15, 0.8),
                diffuse_scattering_intensity=(5.0, 25.0),
                amorphous_intensity=(5.0, 35.0),
                impurity_intensity_range=(0.0, 15.0),
                enable_impurities=True
            )
        }

        return artifacts

    def generate_single_phase_test_patterns(self, target_phase: str, num_patterns: int = 100) -> Dict[str, List]:
        """
        Generate single-phase test patterns with individual artifacts.

        Args:
            target_phase: Name of the target phase
            num_patterns: Number of patterns per artifact type

        Returns:
            Dictionary with patterns organized by artifact type
        """
        if target_phase not in self.reference_phases:
            raise ValueError(f"Target phase {target_phase} not found in references")

        artifact_configs = self.generate_artifact_configs()
        patterns_by_artifact = {}

        structure = self.reference_phases[target_phase]

        for artifact_name, config in artifact_configs.items():
            print(f"Generating {num_patterns} patterns for {target_phase} with {artifact_name}...")

            patterns = []
            for i in range(num_patterns):
                generator = _make_generator(config)
                two_theta, intensity = _generate_one_pattern(generator, structure)
                patterns.append(intensity)

            patterns_by_artifact[artifact_name] = patterns

        return patterns_by_artifact

    def generate_multi_phase_test_patterns(self, target_phase: str, num_patterns: int = 100) -> Dict[str, Dict[str, List]]:
        """
        Generate multi-phase test patterns categorized by target fraction.

        Args:
            target_phase: Name of the target phase
            num_patterns: Number of patterns per category

        Returns:
            Dictionary organized by phase count and fraction range
        """
        if target_phase not in self.reference_phases:
            raise ValueError(f"Target phase {target_phase} not found in references")

        target_structure = self.reference_phases[target_phase]
        results = {}

        # Use configured fraction ranges
        fraction_ranges = self.fraction_ranges

        # Configuration with all artifacts for multi-phase
        config = XRDGenerationConfig(
            min_angle=10.0,
            max_angle=80.0,
            uniform_shift_range=(-0.08, 0.08),
            crystallite_size_range=(8.0, 120.0),
            microstrain_range=(0.001, 0.004),
            lattice_strain_range=(0.003, 0.012),
            texture_range=(0.5, 1.6),
            temperature_range=(230, 330),
            background_level=(1.0, 5.0),
            noise_level=(0.1, 0.6),
            diffuse_scattering_intensity=(3.0, 20.0),
            amorphous_intensity=(2.0, 30.0),
            impurity_intensity_range=(0.0, 12.0),
            enable_impurities=True
        )

        for n_phases in [2, 3, 4]:  # 2-phase, 3-phase, 4-phase
            results[f"{n_phases}_phase"] = {}

            for frac_min, frac_max in fraction_ranges:
                fraction_key = f"{frac_min:.1f}-{frac_max:.1f}"
                patterns = []

                print(f"Generating {num_patterns} {n_phases}-phase patterns with target fraction {fraction_key}...")

                for i in range(num_patterns):
                    # Generate target fraction in range
                    target_fraction = np.random.uniform(frac_min, frac_max)

                    # Select random COD phases for background
                    background_phases = random.sample(self.cod_phases, n_phases - 1)

                    # Generate remaining fractions
                    remaining_fraction = 1.0 - target_fraction
                    background_fractions = np.random.dirichlet([1.0] * (n_phases - 1)) * remaining_fraction

                    # Generate individual patterns
                    target_generator = _make_generator(config)
                    _, target_pattern = _generate_one_pattern(target_generator, target_structure)

                    # Generate background patterns
                    background_patterns = []
                    for bg_structure in background_phases:
                        bg_generator = _make_generator(config)
                        _, bg_pattern = _generate_one_pattern(bg_generator, bg_structure)
                        background_patterns.append(bg_pattern)

                    # Combine patterns
                    combined_pattern = target_fraction * target_pattern
                    for bg_pattern, bg_frac in zip(background_patterns, background_fractions):
                        combined_pattern += bg_frac * bg_pattern

                    # Add noise and normalize
                    noise = np.random.normal(0, 0.3, combined_pattern.shape)
                    combined_pattern += noise
                    combined_pattern = np.maximum(combined_pattern, 0)

                    if np.max(combined_pattern) > 0:
                        combined_pattern = 100 * combined_pattern / np.max(combined_pattern)

                    patterns.append(combined_pattern)

                results[f"{n_phases}_phase"][fraction_key] = patterns

        return results

    def generate_negative_examples(self, target_phase: str,
                                 num_patterns_per_artifact: int = 100,
                                 num_patterns_per_multiphase: int = 100,
                                 num_patterns: int = None) -> Dict[str, List]:
        """
        Generate negative examples where target phase is not present.

        Args:
            target_phase: Name of the target phase (for reference)
            num_patterns_per_artifact: Number of patterns per artifact type
            num_patterns_per_multiphase: Number of patterns per multi-phase combination
            num_patterns: Legacy parameter for backward compatibility

        Returns:
            Dictionary with negative example patterns
        """
        # Backward compatibility
        if num_patterns is not None:
            num_patterns_per_artifact = num_patterns
            num_patterns_per_multiphase = num_patterns

        print(f"\n--- Starting generate_negative_examples for {target_phase} ---")
        print(f"Number of patterns per artifact: {num_patterns_per_artifact}")
        print(f"Number of patterns per multi-phase: {num_patterns_per_multiphase}")
        print(f"Available COD phases: {len(self.cod_phases)}")

        results = {}

        # Configuration for negative examples
        config = XRDGenerationConfig(
            min_angle=10.0,
            max_angle=80.0,
            uniform_shift_range=(-0.1, 0.1),
            crystallite_size_range=(10.0, 100.0),
            microstrain_range=(0.001, 0.005),
            lattice_strain_range=(0.002, 0.015),
            texture_range=(0.5, 1.5),
            temperature_range=(250, 350),
            background_level=(1.0, 4.0),
            noise_level=(0.1, 0.5),
            diffuse_scattering_intensity=(5.0, 25.0),
            amorphous_intensity=(5.0, 40.0),
            impurity_intensity_range=(0.0, 20.0),
            enable_impurities=True
        )

        print(f"Config created with min_angle={config.min_angle}, max_angle={config.max_angle}")

        # Generate unphysical perturbation negative examples using existing method
        if target_phase in self.reference_phases:
            print(f"\n--- Starting unphysical perturbation negative generation ---")
            target_structure = self.reference_phases[target_phase]
            unphysical_patterns = []

            print(f"Generating {num_patterns_per_artifact} unphysical perturbation patterns...")

            for i in range(num_patterns_per_artifact):
                pattern_start = time.time()
                # Show progress every 5 patterns or at start/end
                if i % 5 == 0 or i == num_patterns_per_artifact - 1:
                    print(f"  Progress: {i+1}/{num_patterns_per_artifact}...", end=" ", flush=True)

                try:
                    # Use the existing _generate_negative_perturbed_pattern method
                    perturbed_pattern, metadata = self.base_generator._generate_negative_perturbed_pattern(
                        target_structure,
                        config
                    )

                    unphysical_patterns.append(perturbed_pattern)

                    pattern_elapsed = time.time() - pattern_start
                    if i % 5 == 0 or i == num_patterns_per_artifact - 1:
                        print(f"OK")

                except Exception as e:
                    pattern_elapsed = time.time() - pattern_start
                    error_msg = str(e) if str(e) else f"{type(e).__name__}: {e.__class__.__module__}"
                    if i % 5 == 0 or i == num_patterns_per_artifact - 1:
                        print(f"FAILED: {error_msg}")
                    continue

            print(f"Generated {len(unphysical_patterns)} unphysical perturbation negative patterns")
            results['negative_unphysical_perturbations'] = unphysical_patterns

        # Multi-phase negatives (no target phase)
        for n_phases in [2, 3, 4]:
            print(f"\n--- Starting {n_phases}-phase negative generation ---")
            patterns = []

            print(f"Generating {num_patterns_per_multiphase} {n_phases}-phase patterns without target...")

            for i in range(num_patterns_per_multiphase):
                pattern_start = time.time()
                # Show progress every 10 patterns or at start/end
                if i % 10 == 0 or i == num_patterns_per_multiphase - 1:
                    print(f"  Progress: {i+1}/{num_patterns_per_multiphase}...", end=" ", flush=True)

                try:
                    # Select random COD phases (excluding target)
                    selected_phases = random.sample(self.cod_phases, n_phases)

                    # Generate random fractions
                    fractions = np.random.dirichlet([1.0] * n_phases)

                    # Generate and combine patterns
                    combined_pattern = np.zeros(4501)
                    for j, (structure, fraction) in enumerate(zip(selected_phases, fractions)):
                        generator = _make_generator(config)
                        _, pattern = _generate_one_pattern(generator, structure)
                        combined_pattern += fraction * pattern

                    # Add noise and normalize
                    noise = np.random.normal(0, 0.2, combined_pattern.shape)
                    combined_pattern += noise
                    combined_pattern = np.maximum(combined_pattern, 0)

                    if np.max(combined_pattern) > 0:
                        combined_pattern = 100 * combined_pattern / np.max(combined_pattern)

                    patterns.append(combined_pattern)

                    pattern_elapsed = time.time() - pattern_start
                    if i % 10 == 0 or i == num_patterns_per_multiphase - 1:
                        print(f"OK")

                except Exception as e:
                    pattern_elapsed = time.time() - pattern_start
                    error_msg = str(e) if str(e) else f"{type(e).__name__}: {e.__class__.__module__}"
                    if i % 10 == 0 or i == num_patterns_per_multiphase - 1:
                        print(f"FAILED: {error_msg}")
                    continue

            print(f"Generated {len(patterns)} {n_phases}-phase negative patterns")
            results[f'negative_{n_phases}_phase'] = patterns

        print(f"\n--- generate_negative_examples summary ---")
        total_patterns = sum(len(patterns) for patterns in results.values())
        print(f"Total negative patterns generated: {total_patterns}")
        for category, patterns in results.items():
            print(f"  {category}: {len(patterns)} patterns")

        return results

    def save_comprehensive_test_data(self, target_phase: str, output_dir: str,
                                   num_patterns_per_artifact: int = 100,
                                   num_patterns_per_multiphase: int = 100,
                                   num_patterns: int = None):
        """
        Generate and save comprehensive test data for a target phase.

        Args:
            target_phase: Name of the target phase
            output_dir: Output directory for test data
            num_patterns_per_artifact: Number of patterns per artifact type
            num_patterns_per_multiphase: Number of patterns per multi-phase combination
            num_patterns: Legacy parameter for backward compatibility
        """
        # Backward compatibility
        if num_patterns is not None:
            num_patterns_per_artifact = num_patterns
            num_patterns_per_multiphase = num_patterns
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Create main subdirectories
        (output_path / "positive").mkdir(exist_ok=True)
        (output_path / "negative").mkdir(exist_ok=True)

        ground_truth = {}
        two_theta = np.linspace(10, 80, 4501)

        # Generate positive examples
        print(f"\n=== Generating positive examples for {target_phase} ===")

        # Single-phase with individual artifacts
        single_phase_patterns = self.generate_single_phase_test_patterns(target_phase, num_patterns_per_artifact)

        for artifact, patterns in single_phase_patterns.items():
            artifact_dir = output_path / "positive" / "single_phase" / artifact
            artifact_dir.mkdir(parents=True, exist_ok=True)

            for i, pattern in enumerate(patterns):
                filename = f"{artifact}_{i:04d}.xy"
                filepath = artifact_dir / filename

                pattern_data = np.column_stack([two_theta, pattern])
                np.savetxt(filepath, pattern_data, fmt='%.6f')

                ground_truth[str(filepath.relative_to(output_path))] = {
                    'target_present': True,
                    'target_phase': target_phase,
                    'pattern_type': 'single_phase',
                    'artifact_type': artifact,
                    'target_fraction': 1.0 if artifact != 'all_artifacts' else 0.8
                }

        # Multi-phase with fraction categorization
        multi_phase_patterns = self.generate_multi_phase_test_patterns(target_phase, num_patterns_per_multiphase)

        for phase_count, fraction_data in multi_phase_patterns.items():
            for fraction_range, patterns in fraction_data.items():
                fraction_dir = output_path / "positive" / "multi_phase" / phase_count / fraction_range
                fraction_dir.mkdir(parents=True, exist_ok=True)

                for i, pattern in enumerate(patterns):
                    filename = f"{phase_count}_{fraction_range}_{i:04d}.xy"
                    filepath = fraction_dir / filename

                    pattern_data = np.column_stack([two_theta, pattern])
                    np.savetxt(filepath, pattern_data, fmt='%.6f')

                    frac_min, frac_max = map(float, fraction_range.split('-'))
                    avg_fraction = (frac_min + frac_max) / 2

                    ground_truth[str(filepath.relative_to(output_path))] = {
                        'target_present': True,
                        'target_phase': target_phase,
                        'pattern_type': 'multi_phase',
                        'n_phases': int(phase_count.split('_')[0]),
                        'target_fraction_range': fraction_range,
                        'target_fraction_approx': avg_fraction
                    }

        # Generate negative examples
        print(f"\n=== Generating negative examples for {target_phase} ===")
        print(f"About to call generate_negative_examples with {num_patterns_per_artifact} patterns per artifact and {num_patterns_per_multiphase} patterns per multi-phase...")
        neg_start_time = time.time()

        negative_patterns = self.generate_negative_examples(target_phase, num_patterns_per_artifact, num_patterns_per_multiphase)

        neg_elapsed = time.time() - neg_start_time
        print(f"Negative examples generation completed in {neg_elapsed:.1f}s")

        print("Negative pattern categories generated:")
        for category, patterns in negative_patterns.items():
            print(f"  {category}: {len(patterns)} patterns")

        for category, patterns in negative_patterns.items():
            category_dir = output_path / "negative" / category
            category_dir.mkdir(parents=True, exist_ok=True)

            for i, pattern in enumerate(patterns):
                filename = f"{category}_{i:04d}.xy"
                filepath = category_dir / filename

                pattern_data = np.column_stack([two_theta, pattern])
                np.savetxt(filepath, pattern_data, fmt='%.6f')

                # Set appropriate metadata based on category
                if category == 'negative_unphysical_perturbations':
                    pattern_type_info = 'perturbed_target'
                else:
                    pattern_type_info = 'background_only'

                ground_truth[str(filepath.relative_to(output_path))] = {
                    'target_present': False,
                    'target_phase': target_phase,
                    'pattern_type': 'negative',
                    'negative_category': category,
                    'pattern_type_info': pattern_type_info,
                    'target_fraction': 0.0
                }

        # Save ground truth
        gt_file = output_path / "comprehensive_ground_truth.json"
        with open(gt_file, 'w') as f:
            json.dump(ground_truth, f, indent=2)

        print(f"\n✓ Comprehensive test data saved to {output_dir}")
        print(f"✓ Ground truth saved to {gt_file}")

        return ground_truth
