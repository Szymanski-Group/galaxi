import os
import numpy as np
import random
from pymatgen.core import Structure
from .realistic_xrd import RealisticXRDGenerator
from ..core.pattern_generator import _generate_one_pattern
from itertools import combinations
import multiprocessing
from multiprocessing import Pool, Manager


class MultiPhasePatternGenerator:
    """
    Class used to generate XRD patterns containing multiple phases for phase counting training.
    """

    def __init__(self, reference_dir, num_patterns_per_count=100, max_phases=4,
                 min_phase_fraction=0.1, min_angle=10.0, max_angle=80.0, enable_impurities=False,
                 # Realistic XRD parameters with explicit defaults
                 uniform_shift_range=(-0.05, 0.05), sample_displacement=(-0.2, 0.2), goniometer_radius=240.0,
                 crystallite_size_range=(10.0, 200.0), microstrain_range=(0.0, 0.003),
                 instrumental_broadening={'u': 0.01, 'v': -0.005, 'w': 0.002},
                 pseudo_voigt_eta_range=(0.3, 0.8), temperature_range=(250, 350),
                 atomic_displacement_range=(0.005, 0.02), texture_range=(0.8, 1.2),
                 preferred_orientation_directions=None, lattice_strain_range=(0.0, 0.03),
                 background_level=(0.5, 2.0), noise_level=(0.1, 0.5),
                 diffuse_scattering_intensity=(0.0, 25.0), diffuse_scattering_b_factor=(0.5, 3.0),
                 amorphous_intensity=(0.0, 25.0), amorphous_neighbor_distance=(2.0, 4.0),
                 amorphous_disorder=(0.2, 0.8), impurity_num_peaks_range=(1, 10),
                 impurity_intensity_range=(0.5, 70.0), impurity_width_range=(0.05, 0.3),
                 impurity_eta_range=(0.2, 0.9)):
        """
        Args:
            reference_dir: path to directory containing CIFs for reference phases
            num_patterns_per_count: number of patterns to generate for each phase count
            max_phases: maximum number of phases in a single pattern
            min_phase_fraction: minimum fraction that the smallest phase can have
            min_angle: minimum 2θ angle
            max_angle: maximum 2θ angle
            enable_impurities: if True, include impurity peaks in generated patterns

            # Realistic XRD parameters with explicit defaults from DEFAULT_PARAMS:
            uniform_shift_range: tuple of (min, max) uniform peak shift in degrees 2θ
            sample_displacement: tuple of (min, max) sample displacement in mm
            goniometer_radius: goniometer radius in mm
            crystallite_size_range: tuple of (min, max) crystallite size in nm
            microstrain_range: tuple of (min, max) microstrain (unitless)
            instrumental_broadening: dict with 'u', 'v', 'w' parameters
            pseudo_voigt_eta_range: tuple of (min, max) Lorentzian fraction
            temperature_range: tuple of (min, max) temperature in K
            atomic_displacement_range: tuple of (min, max) B-factors in Angstrom^2
            texture_range: tuple of (min, max) March-Dollase parameter
            preferred_orientation_directions: list of (h, k, l) tuples for random orientation selection
            lattice_strain_range: tuple of (min, max) lattice strain applied to unit cell
            background_level: tuple of (min, max) background as % of max intensity
            noise_level: tuple of (min, max) noise as % of max intensity
            diffuse_scattering_intensity: tuple of (min, max) diffuse scattering % of max intensity
            diffuse_scattering_b_factor: tuple of (min, max) isotropic temperature factor
            amorphous_intensity: tuple of (min, max) amorphous % of max intensity
            amorphous_neighbor_distance: tuple of (min, max) nearest-neighbor distances in Angstrom
            amorphous_disorder: tuple of (min, max) disorder parameter σ in Angstrom
            impurity_num_peaks_range: tuple of (min, max) number of impurity peaks
            impurity_intensity_range: tuple of (min, max) impurity intensity % of max
            impurity_width_range: tuple of (min, max) impurity FWHM in degrees
            impurity_eta_range: tuple of (min, max) impurity pseudo-Voigt mixing parameter
        """
        self.num_cpu = multiprocessing.cpu_count()
        self.ref_dir = reference_dir
        self.num_patterns_per_count = num_patterns_per_count
        self.max_phases = max_phases
        self.min_phase_fraction = min_phase_fraction
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.enable_impurities = enable_impurities

        # Set preferred orientation directions (use default list if None)
        if preferred_orientation_directions is None:
            preferred_orientation_directions = [
                (1, 0, 0), (0, 1, 0), (0, 0, 1),      # Primary axes
                (1, 1, 0), (1, 0, 1), (0, 1, 1),      # Face diagonals
                (1, 1, 1), (-1, 1, 1), (1, -1, 1), (1, 1, -1),  # Body diagonals
                (2, 1, 0), (1, 2, 0), (2, 0, 1), (0, 2, 1), (1, 0, 2), (0, 1, 2),  # Higher index
            ]

        # Store realistic XRD parameters
        self.params = {
            'uniform_shift_range': uniform_shift_range,
            'sample_displacement': sample_displacement,
            'goniometer_radius': goniometer_radius,
            'crystallite_size_range': crystallite_size_range,
            'microstrain_range': microstrain_range,
            'instrumental_broadening': instrumental_broadening,
            'pseudo_voigt_eta_range': pseudo_voigt_eta_range,
            'temperature_range': temperature_range,
            'atomic_displacement_range': atomic_displacement_range,
            'texture_range': texture_range,
            'preferred_orientation_directions': preferred_orientation_directions,
            'lattice_strain_range': lattice_strain_range,
            'background_level': background_level,
            'noise_level': noise_level,
            'diffuse_scattering_intensity': diffuse_scattering_intensity,
            'diffuse_scattering_b_factor': diffuse_scattering_b_factor,
            'amorphous_intensity': amorphous_intensity,
            'amorphous_neighbor_distance': amorphous_neighbor_distance,
            'amorphous_disorder': amorphous_disorder,
            'impurity_num_peaks_range': impurity_num_peaks_range,
            'impurity_intensity_range': impurity_intensity_range,
            'impurity_width_range': impurity_width_range,
            'impurity_eta_range': impurity_eta_range
        }

        # Load all structures
        self.structures = []
        self.phase_names = []
        for filename in sorted(os.listdir(self.ref_dir)):
            if filename.endswith('.cif'):
                structure = Structure.from_file(os.path.join(self.ref_dir, filename))
                phase_name = filename.replace('.cif', '')
                self.structures.append(structure)
                self.phase_names.append(phase_name)

        print(f"Loaded {len(self.structures)} reference phases: {self.phase_names}")

    def generate_phase_weights(self, num_phases):
        """
        Generate random weights for phases ensuring minimum fraction constraint.

        Args:
            num_phases: Number of phases in the mixture

        Returns:
            weights: List of normalized weights
        """
        # Generate random weights
        weights = np.random.uniform(self.min_phase_fraction, 1.0, num_phases)

        # Ensure the smallest weight meets the minimum fraction requirement
        while min(weights) / sum(weights) < self.min_phase_fraction:
            weights = np.random.uniform(self.min_phase_fraction, 1.0, num_phases)

        # Normalize weights to sum to 1
        weights = weights / np.sum(weights)

        return weights.tolist()

    def generate_single_phase_pattern(self, structure_idx):
        """
        Generate a single-phase pattern with realistic artifacts.

        Args:
            structure_idx: Index of the structure to use

        Returns:
            pattern: Generated XRD pattern
        """
        structure = self.structures[structure_idx]

        # Generate pattern with RealisticXRDGenerator. min_angle/max_angle/num_points
        # go inside `params` -- the constructor only accepts a single `params` dict,
        # not separate keyword arguments.
        generator = RealisticXRDGenerator(params={
            **self.params,
            'min_angle': self.min_angle,
            'max_angle': self.max_angle,
            'num_points': 4501,
        })

        # Enable impurity peaks if requested
        if self.enable_impurities:
            generator.enable_impurity_peaks(
                enable=True,
                num_peaks_range=self.params['impurity_num_peaks_range'],
                intensity_range=self.params['impurity_intensity_range'],
                width_range=self.params['impurity_width_range'],
                eta_range=self.params['impurity_eta_range']
            )

        # Generate single pattern
        two_theta, intensity = _generate_one_pattern(generator, structure)
        return intensity

    def generate_multi_phase_pattern(self, structure_indices, weights):
        """
        Generate a multi-phase pattern by combining individual phase patterns.

        Args:
            structure_indices: List of structure indices to combine
            weights: List of weights for each phase

        Returns:
            combined_pattern: Combined XRD pattern
        """
        # Generate individual phase patterns
        phase_patterns = []
        for idx in structure_indices:
            pattern = self.generate_single_phase_pattern(idx)
            phase_patterns.append(pattern)

        # Combine patterns with weights
        combined_pattern = np.zeros_like(phase_patterns[0])
        for pattern, weight in zip(phase_patterns, weights):
            combined_pattern += weight * pattern

        # Add some noise to make it more realistic
        noise = np.random.normal(0, 0.5, combined_pattern.shape)
        combined_pattern += noise

        # Ensure non-negative values
        combined_pattern = np.maximum(combined_pattern, 0)

        # Normalize to 0-100 range
        if np.max(combined_pattern) > 0:
            combined_pattern = 100 * combined_pattern / np.max(combined_pattern)

        return combined_pattern

    def generate_patterns_for_count(self, phase_count):
        """
        Generate patterns containing a specific number of phases.

        Args:
            phase_count: Number of phases to include in each pattern

        Returns:
            patterns: List of generated patterns
        """
        patterns = []

        if phase_count == 1:
            # Single-phase patterns
            for _ in range(self.num_patterns_per_count):
                structure_idx = random.choice(range(len(self.structures)))
                pattern = self.generate_single_phase_pattern(structure_idx)
                patterns.append(pattern.reshape(-1, 1))  # Reshape for consistency
        else:
            # Multi-phase patterns
            if len(self.structures) < phase_count:
                print(f"Warning: Not enough structures ({len(self.structures)}) for {phase_count} phases")
                return []

            for _ in range(self.num_patterns_per_count):
                # Randomly select phases
                structure_indices = random.sample(range(len(self.structures)), phase_count)

                # Generate weights
                weights = self.generate_phase_weights(phase_count)

                # Generate combined pattern
                pattern = self.generate_multi_phase_pattern(structure_indices, weights)
                patterns.append(pattern.reshape(-1, 1))  # Reshape for consistency

        return patterns

    def generate_all_patterns(self):
        """
        Generate patterns for all phase counts from 1 to max_phases.

        Returns:
            patterns_by_count: Dictionary mapping phase count to list of patterns
        """
        patterns_by_count = {}

        for phase_count in range(1, self.max_phases + 1):
            print(f"Generating {self.num_patterns_per_count} patterns with {phase_count} phase(s)...")
            patterns = self.generate_patterns_for_count(phase_count)
            patterns_by_count[phase_count] = patterns
            print(f"Generated {len(patterns)} patterns for {phase_count} phase(s)")

        return patterns_by_count

    def save_patterns(self, output_dir='multi_phase_patterns', patterns_by_count=None):
        """
        Save patterns to files organized by phase count.

        Args:
            output_dir: directory to save generated patterns
            patterns_by_count: pre-generated patterns to save (if None, will generate new ones)
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Use provided patterns or generate new ones
        if patterns_by_count is None:
            patterns_by_count = self.generate_all_patterns()

        # Save patterns organized by phase count
        for phase_count, patterns in patterns_by_count.items():
            phase_dir = os.path.join(output_dir, f"{phase_count}_phases")
            if not os.path.exists(phase_dir):
                os.makedirs(phase_dir)

            for i, pattern in enumerate(patterns):
                pattern_array = np.array(pattern).flatten()
                two_theta = np.linspace(self.min_angle, self.max_angle, len(pattern_array))
                pattern_data = np.column_stack((two_theta, pattern_array))

                filename = f"{phase_count}phases_{i+1:04d}.xy"
                filepath = os.path.join(phase_dir, filename)
                np.savetxt(filepath, pattern_data, fmt='%.6f')

            print(f"Saved {len(patterns)} patterns with {phase_count} phase(s) to {phase_dir}")

        return patterns_by_count


def generate_test_patterns(reference_dir, patterns_by_count, test_fraction=0.2,
                          min_phase_fraction=0.1, output_dir='test_patterns'):
    """
    Generate a separate test set with similar distribution to training data.

    Args:
        reference_dir: Directory containing reference CIF files
        patterns_by_count: Dictionary from training set to maintain similar distribution
        test_fraction: Fraction of training size to generate for testing
        min_phase_fraction: Minimum fraction for smallest phase
        output_dir: Directory to save test patterns

    Returns:
        test_patterns_by_count: Dictionary of test patterns by phase count
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Calculate number of test patterns for each count
    test_patterns_by_count = {}

    for phase_count, train_patterns in patterns_by_count.items():
        num_test_patterns = max(10, int(len(train_patterns) * test_fraction))

        # Create a separate generator for test patterns
        test_generator = MultiPhasePatternGenerator(
            reference_dir=reference_dir,
            num_patterns_per_count=num_test_patterns,
            max_phases=phase_count,
            min_phase_fraction=min_phase_fraction
        )

        print(f"Generating {num_test_patterns} test patterns with {phase_count} phase(s)...")
        test_patterns = test_generator.generate_patterns_for_count(phase_count)
        test_patterns_by_count[phase_count] = test_patterns

        # Save test patterns
        phase_dir = os.path.join(output_dir, f"{phase_count}_phases")
        if not os.path.exists(phase_dir):
            os.makedirs(phase_dir)

        for i, pattern in enumerate(test_patterns):
            pattern_array = np.array(pattern).flatten()
            two_theta = np.linspace(10.0, 80.0, len(pattern_array))
            pattern_data = np.column_stack((two_theta, pattern_array))

            filename = f"test_{phase_count}phases_{i+1:04d}.xy"
            filepath = os.path.join(phase_dir, filename)
            np.savetxt(filepath, pattern_data, fmt='%.6f')

        print(f"Saved {len(test_patterns)} test patterns with {phase_count} phase(s)")

    return test_patterns_by_count
