"""
Unified pattern generation for single and multi-phase XRD patterns.
"""

import os
import json
import random
import warnings
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from contextlib import contextmanager

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # Fallback progress indicator
    class tqdm:
        def __init__(self, iterable=None, total=None, desc=None, **kwargs):
            self.iterable = iterable
            self.total = total
            self.desc = desc
            self.n = 0
            if desc:
                print(f"{desc}: Starting...")

        def __iter__(self):
            if self.iterable:
                for item in self.iterable:
                    yield item
                    self.update(1)
            return self

        def __enter__(self):
            return self

        def __exit__(self, *args):
            if self.desc:
                print(f"{self.desc}: Completed!")

        def update(self, n=1):
            self.n += n
            if self.total and self.n % max(1, self.total // 10) == 0:
                progress = int(100 * self.n / self.total)
                print(f"{self.desc}: {progress}% ({self.n}/{self.total})")

        def set_description(self, desc):
            self.desc = desc

from .config import XRDGenerationConfig, DEFAULT_XRD_CONFIG
from .structures import StructureManager
from galaxi.pattern_generation.realistic_xrd import RealisticXRDGenerator
from galaxi.core.pattern_utils import is_structure_valid


def _make_generator(config: XRDGenerationConfig) -> RealisticXRDGenerator:
    """Construct a RealisticXRDGenerator from an XRDGenerationConfig, wiring up
    impurity peaks the same way every caller in this module needs. The
    generator's angle range/point count/Q-space settings all live inside
    `params` (config.to_dict()) -- RealisticXRDGenerator's constructor only
    accepts `params`, not separate min_angle/max_angle/num_points/structure
    keyword arguments."""
    generator = RealisticXRDGenerator(params=config.to_dict())
    if config.enable_impurities:
        generator.enable_impurity_peaks(
            enable=True,
            num_peaks_range=config.impurity_num_peaks_range,
            intensity_range=config.impurity_intensity_range,
            width_range=config.impurity_width_range,
            eta_range=config.impurity_eta_range,
        )
    return generator


def _generate_one_pattern(generator: RealisticXRDGenerator, structure, apply_all_effects: bool = True):
    """Generate a single (batch_size=1) 2θ/intensity pattern as flat
    numpy arrays, using RealisticXRDGenerator's current batched
    get_peak_list()/generate_realistic_pattern(peak_list, batch_size) API.
    Returns (2θ, intensity), or (2θ, intensity, info) if the
    generator has an active perturbator/augmentor (generate_realistic_pattern
    returns a 3-tuple in that case)."""
    peak_list = generator.get_peak_list(structure)
    result = generator.generate_realistic_pattern(peak_list, 1, apply_all_effects=apply_all_effects)
    two_theta, intensity = result[0], result[1]
    if hasattr(two_theta, "detach"):
        two_theta = two_theta.detach().cpu().numpy()
    if hasattr(intensity, "detach"):
        intensity = intensity.detach().cpu().numpy()
    two_theta = np.asarray(two_theta).reshape(-1)
    intensity = np.asarray(intensity).reshape(-1)
    if len(result) > 2:
        return two_theta, intensity, result[2]
    return two_theta, intensity


@contextmanager
def suppress_pymatgen_warnings():
    """Context manager to suppress common pymatgen warnings during CIF/XRD operations."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*fractional coordinates rounded.*")
        warnings.filterwarnings("ignore", message=".*No Pauling electronegativity.*")
        warnings.filterwarnings("ignore", message=".*Issues encountered while parsing CIF.*")
        warnings.filterwarnings("ignore", message=".*too close distance between atoms.*")
        warnings.filterwarnings("ignore", message=".*Structure contains sites that are less than.*")
        warnings.filterwarnings("ignore", message=".*Crystal system could not be detected.*")
        warnings.filterwarnings("ignore", message=".*Unable to calculate some structure properties.*")
        warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen")
        yield


class UnifiedPatternGenerator:
    """
    Unified pattern generator for single-phase and multi-phase XRD patterns.
    Consolidates all pattern generation functionality with clean interfaces.
    """

    def __init__(self,
                 reference_dir: str = "References",
                 config: Optional[XRDGenerationConfig] = None):
        """
        Initialize pattern generator.

        Args:
            reference_dir: Directory containing reference CIF files
            config: XRD generation configuration (uses default if None)
        """
        self.config = config if config is not None else DEFAULT_XRD_CONFIG
        self.structure_manager = StructureManager(reference_dir)

    def generate_single_phase_patterns(self,
                                     num_patterns: int = 100,
                                     phase_names: Optional[List[str]] = None,
                                     config: Optional[XRDGenerationConfig] = None,
                                     save_patterns: bool = False,
                                     output_dir: str = "single_phase_patterns",
                                     file_format: str = "npy") -> Dict[str, List[np.ndarray]]:
        """
        Generate single-phase patterns.

        Args:
            num_patterns: Number of patterns per phase
            phase_names: Specific phases to generate (None = all phases)
            config: Generation configuration (None = use instance config)
            save_patterns: Whether to save patterns to files
            output_dir: Directory to save patterns
            file_format: File format ("xy" or "npy")

        Returns:
            Dictionary mapping phase name to list of patterns
        """

        config = config if config is not None else self.config
        phases_to_generate = phase_names if phase_names is not None else self.structure_manager.get_phase_names()

        print(f"Generating {num_patterns} single-phase patterns for {len(phases_to_generate)} phases")

        generated_patterns = {}

        with tqdm(total=len(phases_to_generate), desc="Generating single-phase patterns", unit="phase") as pbar:
            for phase_name in phases_to_generate:
                try:
                    structure = self.structure_manager.get_structure(phase_name)
                    patterns = self._generate_patterns_for_phase(structure, num_patterns, config)
                    generated_patterns[phase_name] = patterns

                    if save_patterns:
                        self._save_single_phase_patterns(phase_name, patterns, config, output_dir, file_format)

                except ValueError as e:
                    print(f"Warning: {e}")
                    continue
                finally:
                    pbar.update(1)
                    pbar.set_description(f"Single-phase patterns ({sum(len(p) for p in generated_patterns.values())} total)")

        total_patterns = sum(len(patterns) for patterns in generated_patterns.values())
        print(f"Generated {total_patterns} total single-phase patterns")

        return generated_patterns

    def generate_multi_phase_patterns(self,
                                    pattern_counts: Union[int, Dict[int, int]] = 100,
                                    max_phases: int = 4,
                                    min_phase_fraction: float = 0.1,
                                    config: Optional[XRDGenerationConfig] = None,
                                    save_patterns: bool = False,
                                    output_dir: str = "multi_phase_patterns",
                                    file_format: str = "npy",
                                    save_ground_truth: bool = True) -> Dict[int, List[np.ndarray]]:
        """
        Generate multi-phase patterns.

        Args:
            pattern_counts: Either int (patterns per phase count) or Dict[phase_count, num_patterns]
            max_phases: Maximum number of phases (when pattern_counts is int)
            min_phase_fraction: Minimum mixing coefficient for the smallest phase. Note this is
                NOT a weight or mole fraction: each phase's pattern is independently max-normalized
                before these coefficients are applied, so they weight each phase's strongest
                reflection rather than a physical wt%/mol% scale factor.
            config: Generation configuration (None = use instance config)
            save_patterns: Whether to save patterns to files
            output_dir: Directory to save patterns
            file_format: File format ("xy" or "npy")
            save_ground_truth: Whether to save ground truth JSON

        Returns:
            Dictionary mapping phase count to list of patterns
        """

        config = config if config is not None else self.config

        # Determine pattern counts for each phase number
        if isinstance(pattern_counts, dict):
            counts_dict = pattern_counts
            print(f"Generating custom multi-phase patterns: {counts_dict}")
        else:
            counts_dict = {i: pattern_counts for i in range(1, max_phases + 1)}
            print(f"Generating {pattern_counts} patterns each for 1 to {max_phases} phases")

        patterns_by_count = {}
        ground_truth_data = {}

        total_patterns_to_generate = sum(counts_dict.values())

        with tqdm(total=total_patterns_to_generate, desc="Generating multi-phase patterns", unit="pattern") as pbar:
            for phase_count, num_patterns in counts_dict.items():
                if phase_count < 1:
                    print(f"Warning: Skipping invalid phase count {phase_count}")
                    continue

                phase_patterns = []
                error_count = 0
                for pattern_idx in range(num_patterns):
                    try:
                        pattern, composition = self._generate_multi_phase_pattern(
                            phase_count, min_phase_fraction, config
                        )
                        phase_patterns.append(pattern)

                        # Store ground truth
                        pattern_filename = f"{phase_count}phases_{pattern_idx+1:04d}.{file_format}"
                        ground_truth_data[pattern_filename] = composition

                    except Exception as e:
                        error_count += 1
                    finally:
                        pbar.update(1)
                        total_generated = sum(len(patterns) for patterns in patterns_by_count.values()) + len(phase_patterns)
                        pbar.set_description(f"Multi-phase patterns ({total_generated} completed)")

                if error_count:
                    print(
                        f"Warning: {error_count}/{num_patterns} {phase_count}-phase "
                        f"patterns failed to generate and were skipped"
                    )

                patterns_by_count[phase_count] = phase_patterns

        if save_patterns:
            self._save_multi_phase_patterns(patterns_by_count, config, output_dir, file_format)

            if save_ground_truth:
                self._save_ground_truth(ground_truth_data, output_dir, "ground_truth.json")

        total_patterns = sum(len(patterns) for patterns in patterns_by_count.values())
        print(f"Generated {total_patterns} total multi-phase patterns")

        return patterns_by_count

    def generate_phase_detection_patterns(self,
                                        target_phase_name: str,
                                        num_positive: int = 100,
                                        num_negative: int = 100,
                                        target_fraction_range: Tuple[float, float] = (0.05, 0.95),
                                        cod_cif_dir: Optional[str] = None,
                                        max_cod_structures: int = 1000,
                                        config: Optional[XRDGenerationConfig] = None,
                                        save_patterns: bool = False,
                                        output_dir: str = "phase_detection_patterns",
                                        file_format: str = "npy",
                                        save_ground_truth: bool = True) -> Dict[str, List[np.ndarray]]:
        """
        Generate patterns for phase detection training.

        Args:
            target_phase_name: Name of the target phase to detect
            num_positive: Number of positive examples (containing target)
            num_negative: Number of negative examples (not containing target)
            target_fraction_range: Range of target-phase mixing coefficients for positive examples.
                Note this is NOT a weight or mole fraction: each phase's pattern is independently
                max-normalized before this coefficient is applied, so it weights the target's
                strongest reflection relative to the background rather than a physical wt%/mol%.
            cod_cif_dir: Directory containing COD CIF files for background phases
                (default: the `data/cod` directory of the installed galaxi
                package; see `galaxi.scripts.setup_cod`)
            max_cod_structures: Maximum number of COD structures to load (default 1000)
            config: Generation configuration
            save_patterns: Whether to save patterns to files
            output_dir: Directory to save patterns
            file_format: File format ("xy" or "npy")
            save_ground_truth: Whether to save ground truth JSON

        Returns:
            Dictionary with 'positive' and 'negative' pattern lists
        """

        config = config if config is not None else self.config
        if cod_cif_dir is None:
            from ..paths import get_default_cod_dir
            cod_cif_dir = get_default_cod_dir()

        # Get target structure
        try:
            target_structure = self.structure_manager.get_structure(target_phase_name)
        except ValueError:
            raise ValueError(f"Target phase '{target_phase_name}' not found in reference directory")

        # Load available COD structures (excluding the target phase itself)
        cod_structures = self._load_cod_structures(
            {target_phase_name}, cod_cif_dir, max_cod_structures
        )

        print(f"Generating phase detection patterns for target: {target_phase_name}")
        print(f"Positive examples: {num_positive}, Negative examples: {num_negative}")
        print(f"Target fraction range: {target_fraction_range}")
        print(f"Available COD structures: {len(cod_structures)}")

        # Generate positive examples (containing target phase)
        positive_patterns = []
        positive_metadata = []

        with tqdm(total=num_positive, desc="Generating positive examples", unit="pattern") as pbar:
            attempts = 0
            max_attempts = num_positive * 3  # Safety limit: 3x the target number

            while len(positive_patterns) < num_positive and attempts < max_attempts:
                attempts += 1
                try:
                    target_fraction = np.random.uniform(*target_fraction_range)
                    pattern, metadata = self._generate_positive_detection_pattern(
                        target_structure, target_phase_name, target_fraction, cod_structures, config
                    )
                    positive_patterns.append(pattern)
                    positive_metadata.append({
                        "pattern_id": f"positive_{len(positive_patterns):04d}",
                        "contains_target": True,
                        "target_fraction": target_fraction,
                        **metadata
                    })
                    pbar.update(1)
                except Exception as e:
                    # Continue trying on error
                    print(f'Pattern generation failed with {e}')
                    pass

                pbar.set_description(f"Positive examples ({len(positive_patterns)} successful, {attempts} attempts)")

        # Generate negative examples (not containing target phase)
        negative_patterns = []
        negative_metadata = []

        with tqdm(total=num_negative, desc="Generating negative examples", unit="pattern") as pbar:
            attempts = 0
            max_attempts = num_negative * 3  # Safety limit: 3x the target number

            while len(negative_patterns) < num_negative and attempts < max_attempts:
                attempts += 1
                try:
                    # 50% pure background phases, 50% unphysical perturbations
                    if np.random.random() < 0.5:
                        pattern, metadata = self.generate_negative_background_pattern(
                            cod_structures, config
                        )
                        pattern_type = "background_only"
                    else:
                        pattern, metadata = self._generate_negative_perturbed_pattern(
                            target_structure, config
                        )
                        pattern_type = "perturbed_target"

                    negative_patterns.append(pattern)
                    negative_metadata.append({
                        "pattern_id": f"negative_{len(negative_patterns):04d}",
                        "contains_target": False,
                        "target_fraction": 0.0,
                        "pattern_type": pattern_type,
                        **metadata
                    })
                    pbar.update(1)
                except Exception as e:
                    # Continue trying on error
                    print(f'Pattern generation failed with {e}')
                    pass

                pbar.set_description(f"Negative examples ({len(negative_patterns)} successful, {attempts} attempts)")

        # Check if we hit the attempt limit
        if len(positive_patterns) < num_positive:
            print(f"Warning: Only generated {len(positive_patterns)} positive patterns out of {num_positive} requested after {max_attempts} attempts")
        if len(negative_patterns) < num_negative:
            print(f"Warning: Only generated {len(negative_patterns)} negative patterns out of {num_negative} requested after {max_attempts} attempts")

        # Combine results
        all_patterns = {
            "positive": positive_patterns,
            "negative": negative_patterns
        }

        all_metadata = {
            "positive": positive_metadata,
            "negative": negative_metadata
        }

        # Save patterns and metadata
        if save_patterns:
            self._save_detection_patterns(all_patterns, all_metadata, config, output_dir, file_format)

            if save_ground_truth:
                self._save_detection_ground_truth(all_metadata, output_dir, "detection_ground_truth.json")

        print(f"Generated {len(positive_patterns)} positive and {len(negative_patterns)} negative patterns")

        return all_patterns

    def _generate_patterns_for_phase(self,
                                   structure,
                                   num_patterns: int,
                                   config: XRDGenerationConfig) -> List[np.ndarray]:
        """Generate multiple patterns for a single phase."""
        generator = _make_generator(config)

        patterns = []
        for i in range(num_patterns):
            try:
                two_theta, intensity = _generate_one_pattern(generator, structure)
                patterns.append(intensity)
            except Exception as e:
                print(f"Warning: Error generating pattern {i+1}: {e}")

        return patterns

    def _generate_multi_phase_pattern(self,
                                    phase_count: int,
                                    min_phase_fraction: float,
                                    config: XRDGenerationConfig) -> Tuple[np.ndarray, Dict]:
        """Generate a single multi-phase pattern with composition metadata."""
        structures = self.structure_manager.get_structures()
        phase_names = self.structure_manager.get_phase_names()

        if len(structures) < phase_count:
            raise ValueError(f"Not enough structures ({len(structures)}) for {phase_count} phases")

        # Select random phases
        selected_indices = random.sample(range(len(structures)), phase_count)
        selected_phases = [phase_names[i] for i in selected_indices]
        selected_structures = [structures[i] for i in selected_indices]

        # Generate weights
        weights = self._generate_phase_weights(phase_count, min_phase_fraction)

        # Generate individual patterns
        individual_patterns = []
        for structure in selected_structures:
            generator = _make_generator(config)
            two_theta, intensity = _generate_one_pattern(generator, structure)
            individual_patterns.append(intensity)

        # Combine patterns with weights
        combined_pattern = np.zeros_like(individual_patterns[0])
        for pattern, weight in zip(individual_patterns, weights):
            combined_pattern += weight * pattern

        # Add noise using config and ensure non-negative values
        noise_level = np.random.uniform(*config.noise_level) / 100.0  # Convert percentage to fraction
        noise = np.random.normal(0, noise_level * np.max(combined_pattern), combined_pattern.shape)
        combined_pattern = np.maximum(combined_pattern + noise, 0)

        # Normalize
        if np.max(combined_pattern) > 0:
            combined_pattern = 100 * combined_pattern / np.max(combined_pattern)

        # Create composition metadata. "mixing_coefficients" are NOT weight/mole
        # fractions: each phase's pattern is independently max-normalized before
        # these coefficients are applied, so they weight each phase's strongest
        # reflection rather than a physically calibrated wt%/mol% scale factor.
        composition = {
            "phases": selected_phases,
            "mixing_coefficients": weights,
            "phase_count": phase_count
        }

        return combined_pattern.flatten(), composition

    def _generate_phase_weights(self, num_phases: int, min_phase_fraction: float) -> List[float]:
        """Generate random weights ensuring minimum fraction constraint."""
        max_attempts = 100
        for _ in range(max_attempts):
            weights = np.random.uniform(min_phase_fraction, 1.0, num_phases)
            weights = weights / np.sum(weights)  # Normalize

            if min(weights) >= min_phase_fraction:
                return weights.tolist()

        # Fallback: ensure minimum fraction
        weights = np.full(num_phases, min_phase_fraction)
        remaining = 1.0 - num_phases * min_phase_fraction
        if remaining > 0:
            # Add remaining weight to random phases
            additional = np.random.uniform(0, remaining, num_phases)
            additional = additional / np.sum(additional) * remaining
            weights += additional

        return weights.tolist()

    def _save_single_phase_patterns(self,
                                  phase_name: str,
                                  patterns: List[np.ndarray],
                                  config: XRDGenerationConfig,
                                  output_dir: str,
                                  file_format: str):
        """Save single-phase patterns to files."""
        phase_dir = os.path.join(output_dir, phase_name)
        os.makedirs(phase_dir, exist_ok=True)

        for i, pattern in enumerate(patterns):
            if file_format == "xy":
                x_values = np.linspace(config.min_angle, config.max_angle, len(pattern))
                if config.convert_to_q:
                    from .pattern_utils import convert_two_theta_to_q
                    x_values = convert_two_theta_to_q(x_values)
                pattern_data = np.column_stack((x_values, pattern))
                filename = f"{phase_name}_{i+1:04d}.xy"
                filepath = os.path.join(phase_dir, filename)
                np.savetxt(filepath, pattern_data, fmt='%.6f')
            elif file_format == "npy":
                filename = f"{phase_name}_{i+1:04d}.npy"
                filepath = os.path.join(phase_dir, filename)
                np.save(filepath, pattern)

    def _save_multi_phase_patterns(self,
                                 patterns_by_count: Dict[int, List[np.ndarray]],
                                 config: XRDGenerationConfig,
                                 output_dir: str,
                                 file_format: str):
        """Save multi-phase patterns to files."""
        for phase_count, patterns in patterns_by_count.items():
            phase_dir = os.path.join(output_dir, f"{phase_count}_phases")
            os.makedirs(phase_dir, exist_ok=True)

            for i, pattern in enumerate(patterns):
                if file_format == "xy":
                    x_values = np.linspace(config.min_angle, config.max_angle, len(pattern))
                    if config.convert_to_q:
                        from .pattern_utils import convert_two_theta_to_q
                        x_values = convert_two_theta_to_q(x_values)
                    pattern_data = np.column_stack((x_values, pattern))
                    filename = f"{phase_count}phases_{i+1:04d}.xy"
                    filepath = os.path.join(phase_dir, filename)
                    np.savetxt(filepath, pattern_data, fmt='%.6f')
                elif file_format == "npy":
                    filename = f"{phase_count}phases_{i+1:04d}.npy"
                    filepath = os.path.join(phase_dir, filename)
                    np.save(filepath, pattern)

    def _save_ground_truth(self, ground_truth_data: Dict, output_dir: str, filename: str):
        """Save ground truth metadata to JSON file."""
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(ground_truth_data, f, indent=2)
        print(f"Ground truth saved to {filepath}")

    def generate_single_phase_pattern(self,
                                    cif_file: str,
                                    target_concentration: float = 1.0,
                                    include_metadata: bool = False) -> Dict:
        """
        Generate a single XRD pattern from a CIF file.

        This is a convenience method for the tutorials that generates a single pattern
        using the existing multi-pattern generation infrastructure.

        Args:
            cif_file: Path to the CIF file
            target_concentration: Concentration of the target phase (0.0 to 1.0)
            include_metadata: Whether to include generation metadata

        Returns:
            Dictionary containing 'two_theta', 'intensity', and optionally 'metadata'
        """
        import os
        from pymatgen.core import Structure

        # Load structure from CIF file
        if not os.path.exists(cif_file):
            raise FileNotFoundError(f"CIF file not found: {cif_file}")

        try:
            with suppress_pymatgen_warnings():
                structure = Structure.from_file(cif_file)
        except Exception as e:
            raise ValueError(f"Could not load structure from {cif_file}: {e}")

        # Generate single pattern using the realistic XRD generator
        generator = RealisticXRDGenerator(params=self.config.to_dict())

        if self.config.enable_impurities:
            generator.enable_impurity_peaks(
                enable=True,
                num_peaks_range=self.config.impurity_num_peaks_range,
                intensity_range=self.config.impurity_intensity_range,
                width_range=self.config.impurity_width_range,
                eta_range=self.config.impurity_eta_range
            )

        try:
            peak_list = generator.get_peak_list(structure)
            two_theta, intensity = generator.generate_realistic_pattern(peak_list, 1)

            # generate_realistic_pattern() is batched and returns (batch, N).
            # This wrapper asks for exactly one pattern, so drop the leading
            # axis: the returned dict then holds two 1-D arrays of equal length,
            # and len(intensity) below is the point count the background noise
            # is drawn over.
            if intensity.ndim == 2 and intensity.shape[0] == 1:
                intensity = intensity[0]

            # Apply target concentration scaling if not 1.0
            if target_concentration != 1.0:
                # Scale intensity and add background if needed
                intensity = intensity * target_concentration
                if target_concentration < 1.0:
                    # Add some background noise to simulate other phases
                    background_level = np.random.uniform(*self.config.background_level)
                    background = np.random.normal(background_level, background_level * 0.1, len(intensity))
                    background = np.maximum(background, 0)  # Ensure non-negative
                    intensity = intensity + (1.0 - target_concentration) * background

            result = {
                'two_theta': two_theta,
                'intensity': intensity
            }

            if include_metadata:
                # Extract some metadata from the generation config
                metadata = {
                    'cif_file': cif_file,
                    'target_concentration': target_concentration,
                    'crystallite_size': np.random.uniform(*self.config.crystallite_size_range),
                    'microstrain': np.random.uniform(*self.config.microstrain_range),
                    'background_level': np.random.uniform(*self.config.background_level),
                    'noise_level': np.random.uniform(*self.config.noise_level),
                    'temperature': np.random.uniform(*self.config.temperature_range),
                    'config_used': {
                        'min_angle': self.config.min_angle,
                        'max_angle': self.config.max_angle,
                        'num_points': self.config.num_points,
                        'enable_impurities': self.config.enable_impurities
                    }
                }
                result['metadata'] = metadata

            return result

        except Exception as e:
            raise RuntimeError(f"Failed to generate pattern for {cif_file}: {e}")

    def generate_multi_phase_pattern(self,
                                   cif_files: List[str],
                                   concentrations: List[float],
                                   include_metadata: bool = False) -> Dict:
        """
        Generate a single multi-phase XRD pattern from multiple CIF files.

        This is a convenience method for the tutorials that generates a single pattern
        from multiple phases with specified concentrations.

        Args:
            cif_files: List of paths to CIF files
            concentrations: List of concentrations for each phase (must sum to 1.0)
            include_metadata: Whether to include generation metadata

        Returns:
            Dictionary containing 'two_theta', 'intensity', and optionally 'metadata'
        """
        import os
        from pymatgen.core import Structure

        if len(cif_files) != len(concentrations):
            raise ValueError("Number of CIF files must match number of concentrations")

        if abs(sum(concentrations) - 1.0) > 1e-6:
            raise ValueError(f"Concentrations must sum to 1.0, got {sum(concentrations)}")

        # Load structures
        structures = []
        for cif_file in cif_files:
            if not os.path.exists(cif_file):
                raise FileNotFoundError(f"CIF file not found: {cif_file}")

            try:
                with suppress_pymatgen_warnings():
                    structure = Structure.from_file(cif_file)
                structures.append(structure)
            except Exception as e:
                raise ValueError(f"Could not load structure from {cif_file}: {e}")

        # Generate individual patterns
        individual_patterns = []
        for structure in structures:
            generator = _make_generator(self.config)
            try:
                two_theta, intensity = _generate_one_pattern(generator, structure)
                individual_patterns.append(intensity)
            except Exception as e:
                raise RuntimeError(f"Failed to generate pattern for {structures.index(structure)}: {e}")

        # Combine patterns with weights
        combined_pattern = np.zeros_like(individual_patterns[0])
        for pattern, concentration in zip(individual_patterns, concentrations):
            combined_pattern += concentration * pattern

        # Add noise using config and ensure non-negative values
        noise_level = np.random.uniform(*self.config.noise_level) / 100.0  # Convert percentage to fraction
        noise = np.random.normal(0, noise_level * combined_pattern.max(), combined_pattern.shape)
        combined_pattern = np.maximum(combined_pattern + noise, 0)

        result = {
            'two_theta': two_theta,  # Use two_theta from last generation (should be same for all)
            'intensity': combined_pattern
        }

        if include_metadata:
            metadata = {
                'cif_files': cif_files,
                'concentrations': concentrations,
                'num_phases': len(cif_files),
                'config_used': {
                    'min_angle': self.config.min_angle,
                    'max_angle': self.config.max_angle,
                    'num_points': self.config.num_points,
                    'enable_impurities': self.config.enable_impurities
                }
            }
            result['metadata'] = metadata

        return result

    def get_available_phases(self) -> List[str]:
        """Get list of available phase names."""
        return self.structure_manager.get_phase_names()

    def _load_cod_structures(self, phases, cod_cif_dir: str, max_structures: int = 1000) -> List[Tuple[str, Any]]:
        """Load COD CIF structures for background phases."""
        import os
        import warnings
        from pymatgen.core import Structure

        # Suppress common pymatgen warnings during CIF loading
        with suppress_pymatgen_warnings():
            return self._load_cod_structures_impl(phases, cod_cif_dir, max_structures)

    def _load_cod_structures_impl(self, phases, cod_cif_dir: str, max_structures: int = 1000) -> List[Tuple[str, Any]]:
        """Implementation of COD structure loading with warnings suppressed."""
        import os
        from pymatgen.core import Structure

        cod_structures = []

        if not os.path.exists(cod_cif_dir):
            print(f"Warning: COD directory not found: {cod_cif_dir}")
            return []

        cif_files = [f for f in os.listdir(cod_cif_dir) if f.endswith('.cif')]

        # Randomly sample up to max_structures CIF files to avoid memory issues
        sample_size = min(max_structures, len(cif_files))
        sampled_files = random.sample(cif_files, sample_size)

        loaded_count = 0
        filtered_count = 0
        error_count = 0

        with tqdm(total=sample_size, desc="Loading COD structures", unit="file") as pbar:
            for cif_file in sampled_files:
                try:
                    cif_path = os.path.join(cod_cif_dir, cif_file)
                    structure = Structure.from_file(cif_path)
                    phase_name = cif_file.replace('.cif', '')

                    # Filter out target phases
                    if phase_name in phases:
                        filtered_count += 1
                        continue

                    # Filter out problematic structures
                    if is_structure_valid(structure):
                        cod_structures.append((phase_name, structure))
                        loaded_count += 1
                    else:
                        filtered_count += 1

                except Exception as e:
                    error_count += 1
                    continue  # Skip problematic CIF files
                finally:
                    pbar.update(1)
                    pbar.set_description(f"COD structures ({loaded_count} loaded, {filtered_count} filtered)")

        print(f"COD structure loading: {loaded_count} loaded, {filtered_count} filtered, {error_count} errors from {sample_size} attempted")

        return cod_structures

    def _generate_positive_detection_pattern(self,
                                           target_structure,
                                           target_phase_name: str,
                                           target_fraction: float,
                                           cod_structures: List[Tuple[str, Any]],
                                           config: XRDGenerationConfig) -> Tuple[np.ndarray, Dict]:
        import warnings
        """Generate positive detection pattern containing target phase."""
        # Determine number of background phases (1-3)
        num_background_phases = random.randint(1, 3)

        # Select random background phases
        if len(cod_structures) < num_background_phases:
            num_background_phases = len(cod_structures)

        selected_bg_phases = random.sample(cod_structures, num_background_phases)

        # Calculate background phase fractions
        background_fraction = 1.0 - target_fraction
        if num_background_phases > 1:
            bg_fractions = np.random.dirichlet(np.ones(num_background_phases))
            bg_fractions = bg_fractions * background_fraction
        else:
            bg_fractions = [background_fraction]

        # Generate target pattern
        target_generator = _make_generator(config)
        two_theta, target_intensity = _generate_one_pattern(target_generator, target_structure)

        # Validate target pattern
        if not self._is_pattern_valid(target_intensity):
            raise ValueError(f"Invalid target pattern generated for {target_phase_name}")

        # Generate background patterns
        combined_pattern = target_fraction * target_intensity
        bg_phase_names = []
        bg_error_count = 0

        for (bg_name, bg_structure), bg_fraction in zip(selected_bg_phases, bg_fractions):
            try:
                # Suppress warnings during XRD generation
                with suppress_pymatgen_warnings():
                    bg_generator = _make_generator(config)
                    _, bg_intensity = _generate_one_pattern(bg_generator, bg_structure)

                # Validate generated pattern
                if self._is_pattern_valid(bg_intensity):
                    combined_pattern += bg_fraction * bg_intensity
                    bg_phase_names.append(bg_name)

            except Exception as e:
                # Skip problematic background structures; count rather than
                # print per-occurrence since this runs per generated pattern
                # (thousands of times per training run) and per-iteration
                # printing would be overwhelming.
                bg_error_count += 1
                continue

        if bg_error_count:
            print(
                f"Warning: skipped {bg_error_count}/{len(selected_bg_phases)} "
                f"background phases for {target_phase_name} due to generation errors"
            )

        # Add noise using config and ensure non-negative
        noise_level = np.random.uniform(*config.noise_level) / 100.0  # Convert percentage to fraction
        noise = np.random.normal(0, noise_level * np.max(combined_pattern), combined_pattern.shape)
        combined_pattern = np.maximum(combined_pattern + noise, 0)

        # Normalize
        if np.max(combined_pattern) > 0:
            combined_pattern = 100 * combined_pattern / np.max(combined_pattern)

        metadata = {
            "background_phases": bg_phase_names,
            # NOT weight/mole fractions -- see mixing_coefficients note in _generate_multi_phase_pattern.
            "background_mixing_coefficients": bg_fractions.tolist() if hasattr(bg_fractions, 'tolist') else bg_fractions,
            "total_phases": len(bg_phase_names) + 1
        }

        return combined_pattern.flatten(), metadata

    def generate_negative_background_pattern(self,
                                            cod_structures: List[Tuple[str, Any]],
                                            config: XRDGenerationConfig) -> Tuple[np.ndarray, Dict]:
        import warnings
        """Generate negative pattern with only background phases."""
        # Select 1-4 background phases
        num_phases = random.randint(1, 4)

        if len(cod_structures) < num_phases:
            num_phases = len(cod_structures)

        selected_phases = random.sample(cod_structures, num_phases)

        # Generate equal or random fractions
        if num_phases == 1:
            fractions = [1.0]
        else:
            fractions = np.random.dirichlet(np.ones(num_phases))

        # Generate combined pattern
        combined_pattern = np.zeros(config.num_points)
        phase_names = []
        error_count = 0

        for (phase_name, structure), fraction in zip(selected_phases, fractions):
            try:
                # Suppress warnings during XRD generation
                with suppress_pymatgen_warnings():
                    generator = _make_generator(config)
                    _, intensity = _generate_one_pattern(generator, structure)

                # Validate generated pattern
                if self._is_pattern_valid(intensity):
                    combined_pattern += fraction * intensity
                    phase_names.append(phase_name)

            except Exception as e:
                # Skip problematic structures; count rather than print
                # per-occurrence since this runs per generated pattern.
                error_count += 1
                continue

        if error_count:
            print(
                f"Warning: skipped {error_count}/{len(selected_phases)} "
                f"background phases due to generation errors"
            )

        # Add noise using config and ensure non-negative
        noise_level = np.random.uniform(*config.noise_level) / 100.0  # Convert percentage to fraction
        noise = np.random.normal(0, noise_level * np.max(combined_pattern), combined_pattern.shape)
        combined_pattern = np.maximum(combined_pattern + noise, 0)

        # Normalize
        if np.max(combined_pattern) > 0:
            combined_pattern = 100 * combined_pattern / np.max(combined_pattern)

        metadata = {
            "phases": phase_names,
            # NOT weight/mole fractions -- see mixing_coefficients note in _generate_multi_phase_pattern.
            "mixing_coefficients": fractions.tolist() if hasattr(fractions, 'tolist') else fractions
        }

        return combined_pattern.flatten(), metadata

    def _generate_negative_perturbed_pattern(self,
                                           target_structure,
                                           config: XRDGenerationConfig,
                                           peak_perturbation_config: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Generate negative pattern with unphysical perturbations of target."""
        # Generate base target pattern
        generator = RealisticXRDGenerator(params=config.to_dict())

        # Enable peak perturbations for unphysical negative examples
        generator.enable_peak_perturbations(peak_perturbation_config or {}, enable=True)

        two_theta, perturbed_intensity, perturbations = _generate_one_pattern(generator, target_structure)

        metadata = {
            "perturbations": perturbations,  # Contains perturbations from new peak perturbation system
            "base_target_present": True,
            "physically_realistic": False
        }

        return perturbed_intensity.flatten(), metadata

    def _save_detection_patterns(self,
                               patterns: Dict[str, List[np.ndarray]],
                               metadata: Dict[str, List[Dict]],
                               config: XRDGenerationConfig,
                               output_dir: str,
                               file_format: str):
        """Save detection patterns to files."""
        os.makedirs(output_dir, exist_ok=True)

        total_patterns = sum(len(pattern_list) for pattern_list in patterns.values())

        with tqdm(total=total_patterns, desc="Saving patterns", unit="file") as pbar:
            for label, pattern_list in patterns.items():
                label_dir = os.path.join(output_dir, label)
                os.makedirs(label_dir, exist_ok=True)

                for i, pattern in enumerate(pattern_list):
                    if file_format == "xy":
                        x_values = np.linspace(config.min_angle, config.max_angle, len(pattern))
                        if config.convert_to_q:
                            from .pattern_utils import convert_two_theta_to_q
                            x_values = convert_two_theta_to_q(x_values)
                        pattern_data = np.column_stack((x_values, pattern))
                        filename = f"{label}_{i+1:04d}.xy"
                        filepath = os.path.join(label_dir, filename)
                        np.savetxt(filepath, pattern_data, fmt='%.6f')
                    elif file_format == "npy":
                        filename = f"{label}_{i+1:04d}.npy"
                        filepath = os.path.join(label_dir, filename)
                        np.save(filepath, pattern)

                    pbar.update(1)
                    pbar.set_description(f"Saving {label} patterns")

        print(f"Saved detection patterns to {output_dir}")

    def _save_detection_ground_truth(self, metadata: Dict[str, List[Dict]], output_dir: str, filename: str):
        """Save detection ground truth metadata to JSON file."""
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)

        # Flatten metadata for easier access
        flattened_metadata = {}
        for label, label_metadata in metadata.items():
            for item in label_metadata:
                pattern_filename = f"{item['pattern_id']}.xy"
                flattened_metadata[pattern_filename] = item

        with open(filepath, 'w', encoding="utf-8") as f:
            json.dump(flattened_metadata, f, indent=2, ensure_ascii=False)
        print(f"Detection ground truth saved to {filepath}")

    def _is_pattern_valid(self, pattern: np.ndarray) -> bool:
        """
        Check if a generated XRD pattern is valid.

        Filters out patterns with:
        - NaN or infinite values
        - All zero intensities
        - Unreasonable intensity values
        """
        try:
            # Check for NaN or infinite values
            if not np.isfinite(pattern).all():
                return False

            # Check if pattern is all zeros
            if np.max(pattern) == 0:
                return False

            # Check for reasonable intensity range
            max_intensity = np.max(pattern)
            if max_intensity < 1e-10 or max_intensity > 1e10:
                return False

            # Check for patterns with too many negative values
            negative_fraction = np.sum(pattern < 0) / len(pattern)
            if negative_fraction > 0.1:  # More than 10% negative values is suspicious
                return False

            return True

        except Exception:
            return False
