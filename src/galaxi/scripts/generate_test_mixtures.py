#!/usr/bin/env python
"""
Script to generate test mixture patterns with known ground truth for evaluating
the combined phase identification and counting system.
"""

import os
import sys
import argparse
import numpy as np
import random
import pandas as pd
from pymatgen.core import Structure
from galaxi.pattern_generation.realistic_xrd import RealisticXRDGenerator
from galaxi.core.pattern_generator import _generate_one_pattern


def parse_args():
    parser = argparse.ArgumentParser(description="Generate test mixtures with ground truth")
    parser.add_argument("--ref_dir", type=str, default="References",
                        help="Directory containing reference CIF files")
    parser.add_argument("--output_dir", type=str, default="test_mixtures",
                        help="Directory to save test patterns")
    parser.add_argument("--num_patterns", type=int, default=100,
                        help="Total number of test patterns to generate")
    parser.add_argument("--max_phases", type=int, default=4,
                        help="Maximum number of phases in a mixture")
    parser.add_argument("--min_phase_fraction", type=float, default=0.1,
                        help="Minimum fraction for any phase in mixture")
    parser.add_argument("--min_angle", type=float, default=10.0,
                        help="Minimum 2θ angle")
    parser.add_argument("--max_angle", type=float, default=80.0,
                        help="Maximum 2θ angle")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    return parser.parse_args()


def generate_mixture_pattern(structures, phase_names, selected_indices, weights,
                           min_angle=10.0, max_angle=80.0):
    """
    Generate a mixture XRD pattern from selected phases with given weights.

    Args:
        structures: List of pymatgen Structure objects
        phase_names: List of phase names
        selected_indices: Indices of phases to include in mixture
        weights: Weights for each selected phase
        min_angle: Minimum 2θ angle
        max_angle: Maximum 2θ angle

    Returns:
        combined_pattern: Combined XRD pattern
        composition_info: Dictionary with composition details
    """
    # Normalize weights
    weights = np.array(weights)
    weights = weights / np.sum(weights)

    # Generate individual patterns
    individual_patterns = []
    composition_info = {
        'num_phases': len(selected_indices),
        'phases': [],
        'weights': []
    }

    for i, (idx, weight) in enumerate(zip(selected_indices, weights)):
        structure = structures[idx]
        phase_name = phase_names[idx]

        # Generate pattern using RealisticXRDGenerator. min_angle/max_angle/
        # num_points go inside `params` -- the constructor only accepts a
        # single `params` dict, not separate keyword arguments.
        generator = RealisticXRDGenerator(params={
            'min_angle': min_angle,
            'max_angle': max_angle,
            'num_points': 4501,
        })

        # Set parameter ranges
        generator.set_parameter_range('uniform_shift_range', (-0.15, 0.15))
        generator.set_parameter_range('crystallite_size_range', (5.0, 50.0))
        generator.set_parameter_range('microstrain_range', (0.0, 0.02))
        generator.set_parameter_range('texture_range', (0.6, 1.4))

        # Generate single pattern
        two_theta, intensity = _generate_one_pattern(generator, structure)
        individual_patterns.append(intensity)

        composition_info['phases'].append(phase_name)
        composition_info['weights'].append(weight)

    # Combine patterns with weights
    combined_pattern = np.zeros_like(individual_patterns[0])
    for pattern, weight in zip(individual_patterns, weights):
        combined_pattern += weight * pattern

    # Add some noise to make it realistic
    noise = np.random.normal(0, 0.3, combined_pattern.shape)
    combined_pattern += noise

    # Ensure non-negative values
    combined_pattern = np.maximum(combined_pattern, 0)

    # Normalize to 0-100 range
    if np.max(combined_pattern) > 0:
        combined_pattern = 100 * combined_pattern / np.max(combined_pattern)

    return combined_pattern, composition_info


def generate_phase_weights(num_phases, min_fraction=0.1):
    """Generate random weights ensuring minimum fraction constraint."""
    max_attempts = 100

    for _ in range(max_attempts):
        # Generate random weights
        weights = np.random.uniform(min_fraction, 1.0, num_phases)

        # Normalize
        weights = weights / np.sum(weights)

        # Check if all weights meet minimum fraction
        if np.all(weights >= min_fraction):
            return weights.tolist()

    # Fallback: equal weights
    return [1.0/num_phases] * num_phases


def main():
    args = parse_args()

    # Set random seed
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Check if reference directory exists
    if not os.path.exists(args.ref_dir):
        print(f"Error: Reference directory {args.ref_dir} does not exist")
        return

    print(f"Generating {args.num_patterns} test mixture patterns...")

    # Load all structures
    structures = []
    phase_names = []

    for filename in sorted(os.listdir(args.ref_dir)):
        if filename.endswith('.cif'):
            try:
                structure = Structure.from_file(os.path.join(args.ref_dir, filename))
                phase_name = filename.replace('.cif', '')
                structures.append(structure)
                phase_names.append(phase_name)
            except Exception as e:
                print(f"Warning: Could not load {filename}: {e}")

    if len(structures) < 2:
        print("Error: Need at least 2 reference phases to create mixtures")
        return

    print(f"Loaded {len(structures)} reference phases: {phase_names}")

    # Generate test patterns with ground truth
    ground_truth_data = []

    for i in range(args.num_patterns):
        # Randomly choose number of phases (weighted towards fewer phases)
        phase_count_weights = [0.4, 0.35, 0.2, 0.05][:args.max_phases]  # Favor 1-2 phases
        num_phases = np.random.choice(range(1, min(args.max_phases + 1, len(structures) + 1)),
                                     p=phase_count_weights)

        # Randomly select phases
        selected_indices = random.sample(range(len(structures)), num_phases)

        # Generate weights
        weights = generate_phase_weights(num_phases, args.min_phase_fraction)

        # Generate mixture pattern
        try:
            pattern, composition_info = generate_mixture_pattern(
                structures, phase_names, selected_indices, weights,
                args.min_angle, args.max_angle
            )

            # Save pattern
            two_theta = np.linspace(args.min_angle, args.max_angle, len(pattern))
            pattern_data = np.column_stack((two_theta, pattern))

            filename = f"mixture_{i+1:04d}.xy"
            filepath = os.path.join(args.output_dir, filename)
            np.savetxt(filepath, pattern_data, fmt='%.6f')

            # Record ground truth
            ground_truth_record = {
                'pattern_file': filename,
                'true_phase_count': composition_info['num_phases'],
                'phases_present': ','.join(composition_info['phases']),
                'phase_weights': ','.join([f"{w:.3f}" for w in composition_info['weights']])
            }

            # Add individual phase presence flags
            for phase_name in phase_names:
                ground_truth_record[f'contains_{phase_name}'] = phase_name in composition_info['phases']

                # Add weight for this phase (0 if not present)
                if phase_name in composition_info['phases']:
                    phase_idx = composition_info['phases'].index(phase_name)
                    ground_truth_record[f'weight_{phase_name}'] = composition_info['weights'][phase_idx]
                else:
                    ground_truth_record[f'weight_{phase_name}'] = 0.0

            ground_truth_data.append(ground_truth_record)

            if (i + 1) % 20 == 0:
                print(f"Generated {i + 1}/{args.num_patterns} patterns...")

        except Exception as e:
            print(f"Error generating pattern {i+1}: {e}")

    # Save ground truth to CSV
    ground_truth_df = pd.DataFrame(ground_truth_data)
    ground_truth_file = os.path.join(args.output_dir, "ground_truth.csv")
    ground_truth_df.to_csv(ground_truth_file, index=False)

    # Print summary statistics
    print(f"\n{'='*60}")
    print("GENERATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total patterns generated: {len(ground_truth_data)}")
    print(f"Patterns saved to: {args.output_dir}")
    print(f"Ground truth saved to: {ground_truth_file}")

    # Phase count distribution
    count_dist = ground_truth_df['true_phase_count'].value_counts().sort_index()
    print("\nPhase count distribution:")
    for count, freq in count_dist.items():
        print(f"  {count} phases: {freq} patterns ({100*freq/len(ground_truth_data):.1f}%)")

    # Most common phases
    all_phases_present = []
    for phases_str in ground_truth_df['phases_present']:
        all_phases_present.extend(phases_str.split(','))

    from collections import Counter
    phase_freq = Counter(all_phases_present)
    print(f"\nMost common phases in mixtures:")
    for phase, freq in phase_freq.most_common(10):
        print(f"  {phase}: {freq} patterns ({100*freq/len(ground_truth_data):.1f}%)")

    print(f"\nTest mixtures ready for combined analysis!")
    print(f"Use the ground truth file to evaluate prediction accuracy.")


if __name__ == "__main__":
    main()
