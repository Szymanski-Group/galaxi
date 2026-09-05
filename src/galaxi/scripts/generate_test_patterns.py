#!/usr/bin/env python
"""
This script generates test patterns for each phase in the reference directory.
The patterns are generated with random mixed artifacts for realistic testing.

Standalone counterpart to StreamlinedWorkflow's comprehensive test-data step:
this one takes its parameters straight from the command line and writes plain
.xy files, with no workflow config or COD/background library involved. Use the
workflow step instead when the test set needs to match the config a model was
actually trained under.
"""

import os
import sys
import argparse
import numpy as np
from pymatgen.core import Structure
from galaxi.pattern_generation.realistic_xrd import RealisticXRDGenerator


def parse_args():
    parser = argparse.ArgumentParser(description="Generate test patterns for XRD phase identification")
    parser.add_argument("--ref_dir", type=str, default="References",
                        help="Directory containing reference CIF files")
    parser.add_argument("--output_dir", type=str, default="test_patterns",
                        help="Directory to save generated test patterns")
    parser.add_argument("--num_patterns", type=int, default=1,
                        help="Number of test patterns to generate per phase")
    parser.add_argument("--max_texture", type=float, default=0.6,
                        help="Maximum texture effect (0.0-1.0)")
    parser.add_argument("--min_domain_size", type=float, default=1.0,
                        help="Minimum domain size in nm")
    parser.add_argument("--max_domain_size", type=float, default=100.0,
                        help="Maximum domain size in nm")
    parser.add_argument("--max_strain", type=float, default=0.04,
                        help="Maximum strain (0.0-1.0)")
    parser.add_argument("--max_shift", type=float, default=0.25,
                        help="Maximum peak shift in degrees")
    parser.add_argument("--impur_amt", type=float, default=30.0,
                        help="Maximum impurity peak intensity (percent of max)")
    parser.add_argument("--min_angle", type=float, default=10.0,
                        help="Minimum 2θ angle")
    parser.add_argument("--max_angle", type=float, default=80.0,
                        help="Maximum 2θ angle")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    return parser.parse_args()


def generate_test_patterns(args):
    """
    Generate test patterns for each phase in the reference directory
    using mixed artifacts.

    Args:
        args: Command-line arguments
    """
    # Set random seed if specified
    if args.seed is not None:
        np.random.seed(args.seed)

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Get list of reference phases
    cif_files = [f for f in sorted(os.listdir(args.ref_dir)) if f.endswith('.cif')]

    if not cif_files:
        print(f"Error: No CIF files found in {args.ref_dir}")
        sys.exit(1)

    print(f"Found {len(cif_files)} reference phases in {args.ref_dir}")

    # Generate test patterns for each phase
    for cif_file in cif_files:
        phase_name = cif_file.replace('.cif', '')
        print(f"Generating {args.num_patterns} test patterns for phase: {phase_name}")

        # Load structure
        structure = Structure.from_file(os.path.join(args.ref_dir, cif_file))

        # Generate patterns using RealisticXRDGenerator. min_angle/max_angle/
        # num_points go inside `params` -- the constructor only accepts a
        # single `params` dict, not separate keyword arguments.
        generator = RealisticXRDGenerator(params={
            'min_angle': args.min_angle,
            'max_angle': args.max_angle,
            'num_points': 4501,
        })

        # Set parameter ranges
        generator.set_parameter_range('uniform_shift_range', (-args.max_shift, args.max_shift))
        generator.set_parameter_range('crystallite_size_range', (args.min_domain_size, args.max_domain_size))
        generator.set_parameter_range('microstrain_range', (0.0, args.max_strain))
        generator.set_parameter_range('texture_range', (1.0 - args.max_texture, 1.0 + args.max_texture))

        # Generate multiple patterns
        realistic_patterns = generator.generate_multiple_patterns(structure, args.num_patterns, apply_all_effects=True)

        # Convert to expected format
        patterns = []
        for two_theta, intensity in realistic_patterns:
            patterns.append(intensity)

        # Save patterns as XY files directly in the output directory
        for i, pattern in enumerate(patterns):
            # Pattern is already in the correct format
            intensities = np.array(pattern)

            # Create 2θ values
            two_theta = np.linspace(args.min_angle, args.max_angle, len(intensities))

            # Stack into two columns
            pattern_data = np.column_stack((two_theta, intensities))

            # Save to file
            filename = f"{phase_name}_test_{i+1}.xy"
            filepath = os.path.join(args.output_dir, filename)
            np.savetxt(filepath, pattern_data, fmt='%.6f', header="2Theta Intensity", comments='# ')

            print(f"  Saved {filepath}")

    print(f"\nSuccessfully generated {args.num_patterns} test patterns for each of the {len(cif_files)} phases.")
    print(f"Test patterns saved in {args.output_dir}")


def main():
    args = parse_args()
    generate_test_patterns(args)


if __name__ == "__main__":
    main()
