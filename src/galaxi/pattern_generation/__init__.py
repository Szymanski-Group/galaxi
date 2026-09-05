from .realistic_xrd import RealisticXRDGenerator
from multiprocessing import Pool, Manager
from pymatgen.core import Structure
from scipy import signal
import multiprocessing
import pymatgen as mg
import numpy as np
import math
import os


class PatternGenerator:
    """
    Class used to generate augmented XRD patterns for all reference phases
    """

    def __init__(self, reference_dir, num_patterns=50, min_angle=10.0, max_angle=80.0,
                 is_pdf=False, enable_impurities=False,
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
            num_patterns: number of patterns to generate per technique per phase
            min_angle: minimum 2θ angle
            max_angle: maximum 2θ angle
            is_pdf: if True, convert XRD to PDF patterns
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
            noise_level: tuple of (min, max) noise as percentage of max intensity (e.g., 5.0 = 5%)
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
        self.num_patterns = num_patterns
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.is_pdf = is_pdf
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

    def augment(self, phase_info):
        """
        For a given phase, produce a list of augmented XRD patterns.

        Args:
            phase_info: list containing the pymatgen structure object
                and filename of that structure

        Returns:
            (patterns, filename): tuple with list of augmented patterns and filename
        """
        struc, filename = phase_info[0], phase_info[1]
        patterns, pdf_patterns = [], []

        # Always use RealisticXRDGenerator for all pattern generation.
        # min_angle/max_angle/num_points go inside `params` -- the constructor
        # only accepts a single `params` dict, not separate keyword arguments.
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

        # Always generate mixed-artifact patterns (5 * num_patterns)
        total_patterns = 5 * self.num_patterns

        # Generate multiple patterns using RealisticXRDGenerator
        realistic_patterns = generator.generate_multiple_patterns(struc, total_patterns, apply_all_effects=True)

        # Convert to expected format (list of lists of lists)
        for two_theta, intensity in realistic_patterns:
            pattern_list = [[val] for val in intensity]
            patterns.append(pattern_list)

        if self.is_pdf:
            for xrd in patterns:
                xrd = np.array(xrd).flatten()
                pdf = self.XRD_to_PDF(xrd, self.min_angle, self.max_angle)
                pdf = [[v] for v in pdf]
                pdf_patterns.append(pdf)
            return (pdf_patterns, filename)

        return (patterns, filename)

    @property
    def augmented_patterns(self):
        """
        Generate augmented patterns for all reference phases.

        Returns:
            sorted_patterns: numpy array of augmented patterns sorted by filename
        """
        phases = []
        for filename in sorted(os.listdir(self.ref_dir)):
            if filename.endswith('.cif'):
                phases.append([Structure.from_file(f'{self.ref_dir}/{filename}'), filename])

        with Manager() as manager:
            pool = Pool(self.num_cpu)
            grouped_xrd = pool.map(self.augment, phases)
            sorted_xrd = sorted(grouped_xrd, key=lambda x: x[1])  # Sort by filename
            sorted_patterns = [group[0] for group in sorted_xrd]

            return np.array(sorted_patterns)

    def XRD_to_PDF(self, xrd, min_angle, max_angle):
        """
        Convert XRD pattern to PDF (Pair Distribution Function).

        Args:
            xrd: XRD pattern
            min_angle: minimum 2θ angle
            max_angle: maximum 2θ angle

        Returns:
            pdf: PDF pattern
        """
        thetas = np.linspace(min_angle/2.0, max_angle/2.0, 4501)
        Q = np.array([4*math.pi*math.sin(math.radians(theta))/1.5406 for theta in thetas])
        S = np.array(xrd).flatten()

        pdf = []
        R = np.linspace(1, 40, 1000)  # Only 1000 used to reduce compute time
        integrand = Q * S * np.sin(Q * R[:, np.newaxis])

        pdf = (2*np.trapz(integrand, Q) / math.pi)
        pdf = list(signal.resample(pdf, 4501))

        return pdf

    def save_patterns(self, output_dir='generated_patterns'):
        """
        Generate and save augmented patterns to files.

        Args:
            output_dir: directory to save generated patterns

        Returns:
            None
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # First generate all patterns
        patterns_by_phase = self.augmented_patterns

        # Save each phase's patterns to a subdirectory
        for i, filename in enumerate(sorted(os.listdir(self.ref_dir))):
            if filename.endswith('.cif'):
                phase_name = filename.replace('.cif', '')
                phase_dir = f'{output_dir}/{phase_name}'
                if not os.path.exists(phase_dir):
                    os.makedirs(phase_dir)

                # Save each pattern
                for j, pattern in enumerate(patterns_by_phase[i]):
                    pattern_array = np.array(pattern).flatten()
                    two_theta = np.linspace(self.min_angle, self.max_angle, len(pattern_array))
                    pattern_data = np.column_stack((two_theta, pattern_array))
                    np.savetxt(f'{phase_dir}/{phase_name}_{j+1}.xy', pattern_data, fmt='%.6f')

                print(f"Saved {len(patterns_by_phase[i])} patterns for phase {phase_name}")
