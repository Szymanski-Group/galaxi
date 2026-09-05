"""
Realistic XRD Pattern Generator

This module facilitates XRD pattern simulation with realistic instrumental
and physical effects for powder diffraction data. It implements all major effects
that influence powder XRD patterns to create training data that closely matches
experimental observations.
"""

import numpy as np
import random
import math
from scipy.special import wofz
from scipy.ndimage import gaussian_filter1d
from pymatgen.analysis.diffraction import xrd
from pymatgen.core import Structure
from typing import Tuple, List, Dict, Optional, Union
from .peak_perturbations import PeakPerturbator
from .peak_augmentations import PeakAugmentor
from dataclasses import dataclass
import torch

@dataclass
class PeakList:
    x: np.ndarray  #(L, )
    y: np.ndarray   #(L, )
    hkls: np.ndarray        #(L, 3)
    d_hkls: np.ndarray      #(L, )

class RealisticXRDGenerator:
    """
    Enhanced XRD pattern generator with realistic physical and instrumental effects.

    This class generates powder XRD patterns that include:
    - Cu Kα1/Kα2 doublet with proper intensity ratios
    - Pseudo-Voigt peak profiles
    - Temperature effects (Debye-Waller factors)
    - Instrumental peak broadening
    - Sample displacement effects
    - Temperature diffuse scattering background
    - Amorphous phase contributions (broad humps)
    - Realistic parameter ranges for powder XRD

    All effects are applied simultaneously to create mixed artifact patterns.
    """

    # X-ray wavelengths (in Angstroms)
    CU_KA1_WAVELENGTH = 1.5405929  # Cu Kα1
    CU_KA2_WAVELENGTH = 1.5444260  # Cu Kα2
    CU_KA1_KA2_RATIO = 2.0         # Intensity ratio Kα1:Kα2

    # Typical instrumental parameters for powder XRD
    DEFAULT_PARAMS = {
        'min_angle': 5.0,                      # Minimum 2θ angle (degrees)
        'max_angle': 105.0,                     # Maximum 2θ angle (degrees)
        'num_points': 5001,
        'convert_to_q': False,
        # Peak position shifts (degrees 2θ)
        'uniform_shift_range': (-0.05, 0.05),      # Smaller, realistic range
        'sample_displacement': (-0.2, 0.2),        # mm, typical sample displacement
        'goniometer_radius': 240.0,                # mm, typical goniometer radius

        # Peak broadening parameters
        'crystallite_size_range': (10.0, 200.0),   # nm, realistic for powder samples
        'crystallite_size_log_scale': False,       # sample log-uniformly instead of linearly over the range above
        'microstrain_range': (0.0, 0.003),         # Unitless, typical microstrain
        'instrumental_broadening': {
            'u': 0.01,    # Instrumental parameter U
            'v': -0.005,  # Instrumental parameter V
            'w': 0.002,   # Instrumental parameter W
        },

        # Peak shape parameters
        'pseudo_voigt_eta_range': (0.3, 0.8),      # Lorentzian fraction

        # Temperature effects
        'temperature_range': (250, 350),           # K, room temperature range
        'atomic_displacement_range': (0.005, 0.02), # U_iso, mean-square displacement <u^2> in Angstrom^2 (not the B-factor, which is 8*pi^2*U)

        # Preferred orientation
        'texture_range': (0.8, 1.2),               # March-Dollase parameter range
        'weights_low_index': 0.7,
        'low_index_orientation': [
                (1, 0, 0), (0, 1, 0), (0, 0, 1),      # Primary axes
                (1, 1, 0), (1, 0, 1), (0, 1, 1),      # Face diagonals
                (1, 1, 1), (-1, 1, 1), (1, -1, 1), (1, 1, -1)  # Body diagonals
            ],
        'high_index_orientation': [(2, 1, 0), (1, 2, 0), (2, 0, 1), (0, 2, 1), (1, 0, 2), (0, 1, 2),
                (1, 1, 2), (1, 2, 1), (2, 1, 1), (3, 1, 1), (1, 3, 1), (1, 1, 3),
                (3, 2, 1), (3, 1, 2), (1, 3, 2), (2, 3, 1), (1, 2, 3), (2, 1, 3),
                (3, 2, 2), (2, 3, 2), (2, 2, 3), (3, 3, 2), (3, 2, 3), (2, 3, 3)
            ],

        # Lattice strain (macroscopic strain on unit cell)
        'lattice_strain_range': (0.0, 0.03),       # Maximum lattice strain (unitless)

        # Background and noise
        'background_level': (0.5, 2.0),            # % of max intensity
        'noise_level': (1.0, 5.0),                 # percentage of max intensity (e.g., 5.0 = 5%)

        # Temperature diffuse scattering background
        'diffuse_scattering_intensity': (0.0, 25.0), # % of max peak intensity
        'diffuse_scattering_b_factor': (0.5, 3.0),   # Isotropic temperature factor range

        # Amorphous phase contributions
        'amorphous_intensity': (0.0, 25.0),          # % of max peak intensity
        'amorphous_neighbor_distance': (2.0, 4.0),   # Angstrom, typical nearest-neighbor distances
        'amorphous_disorder': (0.2, 0.8),            # Disorder parameter σ (Angstrom)

        # Impurity peak parameters
        'impurity_enable': False,                     # Whether to include impurity peaks
        'impurity_num_peaks_range': (1, 10),         # Number of impurity peaks to generate
        'impurity_intensity_range': (0.5, 70.0),     # % of max peak intensity for each impurity peak
        'impurity_width_range': (0.05, 0.3),         # FWHM range for impurity peaks (degrees)
        'impurity_eta_range': (0.2, 0.9),            # Pseudo-Voigt mixing parameter range for impurities
    }

    def __init__(self, params: Optional[Dict] = None):
        """
        Initialize the realistic XRD generator.

        Args:
            params: Custom parameters (uses defaults if None)
            min_angle: Minimum 2θ angle (degrees)
            max_angle: Maximum 2θ angle (degrees)
            num_points: Number of data points in pattern
            convert_to_q: Whether to convert patterns to Q-space
            perturbator: stores perturbator configs
            augmentor: stores augmentor configs
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.min_angle = self.params['min_angle']
        self.max_angle = self.params['max_angle']
        self.num_points = self.params['num_points']
        self.convert_to_q = self.params['convert_to_q']

        # Create angle grid
        self.two_theta_grid = torch.linspace(self.min_angle, self.max_angle, self.num_points, device=self.device, dtype=torch.float32)   #(num_points,)
        self.step_size = (self.max_angle - self.min_angle) / (self.num_points - 1)

        # Initialize perturbator (None by default)
        self.perturbator = None

        # Initialize augmentor (None by default)
        self.augmentor = None

    def enable_peak_perturbations(self, peak_perturbation_config: Dict = {}, enable: bool = True):
        """
        Enable or disable peak perturbations for unphysical negative examples.

        Args:
            enable: Whether to enable perturbations
        """

        if enable:
            # perterbation peak threshold
            perturbation_intensity_threshold = peak_perturbation_config.get('perturbation_intensity_threshold', 0.5)

            # probabilities
            removal_probability = peak_perturbation_config.get('probabilities', {}).get('removal_probability', 0.5)
            intensity_probability= peak_perturbation_config.get('probabilities', {}).get('intensity_change_probability', 0.7)
            shift_probability = peak_perturbation_config.get('probabilities', {}).get('shift_probability', 0.8)

            # perturbation strength
            perturbation_fraction_range = peak_perturbation_config.get('perturbation_strength', {}).get('perturbation_fraction_range', [0, 0.8])
            shift_range = peak_perturbation_config.get('perturbation_strength', {}).get('shift_range', [0.3, 1.5])
            intensity_factor_range = peak_perturbation_config.get('perturbation_strength', {}).get('intensity_factor_range', [0.1, 10])

            self.perturbator = PeakPerturbator(
                perturbation_intensity_threshold=perturbation_intensity_threshold,
                perturbation_fraction_range=perturbation_fraction_range,
                removal_probability=removal_probability,
                shift_probability=shift_probability,
                intensity_probability=intensity_probability,
                shift_range=shift_range,
                intensity_factor_range=intensity_factor_range
                )
        else:
            self.perturbator = None

    def enable_peak_augmentations(self, peak_augmentation_config: Dict = {}, enable: bool = True):

        augmentation_intensity_threshold = peak_augmentation_config.get('augmentation_intensity_threshold', 0.25)
        augmentation_fraction = peak_augmentation_config.get('augmentation_fraction', 0.3)
        overlapping_peak_intensity_range = peak_augmentation_config.get('overlapping_peak_intensity_range', [0.2, 2])
        overlapping_peak_shift_range = peak_augmentation_config.get('overlapping_peak_shift_range', [0, 0.5])

        if enable:
            self.augmentor = PeakAugmentor(
                augmentation_intensity_threshold=augmentation_intensity_threshold,
                augmentation_fraction=augmentation_fraction,
                overlapping_peak_intensity_range=overlapping_peak_intensity_range,
                overlapping_peak_shift_range=overlapping_peak_shift_range
            )
        else:
            self.augmentor = None

    def _calculate_debye_waller_factor(self, d_spacing: torch.Tensor, temperature: torch.Tensor, u_iso: torch.Tensor) -> torch.Tensor:
        """
        d_spacing: (B, N) tensor  [Angstrom]
        temperature: (B, 1) tensor from temperature_range
        u_iso: (B, 1) tensor from atomic_displacement_range
        Returns: Debye–Waller factor (B, N)
        """
        # sinθ/λ = 1 / (2d)
        sin_theta_over_lambda = 1.0 / (2.0 * d_spacing)   #(B, N)
        sin_theta_over_lambda_sq = sin_theta_over_lambda ** 2       #(B, N)

        # Debye-Waller factor = 8π² u_iso sin²θ/λ²
        m_factor = 8.0 * (torch.pi**2) * u_iso * sin_theta_over_lambda_sq       #(B, N)

        # Temperature scaling
        temp_factor = temperature / 300.0       #(B, 1)
        m_factor = m_factor * temp_factor       #(B, N)

        # Debye–Waller: exp(-2M)
        return torch.exp(-2.0 * m_factor)       #(B, N)

    def _calculate_sample_displacement_shift(self, x: torch.Tensor, displacement: torch.Tensor) -> torch.Tensor:
        """
        Calculate peak shift due to sample displacement.

        Formula: Δ(2θ) = -2s·cosθ/R
        where s is displacement, R is goniometer radius

        Args:
            x: 2θ angle in degrees in (B, n_peaks)
            displacement: Sample displacement in mm in (B, 1)

        Returns:
            Peak shift in degrees 2θ
        """
        goniometer_radius = self.params['goniometer_radius']
        theta_rad = torch.deg2rad(x * 0.5)
        cos_theta = torch.cos(theta_rad)        # (B, n_peaks)

        # Δ(2θ) = -2s·cosθ/R  (radians)
        shift_rad = -2.0 * displacement * cos_theta / goniometer_radius

        # return shift in degrees
        return torch.rad2deg(shift_rad)

    def _calculate_instrumental_broadening(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate instrumental broadening using Caglioti equation.

        FWHM_inst² = U*tan²θ + V*tanθ + W

        Args:
            x: 2θ angle in degrees

        Returns:
            Instrumental FWHM in degrees
        """
        theta_rad = torch.deg2rad(x / 2.0)
        tan_theta = torch.tan(theta_rad)

        u = self.params['instrumental_broadening']['u']
        v = self.params['instrumental_broadening']['v']
        w = self.params['instrumental_broadening']['w']

        fwhm_sq = u * tan_theta**2 + v * tan_theta + w
        fwhm_sq = torch.clamp(fwhm_sq, min=1e-4)

        return torch.sqrt(fwhm_sq)  # Minimum broadening

    def _calculate_size_broadening(self, x: torch.Tensor, crystallite_size: torch.Tensor) -> torch.Tensor:
        """
        Calculate size broadening using Scherrer equation.

        Args:
            x: 2θ angle in degrees
            crystallite_size: Crystallite size in nm

        Returns:
            Size broadening FWHM in degrees
        """
        theta_rad = torch.deg2rad(x / 2.0)
        k_factor = 0.9  # Scherrer constant
        wavelength_nm = self.CU_KA1_WAVELENGTH / 10.0  # Convert to nm

        # Scherrer equation: β = Kλ/(D*cosθ)
        beta_rad = k_factor * wavelength_nm / (crystallite_size * torch.cos(theta_rad))
        return torch.rad2deg(beta_rad)

    def _calculate_strain_broadening(self, two_theta: torch.Tensor,
                                   microstrain: torch.Tensor) -> torch.Tensor:
        """
        Calculate strain broadening.

        Args:
            x: 2θ angle in degrees
            microstrain: Microstrain (unitless)

        Returns:
            Strain broadening FWHM in degrees
        """
        theta_rad = torch.deg2rad(two_theta / 2.0)
        # Strain broadening: β = 4ε*tanθ
        beta_rad = 4.0 * microstrain * torch.tan(theta_rad)
        return torch.rad2deg(beta_rad)

    def _pseudo_voigt_profile(
        self,
        two_theta_grid: torch.Tensor,  # (B, N)
        x: torch.Tensor,               # (B, P)
        fwhm: torch.Tensor,            # (B, P)
        eta: torch.Tensor,             # (B, 1)
        chunk: int = 512
    ):
        B, N = two_theta_grid.shape
        P = x.shape[1]

        # Precompute widths
        sigma = fwhm / (2.0 * torch.sqrt(2.0 * torch.log(torch.tensor(2.0, device=fwhm.device))))
        sigma = torch.clamp(sigma, min=1e-6).unsqueeze(2)  # (B,P,1)

        gamma = torch.clamp(fwhm / 2.0, min=1e-6).unsqueeze(2)  # (B,P,1)
        gamma2 = gamma ** 2

        eta = eta.unsqueeze(2)  # (B,1,1)

        # Allocate output
        peak_profile = torch.empty((B, P, N), device=two_theta_grid.device)

        with torch.no_grad():
            for i in range(0, N, chunk):
                j = min(i + chunk, N)

                grid = two_theta_grid[:, i:j].unsqueeze(1)  # (B,1,chunk)
                dx = grid - x.unsqueeze(2)                  # (B,P,chunk)

                gauss = torch.exp(-0.5 * (dx / sigma) ** 2)
                gauss.mul_(1 - eta)

                lor = gamma2 / (dx * dx + gamma2)
                lor.mul_(eta)

                peak_profile[:, :, i:j].copy_(gauss + lor)

        return peak_profile


    def _add_background(self, pattern: torch.Tensor, background_level: torch.Tensor) -> torch.Tensor:
        """
        Add gaussian+linear background to pattern.

        Args:
            pattern: XRD pattern intensities
            level: Background level as fraction of max intensity

        Returns:
            Pattern with background added
        """

        N = self.num_points
        B = pattern.shape[0]

        # x-grid from 0 to 1 (representing normalized 2θ)
        x = torch.linspace(0, 1, N, device=self.device).unsqueeze(0)   # (1, N)

        # Hump center between 0 and 0.2
        hump_center = torch.rand(B, 1, device=self.device) * 0.20   # (B, 1)

        # Hump width between 0.1 and 0.3
        hump_width = 0.10 + torch.rand(B, 1, device=self.device) * 0.20  # (B, 1)

        # Gaussian hump: exp(-(x - c)^2 / w^2)
        hump = torch.exp(-((x - hump_center)**2) / (hump_width**2))  # (B, N)

        # Random ramp: 0.05 to 0.2 slope
        ramp_slope = 0.05 + 0.15 * torch.rand(B, 1, device=self.device)  # (B, 1)
        ramp = ramp_slope * x                                       # (B, N)

        # Random mixing: 0.6 to 0.9 hump weight
        mix = 0.6 + 0.3 * torch.rand(B, 1, device=self.device)           # (B, 1)

        # Combine hump + ramp
        shape = mix * hump + (1 - mix) * ramp                       # (B, N)

        # Scale by background level * peak intensity
        max_int = pattern.max(dim=1, keepdim=True).values           # (B, 1)
        background = background_level * max_int * shape              # (B, N)

        return pattern + background

    def _calculate_temperature_diffuse_scattering(self, two_theta_array: torch.Tensor,
                                                 i_zero: torch.Tensor, b_factor: torch.Tensor) -> torch.Tensor:
        """
        Calculate temperature diffuse scattering background.

        Uses the equation: I_diffuse ∝ I₀ × exp(-2M) × [1 - exp(-2M)]
        Where M = (B sin²θ)/λ²

        Args:
            two_theta_array: Array of 2θ angles in degrees in (B, num_points)
            i_zero: Incident beam intensity scaling factor in (B, 1)
            b_factor: Isotropic temperature factor B (Å²) in (B, 1)

        Returns:
            Array of diffuse scattering intensities
        """
        # Convert 2θ to θ in radians
        theta_rad = torch.deg2rad(two_theta_array / 2.0)

        # Calculate sin²θ
        sin_theta_sq = torch.sin(theta_rad)**2      # (B, num_points)

        # Calculate M = (B sin²θ)/λ²
        # Using average wavelength for simplicity
        wavelength_avg = (self.CU_KA1_WAVELENGTH + self.CU_KA2_WAVELENGTH) / 2.0
        m_factor = (b_factor * sin_theta_sq) / (wavelength_avg**2)      # (B, num_points)

        # Calculate exp(-2M)
        exp_2m = torch.exp(-2.0 * m_factor)     # (B, num_points)

        # Calculate diffuse scattering: I₀ × exp(-2M) × [1 - exp(-2M)]
        diffuse_intensity = i_zero * exp_2m * (1.0 - exp_2m)        # (B, num_points)

        return diffuse_intensity

    def _calculate_amorphous_contribution(self, two_theta_array: torch.Tensor,
                                        amplitude: torch.Tensor, r1: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        Calculate amorphous phase contribution using simplified Debye scattering.

        Uses approximation: I(Q) ≈ A × sin(Qr₁)/(Qr₁) × exp(-Q²σ²/2)
        where Q = 4π sin(θ)/λ

        Args:
            two_theta_array: Array of 2θ angles in degrees in (B, num_points)
            amplitude: Amplitude factor A in (B, 1)
            r1: Average nearest-neighbor distance (Angstrom) in (B, 1)
            sigma: Disorder parameter (Angstrom) in (B, 1)

        Returns:
            Array of amorphous scattering intensities
        """
        # Convert 2θ to θ in radians
        theta_rad = torch.deg2rad(two_theta_array / 2.0)        # (B, num_points)

        # Calculate Q = 4π sin(θ)/λ
        # Using average wavelength for simplicity
        wavelength_avg = (self.CU_KA1_WAVELENGTH + self.CU_KA2_WAVELENGTH) / 2.0
        q_vector = 4.0 * torch.pi * torch.sin(theta_rad) / wavelength_avg

        # Avoid division by zero
        q_vector = torch.clamp(q_vector, min=1e-6)      # (B, num_points)

        # Calculate Qr₁
        qr1 = q_vector * r1

        # Calculate sin(Qr₁)/(Qr₁) - the interference function
        # Use sinc function: sinc(x) = sin(πx)/(πx), so sin(x)/x = sinc(x/π)
        interference = torch.sinc(qr1 / torch.pi)

        # Calculate disorder factor: exp(-Q²σ²/2)
        disorder_factor = torch.exp(-q_vector**2 * sigma**2 / 2.0)

        # Combine to get amorphous intensity
        amorphous_intensity = amplitude * interference * disorder_factor

        # Ensure non-negative values
        amorphous_intensity = torch.clamp(amorphous_intensity, min=0)

        return amorphous_intensity

    def _generate_impurity_peaks(self, max_intensity: torch.Tensor):
        """
        Generate random impurity peaks with realistic pseudo-Voigt profiles.

        Args:
            max_intensity: Maximum intensity in the main pattern for scaling (B,1)

        Returns:
            Tuple of (angles, intensities, widths, eta_values, mask)
            - angles:      (B, n_peaks)
            - intensities: (B, n_peaks)
            - widths:      (B, n_peaks)
            - eta_values:  (B, n_peaks)
            - mask:        (B, n_peaks)  boolean mask of valid peaks
        """
        if not self.params['impurity_enable']:
            B = max_intensity.shape[0]
            return (
                torch.empty(B, 0, device=self.device),
                torch.empty(B, 0, device=self.device),
                torch.empty(B, 0, device=self.device),
                torch.empty(B, 0, device=self.device),
                torch.empty(B, 0, dtype=torch.bool, device=self.device)
            )

        B = max_intensity.shape[0]
        lowP, highP = self.params['impurity_num_peaks_range']
        P_max = highP     # maximum possible peaks per batch

        # Sample number of impurity peaks per batch item (B,)
        num_peaks = torch.randint(lowP, highP + 1, (B,), device=self.device)    # (B,)

        # Create a mask for valid peaks (B, n_peaks)
        peak_idx = torch.arange(P_max, device=self.device).unsqueeze(0)  # (1, n_peaks)
        mask = peak_idx < num_peaks.unsqueeze(1)                         # (B, n_peaks)

        # Generate random angles for impurity peaks
        angles = torch.empty(B, P_max, device=self.device).uniform_(
            self.min_angle, self.max_angle
        )       # (B, n_peaks)

        # Generate random intensities as percentage of max intensity
        frac_low, frac_high = self.params['impurity_intensity_range']
        intensity_fraction = torch.empty(B, P_max, device=self.device).uniform_(
            frac_low, frac_high
        ) / 100.0       # (B, n_peaks)

        # Broadcast to match shape (B, n_peaks)
        intensities = intensity_fraction * max_intensity          # (B, n_peaks)

        # Generate random peak widths (FWHM)
        w_low, w_high = self.params['impurity_width_range']
        widths = torch.empty(B, P_max, device=self.device).uniform_(w_low, w_high)   # (B, n_peaks)

        # Generate random pseudo-Voigt mixing parameters
        eta_low, eta_high = self.params['impurity_eta_range']
        eta_values = torch.empty(B, P_max, device=self.device).uniform_(eta_low, eta_high)     # (B, n_peaks)

        return angles, intensities, widths, eta_values, mask

    def _add_impurity_peaks_to_pattern(self, intensity_grid: torch.Tensor,
                                    max_intensity: torch.Tensor) -> torch.Tensor:
        """
        Add impurity peaks to the existing intensity pattern.

        Args:
            intensity_grid: Current intensity pattern in (B, num_points)
            max_intensity: Maximum intensity for scaling impurity peaks in (B, 1)

        Returns:
            Updated intensity pattern with impurity peaks added
        """
        if not self.params['impurity_enable']:
            return intensity_grid

        B, N = intensity_grid.shape

        # Generate impurity peak parameters
        angles, intensities, widths, eta_values, mask = self._generate_impurity_peaks(max_intensity)

        # If no peaks exist (edge case)
        if angles.shape[1] == 0:
            return intensity_grid

        # Prepare shapes for broadcasting
        # two_theta_grid: (B, N)
        # angles:         (B, P)
        # intensities:    (B, P)
        # widths:         (B, P)
        # eta_values:     (B, P)
        # mask:           (B, P)

        # Compute pseudo-Voigt profiles: (B, P, N). self.two_theta_grid is
        # (N,) -- a single grid shared across the batch -- but
        # _pseudo_voigt_profile requires (B, N), so broadcast it to match.
        peak_profile = self._pseudo_voigt_profile(
            self.two_theta_grid.unsqueeze(0).expand(B, -1),  # (B, N)
            angles,               # (B, P)
            widths,               # (B, P)
            eta_values            # (B, P)
        )

        # Apply mask
        mask_3d = mask.unsqueeze(2)           # (B, P, 1)
        peak_profile = peak_profile * mask_3d # (B, P, N)

        # Multiply each peak by its amplitude
        peak_profile = peak_profile * intensities.unsqueeze(2)     # (B, P, N)

        # Sum impurity peaks
        impurity_sum = peak_profile.sum(dim=1)    # (B, N)

        # Add to main intensity
        return intensity_grid + impurity_sum

    def _convert_pattern_to_q_space(self, two_theta_grid, intensity_grid):
        """
        Convert 2θ pattern to Q-space.

        Args:
            two_theta_grid: 2θ angles in degrees in (B, num_points)
            intensity_grid: Intensities in (B, num_points)
        Returns:
            Q-space grid and intensities
        """
        theta_rad = torch.deg2rad(two_theta_grid / 2.0)
        wavelength = self.CU_KA1_WAVELENGTH
        q = (4.0 * torch.pi / wavelength) * torch.sin(theta_rad)
        return q, intensity_grid

    def _get_crystal_system(self, structure) -> str:
        """
        Determine crystal system from space group number.

        Args:
            structure: Pymatgen Structure object

        Returns:
            Crystal system classification string
        """

        try:
            sg_num = structure.get_space_group_info()[1]
        except Exception:
            # Fallback for structures with undetermined symmetry
            return 'triclinic'  # Default to lowest symmetry

        if 1 <= sg_num <= 2:
            return 'triclinic'
        elif 3 <= sg_num <= 15:
            return 'monoclinic'
        elif 16 <= sg_num <= 74:
            return 'orthorhombic'
        elif 75 <= sg_num <= 142:
            return 'tetragonal'
        elif 143 <= sg_num <= 167:
            return 'hexagonal'
        elif 168 <= sg_num <= 194:
            return 'hexagonal'  # Also includes trigonal
        elif 195 <= sg_num <= 230:
            return 'cubic'
        else:
            return 'triclinic'  # Default fallback

    def _apply_lattice_strain(self, structure, max_strain: float):
        """
        Apply symmetry-preserving lattice strain to structure.

        Args:
            structure: Pymatgen Structure object
            max_strain: Maximum strain magnitude

        Returns:
            Modified structure with strained lattice
        """
        crystal_system = self._get_crystal_system(structure)

        # Generate strain components
        diag_range = (-max_strain, max_strain)
        off_diag_range = (-max_strain/2, max_strain/2)  # Smaller off-diagonal terms

        s11, s22, s33 = [np.random.uniform(*diag_range) for _ in range(3)]
        s12, s13, s21, s23, s31, s32 = [np.random.uniform(*off_diag_range) for _ in range(6)]

        # Apply symmetry constraints
        if crystal_system == 'cubic':
            # Cubic: all diagonal equal, no off-diagonal
            strain_tensor = np.array([
                [s11, 0, 0],
                [0, s11, 0],
                [0, 0, s11]
            ])
        elif crystal_system == 'tetragonal':
            # Tetragonal: a=b≠c, limited off-diagonal
            strain_tensor = np.array([
                [s11, 0, 0],
                [0, s11, 0],
                [0, 0, s33]
            ])
        elif crystal_system == 'hexagonal':
            # Hexagonal: a=b≠c, s12=-s11/2
            s12_hex = -s11/2
            strain_tensor = np.array([
                [s11, s12_hex, 0],
                [s12_hex, s11, 0],
                [0, 0, s33]
            ])
        elif crystal_system == 'orthorhombic':
            # Orthorhombic: all different diagonal, no off-diagonal
            strain_tensor = np.array([
                [s11, 0, 0],
                [0, s22, 0],
                [0, 0, s33]
            ])
        elif crystal_system == 'monoclinic':
            # Monoclinic: only s23≠0 off-diagonal
            strain_tensor = np.array([
                [s11, 0, 0],
                [0, s22, s23],
                [0, s23, s33]
            ])
        else:  # triclinic
            # Triclinic: all components allowed
            strain_tensor = np.array([
                [s11, s12, s13],
                [s21, s22, s23],
                [s31, s32, s33]
            ])

        # Apply strain: ε = (F^T F - I)/2, F = I + ∇u, for small strain F ≈ I + ε
        # For small strains, new_lattice = old_lattice * (I + strain_tensor)
        lattice_matrix = structure.lattice.matrix
        identity = np.eye(3)
        deformation_gradient = identity + strain_tensor

        # Apply deformation
        new_lattice_matrix = lattice_matrix @ deformation_gradient

        # Create new structure with strained lattice
        from pymatgen.core import Lattice
        new_lattice = Lattice(new_lattice_matrix)
        strained_structure = structure.copy()
        strained_structure._lattice = new_lattice

        return strained_structure

    def _apply_march_dollase_correction(self, y: float, hkls: tuple,
                                      march_dollase_param: float,
                                      preferred_directions: tuple = (0, 0, 1)) -> float:
        """
        Apply March-Dollase preferred orientation correction.

        Args:
            y: peak intensity in (b, n_peaks)
            hkls: Miller indices of the reflection in (b, n_peaks, 3)
            march_dollase_param: March-Dollase parameter (1.0 = random, <1.0 or >1.0 = preferred) ; in (b, 1)
            preferred_direction: Preferred orientation direction in (b, 3)

        Returns:
            Corrected y
        """


        pref_h, pref_k, pref_l = preferred_directions.split(1, dim=1)       #(b, 1)
        h, k, l = hkls[..., 0], hkls[..., 1], hkls[..., 2]       #(b, n_peaks)
        illegal_mask = (h == 0) & (k == 0) & (l == 0)

        # Dot product and magnitudes
        dot_product = h*pref_h + k*pref_k + l*pref_l
        hkl_magnitude = torch.sqrt(h*h + k*k + l*l)
        hkl_magnitude[illegal_mask] = 1.0
        pref_magnitude = torch.sqrt(pref_h*pref_h + pref_k*pref_k + pref_l*pref_l)

        # Cosine of angle between hkl and preferred direction
        cos_alpha = dot_product / (hkl_magnitude * pref_magnitude)
        cos_alpha = torch.clamp(cos_alpha, -1.0, 1.0)       #(b, n_peaks)

        # March-Dollase correction factor P(α) = (r²cos²α + sin²α/r)^(-3/2)
        r = march_dollase_param     #(b, 1)
        r = torch.clamp(r, min=1e-6)     # Make sure no zero or negative values
        sin_alpha_sq = 1.0 - cos_alpha*cos_alpha
        numerator = r*r * cos_alpha*cos_alpha + sin_alpha_sq / r
        numerator = torch.clamp(numerator, min=1e-12)
        correction = numerator**(-1.5)      #(b, n_peaks)

        y_corrected = y * correction
        # Restore original y for padded HKL=000 to avoid NaN
        y_corrected[illegal_mask] = y[illegal_mask]

        return y_corrected      #(b, n_peaks)

    def get_peak_list(self, structure, apply_all_effects: bool = True):
        # Initialize XRD calculators for both wavelengths
        calc_ka1 = xrd.XRDCalculator(wavelength=self.CU_KA1_WAVELENGTH)
        calc_ka2 = xrd.XRDCalculator(wavelength=self.CU_KA2_WAVELENGTH)

        # Apply lattice strain to structure if requested
        working_structure = structure
        lattice_strain = np.random.uniform(*self.params['lattice_strain_range'])
        if apply_all_effects and lattice_strain > 0:
            working_structure = self._apply_lattice_strain(structure, lattice_strain)

        # Generate patterns for both Kα1 and Kα2
        pattern_ka1 = calc_ka1.get_pattern(
            working_structure,
            two_theta_range=(self.min_angle, self.max_angle)
        )
        pattern_ka2 = calc_ka2.get_pattern(
            working_structure,
            two_theta_range=(self.min_angle, self.max_angle)
        )

        # 1. Extract the raw arrays
        ka1_x = np.array(pattern_ka1.x)
        ka1_y = np.array(pattern_ka1.y)
        ka1_hkls = np.array([h[0]["hkl"] for h in pattern_ka1.hkls])
        ka1_dhkls = np.array(pattern_ka1.d_hkls)

        ka2_x = np.array(pattern_ka2.x)
        ka2_y = np.array(pattern_ka2.y)
        ka2_hkls = np.array([h[0]["hkl"] for h in pattern_ka2.hkls])
        ka2_dhkls = np.array(pattern_ka2.d_hkls)

        # 2. Calculate global max intensity and threshold
        # Safety check prevents ValueError if an array is completely empty
        max_ka1 = ka1_y.max() if len(ka1_y) > 0 else 0.0
        max_ka2 = ka2_y.max() if len(ka2_y) > 0 else 0.0
        max_intensity = max(max_ka1, max_ka2)

        threshold = 0.005 * max_intensity

        # 3. Define the boolean masks
        mask_ka1 = ka1_y >= threshold
        mask_ka2 = ka2_y >= threshold

        # 4. Apply the mask to ALL arrays simultaneously and build the PeakLists
        filtered_ka1 = PeakList(
            x = ka1_x[mask_ka1],
            y = ka1_y[mask_ka1],
            hkls = ka1_hkls[mask_ka1],
            d_hkls = ka1_dhkls[mask_ka1]
        )

        filtered_ka2 = PeakList(
            x = ka2_x[mask_ka2],
            y = ka2_y[mask_ka2],
            hkls = ka2_hkls[mask_ka2],
            d_hkls = ka2_dhkls[mask_ka2]
        )

        peak_list = {
            'ka1': filtered_ka1,
            'ka2': filtered_ka2
        }
        return peak_list

    def generate_realistic_pattern(self, peak_list, batch_size, apply_all_effects=True, enable_impurity=True):
        # Initiallizing intensity grid
        intensity_grid = torch.zeros(batch_size, self.num_points, device=self.device)   # (b, num_points)

        # Handling peak augmentation and perturbation by treating Ka1 and Ka2 together to make sure
        # that augmentation and perturbation are consistent between the two wavelengths
        x_all = torch.tensor(
            np.concatenate([peak_list["ka1"].x, peak_list["ka2"].x]),
            dtype=torch.float32, device=self.device
        ).unsqueeze(0).repeat(batch_size, 1)

        y_all = torch.tensor(
            np.concatenate([peak_list["ka1"].y, peak_list["ka2"].y]),
            dtype=torch.float32, device=self.device
        ).unsqueeze(0).repeat(batch_size, 1)

        hkls_all = torch.tensor(
            np.concatenate([peak_list["ka1"].hkls, peak_list["ka2"].hkls]),
            dtype=torch.float32, device=self.device
        ).unsqueeze(0).repeat(batch_size, 1, 1)

        d_hkls_all = torch.tensor(
            np.concatenate([peak_list["ka1"].d_hkls, peak_list["ka2"].d_hkls]),
            dtype=torch.float32, device=self.device
        ).unsqueeze(0).repeat(batch_size, 1)
        n_ka1 = len(peak_list["ka1"].x)
        n_ka2 = len(peak_list["ka2"].x)

        # Apply augmentation
        if self.augmentor is not None:
            x_all, y_all, hkls_all, d_hkls_all, aug_info = self.augmentor.augment_peak_list(x_all, y_all, hkls_all, d_hkls_all)

        # Apply perturbation
        if self.perturbator is not None:
            x_all, y_all, pert_info = self.perturbator.perturb_peak_list(x_all, y_all)

        # Split after augmentation
        x_ka1 = x_all[:, :n_ka1]
        y_ka1 = y_all[:, :n_ka1]
        hkls_ka1 = hkls_all[:, :n_ka1]
        d_hkls_ka1 = d_hkls_all[:, :n_ka1]

        x_ka2 = x_all[:, n_ka1:]
        y_ka2 = y_all[:, n_ka1:]
        hkls_ka2 = hkls_all[:, n_ka1:]
        d_hkls_ka2 = d_hkls_all[:, n_ka1:]

        # Sample the physical latent variables (temperature, texture, shifts, broadening,
        # peak shape) once per pattern: Ka1 and Ka2 are the same physical specimen, so they
        # must share these values rather than each independently sampling its own.
        effects = self._sample_physical_effects(batch_size) if apply_all_effects else None

        # ========== Handling Ka1 ==========
        intensity_grid += self._discrete_to_continuous(apply_all_effects=apply_all_effects,
                                                       x=x_ka1,
                                                       y=y_ka1,
                                                       hkls=hkls_ka1,
                                                       d_hkls=d_hkls_ka1,
                                                       batch_size=batch_size,
                                                       effects=effects) * self.CU_KA1_KA2_RATIO / (1 + self.CU_KA1_KA2_RATIO)  # Weight by Kα1 intensity

        #  ========== Handling Ka2 ==========
        intensity_grid += self._discrete_to_continuous(apply_all_effects=apply_all_effects,
                                                       x=x_ka2,
                                                       y=y_ka2,
                                                       hkls=hkls_ka2,
                                                       d_hkls=d_hkls_ka2,
                                                       batch_size=batch_size,
                                                       effects=effects) * 1 / (1 + self.CU_KA1_KA2_RATIO)  # Weight by Kα2 intensity


        # Handling other artifacts on continuous patterns
        # Normalize intensities
        max_val = intensity_grid.max(dim=1, keepdim=True).values   # (B, 1)
        min_val = intensity_grid.min(dim=1, keepdim=True).values   # (B, 1)
        intensity_grid = (intensity_grid - min_val) / (max_val - min_val + 1e-8)   # (B, num_points)

        if apply_all_effects:
            # Add background
            background_level = torch.empty(batch_size, 1, device=self.device).uniform_(*self.params['background_level']) / 100      #(B, 1)
            intensity_grid = self._add_background(intensity_grid, background_level)
            # Add temperature diffuse scattering background
            diffuse_fraction = torch.empty(batch_size, 1, device=self.device).uniform_(*self.params['diffuse_scattering_intensity']) / 100      #(B, 1)
            diffuse_b_factor = torch.empty(batch_size, 1, device=self.device).uniform_(*self.params['diffuse_scattering_b_factor'])      #(B, 1)
            max_intensity_before_diffuse = intensity_grid.max(dim=1, keepdim=True).values      # (B, 1)
            i_zero = diffuse_fraction * max_intensity_before_diffuse        # (B, 1)

            diffuse_scattering = self._calculate_temperature_diffuse_scattering(
                self.two_theta_grid, i_zero, diffuse_b_factor
            )
            intensity_grid += diffuse_scattering

            # Add amorphous phase contribution (broad humps at low 2θ)
            amorphous_fraction = torch.empty(batch_size, 1, device=self.device).uniform_(*self.params['amorphous_intensity']) / 100      #(B, 1)
            neighbor_distance = torch.empty(batch_size, 1, device=self.device).uniform_(*self.params['amorphous_neighbor_distance'])      #(B, 1)
            disorder = torch.empty(batch_size, 1, device=self.device).uniform_(*self.params['amorphous_disorder'])      #(B, 1)
            max_intensity_before_amorphous = intensity_grid.max(dim=1, keepdim=True).values      # (B, 1)
            amorphous_amplitude = amorphous_fraction * max_intensity_before_amorphous
            amorphous_contribution = self._calculate_amorphous_contribution(
                self.two_theta_grid, amorphous_amplitude, neighbor_distance, disorder
            )
            intensity_grid += amorphous_contribution

            # Add impurity peaks if enabled
            if enable_impurity:
                max_intensity_before_impurity = intensity_grid.max(dim=1, keepdim=True).values      # (B, 1)
                intensity_grid = self._add_impurity_peaks_to_pattern(intensity_grid, max_intensity_before_impurity)

            # Add noise
            noise_level = torch.empty(batch_size, 1, device=self.device).uniform_(*self.params['noise_level']) / 100.0     # (B, 1)
            max_intensity = intensity_grid.max(dim=1, keepdim=True).values     # (B, 1)
            std = noise_level * max_intensity                                 # (B, 1)
            noise = torch.randn_like(intensity_grid) * std                    # (B, N)
            intensity_grid = intensity_grid + noise

            # Ensure non-negative values
            intensity_grid = torch.clamp(intensity_grid, min=0.0)

        # Convert to Q-space if requested
        if self.convert_to_q:
            x_grid, intensity_grid = self._convert_pattern_to_q_space(
                self.two_theta_grid, intensity_grid
            )
        else:
            x_grid = self.two_theta_grid

        # handle perturbation info

        if hasattr(self, 'perturbator') and self.perturbator is not None:
            return x_grid, intensity_grid, pert_info
        elif hasattr(self, 'augmentor') and self.augmentor is not None:
            return x_grid, intensity_grid, aug_info
        else:
            return x_grid, intensity_grid


    def _sample_physical_effects(self, batch_size):
        """
        Sample the per-specimen physical latent variables (temperature, texture,
        peak-position shifts, broadening, peak-shape) once per pattern. Ka1 and Ka2
        come from the same physical specimen/measurement, so both must reuse the
        same sampled values rather than each drawing independently.
        """
        temperature = torch.empty(batch_size, 1, device=self.device).uniform_(*self.params['temperature_range'])    # (B, 1)
        u_iso = torch.empty(batch_size, 1, device=self.device).uniform_(*self.params['atomic_displacement_range'])      #(B, 1)

        texture_param = torch.empty(batch_size, 1, device=self.device).uniform_(*self.params['texture_range'])      #(B, 1)
        LOW_INDEX = self.params['low_index_orientation']
        HIGH_INDEX = self.params['high_index_orientation']
        frac = self.params['weights_low_index']
        preferred_directions = LOW_INDEX + HIGH_INDEX
        weights = [frac] * len(LOW_INDEX) + [1-frac] * len(HIGH_INDEX)
        weights = np.array(weights, dtype=np.float32)
        weights /= weights.sum()
        chosen_idx = np.random.choice(
            len(preferred_directions),
            size=batch_size,
            p=weights
        )
        preferred_orientations = torch.tensor(
            [preferred_directions[i] for i in chosen_idx],
            dtype=torch.float32,
            device=self.device
        )   # shape: (B, 3)

        uniform_shift = torch.empty(batch_size, 1, device=self.device).uniform_(*self.params['uniform_shift_range'])      #(B, 1)
        sample_displacement = torch.empty(batch_size, 1, device=self.device).uniform_(*self.params['sample_displacement'])      #(B, 1)

        if self.params.get('crystallite_size_log_scale', False):
            log_low, log_high = math.log(self.params['crystallite_size_range'][0]), math.log(self.params['crystallite_size_range'][1])
            crystallite_size = torch.exp(
                torch.empty(batch_size, 1, device=self.device).uniform_(log_low, log_high)
            )      #(B, 1)
        else:
            crystallite_size = torch.empty(batch_size, 1, device=self.device).uniform_(*self.params['crystallite_size_range'])      #(B, 1)
        microstrain = torch.empty(batch_size, 1, device=self.device).uniform_(*self.params['microstrain_range'])      #(B, 1)
        eta = torch.empty(batch_size, 1, device=self.device).uniform_(*self.params['pseudo_voigt_eta_range'])      #(B, 1)

        return {
            'temperature': temperature,
            'u_iso': u_iso,
            'texture_param': texture_param,
            'preferred_orientations': preferred_orientations,
            'uniform_shift': uniform_shift,
            'sample_displacement': sample_displacement,
            'crystallite_size': crystallite_size,
            'microstrain': microstrain,
            'eta': eta,
        }

    def _discrete_to_continuous(self, apply_all_effects, x, y, hkls, d_hkls, batch_size, effects=None):
        """
        Converting discrete peak lists in (b, n_peaks) to continuous patterns (b, num_points)
        """

        two_theta_grid_batch = self.two_theta_grid.unsqueeze(0).expand(batch_size, -1)      # (B, num_points)

        if apply_all_effects:
            if effects is None:
                effects = self._sample_physical_effects(batch_size)

            # Apply temperature effects
            dw_factor = self._calculate_debye_waller_factor(d_hkls, effects['temperature'], effects['u_iso'])     #(B, n_peaks)
            y *= dw_factor      #(B, n_peaks)

            # Apply texture (March-Dollase) correction
            y = self._apply_march_dollase_correction(
                y, hkls, effects['texture_param'], effects['preferred_orientations']
            )

            # Apply peak position shifts
            x += effects['uniform_shift']
            x += self._calculate_sample_displacement_shift(x, effects['sample_displacement'])

            # Calculate total broadening
            crystallite_size = effects['crystallite_size']
            microstrain = effects['microstrain']
            inst_fwhm = self._calculate_instrumental_broadening(x)              #(B, n_peaks)
            size_fwhm = self._calculate_size_broadening(x, crystallite_size)    #(B, n_peaks)
            strain_fwhm = self._calculate_strain_broadening(x, microstrain)     #(B, n_peaks)

            # Combine broadening effects (add in quadrature)
            total_fwhm = torch.sqrt(inst_fwhm**2 + size_fwhm**2 + strain_fwhm**2)      #(B, n_peaks)

            # Generate pseudo-Voigt peak
            peak_profile = self._pseudo_voigt_profile(two_theta_grid_batch, x, total_fwhm, effects['eta'])     # (B, n_peaks, num_points) the peak profile of each discrete peak

            # Sum peaks profiles to form continuous pattern
            intensity_grid = (y.unsqueeze(2) * peak_profile).sum(dim=1)     # (B, num_points) summing over all peaks weighed by their intensities y
        else:
            # Simple Gaussian peaks for comparison
            sigma = 0.1  # Fixed width
            grid_expanded = two_theta_grid_batch.unsqueeze(1)      # (B, 1, N)
            x_expanded = x.unsqueeze(2)                         # (B, P, 1)
            gaussian = torch.exp(-0.5 * ((grid_expanded - x_expanded) / sigma) ** 2)        # (B, P, N)
            weighted = y.unsqueeze(2) * gaussian     # (B, P, N)
            intensity_grid = weighted.sum(dim=1)     # (B, N)

        return intensity_grid

    def generate_multiple_patterns(self, structure, num_patterns: int,
                                 apply_all_effects: bool = True) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate multiple realistic XRD patterns for one structure, each with
        independently randomized artifact parameters.

        Args:
            structure: pymatgen Structure to generate patterns for
            num_patterns: Number of patterns to generate
            apply_all_effects: Whether to apply all effects simultaneously

        Returns:
            List of (2θ, intensity) tuples, each a flat 1-D numpy array
        """
        peak_list = self.get_peak_list(structure)
        patterns = []
        for i in range(num_patterns):
            result = self.generate_realistic_pattern(peak_list, 1, apply_all_effects=apply_all_effects)
            two_theta, intensity = result[0], result[1]
            if hasattr(two_theta, "detach"):
                two_theta = two_theta.detach().cpu().numpy()
            if hasattr(intensity, "detach"):
                intensity = intensity.detach().cpu().numpy()
            two_theta = np.asarray(two_theta).reshape(-1)
            intensity = np.asarray(intensity).reshape(-1)
            patterns.append((two_theta, intensity))

        return patterns

    def get_pattern_parameters(self) -> Dict:
        """
        Get the current parameter ranges used for pattern generation.

        Returns:
            Dictionary of parameter ranges
        """
        return self.params.copy()

    def set_parameter_range(self, param_name: str, param_range: Tuple[float, float]):
        """
        Set a specific parameter range.

        Args:
            param_name: Name of parameter to modify
            param_range: (min, max) tuple for parameter range
        """
        if param_name in self.params:
            self.params[param_name] = param_range
        else:
            raise ValueError(f"Unknown parameter: {param_name}")

    def enable_impurity_peaks(self, enable: bool = True,
                            num_peaks_range: Tuple[int, int] = (1, 10),
                            intensity_range: Tuple[float, float] = (0.5, 70.0),
                            width_range: Tuple[float, float] = (0.05, 0.3),
                            eta_range: Tuple[float, float] = (0.2, 0.9)):
        """
        Enable or disable impurity peaks in generated patterns.

        Args:
            enable: Whether to include impurity peaks
            num_peaks_range: Range for number of impurity peaks to generate
            intensity_range: Range for impurity peak intensities (% of max intensity)
            width_range: Range for impurity peak widths (FWHM in degrees)
            eta_range: Range for pseudo-Voigt mixing parameter
        """
        self.params['impurity_enable'] = enable
        if enable:
            self.params['impurity_num_peaks_range'] = num_peaks_range
            self.params['impurity_intensity_range'] = intensity_range
            self.params['impurity_width_range'] = width_range
            self.params['impurity_eta_range'] = eta_range

    def get_impurity_settings(self) -> Dict:
        """
        Get current impurity peak settings.

        Returns:
            Dictionary with current impurity peak parameters
        """
        return {
            'enabled': self.params['impurity_enable'],
            'num_peaks_range': self.params['impurity_num_peaks_range'],
            'intensity_range': self.params['impurity_intensity_range'],
            'width_range': self.params['impurity_width_range'],
            'eta_range': self.params['impurity_eta_range']
        }
