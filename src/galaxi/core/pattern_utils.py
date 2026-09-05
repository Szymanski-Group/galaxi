"""
Pattern utility functions for XRD data processing and resampling.
"""

import json
import re
from pathlib import Path
from unicodedata import name

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from scipy import interpolate as ip
from scipy.ndimage import gaussian_filter1d, grey_opening, minimum_filter1d
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import savgol_filter
from galaxi.core.config import ModelConfig
import matplotlib.pyplot as plt


def convert_two_theta_to_q(two_theta: np.ndarray, wavelength: float = 1.5405929) -> np.ndarray:
    """
    Convert 2θ angles to Q-space values.

    Args:
        two_theta: Array of 2θ angles in degrees
        wavelength: X-ray wavelength in Angstroms (default: Cu Kα1)

    Returns:
        Array of Q values in Å^-1
    """
    theta_rad = np.radians(two_theta / 2.0)
    q = (4.0 * np.pi / wavelength) * np.sin(theta_rad)
    return q


def convert_q_to_two_theta(q: np.ndarray, wavelength: float = 1.5405929) -> np.ndarray:
    """
    Convert Q-space values to 2θ angles.

    Args:
        q: Array of Q values in Å^-1
        wavelength: X-ray wavelength in Angstroms (default: Cu Kα1)

    Returns:
        Array of 2θ angles in degrees
    """
    sin_theta = (q * wavelength) / (4.0 * np.pi)
    sin_theta = np.clip(sin_theta, 0, 1)  # Ensure valid sine values
    theta_rad = np.arcsin(sin_theta)
    two_theta = 2.0 * np.degrees(theta_rad)
    return two_theta


def convert_pattern_to_q_space(two_theta: np.ndarray, intensity: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert XRD pattern from 2θ space to Q-space.

    Args:
        two_theta: Array of 2θ angles in degrees
        intensity: Array of intensity values
        wavelength: X-ray wavelength in Angstroms (default: Cu Kα1)

    Returns:
        Tuple of (q, intensity) arrays
    """
    wavelength: float = 1.5405929  # Cu Kα1 wavelength in Angstroms
    q = convert_two_theta_to_q(two_theta, wavelength)
    return q, intensity


def resample_pattern(pattern_data: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
                    target_num_points: int,
                    source_min_angle: float = 10.0,
                    source_max_angle: float = 80.0,
                    target_min_angle: Optional[float] = None,
                    target_max_angle: Optional[float] = None) -> np.ndarray:
    """
    Resample XRD pattern to target number of points.

    Args:
        pattern_data: Either 1D intensity array or tuple of (2θ, intensity)
        target_num_points: Target number of points
        source_min_angle: Minimum angle of source pattern (if pattern_data is 1D)
        source_max_angle: Maximum angle of source pattern (if pattern_data is 1D)
        target_min_angle: Target minimum angle (if None, uses source_min_angle)
        target_max_angle: Target maximum angle (if None, uses source_max_angle)

    Returns:
        Resampled intensity array with target_num_points points
    """
    if target_min_angle is None:
        target_min_angle = source_min_angle
    if target_max_angle is None:
        target_max_angle = source_max_angle

    if isinstance(pattern_data, tuple):
        two_theta, intensity = pattern_data
    else:
        # Pattern data is 1D intensity array
        intensity = pattern_data
        two_theta = np.linspace(source_min_angle, source_max_angle, len(intensity))

    # Create interpolation function
    f = ip.CubicSpline(two_theta, intensity)

    # Create target 2θ range
    target_two_theta = np.linspace(target_min_angle, target_max_angle, target_num_points)

    # Interpolate to target points
    resampled_intensity = f(target_two_theta)

    return resampled_intensity


def get_model_metadata(model_path: str) -> dict:
    """
    Extract metadata from a trained model file.

    Args:
        model_path: Path to the model file

    Returns:
        Dictionary containing model metadata including num_points
    """
    import torch
    import os

    metadata = {
        'num_points': 4501,  # Default fallback
        'min_angle': 10.0,
        'max_angle': 80.0,
        'input_size': 4501
    }

    if not os.path.exists(model_path):
        return metadata

    try:
        # Try to load model and extract metadata
        device = torch.device('cpu')  # Load on CPU for metadata extraction
        checkpoint = torch.load(model_path, map_location=device)

        # Check if metadata is stored in the checkpoint
        if isinstance(checkpoint, dict):
            if 'metadata' in checkpoint:
                metadata.update(checkpoint['metadata'])
            elif 'config' in checkpoint:
                config = checkpoint['config']
                if isinstance(config, dict):
                    metadata['num_points'] = config.get('num_points', 4501)
                    metadata['min_angle'] = config.get('min_angle', 10.0)
                    metadata['max_angle'] = config.get('max_angle', 80.0)
                    metadata['input_size'] = config.get('input_size', 4501)

        # Try to infer input size from model architecture
        if 'input_size' not in metadata or metadata['input_size'] == 4501:
            model = checkpoint if not isinstance(checkpoint, dict) else checkpoint.get('model', checkpoint)

            # Try to get input size from the first layer
            if hasattr(model, 'state_dict'):
                state_dict = model.state_dict()
            elif isinstance(model, dict):
                state_dict = model
            else:
                state_dict = {}

            # Look for conv1d input size or linear layer input size
            for name, param in state_dict.items():
                if 'conv1' in name and 'weight' in name:
                    # Conv1d weight shape: (out_channels, in_channels, kernel_size)
                    if param.dim() == 3:
                        continue  # This is the convolution kernel, not input size
                elif 'fc1' in name and 'weight' in name:
                    # Linear layer weight shape: (out_features, in_features)
                    if param.dim() == 2:
                        # This might be after convolution, so not directly the input size
                        continue

    except Exception as e:
        print(f"Warning: Could not extract metadata from {model_path}: {e}")

    return metadata


def save_model_with_metadata(model, model_path: str, metadata: dict):
    """
    Save model with metadata for later inference configuration.

    Args:
        model: The model to save
        model_path: Path to save the model
        metadata: Metadata dictionary to include
    """
    import torch

    # Create checkpoint with model and metadata
    checkpoint = {
        'model': model,
        'metadata': metadata
    }

    torch.save(checkpoint, model_path)


def normalize_pattern(pattern: np.ndarray, method: str = 'max') -> np.ndarray:
    """
    Normalize XRD pattern.

    Args:
        pattern: Input pattern array
        method: Normalization method ('max', 'area', 'std')

    Returns:
        Normalized pattern array
    """
    pattern = np.array(pattern)

    # Handle negative values by shifting to zero baseline
    if np.min(pattern) < 0:
        pattern = pattern - np.min(pattern)

    if method == 'max':
        # Normalize to 0-100 range
        if np.max(pattern) > 0:
            pattern = 100 * pattern / np.max(pattern)
    elif method == 'area':
        # Normalize by area under curve
        area = np.trapz(pattern)
        if area > 0:
            pattern = pattern / area
    elif method == 'std':
        # Z-score normalization
        mean = np.mean(pattern)
        std = np.std(pattern)
        if std > 0:
            pattern = (pattern - mean) / std

    return pattern

def preprocess_xrd_pattern(pattern: np.ndarray, mask: np.ndarray, config: ModelConfig) -> np.ndarray:
    # 1. SETUP
    y = np.asarray(pattern, dtype=np.float64).flatten()
    m = np.asarray(mask, dtype=bool).flatten()
    n = len(y)

    if mask is None or not np.any(mask):
        m = np.ones(n, dtype=bool)
    else:
        m = np.asarray(mask, dtype=bool).flatten()

    snip_iter_local = config.snip_iter
    snip_iter_global = config.snip_iter * 5
    smoothing_window_length = config.smoothing_window_length
    noise_sensitivity = config.noise_sensitivity
    gate_sharpness = config.gate_sharpness
    magnification_power = config.magnification_power

    valid_coords = np.where(m)[0]
    if len(valid_coords) < config.smoothing_window_length:
        return np.zeros_like(y, dtype=np.float32)

    m_start, m_end = valid_coords[0], valid_coords[-1]

    # 2. ISOLATE VALID REGION
    # We only smooth what is actually inside the mask
    y_valid = y[m_start : m_end + 1]

    # 3. SMOOTHING
    y_smooth_valid = savgol_filter(y_valid, window_length=smoothing_window_length, polyorder=2, mode='mirror')
    y_smooth = np.zeros_like(y)
    y_smooth[m_start : m_end + 1] = y_smooth_valid

    # 4. BACKGROUND (Global)
    global_bg = snip(y_smooth, m, iterations=snip_iter_global)
    y_leveled = np.maximum(y - global_bg, 1e-6)

    # 5. SECOND PASS (Local Background)
    y_leveled_valid = y_leveled[m_start : m_end + 1]
    y_smooth_leveled_valid = savgol_filter(y_leveled_valid, window_length=smoothing_window_length, polyorder=2, mode='mirror')

    y_smooth_leveled = np.zeros_like(y)
    y_smooth_leveled[m_start : m_end + 1] = y_smooth_leveled_valid

    bg_local = snip(y_smooth_leveled, m, iterations=snip_iter_local)
    raw_residuals = y_smooth_leveled - bg_local

    # 3. MASK-AWARE NOISE CALCULATION
    boundary_mask = np.zeros(n, dtype=bool)
    valid_coords = np.where(m)[0]

    if len(valid_coords) > 2 * snip_iter_local:
        start_idx = valid_coords[0] + snip_iter_local
        end_idx = valid_coords[-1] - snip_iter_local
        boundary_mask[start_idx:end_idx] = True
    else:
        boundary_mask[:] = True

    # Combine: Must be inside boundaries AND allowed by input mask
    final_stats_mask = boundary_mask & m
    noise_region = raw_residuals[final_stats_mask]

    if noise_region.size > 0:
        median_val = np.median(noise_region)
        mad = np.median(np.abs(noise_region - median_val))
        estimated_sigma_abs = 1.4826 * mad
        noise_floor = np.percentile(noise_region, 10)
    else:
        estimated_sigma_abs = 0.05
        noise_floor = 0

    # 4. NORMALIZATION & MAGNIFICATION
    y_clean = np.maximum(raw_residuals - noise_floor, 0)

    valid_data = y_clean[m]
    max_intensity = np.max(valid_data) if valid_data.size > 0 else 0

    if max_intensity > 1e-9:
        y_final = y_clean / max_intensity
        normalized_sigma = estimated_sigma_abs / max_intensity
        dynamic_sigma = np.clip(normalized_sigma * noise_sensitivity, 0.005, 0.1)
    else:
        y_final = np.zeros_like(y_clean)
        dynamic_sigma = 0.02

    # Magnification
    gain_map = 1 - np.exp(-(y_final / dynamic_sigma) ** gate_sharpness)
    y_final = (y_final * gain_map) ** magnification_power

    # 5. FINAL CLEANUP
    y_final[~m] = 0 # Explicitly zero out masked regions

    return y_final.astype(np.float32)

def snip(y, mask, iterations=24):
    """Statistics-sensitive Non-linear Iterative Peak-clipping background estimate.

    Iteratively clips each point down to the average of its two neighbors at
    distance p, with p shrinking from `iterations` to 1 -- points under a
    peak get pulled toward the smooth baseline while the baseline itself is
    left alone, since a true background point already sits below that
    average. Edges are linearly extrapolated first (step 1) so the clipping
    window has real values to compare against near the mask boundary,
    rather than clipping against the mask's hard edge.
    """
    z_orig = np.copy(y)
    n = len(y)
    m = np.asarray(mask, dtype=bool).flatten()
    valid_indices = np.where(m)[0]

    if len(valid_indices) < 10: # Need enough points for a stable slope
        return z_orig

    m_start = valid_indices[0]
    m_end = valid_indices[-1]

    # 1. Padding with Linear Extrapolation to keep p constant at edges
    pad = iterations

    def get_slope(idx, direction):
        if direction == 'left':
            pts = y[idx : idx + 50]
        else:
            pts = y[idx - 50 : idx]
        return np.mean(np.diff(pts))

    slope_left = get_slope(m_start, 'left')
    slope_right = get_slope(m_end, 'right')

    left_wing = y[m_start] + (np.arange(-pad, 0) * slope_left)
    right_wing = y[m_end] + (np.arange(1, pad + 1) * slope_right)

    z_padded = np.concatenate([left_wing, y[m_start : m_end + 1], right_wing])

    # 2. Standard SNIP on padded data
    p_start = pad
    p_end = len(z_padded) - pad - 1

    for p in range(iterations, 0, -1):
        start = p_start
        end = p_end + 1

        target = z_padded[start:end]
        left = z_padded[start - p : end - p]
        right = z_padded[start + p : end + p]

        avg = (left + right) / 2.0
        z_padded[start:end] = np.minimum(target, avg)

    # 3. Reconstruct final array
    final_bg = np.copy(y)
    final_bg[m_start : m_end + 1] = z_padded[pad : pad + (m_end - m_start + 1)]

    # For points outside the mask, use the edge background value (or 0)
    final_bg[:m_start] = 0
    final_bg[m_end + 1:] = 0

    return final_bg

def calculate_pattern_f1_metrics(
    true_phases: List[str],
    predicted_groups: List[List[str]],
    grouping_catalog: List[List[str]]
) -> Dict:

    def clean_name(x):
        if not isinstance(x, str):
            return x

        # Remove any _mp suffix
        x = x.replace("_mp", "")

        # Split by underscore to isolate formula, space group, and variant
        parts = x.split('_')

        # Reconstruct as Formula_SG, intentionally dropping variant numbers like _1, _2
        if len(parts) >= 2 and parts[1].isdigit():
            return f"{parts[0]}_{parts[1]}"
        return parts[0]

    phase_to_group = {}
    for group in grouping_catalog:
        group_set = frozenset(
            clean_name(x)
            for x in group
        )
        for phase in group_set:
            phase_to_group[phase] = group_set

    predicted_units = set()
    for group in predicted_groups:
        group_set = frozenset(
            clean_name(x)
            for x in group
        )
        if group_set:
            predicted_units.add(group_set)

    true_units = set()
    for phase in true_phases:
        phase = clean_name(phase)
        if phase in phase_to_group:
            true_units.add(
                phase_to_group[phase]
            )
        else:
            true_units.add(
                frozenset([phase])
            )

    tp = len(
        predicted_units & true_units
    )
    fp = len(
        predicted_units - true_units
    )
    fn = len(
        true_units - predicted_units
    )
    precision = (
        tp / (tp + fp)
        if tp + fp > 0 else 0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0 else 0
    )

    f1 = (
        2 * precision * recall /
        (precision + recall)
        if precision + recall > 0
        else 0
    )

    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "true_formulas": [
            sorted(list(x))
            for x in true_units
        ],
        "predicted_formulas": [
            sorted(list(x))
            for x in predicted_units
        ]
    }

def generate_experimental_summary_csv(exp_results: Dict, csv_path: str) -> None:
    """
    Generate a simplified CSV summary of experimental evaluation results,
    including group-level predictions if available.
    """
    import csv

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Modified header
        writer.writerow([
            'File name', 'True phases', 'CNN Predicted phases', 'CNN Grouped phases',
            'DARA Predicted phases', 'DARA Grouped phases',
            'F1-Score', 'Precision', 'Recall', 'Time'
        ])

        for pattern_name, pattern_results in exp_results.items():
            if isinstance(pattern_results, dict) and 'phase_metrics' in pattern_results:
                phase_metrics = pattern_results['phase_metrics']
                filename = pattern_results.get('filename', 'None')

                # True phases
                true_formulas = phase_metrics.get('true_formulas', [])

                if true_formulas:
                    formatted_true_groups = []

                    for g in true_formulas:
                        if isinstance(g, list):
                            formatted_true_groups.append(
                                f"[{', '.join(g)}]"
                            )
                        else:
                            formatted_true_groups.append(
                                f"[{g}]"
                            )

                    true_phases = ", ".join(formatted_true_groups)
                else:
                    true_phases = "None"

                # CNN predicted phases
                cnn_predicted_list = [
                    phase for phase, is_pred in pattern_results.get('cnn_predicted_labels', {}).items()
                    if is_pred == 1
                ]
                cnn_predicted_phases = '-'.join(cnn_predicted_list) if cnn_predicted_list else 'None'

                # Fetch DARA phases
                dara_predicted_dict = pattern_results.get('dara_predicted_labels', {})
                dara_predicted_formulas = [k for k, v in dara_predicted_dict.items() if v == 1]
                dara_predicted_phases = '-'.join(dara_predicted_formulas) if dara_predicted_formulas else 'None'

                # Grouped phases formatting
                cnn_grouped_phases_list = pattern_results.get('cnn_grouped_phases', [])
                dara_grouped_phases_list = pattern_results.get('dara_grouped_phases', [])
                if cnn_grouped_phases_list:
                    formatted_groups = []
                    for g in cnn_grouped_phases_list:
                        if isinstance(g, list):
                            formatted_groups.append(f"[{', '.join(g)}]")
                        else:
                            formatted_groups.append(f"[{g}]")
                    cnn_grouped_phases_str = ", ".join(formatted_groups)
                else:
                    cnn_grouped_phases_str = 'None'

                if dara_grouped_phases_list:
                    formatted_groups = []
                    for g in dara_grouped_phases_list:
                        if isinstance(g, list):
                            formatted_groups.append(f"[{', '.join(g)}]")
                        else:
                            formatted_groups.append(f"[{g}]")
                    dara_grouped_phases_str = ", ".join(formatted_groups)
                else:
                    dara_grouped_phases_str = 'None'

                precision = phase_metrics.get('precision', 0.0)
                recall = phase_metrics.get('recall', 0.0)
                f1_score = phase_metrics.get('f1_score', 0.0)
                time_taken = pattern_results.get('time', 0.0)

                writer.writerow([
                    filename, true_phases, cnn_predicted_phases, cnn_grouped_phases_str,
                    dara_predicted_phases, dara_grouped_phases_str,
                    f"{f1_score:.3f}", f"{precision:.3f}", f"{recall:.3f}", f"{time_taken:.2f}"
                ])

    print(f"✓ CSV summary saved to {csv_path}")

def regularize_input(file_name, pattern: np.ndarray,
                     min_angle: float, max_angle: float, target_length: int,
                     use_mask: bool = False, model_config: ModelConfig = None) -> np.ndarray:

    """
    1. Truncate pattern to the model angular range
    2. Pad pattern to the angular range with zero intensity, plus a mask row if use_mask
    3. Interpolate to standard grid

    Args:
        file_name: filename of the pattern file
        pattern: [[2θ], [intensity]] pattern
        target_length: Expected length for the model
        use_mask: whether to use mask for missing data points. If enabled, then pattern would become (2, N) with the second row being mask

    Returns:
        Preprocessed pattern with correct length (no angle data)
    """
    two_theta = pattern[:, 0]
    intensity = pattern[:, 1]

    pattern_min, pattern_max = np.min(two_theta), np.max(two_theta)
    model_min, model_max = min_angle, max_angle

    standard_two_theta = np.linspace(model_min, model_max, target_length)

    # Truncate if data extends beyond model angular range. Applied with or without mask!
    if pattern_min < model_min or pattern_max > model_max:
        valid_index = np.where((two_theta >= model_min) & (two_theta <= model_max))[0]
        if len(valid_index) == 0:
            print(f"Warning: {file_name} has no data in required 2θ range.")
            return np.zeros((1, target_length))
        two_theta = two_theta[valid_index]
        intensity = intensity[valid_index]
        pattern_min, pattern_max = np.min(two_theta), np.max(two_theta)

    if use_mask:
        # Pad pattern to model angular range if pattern angular range is smaller, keep intensity outside pattern angular range as 0
        if pattern_min > model_min:
            two_theta = np.concatenate([[model_min, pattern_min], two_theta])
            intensity = np.concatenate([[0.0, 0.0], intensity])

        # Pad missing right region with zeros
        if pattern_max < model_max:
            two_theta = np.concatenate([two_theta, [pattern_max, model_max]])
            intensity = np.concatenate([intensity, [0.0, 0.0]])

        # Interpolate to standard grid
        standard_intensity = np.interp(standard_two_theta, two_theta, intensity)

        # Create mask
        mask = np.ones_like(standard_two_theta)
        mask[(standard_two_theta < pattern_min) | (standard_two_theta > pattern_max)] = 0

        # preprocessing
        standard_intensity = preprocess_xrd_pattern(standard_intensity, mask, model_config)
        # Stack intensity and mask
        standard_intensity = np.stack((standard_intensity, mask), axis=0)
    else:
        # Pad pattern to model angular range if pattern angular range is smaller, keep intensity outside pattern angular range as 0
        if pattern_min > model_min:
            two_theta = np.concatenate([[model_min, pattern_min], two_theta])
            intensity = np.concatenate([[0.0, 0.0], intensity])

        # Pad missing right region with zeros
        if pattern_max < model_max:
            two_theta = np.concatenate([two_theta, [pattern_max, model_max]])
            intensity = np.concatenate([intensity, [0.0, 0.0]])

        # Interpolate to standard grid
        standard_intensity = np.interp(standard_two_theta, two_theta, intensity)

        # preprocessing
        standard_intensity = preprocess_xrd_pattern(standard_intensity, None, model_config)
        standard_intensity = standard_intensity.reshape(1, -1)

    return standard_intensity   # (1, N) or (2, N) if use_mask

def is_structure_valid(structure) -> bool:

    """
    Check if a structure is valid for XRD pattern generation.

    Filters out structures with:
    - Unreasonably close atomic distances
    - Too many/few atoms
    - Unreasonable unit cell parameters
    - Other problematic features
    """
    try:
        # Basic structure checks
        if len(structure) == 0:
            return False

        # Check for reasonable number of atoms (avoid huge structures)
        if len(structure) > 500:
            return False

        # Check unit cell parameters
        lattice = structure.lattice

        # Check for reasonable lattice parameters (0.5 to 100 Angstroms)
        params = lattice.abc
        if any(p < 0.5 or p > 100.0 for p in params):
            return False

        # Check for reasonable angles (10 to 170 degrees)
        angles = lattice.angles
        if any(a < 10.0 or a > 170.0 for a in angles):
            return False

        # Check for reasonable volume (avoid near-zero or huge volumes)
        volume = lattice.volume
        if volume < 10.0 or volume > 50000.0:
            return False

        # Check minimum interatomic distances
        min_distance = get_minimum_distance(structure)
        if min_distance < 0.5:  # Atoms closer than 0.5 Angstrom are unrealistic
            return False

        # Check for reasonable density (avoid structures that are too dense or sparse)
        density = structure.density
        if density < 0.1 or density > 50.0:  # g/cm³
            return False

        # Additional checks for common problematic cases

        # Check for structures with only hydrogen atoms
        if all(site.specie.symbol == 'H' for site in structure):
            return False

        # Check for structures with unreasonable chemical compositions
        # (e.g., 99% of one element mixed with trace amounts)
        composition = structure.composition
        if len(composition) > 1:
            max_fraction = max(composition.get_atomic_fraction(el) for el in composition.elements)
            if max_fraction > 0.99:
                return False

        return True

    except Exception:
        # If any check fails due to structure issues, reject the structure
        return False

def get_minimum_distance(structure) -> float:
    """Get minimum interatomic distance in the structure."""
    try:
        # Use a reasonable cutoff to avoid checking all pairs in large structures
        max_sites_to_check = min(50, len(structure))
        min_dist = float('inf')

        for i in range(max_sites_to_check):
            for j in range(i + 1, max_sites_to_check):
                # Get distance between sites
                dist = structure[i].distance(structure[j])
                min_dist = min(min_dist, dist)

                # Early exit if we find a clearly problematic distance
                if min_dist < 0.5:
                    break

            if min_dist < 0.5:
                break

        return min_dist

    except Exception:
        # If distance calculation fails, assume problematic structure
        return 0.0

from collections import defaultdict
from typing import List, Set, Dict

def calculate_peak_similarity(data_a: Dict, data_b: Dict, theta_tol: float = 0.2, sigma: float = 0.1) -> float:
    theta_a, int_a = np.array(data_a.get("theta", [])), np.array(data_a.get("intensity", []))
    theta_b, int_b = np.array(data_b.get("theta", [])), np.array(data_b.get("intensity", []))

    if len(theta_a) == 0 or len(theta_b) == 0:
        return 0.0

    # L2 Normalization (Unit Vectors) makes the cosine similarity mathematically sound
    int_a = int_a / np.sqrt(np.sum(int_a**2) + 1e-9)
    int_b = int_b / np.sqrt(np.sum(int_b**2) + 1e-9)

    # 1. Compute pairwise absolute distance matrix (Shape: len(A) x len(B))
    # Using broadcasting: theta_a[:, None] is column, theta_b[None, :] is row
    dist_matrix = np.abs(theta_a[:, None] - theta_b[None, :])

    # 2. Calculate Gaussian weights for all pairs
    weight_matrix = np.exp(-0.5 * (dist_matrix / sigma) ** 2)

    # Zero out weights for peaks that exceed our hard tolerance cutoff
    weight_matrix[dist_matrix > theta_tol] = 0.0

    # 3. Compute intensity compatibility matrix
    intensity_matrix = int_a[:, None] * int_b[None, :]

    # 4. Total overlap score
    # This accounts for all cross-peak interactions seamlessly
    total_score = np.sum(intensity_matrix * weight_matrix)

    # Clip to 1.0 due to potential minor floating point inaccuracies
    return float(np.clip(total_score, 0.0, 1.0))

# Single source of truth for the grouping threshold's default value --
# referenced wherever a caller needs to resolve an unset/None threshold
# instead of hardcoding a second literal (see model_evaluator.py and
# streamlined_workflow.py).
DEFAULT_GROUP_SIMILARITY_THRESHOLD = 0.80


def _get_chem_space_and_sg(name: str):
    clean_name = name.replace("_mp", "")
    parts = clean_name.split("_")

    if (
        len(parts) >= 3
        and parts[-1].isdigit()
        and parts[-2].isdigit()
    ):
        formula_part = "_".join(parts[:-2])
        sg_part = parts[-2]

    elif (
        len(parts) >= 2
        and parts[-1].isdigit()
    ):
        formula_part = "_".join(parts[:-1])
        sg_part = parts[-1]

    else:
        formula_part = clean_name
        # Unique per-name fallback: two differently-unparseable names must
        # never spuriously satisfy sg_i == sg_j and merge just because both
        # failed to parse the same way.
        sg_part = f"UNKNOWN::{name}"

    elements = frozenset(
        re.findall(r"[A-Z][a-z]?", formula_part)
    )

    return elements, sg_part


def build_phase_groups_from_peaks(
    models_dir,
    group_similarity_threshold: Optional[float] = None,
    enable_chemistry_prefilter: bool = False,
    grouping_backend: str = "gaussian_cosine",
) -> List[List[str]]:
    """
    Loads peak_list.json for all models and groups highly similar phases.

    Args:
        group_similarity_threshold: minimum similarity to merge two phases.
            ``None`` resolves to `DEFAULT_GROUP_SIMILARITY_THRESHOLD`.
        enable_chemistry_prefilter: only compare phases whose parsed
            element-set and space-group string match. Defaults to False:
            grouping should reflect XRD-indistinguishability (profile
            similarity) only, not composition -- the chemistry gate was not
            stoichiometry-aware and blocked legitimate XRD-indistinguishable
            merges just because two formula strings parsed differently.
        grouping_backend: "gaussian_cosine" (default) imports the shared
            `peak_pattern_similarity`/`cluster_phases` from
            `dara.search.phase_grouping` (DARA's own editable-installed
            grouping module) and clusters with bounded-linkage
            AgglomerativeClustering -- this is the SAME code path DARA's own
            internal search grouping uses (see `DaraConfig.grouping_metric`),
            so CNN-side and DARA-side grouping are identical algorithms, not
            just both "profile-similarity-based". "legacy_bfs" (this
            module's own `calculate_peak_similarity` plus unbounded
            BFS/connected-components chaining) is kept only for historical
            comparison.
    """

    import json
    import re
    from collections import defaultdict

    if group_similarity_threshold is None:
        group_similarity_threshold = DEFAULT_GROUP_SIMILARITY_THRESHOLD

    phase_peaks = {}

    for phase_dir in models_dir.glob("models_*"):
        phase_name = phase_dir.name.replace("models_", "")
        peak_files = list(phase_dir.rglob("peak_list.json"))

        if not peak_files:
            continue
        try:
            with open(peak_files[0], "r") as f:
                phase_peaks[phase_name] = json.load(f)
        except Exception as e:
            print(
                f"Warning: Could not load peak list "
                f"for {phase_name}: {e}"
            )

    phases = list(phase_peaks.keys())

    if grouping_backend == "gaussian_cosine":
        from dara.search.phase_grouping import cluster_phases, peak_pattern_similarity

        n = len(phases)
        peak_arrays = []
        for name in phases:
            data = phase_peaks[name]
            theta = np.asarray(data.get("theta", []), dtype=float)
            intensity = np.asarray(data.get("intensity", []), dtype=float)
            peak_arrays.append(
                np.column_stack([theta, intensity]) if len(theta) else np.empty((0, 2))
            )

        chem_sg = [_get_chem_space_and_sg(name) for name in phases]

        distance_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                if enable_chemistry_prefilter and chem_sg[i] != chem_sg[j]:
                    d = 1.0
                else:
                    sim = peak_pattern_similarity(peak_arrays[i], peak_arrays[j])
                    d = 1.0 - sim
                distance_matrix[i, j] = d
                distance_matrix[j, i] = d

        if n <= 1:
            grouped_phases = [[p] for p in phases]
        else:
            labels = cluster_phases(distance_matrix, 1.0 - group_similarity_threshold)
            clusters = defaultdict(list)
            for name, label in zip(phases, labels):
                clusters[label].append(name)
            grouped_phases = [sorted(members) for members in clusters.values()]

        print("\nPhase groups found:")
        for group in grouped_phases:
            if len(group) > 1:
                print(group)
        print(f"Total groups: {len(grouped_phases)}")
        return grouped_phases

    elif grouping_backend != "legacy_bfs":
        raise ValueError(f"Unknown grouping_backend: {grouping_backend!r}")

    adj = defaultdict(list)

    for i in range(len(phases)):

        phase_i = phases[i]

        space_i, sg_i = _get_chem_space_and_sg(
            phase_i
        )

        for j in range(i + 1, len(phases)):
            phase_j = phases[j]
            # Only compare phases with same chemistry and same space group
            if enable_chemistry_prefilter:
                space_j, sg_j = _get_chem_space_and_sg(
                    phase_j
                )
                if space_i != space_j:
                    continue
                if sg_i != sg_j:
                    continue
            sim = calculate_peak_similarity(
                phase_peaks[phase_i],
                phase_peaks[phase_j]
            )
            if sim >= group_similarity_threshold:
                adj[phase_i].append(phase_j)
                adj[phase_j].append(phase_i)

    visited = set()
    grouped_phases = []

    for phase in phases:
        if phase in visited:
            continue
        component = []
        queue = [phase]
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            component.append(current)
            queue.extend(adj[current])
        grouped_phases.append(
            sorted(component)
        )
    print("\nPhase groups found:")

    for group in grouped_phases:
        if len(group) > 1:
            print(group)
    print(
        f"Total groups: {len(grouped_phases)}"
    )
    return grouped_phases

def group_phases_func(
    phase_list: List[str],
    grouped_phase_catalog: List[List[str]]
) -> List[List[str]]:
    """
    Convert a flat phase list into grouped representation.
    """

    phase_set = set(phase_list)

    results = []
    assigned = set()

    for group in grouped_phase_catalog:

        overlap = sorted(
            p for p in group
            if p in phase_set
        )

        if overlap:
            results.append(overlap)
            assigned.update(overlap)

    for phase in sorted(phase_set - assigned):
        results.append([phase])
    return results


def count_header_lines(path: Union[str, Path], max_check: int = 3) -> int:
    """How many leading lines of a two-column XY file fail to parse as two
    whitespace-separated floats. A hardcoded skiprows=2 silently eats real
    data rows from files with no header (e.g. some GALAXI-generated XY
    output) while still being required for genuinely 2-line-header files
    (e.g. PyD8/worked-example experimental patterns) -- detect it per-file
    instead of assuming either convention."""
    with open(path) as f:
        lines = f.readlines()
    n = 0
    for line in lines[:max_check]:
        parts = line.split()
        try:
            float(parts[0])
            float(parts[1])
            break
        except (ValueError, IndexError):
            n += 1

    return n
