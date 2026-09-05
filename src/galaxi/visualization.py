"""
Module for visualizing XRD patterns with enhanced aesthetics.

This module provides the XRDVisualizer class for creating publication-quality
visualizations of X-ray diffraction (XRD) patterns including single patterns,
multiple pattern comparisons, and file-based plotting.

Classes:
    XRDVisualizer: Main class for XRD pattern visualization

Example:
    >>> from galaxi.visualization import XRDVisualizer
    >>> import numpy as np
    >>>
    >>> visualizer = XRDVisualizer(figsize=(10, 6))
    >>> two_theta = np.linspace(10, 80, 1000)
    >>> intensity = np.random.rand(1000) * 100
    >>> fig, ax = visualizer.plot_pattern(two_theta, intensity, title="Sample XRD")
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib.ticker import MaxNLocator
import json

from .core.pattern_utils import count_header_lines


def determine_x_axis_label(x_values: np.ndarray, convert_to_q: bool = False) -> str:
    """
    Determine appropriate x-axis label based on data characteristics.

    Args:
        x_values: Array of x-axis values
        convert_to_q: Whether the data is in Q-space

    Returns:
        Appropriate x-axis label string
    """
    if convert_to_q:
        return 'Q (Å⁻¹)'
    else:
        # Check if values are in typical 2θ range
        max_val = np.max(x_values)
        if max_val > 10 and max_val < 200:  # Typical 2θ range
            return '2θ (degrees)'
        else:
            return 'Q (Å⁻¹)'


class XRDVisualizer:
    """
    Class for creating publication-quality visualizations of XRD patterns.

    This class provides methods to visualize single XRD patterns, compare multiple patterns,
    and create plots directly from data files. All plots are optimized for publication
    quality with customizable styling options.

    Attributes:
        figsize: Tuple specifying default figure dimensions (width, height) in inches
        dpi: Dots per inch for saved figure resolution
        style: Matplotlib style name for plot appearance

    Example:
        >>> visualizer = XRDVisualizer(figsize=(10, 6), dpi=300)
        >>> fig, ax = visualizer.plot_pattern(two_theta, intensity, title="My Pattern")
    """

    def __init__(
        self,
        figsize: Tuple[int, int] = (12, 8),
        dpi: int = 400,
        style: str = 'seaborn-v0_8-whitegrid'
    ) -> None:
        """
        Initialize the XRD visualizer with plotting parameters.

        Args:
            figsize: Figure dimensions (width, height) in inches. Default: (12, 8)
            dpi: Resolution for saved figures in dots per inch. Default: 400
            style: Matplotlib style for plot appearance. Default: 'seaborn-v0_8-whitegrid'

        Raises:
            ValueError: If figsize contains non-positive values or dpi is not positive
        """
        if any(dim <= 0 for dim in figsize):
            raise ValueError("Figure dimensions must be positive")
        if dpi <= 0:
            raise ValueError("DPI must be positive")

        self.figsize = figsize
        self.dpi = dpi
        self.style = style

        self._setup_plot_style()

    def _setup_plot_style(self) -> None:
        """Configure matplotlib settings for consistent plot appearance."""
        plt.style.use(self.style)
        rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
            'font.size': 20,
            'figure.dpi': 100,  # Display DPI for on-screen viewing
            'savefig.dpi': self.dpi  # High DPI for saved figures
        })

    def plot_pattern(
        self,
        x_values: np.ndarray,
        intensity: np.ndarray,
        title: Optional[str] = None,
        color: str = '#1E88E5',
        fill_alpha: float = 0.3,
        grid_alpha: float = 0.3,
        linewidth: float = 2,
        xlabel: Optional[str] = None,
        ylabel: str = 'Intensity (a.u.)',
        fontsize: int = 20,
        xmin: Optional[float] = None,
        xmax: Optional[float] = None,
        show: bool = True,
        save: bool = False,
        filename: Optional[str] = None,
        save_format: str = 'png',
        convert_to_q: bool = False
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot a single XRD pattern with publication-quality styling.

        Args:
            x_values: Array of x-axis values (2θ in degrees or Q in Å⁻¹)
            intensity: Array of corresponding intensity values
            title: Plot title. If None, no title is added
            color: Color for the line and fill area
            fill_alpha: Transparency for the fill area (0-1)
            grid_alpha: Transparency for the grid (0-1)
            linewidth: Width of the XRD pattern line
            xlabel: Label for the x-axis. If None, auto-determined from data
            ylabel: Label for the y-axis
            fontsize: Base font size for plot elements
            xmin: Minimum x-axis limit. If None, uses data minimum
            xmax: Maximum x-axis limit. If None, uses data maximum
            show: Whether to display the plot
            save: Whether to save the plot to a file
            filename: Filename for saved plot (without extension)
            save_format: File format for saved plot (png, pdf, svg, etc.)
            convert_to_q: Whether the data is in Q-space (for auto x-axis labeling)

        Returns:
            Tuple of (figure, axes) objects for further customization

        Raises:
            ValueError: If input arrays have different lengths or invalid parameters
        """
        self._validate_pattern_data(x_values, intensity)
        self._validate_plot_parameters(fill_alpha, grid_alpha, linewidth, fontsize)

        # Auto-determine xlabel if not provided
        if xlabel is None:
            xlabel = determine_x_axis_label(x_values, convert_to_q)

        fig, ax = self._create_pattern_plot(
            x_values, intensity, color, fill_alpha, grid_alpha, linewidth,
            xlabel, ylabel, title, fontsize, xmin, xmax
        )

        self._handle_plot_output(fig, show, save, filename, save_format)
        return fig, ax

    def _validate_pattern_data(self, x_values: np.ndarray, intensity: np.ndarray) -> None:
        """Validate input data arrays for pattern plotting."""
        if len(x_values) != len(intensity):
            raise ValueError("x_values and intensity arrays must have the same length")
        if len(x_values) == 0:
            raise ValueError("Input arrays cannot be empty")

    def _validate_plot_parameters(
        self,
        fill_alpha: float,
        grid_alpha: float,
        linewidth: float,
        fontsize: int
    ) -> None:
        """Validate plot styling parameters."""
        if not 0 <= fill_alpha <= 1:
            raise ValueError("fill_alpha must be between 0 and 1")
        if not 0 <= grid_alpha <= 1:
            raise ValueError("grid_alpha must be between 0 and 1")
        if linewidth <= 0:
            raise ValueError("linewidth must be positive")
        if fontsize <= 0:
            raise ValueError("fontsize must be positive")

    def _create_pattern_plot(
        self,
        x_values: np.ndarray,
        intensity: np.ndarray,
        color: str,
        fill_alpha: float,
        grid_alpha: float,
        linewidth: float,
        xlabel: str,
        ylabel: str,
        title: Optional[str],
        fontsize: int,
        xmin: Optional[float],
        xmax: Optional[float]
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Create the actual plot with all styling applied."""
        fig, ax = plt.subplots(figsize=self.figsize)

        # Add small padding to y-axis minimum to prevent label overlap
        y_max = intensity.max()
        y_padding = 0.02 * y_max  # 2% padding
        y_min = -y_padding

        # Plot the pattern with fill
        ax.plot(x_values, intensity, color=color, linewidth=linewidth, zorder=3)
        ax.fill_between(x_values, y_min, intensity, color=color, alpha=fill_alpha, zorder=2)

        # Configure appearance
        ax.grid(True, alpha=grid_alpha, linestyle='--', zorder=1)
        ax.set_xlabel(xlabel, fontsize=fontsize, fontweight='bold', labelpad=12)
        ax.set_ylabel(ylabel, fontsize=fontsize, fontweight='bold', labelpad=16)

        if title:
            ax.set_title(title, fontsize=fontsize+1, fontweight='bold', pad=10)

        # Set axis limits and ticks
        ax.set_xlim(xmin or x_values.min(), xmax or x_values.max())
        ax.set_ylim(bottom=y_min, top=y_max + y_padding)

        ax.tick_params(axis='both', which='major', labelsize=fontsize-2)

        if intensity.max() > 10:
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        fig.tight_layout()
        return fig, ax

    def _create_pattern_plot_with_sticks(
        self,
        x_values: np.ndarray,
        intensity: np.ndarray,
        phase_name: str,
        xy_file_path: str,
        sticks_cif_dir: str,
        sticks_title: str,
        eval_path: str,
        color: str,
        fill_alpha: float,
        grid_alpha: float,
        linewidth: float,
        xlabel: str,
        ylabel: str,
        title: Optional[str],
        fontsize: int,
        xmin: Optional[float],
        xmax: Optional[float]
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Create plot with stick pattern below the curve."""
        if phase_name is None:
            raise ValueError("phase_name must be provided when plotting sticks")

        from pymatgen.core import Structure
        from pymatgen.analysis.diffraction.xrd import XRDCalculator

        # Load structure and simulate XRD sticks
        structure = Structure.from_file(sticks_cif_dir)
        xrd_calculator = XRDCalculator()
        pattern = xrd_calculator.get_pattern(structure, two_theta_range=(x_values.min(), x_values.max()))

        sim_two_theta = pattern.x
        sim_intensity = pattern.y

        # Normalize simulated intensity to 100
        sim_intensity = sim_intensity / max(sim_intensity) * 100

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=self.figsize, sharex=True,
            gridspec_kw={'height_ratios': [3, 1]}
        )

        # Add small padding to y-axis minimum to prevent label overlap
        y_max = intensity.max()
        y_padding = 0.02 * y_max  # 2% padding
        y_min = -y_padding

        # Plot the experimental pattern with fill
        ax1.plot(x_values, intensity, color=color, linewidth=linewidth, zorder=3)
        ax1.fill_between(x_values, y_min, intensity, color=color, alpha=fill_alpha, zorder=2)
        ax1.grid(True, alpha=grid_alpha, linestyle='--', zorder=1)
        ax1.set_ylabel(ylabel, fontsize=fontsize, fontweight='bold', labelpad=16)
        plot_title = title

        material_name = Path(xy_file_path).stem
        if eval_path:
            try:
                with open(eval_path, "r") as f:
                    eval_data = json.load(f)
                score = eval_data[phase_name][material_name]["probabilities"].get(phase_name, None)
                if score:
                    plot_title = f"{title} (score={score:.3f})"
                else:
                    print(f"[INFO] No match for {phase_name} in {eval_path}")
            except Exception as e:
                print(f"[ERROR] Could not read eval file {eval_path}: {e}")

        if plot_title:
            ax1.set_title(plot_title, fontsize=fontsize+1, fontweight="bold", pad=10)
        ax1.set_xlim(xmin or x_values.min(), xmax or x_values.max())
        ax1.set_ylim(bottom=y_min, top=y_max + y_padding)
        ax1.tick_params(axis='both', which='major', labelsize=fontsize-2)

        # Plot the simulated sticks
        ax2.vlines(sim_two_theta, ymin=0, ymax=sim_intensity, color='red', lw=1.2)
        ax2.grid(True, alpha=grid_alpha, linestyle='--', zorder=1)
        ax2.set_xlabel(xlabel, fontsize=fontsize, fontweight='bold', labelpad=12)
        ax2.set_ylabel(sticks_title, fontsize=fontsize, fontweight='bold', labelpad=16)
        ax2.tick_params(axis='both', which='major', labelsize=fontsize-2)

        return fig, ax1

    def _handle_plot_output(
        self,
        fig: plt.Figure,
        show: bool,
        save: bool,
        filename: Optional[str],
        save_format: str,
        save_dir: Union[str, Path] = None
    ) -> None:
        """Handle plot display and saving."""
        if save:
            save_name = f"{filename or 'xrd_pattern'}.{save_format}"
            if save_dir:
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir, exist_ok=True)
                save_path = Path(save_dir) / save_name
            else:
                save_path = Path.cwd() / save_name
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Figure saved as {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

    def plot_multiple_patterns(
        self,
        patterns: List[Tuple[np.ndarray, np.ndarray]],
        labels: Optional[List[str]] = None,
        colors: Optional[List[str]] = None,
        title: Optional[str] = None,
        fill_alpha: float = 0.2,
        grid_alpha: float = 0.3,
        linewidth: float = 2,
        xlabel: str = '2θ (degrees)',
        ylabel: str = 'Intensity (a.u.)',
        show_legend: bool = True,
        show: bool = True,
        save: bool = False,
        filename: Optional[str] = None,
        save_format: str = 'png'
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot multiple XRD patterns on the same graph for comparison.

        Args:
            patterns: List of (2θ, intensity) tuples for each pattern
            labels: Labels for each pattern. If None, uses "Pattern 1", "Pattern 2", etc.
            colors: Colors for each pattern. If None, uses default matplotlib colors
            title: Plot title. If None, no title is added
            fill_alpha: Transparency for fill areas (0-1)
            grid_alpha: Transparency for grid (0-1)
            linewidth: Width of pattern lines
            xlabel: Label for x-axis
            ylabel: Label for y-axis
            show_legend: Whether to display legend
            show: Whether to display the plot
            save: Whether to save the plot
            filename: Filename for saved plot (without extension)
            save_format: File format for saved plot

        Returns:
            Tuple of (figure, axes) objects

        Raises:
            ValueError: If patterns list is empty or contains invalid data
        """
        if not patterns:
            raise ValueError("Patterns list cannot be empty")

        # Validate each pattern
        for i, pattern in enumerate(patterns):
            if len(pattern) != 2:
                raise ValueError(f"Pattern {i} must be a tuple of (2θ, intensity)")
            self._validate_pattern_data(pattern[0], pattern[1])

        colors = self._prepare_colors(colors, len(patterns))
        labels = labels or [f"Pattern {i+1}" for i in range(len(patterns))]

        fig, ax = self._create_multiple_patterns_plot(
            patterns, colors, labels, fill_alpha, grid_alpha, linewidth,
            xlabel, ylabel, title, show_legend
        )

        self._handle_plot_output(fig, show, save, filename or 'xrd_patterns_comparison', save_format)
        return fig, ax

    def _prepare_colors(self, colors: Optional[List[str]], num_patterns: int) -> List[str]:
        """Prepare color list for multiple patterns."""
        if colors is None:
            return list(plt.cm.tab10.colors[:num_patterns])
        elif len(colors) < num_patterns:
            additional_colors = list(plt.cm.tab10.colors[:num_patterns - len(colors)])
            return colors + additional_colors
        return colors[:num_patterns]

    def _create_multiple_patterns_plot(
        self,
        patterns: List[Tuple[np.ndarray, np.ndarray]],
        colors: List[str],
        labels: List[str],
        fill_alpha: float,
        grid_alpha: float,
        linewidth: float,
        xlabel: str,
        ylabel: str,
        title: Optional[str],
        show_legend: bool
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Create plot for multiple XRD patterns."""
        fig, ax = plt.subplots(figsize=self.figsize)

        # Calculate global x-limits
        all_x_values = [two_theta for two_theta, _ in patterns]
        global_xmin = min(x.min() for x in all_x_values)
        global_xmax = max(x.max() for x in all_x_values)
        x_padding = 0.02 * (global_xmax - global_xmin)

        # Calculate y-axis padding
        global_y_max = max(intensity.max() for _, intensity in patterns)
        y_padding = 0.02 * global_y_max  # 2% padding
        y_min = -y_padding

        # Plot each pattern
        for i, (two_theta, intensity) in enumerate(patterns):
            ax.plot(two_theta, intensity, color=colors[i], linewidth=linewidth,
                   label=labels[i], zorder=3+i)
            ax.fill_between(two_theta, y_min, intensity, color=colors[i],
                           alpha=fill_alpha, zorder=2+i)

        # Configure appearance
        ax.grid(True, alpha=grid_alpha, linestyle='--', zorder=1)
        ax.set_xlabel(xlabel, fontsize=16, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=16, fontweight='bold')
        ax.set_xlim(global_xmin - x_padding, global_xmax + x_padding)
        ax.set_ylim(bottom=y_min, top=global_y_max + y_padding)

        ax.tick_params(axis='both', which='major', labelsize=14)

        if title:
            ax.set_title(title, fontsize=18, fontweight='bold', pad=20)

        if show_legend:
            ax.legend(fontsize=14, loc='best', framealpha=0.7)

        fig.tight_layout()
        return fig, ax

    def plot_pattern_from_file(
        self,
        file_path: Union[str, Path],
        phase_name: Optional[str] = None,
        sticks: bool = False,
        sticks_cif_path: Optional[str] = None,
        sticks_title: Optional[str] = "Simulated Pattern",
        eval_path: Optional[str] = None,
        title: Optional[str] = None,
        color: str = '#1E88E5',
        fill_alpha: float = 0.3,
        grid_alpha: float = 0.3,
        linewidth: float = 2,
        xlabel: Optional[str] = None,
        ylabel: str = 'Intensity (a.u.)',
        fontsize: int = 20,
        show: bool = True,
        save: bool = False,
        filename: Optional[str] = None,
        save_format: str = 'png',
        save_dir: Union[str, Path] = None,
        convert_to_q: bool = False
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Load and plot an XRD pattern from a data file, with optional stick plot."""
        two_theta, intensity = self._load_pattern_data(file_path)

        if xlabel is None:
            xlabel = determine_x_axis_label(two_theta, convert_to_q)

        if not sticks:
            fig, ax = self._create_pattern_plot(
                two_theta, intensity, color, fill_alpha, grid_alpha, linewidth,
                xlabel, ylabel, title or Path(file_path).stem,
                fontsize, None, None
            )
        else:
            if phase_name is None or sticks_cif_path is None:
                raise ValueError("phase_name and sticks_cif_path must be provided when sticks=True")

            fig, ax = self._create_pattern_plot_with_sticks(
                two_theta, intensity, phase_name, file_path,
                sticks_cif_path, sticks_title, eval_path,
                color, fill_alpha, grid_alpha, linewidth,
                xlabel, ylabel, title or Path(file_path).stem,
                fontsize, None, None
            )

        self._handle_plot_output(fig, show, save, filename, save_format, save_dir)
        return fig, ax

    def _load_pattern_data(self, file_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
        """Load XRD pattern data from file."""
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Pattern file not found: {file_path}")

        try:
            # Real .xy patterns (PyD8 exports, the bundled experimental
            # patterns) carry a short non-numeric header; GALAXI-generated ones
            # do not. Detect it per-file rather than assuming either convention.
            data = np.loadtxt(file_path, skiprows=count_header_lines(file_path))
            if data.ndim != 2 or data.shape[1] < 2:
                raise ValueError("File must contain at least two columns (2θ, intensity)")

            two_theta = data[:, 0]
            intensity = data[:, 1]

            if len(two_theta) == 0:
                raise ValueError("File contains no data")

            return two_theta, intensity

        except Exception as e:
            raise ValueError(f"Error loading XRD data from {file_path}: {e}") from e

    def plot_multiple_files(
        self,
        file_paths: List[Union[str, Path]],
        labels: Optional[List[str]] = None,
        colors: Optional[List[str]] = None,
        title: Optional[str] = None,
        fill_alpha: float = 0.2,
        grid_alpha: float = 0.3,
        linewidth: float = 2,
        xlabel: str = '2θ (degrees)',
        ylabel: str = 'Intensity (a.u.)',
        show_legend: bool = True,
        show: bool = True,
        save: bool = False,
        filename: Optional[str] = None,
        save_format: str = 'png'
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Load and plot multiple XRD patterns from data files.

        Args:
            file_paths: List of paths to XRD pattern files
            labels: Labels for each pattern. If None, uses filenames
            colors: Colors for each pattern. If None, uses default colors
            title: Plot title. If None, no title is added
            fill_alpha: Transparency for fill areas (0-1)
            grid_alpha: Transparency for grid (0-1)
            linewidth: Width of pattern lines
            xlabel: Label for x-axis
            ylabel: Label for y-axis
            show_legend: Whether to display legend
            show: Whether to display the plot
            save: Whether to save the plot
            filename: Filename for saved plot (without extension)
            save_format: File format for saved plot

        Returns:
            Tuple of (figure, axes) objects

        Raises:
            ValueError: If no valid patterns could be loaded
        """
        if not file_paths:
            raise ValueError("File paths list cannot be empty")

        patterns = self._load_multiple_pattern_files(file_paths)

        if not patterns:
            raise ValueError("No valid patterns could be loaded from the provided files")

        # Generate labels from filenames if not provided
        if labels is None:
            labels = [Path(fp).stem for fp in file_paths[:len(patterns)]]
        else:
            labels = labels[:len(patterns)]  # Trim to match loaded patterns

        return self.plot_multiple_patterns(
            patterns, labels=labels, colors=colors, title=title,
            fill_alpha=fill_alpha, grid_alpha=grid_alpha, linewidth=linewidth,
            xlabel=xlabel, ylabel=ylabel, show_legend=show_legend,
            show=show, save=save, filename=filename, save_format=save_format
        )

    def _load_multiple_pattern_files(
        self,
        file_paths: List[Union[str, Path]]
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Load multiple pattern files, skipping invalid ones."""
        patterns = []

        for file_path in file_paths:
            try:
                two_theta, intensity = self._load_pattern_data(file_path)
                patterns.append((two_theta, intensity))
            except (FileNotFoundError, ValueError) as e:
                print(f"Warning: Could not load XRD data from {file_path}: {e}")
                continue

        return patterns
