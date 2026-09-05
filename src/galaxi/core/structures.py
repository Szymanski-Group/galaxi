"""
Structure loading and management utilities.
"""

import os
from typing import List, Tuple
from pymatgen.core import Structure


class StructureManager:
    """Manages loading and accessing crystal structures from CIF files using lazy loading."""

    def __init__(self, reference_dir: str = "References"):
        """
        Initialize structure manager.

        Args:
            reference_dir: Directory containing reference CIF files
        """
        self.reference_dir = reference_dir
        self.phase_names: List[str] = []

        # Dictionary to cache structures once they are loaded
        self._structures_cache = {}

        self._discover_phases()

    def _discover_phases(self):
        """Find all CIF files and populate phase names without loading the structures (Lazy Loading)."""
        if not os.path.exists(self.reference_dir):
            raise FileNotFoundError(f"Reference directory {self.reference_dir} does not exist")

        self.phase_names = []
        self._structures_cache.clear()

        for filename in sorted(os.listdir(self.reference_dir)):
            if filename.endswith('.cif'):
                phase_name = filename.replace('.cif', '')
                self.phase_names.append(phase_name)

        print(f"Discovered {len(self.phase_names)} reference phases (structures will be lazily loaded)")

    def get_structure(self, phase_name: str) -> Structure:
        """Get structure by phase name. Loads from disk if not already cached."""
        if phase_name not in self.phase_names:
            raise ValueError(f"Phase {phase_name} not found")

        # Load and cache the structure if it hasn't been accessed yet
        if phase_name not in self._structures_cache:
            filename = f"{phase_name}.cif"
            try:
                structure = Structure.from_file(os.path.join(self.reference_dir, filename))
                self._structures_cache[phase_name] = structure
            except Exception as e:
                print(f"Warning: Could not load {filename}: {e}")
                raise ValueError(f"Failed to load structure for {phase_name}") from e

        return self._structures_cache[phase_name]

    def get_all_structures(self) -> List[Tuple[str, Structure]]:
        """Get all phase names and structures (Warning: This will force load ALL structures)."""
        return [(phase_name, self.get_structure(phase_name)) for phase_name in self.phase_names]

    def get_phase_names(self) -> List[str]:
        """Get list of all phase names."""
        return self.phase_names.copy()

    def get_structures(self) -> List[Structure]:
        """Get list of all structures (Warning: This will force load ALL structures)."""
        return [self.get_structure(phase_name) for phase_name in self.phase_names]

    def reload(self):
        """Reload phase list and clear cache."""
        self._discover_phases()