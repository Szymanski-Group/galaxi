#!/usr/bin/env python3
"""
COD CIF Query Module

This module provides functionality to query CIF files from the COD (Crystallography Open Database)
based on chemical systems. It allows filtering by specific element combinations
and supports both exact matches and subsystem queries.
"""

import os
import shutil
from pathlib import Path
from typing import List, Set, Optional, Union
from pymatgen.core import Composition, Structure
import logging

# Library code does not configure the root logger; console entry points call
# galaxi.log_config.configure_cli_logging() instead.
logger = logging.getLogger(__name__)


class CODQuery:
    """
    Class for querying CIF files from the COD database based on chemical systems.

    This class provides methods to filter CIF files by element composition,
    allowing users to extract structures containing specific elements or
    exact chemical systems.
    """

    def __init__(self, cod_path: Optional[str] = None):
        """
        Initialize the COD query object.

        Args:
            cod_path: Path to the COD database directory. If None, defaults to
                      the package data directory.
        """
        if cod_path is None:
            from galaxi.paths import get_default_cod_dir
            self.cod_path = Path(get_default_cod_dir())
        else:
            self.cod_path = Path(cod_path)

        if not self.cod_path.exists():
            raise FileNotFoundError(
                f"COD database not found at {self.cod_path}. "
                f"Please download and extract the FilteredCIFs.tar.gz file. "
                f"See README for installation instructions."
            )

    def _extract_composition_from_filename(self, filename: str) -> str:
        """
        Extract composition string from CIF filename.

        Args:
            filename: Name of the CIF file

        Returns:
            Composition string (everything before the first underscore)
        """
        return filename.split('_')[0]

    def _parse_composition(self, composition_str: str) -> Optional[Composition]:
        """
        Parse composition string into a Composition object.

        Args:
            composition_str: String representation of composition

        Returns:
            Composition object or None if parsing fails
        """
        try:
            return Composition(composition_str)
        except Exception as e:
            logger.debug(f"Failed to parse composition '{composition_str}': {e}")
            return None

    def _get_elements(self, composition: Composition) -> Set[str]:
        """
        Get set of element symbols from composition.

        Args:
            composition: Composition object

        Returns:
            Set of element symbols
        """
        return {str(el) for el in composition.elements}

    def query_exact_system(self, elements: List[str], output_dir: str = "queried_cifs",
                          ordered_only: bool = False) -> List[Path]:
        """
        Query CIF files containing exactly the specified elements.

        Args:
            elements: List of element symbols (e.g., ['Pd', 'Mn', 'O'])
            output_dir: Directory to copy matching CIF files
            ordered_only: If True, only include ordered structures

        Returns:
            List of paths to copied CIF files
        """
        elements_set = set(elements)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        matching_files = []
        cif_files = list(self.cod_path.glob('*.cif'))

        logger.info(f"Searching {len(cif_files)} CIF files for exact system: {elements}")
        if ordered_only:
            logger.info("Filtering for ordered structures only")

        for cif_file in cif_files:
            composition_str = self._extract_composition_from_filename(cif_file.name)
            composition = self._parse_composition(composition_str)

            if composition is None:
                continue

            file_elements = self._get_elements(composition)

            if file_elements == elements_set:
                # Check if structure is ordered if requested
                if ordered_only:
                    try:
                        structure = Structure.from_file(str(cif_file))
                        if not structure.is_ordered:
                            logger.debug(f"Skipping disordered structure: {cif_file.name}")
                            continue
                    except Exception as e:
                        logger.debug(f"Failed to load structure {cif_file.name}: {e}")
                        continue

                target_path = output_path / cif_file.name
                shutil.copy2(cif_file, target_path)
                matching_files.append(target_path)
                logger.debug(f"Match found: {cif_file.name} -> {composition_str}")

        logger.info(f"Found {len(matching_files)} files for exact system {elements}")
        return matching_files

    def query_containing_elements(self, elements: List[str], output_dir: str = "queried_cifs",
                                 ordered_only: bool = False) -> List[Path]:
        """
        Query CIF files containing all specified elements (but may contain others).

        Args:
            elements: List of element symbols that must be present
            output_dir: Directory to copy matching CIF files
            ordered_only: If True, only include ordered structures

        Returns:
            List of paths to copied CIF files
        """
        elements_set = set(elements)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        matching_files = []
        cif_files = list(self.cod_path.glob('*.cif'))

        logger.info(f"Searching {len(cif_files)} CIF files containing elements: {elements}")
        if ordered_only:
            logger.info("Filtering for ordered structures only")

        for cif_file in cif_files:
            composition_str = self._extract_composition_from_filename(cif_file.name)
            composition = self._parse_composition(composition_str)

            if composition is None:
                continue

            file_elements = self._get_elements(composition)

            if elements_set.issubset(file_elements):
                # Check if structure is ordered if requested
                if ordered_only:
                    try:
                        structure = Structure.from_file(str(cif_file))
                        if not structure.is_ordered:
                            logger.debug(f"Skipping disordered structure: {cif_file.name}")
                            continue
                    except Exception as e:
                        logger.debug(f"Failed to load structure {cif_file.name}: {e}")
                        continue

                target_path = output_path / cif_file.name
                shutil.copy2(cif_file, target_path)
                matching_files.append(target_path)
                logger.debug(f"Match found: {cif_file.name} -> {composition_str}")

        logger.info(f"Found {len(matching_files)} files containing elements {elements}")
        return matching_files

    def query_subsystems(self, elements: List[str], output_dir: str = "queried_cifs",
                        include_parent: bool = True, ordered_only: bool = False,
                        min_elems: int = 1, skip_sets: Optional[List[str]] = None) -> List[Path]:
        """
        Query CIF files for all subsystems of the specified elements.

        Args:
            elements: List of element symbols to generate subsystems from
            output_dir: Directory to copy matching CIF files
            include_parent: Whether to include the full system in addition to subsystems
            ordered_only: If True, only include ordered structures
            min_elems: Minimum number of elements required in subsystems
            skip_sets: List of element sets to skip (e.g., ["H-O", "C-O"]). Order doesn't matter.

        Returns:
            List of paths to copied CIF files
        """
        from itertools import combinations

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        all_matching_files = []

        # Parse skip_sets into normalized sets for comparison
        skip_element_sets = set()
        if skip_sets:
            for skip_set in skip_sets:
                # Split by '-' and normalize to sorted tuple for order-independent comparison
                skip_elements = tuple(sorted(skip_set.split('-')))
                skip_element_sets.add(skip_elements)

        # Generate all possible subsystems
        subsystems = []
        for r in range(max(1, min_elems), len(elements) + 1):
            if r == len(elements) and not include_parent:
                continue
            for combo in combinations(elements, r):
                # Check if this combination should be skipped
                combo_tuple = tuple(sorted(combo))
                if combo_tuple in skip_element_sets:
                    logger.debug(f"Skipping subsystem: {list(combo)}")
                    continue
                subsystems.append(list(combo))

        logger.info(f"Querying {len(subsystems)} subsystems from elements {elements}")

        for subsystem in subsystems:
            logger.info(f"Querying subsystem: {subsystem}")

            matching_files = self.query_exact_system(subsystem, str(output_path), ordered_only)
            all_matching_files.extend(matching_files)

        logger.info(f"Found {len(all_matching_files)} total files across all subsystems")
        return all_matching_files

    def query_by_n_elements(self, n: int, required_elements: Optional[List[str]] = None,
                           output_dir: str = "queried_cifs", ordered_only: bool = False) -> List[Path]:
        """
        Query CIF files containing exactly n elements.

        Args:
            n: Number of elements the composition must contain
            required_elements: Optional list of elements that must be present
            output_dir: Directory to copy matching CIF files
            ordered_only: If True, only include ordered structures

        Returns:
            List of paths to copied CIF files
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        matching_files = []
        cif_files = list(self.cod_path.glob('*.cif'))

        required_set = set(required_elements) if required_elements else set()

        logger.info(f"Searching {len(cif_files)} CIF files with exactly {n} elements")
        if required_elements:
            logger.info(f"Required elements: {required_elements}")
        if ordered_only:
            logger.info("Filtering for ordered structures only")

        for cif_file in cif_files:
            composition_str = self._extract_composition_from_filename(cif_file.name)
            composition = self._parse_composition(composition_str)

            if composition is None:
                continue

            file_elements = self._get_elements(composition)

            # Check if it has exactly n elements
            if len(file_elements) != n:
                continue

            # Check if required elements are present
            if required_elements and not required_set.issubset(file_elements):
                continue

            # Check if structure is ordered if requested
            if ordered_only:
                try:
                    structure = Structure.from_file(str(cif_file))
                    if not structure.is_ordered:
                        logger.debug(f"Skipping disordered structure: {cif_file.name}")
                        continue
                except Exception as e:
                    logger.debug(f"Failed to load structure {cif_file.name}: {e}")
                    continue

            target_path = output_path / cif_file.name
            shutil.copy2(cif_file, target_path)
            matching_files.append(target_path)
            logger.debug(f"Match found: {cif_file.name} -> {composition_str}")

        logger.info(f"Found {len(matching_files)} files with exactly {n} elements")
        return matching_files

    def get_available_elements(self) -> Set[str]:
        """
        Get set of all elements present in the COD database.

        Returns:
            Set of all element symbols found in the database
        """
        all_elements = set()
        cif_files = list(self.cod_path.glob('*.cif'))

        logger.info(f"Scanning {len(cif_files)} CIF files for available elements")

        for cif_file in cif_files:
            composition_str = self._extract_composition_from_filename(cif_file.name)
            composition = self._parse_composition(composition_str)

            if composition is not None:
                all_elements.update(self._get_elements(composition))

        logger.info(f"Found {len(all_elements)} unique elements in database")
        return all_elements

    def get_database_stats(self) -> dict:
        """
        Get statistics about the COD database.

        Returns:
            Dictionary containing database statistics
        """
        cif_files = list(self.cod_path.glob('*.cif'))
        total_files = len(cif_files)

        element_counts = {}
        composition_counts = {}

        for cif_file in cif_files:
            composition_str = self._extract_composition_from_filename(cif_file.name)
            composition = self._parse_composition(composition_str)

            if composition is not None:
                elements = self._get_elements(composition)
                n_elements = len(elements)

                # Count compositions by number of elements
                composition_counts[n_elements] = composition_counts.get(n_elements, 0) + 1

                # Count individual elements
                for element in elements:
                    element_counts[element] = element_counts.get(element, 0) + 1

        return {
            'total_files': total_files,
            'parsed_files': sum(composition_counts.values()),
            'composition_counts': composition_counts,
            'element_counts': element_counts,
            'unique_elements': len(element_counts)
        }


def query_chemical_system(elements: List[str], query_type: str = "exact",
                         output_dir: str = "queried_cifs",
                         cod_path: Optional[str] = None) -> List[Path]:
    """
    Convenience function to query the COD database for specific chemical systems.

    Args:
        elements: List of element symbols
        query_type: Type of query ("exact", "containing", "subsystems", or "n_elements")
        output_dir: Directory to copy matching CIF files
        cod_path: Path to COD database (uses default if None)

    Returns:
        List of paths to copied CIF files
    """
    querier = CODQuery(cod_path)

    if query_type == "exact":
        return querier.query_exact_system(elements, output_dir)
    elif query_type == "containing":
        return querier.query_containing_elements(elements, output_dir)
    elif query_type == "subsystems":
        return querier.query_subsystems(elements, output_dir)
    elif query_type == "n_elements":
        if len(elements) == 1 and elements[0].isdigit():
            n = int(elements[0])
            return querier.query_by_n_elements(n, output_dir=output_dir)
        else:
            raise ValueError("For n_elements query, provide number as single element")
    else:
        raise ValueError(f"Unknown query type: {query_type}")


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Query COD database for chemical systems")
    parser.add_argument("elements", nargs="+", help="Element symbols to query")
    parser.add_argument("--type", choices=["exact", "containing", "subsystems", "n_elements"],
                       default="exact", help="Type of query to perform")
    parser.add_argument("--output", default="queried_cifs", help="Output directory")
    parser.add_argument("--cod-path", help="Path to COD database")
    parser.add_argument("--stats", action="store_true", help="Show database statistics")

    args = parser.parse_args()

    querier = CODQuery(args.cod_path)

    if args.stats:
        stats = querier.get_database_stats()
        print(f"\nCOD Database Statistics:")
        print(f"Total files: {stats['total_files']}")
        print(f"Parsed files: {stats['parsed_files']}")
        print(f"Unique elements: {stats['unique_elements']}")
        print(f"\nComposition distribution:")
        for n, count in sorted(stats['composition_counts'].items()):
            print(f"  {n} elements: {count} files")

    else:
        files = query_chemical_system(args.elements, args.type, args.output, args.cod_path)
        print(f"\nCopied {len(files)} matching CIF files to {args.output}")