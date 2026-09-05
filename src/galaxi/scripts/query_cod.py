#!/usr/bin/env python3
"""
Command-line script for querying the COD database.

This script provides a convenient command-line interface for querying
CIF files from the COD (Crystallography Open Database) based on chemical systems.
"""

import argparse
import sys
from pathlib import Path
from galaxi.cod_query import CODQuery
from galaxi.log_config import configure_cli_logging


def main():
    """Main function for the COD query script."""
    parser = argparse.ArgumentParser(
        description="Query COD database for chemical systems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Query for exact Pd-Mn-O ternary system
  python -m galaxi.scripts.query_cod Pd Mn O --type exact --output PdMnO_cifs

  # Query for all subsystems of Li-Mn-Ti-O-F
  python -m galaxi.scripts.query_cod Li Mn Ti O F --type subsystems --output LiMnTiOF_subsystems

  # Query for ordered structures only
  python -m galaxi.scripts.query_cod Li Mn O --type exact --output LiMnO_ordered --ordered-only

  # Query for structures containing both Pd and O (may have other elements)
  python -m galaxi.scripts.query_cod Pd O --type containing --output PdO_containing

  # Query for all ternary systems (exactly 3 elements)
  python -m galaxi.scripts.query_cod 3 --type n_elements --output ternary_systems

  # Show database statistics
  python -m galaxi.scripts.query_cod --stats
        """
    )

    parser.add_argument(
        "elements",
        nargs="*",
        help="Element symbols to query (e.g., Pd Mn O) or number for n_elements query"
    )

    parser.add_argument(
        "--type",
        choices=["exact", "containing", "subsystems", "n_elements"],
        default="exact",
        help="Type of query to perform (default: exact)"
    )

    parser.add_argument(
        "--output",
        default="queried_cifs",
        help="Output directory for copied CIF files (default: queried_cifs)"
    )

    parser.add_argument(
        "--cod-path",
        help="Path to COD database directory (uses package default if not specified)"
    )

    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show database statistics and exit"
    )

    parser.add_argument(
        "--list-elements",
        action="store_true",
        help="List all available elements in the database and exit"
    )

    parser.add_argument(
        "--ordered-only",
        action="store_true",
        help="Only include ordered structures (no site occupancy disorders)"
    )

    args = parser.parse_args()

    configure_cli_logging()

    # Validate arguments
    if not args.stats and not args.list_elements and not args.elements:
        parser.error("Must provide elements to query, or use --stats or --list-elements")

    try:
        querier = CODQuery(args.cod_path)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("\nPlease follow the installation instructions in the README to set up the COD database.", file=sys.stderr)
        sys.exit(1)

    if args.stats:
        stats = querier.get_database_stats()
        print(f"\nCOD Database Statistics:")
        print(f"Total files: {stats['total_files']:,}")
        print(f"Successfully parsed files: {stats['parsed_files']:,}")
        print(f"Unique elements: {stats['unique_elements']}")

        print(f"\nComposition distribution:")
        for n, count in sorted(stats['composition_counts'].items()):
            print(f"  {n} element{'s' if n != 1 else ''}: {count:,} files")

        print(f"\nTop 20 most common elements:")
        sorted_elements = sorted(stats['element_counts'].items(), key=lambda x: x[1], reverse=True)
        for element, count in sorted_elements[:20]:
            print(f"  {element}: {count:,} occurrences")

        return

    if args.list_elements:
        elements = sorted(querier.get_available_elements())
        print(f"\nAvailable elements in COD database ({len(elements)} total):")
        # Print in columns
        for i in range(0, len(elements), 10):
            row = elements[i:i+10]
            print("  " + "  ".join(f"{el:>2}" for el in row))
        return

    # Perform the query
    try:
        if args.type == "exact":
            files = querier.query_exact_system(args.elements, args.output, args.ordered_only)
        elif args.type == "containing":
            files = querier.query_containing_elements(args.elements, args.output, args.ordered_only)
        elif args.type == "subsystems":
            files = querier.query_subsystems(args.elements, args.output, ordered_only=args.ordered_only)
        elif args.type == "n_elements":
            if len(args.elements) != 1 or not args.elements[0].isdigit():
                parser.error("For n_elements query, provide exactly one number")
            n = int(args.elements[0])
            files = querier.query_by_n_elements(n, output_dir=args.output, ordered_only=args.ordered_only)

        print(f"\nQuery completed successfully!")
        print(f"Found and copied {len(files)} matching CIF files to '{args.output}'")

        if files:
            print(f"\nFirst few matches:")
            for file_path in files[:5]:
                composition = file_path.name.split('_')[0]
                print(f"  {file_path.name} -> {composition}")

            if len(files) > 5:
                print(f"  ... and {len(files) - 5} more files")

    except Exception as e:
        print(f"Error during query: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()