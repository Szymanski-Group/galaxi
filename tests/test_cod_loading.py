"""COD background structures must be loaded with the arguments they expect.

`_load_cod_structures(phases, cod_cif_dir, max_structures)` takes the phase
filter first, so every caller has to pass its arguments by the right name or
position -- a directory path bound to `phases` fails only later, at the first
filesystem access, and looks like a missing-directory problem.
"""

from pathlib import Path

from galaxi.core.pattern_generator import UnifiedPatternGenerator

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_CIF_DIR = REPO_ROOT / "tutorials" / "data" / "cif_files"


def test_load_cod_structures_accepts_phases_dir_and_max_structures():
    generator = UnifiedPatternGenerator(reference_dir=str(FIXTURE_CIF_DIR))

    structures = generator._load_cod_structures(
        phases=set(), cod_cif_dir=str(FIXTURE_CIF_DIR), max_structures=2
    )

    assert len(structures) <= 2
    for name, structure in structures:
        assert isinstance(name, str)


def test_load_cod_structures_filters_out_named_phases():
    generator = UnifiedPatternGenerator(reference_dir=str(FIXTURE_CIF_DIR))
    all_names = {p.stem for p in FIXTURE_CIF_DIR.glob("*.cif")}
    excluded = next(iter(all_names))

    structures = generator._load_cod_structures(
        phases={excluded}, cod_cif_dir=str(FIXTURE_CIF_DIR), max_structures=len(all_names)
    )

    assert excluded not in {name for name, _ in structures}
