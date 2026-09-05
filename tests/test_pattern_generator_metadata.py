"""Multi-phase metadata names its weights for what they are.

Each phase's pattern is independently max-normalized before the synthetic
mixing weights are applied, so those weights are not physical weight or mole
fractions. The metadata keys are "mixing_coefficients" and
"background_mixing_coefficients" to keep that distinction explicit.
"""

from pymatgen.core import Lattice, Structure

from galaxi.core.config import XRDGenerationConfig
from galaxi.core.pattern_generator import UnifiedPatternGenerator


def _make_generator(tmp_path):
    lattice1 = Lattice.cubic(4.2)
    structure1 = Structure(lattice1, ["Li", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    lattice2 = Lattice.cubic(3.0)
    structure2 = Structure(lattice2, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])

    structure1.to(filename=str(tmp_path / "Li2O.cif"))
    structure2.to(filename=str(tmp_path / "NaCl.cif"))

    return UnifiedPatternGenerator(reference_dir=str(tmp_path))


def test_multi_phase_metadata_uses_mixing_coefficients_key(tmp_path):
    generator = _make_generator(tmp_path)
    config = XRDGenerationConfig(min_angle=10, max_angle=80, num_points=500)

    _, composition = generator._generate_multi_phase_pattern(2, 0.1, config)

    assert "mixing_coefficients" in composition
    assert "fractions" not in composition


def test_negative_background_metadata_uses_mixing_coefficients_key(tmp_path):
    generator = _make_generator(tmp_path)
    config = XRDGenerationConfig(min_angle=10, max_angle=80, num_points=500)

    structures = generator.structure_manager.get_structures()
    names = generator.structure_manager.get_phase_names()
    cod_structures = list(zip(names, structures))

    _, metadata = generator.generate_negative_background_pattern(cod_structures, config)

    assert "mixing_coefficients" in metadata
    assert "fractions" not in metadata


def test_positive_detection_metadata_uses_background_mixing_coefficients_key(tmp_path):
    generator = _make_generator(tmp_path)
    config = XRDGenerationConfig(min_angle=10, max_angle=80, num_points=500)

    structures = generator.structure_manager.get_structures()
    names = generator.structure_manager.get_phase_names()
    target_structure = structures[0]
    background = list(zip(names[1:], structures[1:])) or [(names[0], structures[0])]

    _, metadata = generator._generate_positive_detection_pattern(
        target_structure, names[0], 0.7, background, config
    )

    assert "background_mixing_coefficients" in metadata
    assert "background_fractions" not in metadata
