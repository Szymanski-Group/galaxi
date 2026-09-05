"""Saved training patterns default to the format the training path reads.

UnifiedPatternGenerator writes "npy" by default, matching what PatternsDataset
and StreamlinedWorkflow expect. A two-column ".xy" file is a raw
(2theta, intensity) pattern, which is a different thing from the preprocessed
(intensity, mask) cache training reads back.
"""

from pymatgen.core import Lattice, Structure

from galaxi.core.pattern_generator import UnifiedPatternGenerator


def test_generate_phase_detection_patterns_defaults_to_npy(tmp_path):
    lattice = Lattice.cubic(4.2)
    structure = Structure(lattice, ["Li", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    structure.to(filename=str(tmp_path / "Li2O.cif"))

    generator = UnifiedPatternGenerator(reference_dir=str(tmp_path))
    output_dir = tmp_path / "out"
    generator.generate_phase_detection_patterns(
        target_phase_name="Li2O",
        num_positive=1,
        num_negative=1,
        save_patterns=True,
        output_dir=str(output_dir),
        # file_format intentionally omitted -- must default to npy
    )

    saved_files = [p for p in output_dir.rglob("*") if p.is_file() and p.suffix]
    assert saved_files, "expected at least one saved pattern file"
    assert all(p.suffix == ".npy" for p in saved_files if p.name != "detection_ground_truth.json")
    assert not any(p.suffix == ".xy" for p in saved_files)
