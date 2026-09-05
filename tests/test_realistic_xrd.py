"""Physical consistency of RealisticXRDGenerator.

Two properties the simulation must hold to:

1. `apply_all_effects=False` suppresses every stochastic effect, so repeated
   calls under different seeds return identical patterns.
2. The Ka1 and Ka2 components of one pattern come from the same specimen, so
   they share their physical latent variables -- temperature, texture, sample
   displacement, crystallite size, microstrain and the pseudo-Voigt mixing
   parameter -- rather than sampling them independently.
"""

import torch
from pymatgen.core import Lattice, Structure

from galaxi.pattern_generation.realistic_xrd import RealisticXRDGenerator


def _build_generator():
    lattice = Lattice.cubic(4.2)
    structure = Structure(lattice, ["Li", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    generator = RealisticXRDGenerator(params={"min_angle": 10, "max_angle": 80, "num_points": 2001})
    peak_list = generator.get_peak_list(structure)
    return generator, peak_list


def test_apply_all_effects_false_is_deterministic_across_seeds():
    generator, peak_list = _build_generator()

    torch.manual_seed(0)
    result_seed0 = generator.generate_realistic_pattern(peak_list, 2, apply_all_effects=False)
    torch.manual_seed(1)
    result_seed1 = generator.generate_realistic_pattern(peak_list, 2, apply_all_effects=False)

    # With apply_all_effects=False, no randomized Ka1/Ka2-level physics
    # (Debye-Waller, texture, shifts, broadening) should be sampled, so the
    # simple-Gaussian fallback path is fully deterministic given the same peaks.
    assert torch.allclose(result_seed0[1], result_seed1[1], atol=1e-6)


def test_apply_all_effects_true_varies_across_seeds():
    generator, peak_list = _build_generator()

    torch.manual_seed(0)
    result_seed0 = generator.generate_realistic_pattern(peak_list, 2, apply_all_effects=True)
    torch.manual_seed(1)
    result_seed1 = generator.generate_realistic_pattern(peak_list, 2, apply_all_effects=True)

    assert not torch.allclose(result_seed0[1], result_seed1[1], atol=1e-6)


def test_ka1_ka2_share_one_sampled_set_of_physical_effects(monkeypatch):
    generator, peak_list = _build_generator()

    sample_calls = []
    original_sample = generator._sample_physical_effects

    def spy_sample(batch_size):
        effects = original_sample(batch_size)
        sample_calls.append(effects)
        return effects

    monkeypatch.setattr(generator, "_sample_physical_effects", spy_sample)

    effects_seen = []
    original_dtc = generator._discrete_to_continuous

    def spy_dtc(*args, **kwargs):
        effects_seen.append(kwargs.get("effects"))
        return original_dtc(*args, **kwargs)

    monkeypatch.setattr(generator, "_discrete_to_continuous", spy_dtc)

    torch.manual_seed(0)
    generator.generate_realistic_pattern(peak_list, 3, apply_all_effects=True)

    assert len(sample_calls) == 1, "physical latents must be sampled once per pattern, not once per wavelength"
    assert len(effects_seen) == 2, "expected exactly one Ka1 call and one Ka2 call"
    assert effects_seen[0] is effects_seen[1], "Ka1 and Ka2 must reuse the same sampled physical latent variables"
