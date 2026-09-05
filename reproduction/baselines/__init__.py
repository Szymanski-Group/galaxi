"""
Baseline phase-identification methods benchmarked against GALAXI in the
manuscript (Fig. 3a/b): a custom classical search-match implementation
(`peak_search_match`, vendored here in full since it's this project's own
code) and thin inference adapters for three published ML baselines --
`xca`, `xqueryer`, `autoanalyzer` -- each of which depends on its own
upstream repository (not vendored here; see each module's docstring for
the exact commit used) plus a pretrained-weights file fetched on demand via
`fetch_baseline_weights.py` rather than committed to git.
"""

from .peak_search_match import match_phases

__all__ = ["match_phases"]
