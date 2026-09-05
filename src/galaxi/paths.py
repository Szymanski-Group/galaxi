"""Shared filesystem-location helpers.

Centralizes logic for locating package-installed data (e.g. the COD
background-phase CIFs) so no call site has to hardcode a path that only
happens to be correct on the machine it was written on.
"""

import importlib.util
import os
from pathlib import Path


def get_galaxi_package_path() -> Path:
    """Locate the installed (or in-source) `galaxi` package directory.

    Prefers an actually-installed location (site-packages/dist-packages)
    over a source checkout, but falls back to whatever `galaxi` resolves
    to via `importlib` if nothing better is found.
    """
    spec = importlib.util.find_spec("galaxi")
    if spec is None or spec.origin is None:
        raise RuntimeError(
            "Could not locate the galaxi package. Please ensure it is "
            "installed (e.g. `pip install -e .`)."
        )

    package_path = Path(spec.origin).parent
    return package_path


def _legacy_cod_dir() -> Path:
    """Legacy location, inside the installed package directory itself.

    Package directories are often read-only for system-wide installs and are
    replaced on reinstall, so setup writes to the user data directory instead.
    An install that already populated this path keeps working.
    """
    return get_galaxi_package_path() / "data" / "cod"


def get_default_cod_dir() -> str:
    """Default location of the COD background-phase CIFs (see
    `galaxi.scripts.setup_cod`).

    Resolution order:
    1. `GALAXI_COD_DIR` environment variable, if set (explicit override).
    2. The legacy in-package location, if it already holds data, so an install
       that populated it keeps working without re-downloading.
    3. A user-writable data directory: `$XDG_DATA_HOME/galaxi/cod`, falling
       back to `~/.local/share/galaxi/cod`.
    """
    env_override = os.environ.get("GALAXI_COD_DIR")
    if env_override:
        return env_override

    legacy_dir = _legacy_cod_dir()
    if legacy_dir.exists() and any(legacy_dir.glob("*.cif")):
        return str(legacy_dir)

    xdg_data_home = os.environ.get("XDG_DATA_HOME")
    data_home = Path(xdg_data_home) if xdg_data_home else Path.home() / ".local" / "share"
    return str(data_home / "galaxi" / "cod")


BG_PROFILES_FILENAME = "public_phase_profile_50000.h5"


def get_default_bg_profiles_path() -> str:
    """Default location of the background-profile HDF5 library (see
    `galaxi.scripts.setup_bg_profiles`).

    This is the pool of ~50,000 pre-simulated single-phase patterns that
    training-data generation samples from to build the "other phases" in
    multi-phase positives and the negatives. It is a large (~670MB) downloaded
    asset, so it lives next to the COD CIFs in a user-writable data directory
    rather than inside the package.

    Resolution order:
    1. `GALAXI_BG_PROFILES` environment variable, if set (explicit override).
    2. A user-writable data directory: `$XDG_DATA_HOME/galaxi/`, falling back
       to `~/.local/share/galaxi/`.

    An absolute path, so training-data generation resolves the library from
    any working directory.
    """
    env_override = os.environ.get("GALAXI_BG_PROFILES")
    if env_override:
        return env_override

    xdg_data_home = os.environ.get("XDG_DATA_HOME")
    data_home = Path(xdg_data_home) if xdg_data_home else Path.home() / ".local" / "share"
    return str(data_home / "galaxi" / BG_PROFILES_FILENAME)
