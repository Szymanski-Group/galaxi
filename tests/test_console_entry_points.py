"""Every [project.scripts] entry point must be importable and expose main().

Importing each entry-point target catches module paths that no longer resolve,
renamed packages, and entry points aimed at modules that no longer exist --
without running any of the commands.
"""

import importlib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"


def _load_entry_points():
    try:
        import tomllib
    except ModuleNotFoundError:  # Python < 3.11
        import tomli as tomllib

    with open(PYPROJECT, "rb") as f:
        data = tomllib.load(f)
    return data.get("project", {}).get("scripts", {})


ENTRY_POINTS = _load_entry_points()


def test_entry_points_are_declared():
    assert ENTRY_POINTS, "no [project.scripts] entry points found in pyproject.toml"


@pytest.mark.parametrize("name,target", sorted(ENTRY_POINTS.items()))
def test_entry_point_target_is_importable(name, target):
    module_path, _, attr = target.partition(":")
    assert attr, f"{name} target {target!r} does not name a callable"

    module = importlib.import_module(module_path)

    func = getattr(module, attr, None)
    assert func is not None, f"{module_path} has no attribute {attr!r} (entry point {name})"
    assert callable(func), f"{module_path}:{attr} is not callable (entry point {name})"
