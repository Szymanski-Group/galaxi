"""The README Quick Start must be wired to assets a user can actually obtain.

`create_default_config()` must point at absolute paths so a config resolves from
any working directory, and a missing background-profile library must report
which file is missing and how to install it.

These tests need none of the multi-GB assets themselves.
"""

import os
from pathlib import Path

import pytest

from galaxi.paths import get_default_bg_profiles_path, get_default_cod_dir
from galaxi.workflows.streamlined_workflow import StreamlinedWorkflow, create_default_config


def test_default_config_paths_are_absolute(tmp_path, monkeypatch):
    """A default config must resolve from any working directory."""
    monkeypatch.chdir(tmp_path)
    config = create_default_config()

    for key in ("bg_profiles_path", "cod_dir"):
        value = config["directories"][key]
        assert os.path.isabs(value), f"{key} default {value!r} is not an absolute path"


def test_bg_profiles_path_honors_env_override(monkeypatch):
    monkeypatch.setenv("GALAXI_BG_PROFILES", "/tmp/some/other/library.h5")
    assert get_default_bg_profiles_path() == "/tmp/some/other/library.h5"


def test_missing_bg_profiles_raises_actionable_error(tmp_path, monkeypatch):
    """Step 1 must name the file and the command that installs it."""
    monkeypatch.chdir(tmp_path)

    references = tmp_path / "References"
    references.mkdir()

    config = create_default_config()
    config["directories"]["references_dir"] = str(references)
    config["directories"]["bg_profiles_path"] = str(tmp_path / "definitely_absent.h5")
    config["directories"]["output_dir"] = str(tmp_path / "out")

    workflow = StreamlinedWorkflow(config=config)

    # Must fail before any expensive work (COD loading), not deep inside
    # per-phase generation.
    with pytest.raises(FileNotFoundError) as excinfo:
        workflow.step_1_generate_training_data(phases=["AnyPhase"])

    message = str(excinfo.value)
    assert "definitely_absent.h5" in message
    assert "galaxi-setup-bg-profiles" in message


def test_setup_bg_profiles_reports_expected_schema():
    """The setup script and the workflow must agree on the datasets required."""
    from galaxi.scripts import setup_bg_profiles
    from galaxi.paths import BG_PROFILES_FILENAME

    assert BG_PROFILES_FILENAME.endswith(".h5")
    assert len(setup_bg_profiles.EXPECTED_SHA256) == 64
    assert setup_bg_profiles.EXPECTED_SIZE_BYTES > 0


def test_bg_profiles_download_location_is_configured():
    """A release must ship a working default download location.

    Without it `galaxi-setup-bg-profiles` cannot fetch the library on a clean
    install, and the Quick Start has no way to run.
    """
    from galaxi.scripts import setup_bg_profiles

    assert setup_bg_profiles.DRIVE_FILE_ID, (
        "DRIVE_FILE_ID is unset: `galaxi-setup-bg-profiles` cannot download the "
        "background-profile library, so the README Quick Start is unrunnable."
    )
