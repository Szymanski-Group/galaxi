#!/usr/bin/env python3
"""
Background-Profile Library Setup Script

Downloads the HDF5 library of pre-simulated single-phase XRD patterns that
`StreamlinedWorkflow.step_1_generate_training_data` samples from to build the
"other phases" in multi-phase positive patterns and the negative patterns.

The file holds ~50,000 phases simulated from the COD:

    patterns     (N, L) float  -- one simulated pattern per phase
    phase_names  (N,)   bytes  -- used to mask the target phase out of the pool
    two_theta    (L,)   float  -- the grid the patterns were simulated on
                                  (interpolated if your config uses another)

It is ~670MB, so it is downloaded on demand into a user-writable data
directory rather than shipped in the package -- the same treatment the COD
CIFs get from `galaxi-setup-cod`.
"""

import argparse
import hashlib
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from galaxi.log_config import configure_cli_logging
from galaxi.paths import BG_PROFILES_FILENAME, get_default_bg_profiles_path

logger = logging.getLogger(__name__)

# Google Drive file id for public_phase_profile_50000.h5.
# Share link: https://drive.google.com/file/d/1R7wcsjchbY_1hBTf_vTeRf_LLL7CmEkU/view
DRIVE_FILE_ID = "1R7wcsjchbY_1hBTf_vTeRf_LLL7CmEkU"

# sha256 of public_phase_profile_50000.h5 (702,781,444 bytes). Google Drive is
# a mutable, third-party-controlled source with no integrity guarantee, so this
# is the only way to detect content that silently changed or was truncated
# mid-download -- the same reasoning behind galaxi-setup-cod's --sha256 flag.
EXPECTED_SHA256 = "1ac44927374c8e5817058f83444cb69219869751e29ab77870d1672d5cf34610"

EXPECTED_SIZE_BYTES = 702781444


def _drive_url(file_id: str) -> str:
    return f"https://drive.google.com/uc?id={file_id}"


def _normalize_permissions(path: Path) -> None:
    """Give the installed library ordinary data-file permissions.

    Both install paths would otherwise propagate a restrictive mode: the
    download stages through a TemporaryDirectory (0700) and shutil.move carries
    that over, and shutil.copy2 copies the source file's mode. Either can leave
    a 0600 library in a shared data directory that no other user can read.
    """
    mask = os.umask(0)
    os.umask(mask)
    try:
        os.chmod(path, 0o666 & ~mask)
    except OSError:
        pass  # non-fatal: the file is installed and readable by its owner


def install_gdown() -> bool:
    """Ensure gdown is importable (it is a declared dependency, but installs
    from source checkouts occasionally miss it)."""
    try:
        import gdown  # noqa: F401
        return True
    except ImportError:
        logger.info("gdown not found, installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
            logger.info("Successfully installed gdown")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install gdown: {e}")
            return False


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_bg_profiles(path: Path, check_checksum: bool = False) -> bool:
    """Check that `path` is a usable background-profile library."""
    if not path.exists():
        logger.error(f"Background-profile library not found: {path}")
        return False

    size = path.stat().st_size
    if size != EXPECTED_SIZE_BYTES:
        logger.warning(
            f"{path} is {size} bytes; expected {EXPECTED_SIZE_BYTES}. "
            "This may be a truncated download or a different build of the library."
        )

    if check_checksum:
        logger.info("Verifying sha256 (this reads the whole file)...")
        digest = sha256_of(path)
        if digest != EXPECTED_SHA256:
            logger.error(f"sha256 mismatch: expected {EXPECTED_SHA256}, got {digest}")
            return False
        logger.info("Checksum OK.")

    try:
        import h5py
        with h5py.File(path, "r") as f:
            missing = [k for k in ("patterns", "phase_names", "two_theta") if k not in f]
            if missing:
                logger.error(f"{path} is missing required dataset(s): {', '.join(missing)}")
                return False
            n_phases, n_points = f["patterns"].shape
            two_theta = f["two_theta"][:]
            logger.info(
                f"Background-profile library verified: {n_phases} phases x {n_points} points, "
                f"2theta {two_theta[0]:.2f}-{two_theta[-1]:.2f} deg ({path})"
            )
        return True
    except Exception as e:
        logger.error(f"Failed to open {path} as HDF5: {e}")
        return False


def download_bg_profiles(dest: Path, url: str, force: bool = False) -> bool:
    """Download the library to `dest` via gdown, atomically."""
    if dest.exists() and not force:
        logger.info(f"Background-profile library already exists: {dest}")
        return True

    if not install_gdown():
        logger.error("Could not install gdown. Please install manually: pip install gdown")
        return False

    import gdown

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading background-profile library (~670MB) from {url}")
    logger.info("This may take several minutes depending on your connection.")

    # Download to a temp file next to the destination, then move into place, so
    # an interrupted download never leaves a half-written file that later runs
    # would treat as a valid library.
    with tempfile.TemporaryDirectory(dir=str(dest.parent)) as tmpdir:
        tmp_path = Path(tmpdir) / BG_PROFILES_FILENAME
        try:
            gdown.download(url, str(tmp_path), quiet=False)
        except Exception as e:
            logger.error(f"Download failed: {e}")
            logger.error(f"You can also download it by hand and pass it with --file <path>.")
            return False

        if not tmp_path.exists() or tmp_path.stat().st_size == 0:
            logger.error(
                "Download produced no data. Google Drive sometimes refuses large "
                "files when a share quota is exceeded; try again later, or "
                "download by hand and pass it with --file <path>."
            )
            return False

        digest = sha256_of(tmp_path)
        if digest != EXPECTED_SHA256:
            logger.error(
                f"sha256 mismatch: expected {EXPECTED_SHA256}, got {digest}. "
                "The download was discarded. This can mean the Drive content "
                "changed, or the transfer was corrupted/truncated."
            )
            return False
        logger.info("Checksum OK.")

        shutil.move(str(tmp_path), str(dest))

    logger.info(f"Background-profile library installed at: {dest}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Set up the background-profile HDF5 library for GALAXI training-data generation"
    )
    parser.add_argument("--force-download", action="store_true",
                        help="Redownload even if the file already exists")
    parser.add_argument("--verify-only", action="store_true",
                        help="Only verify an existing installation")
    parser.add_argument("--check-checksum", action="store_true",
                        help="With --verify-only, also verify the sha256 (reads the whole file)")
    parser.add_argument("--dest", default=None,
                        help=f"Where to install it (default: {get_default_bg_profiles_path()})")
    parser.add_argument("--url", default=None,
                        help="Override the download URL")
    parser.add_argument("--file-id", default=None,
                        help="Google Drive file id, if the built-in default is not set")
    parser.add_argument("--file", default=None,
                        help="Install an already-downloaded .h5 from this path instead of downloading")

    args = parser.parse_args()
    configure_cli_logging()

    dest = Path(args.dest) if args.dest else Path(get_default_bg_profiles_path())

    if args.verify_only:
        return 0 if verify_bg_profiles(dest, check_checksum=args.check_checksum) else 1

    if args.file:
        source = Path(args.file)
        if not source.exists():
            logger.error(f"No such file: {source}")
            return 1
        dest.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != dest.resolve():
            logger.info(f"Copying {source} -> {dest}")
            shutil.copy2(source, dest)
    else:
        if dest.exists() and not args.force_download:
            logger.info("Background-profile library appears to already be set up.")
            if verify_bg_profiles(dest):
                logger.info("Setup verified. Use --force-download to reinstall.")
                return 0

        if args.url:
            url = args.url
        else:
            file_id = args.file_id or DRIVE_FILE_ID
            if not file_id:
                logger.error(
                    "No download location is configured for the background-profile "
                    "library in this build of GALAXI."
                )
                logger.error(
                    "Pass one explicitly with --file-id <google drive id> or "
                    "--url <direct link>, or install an already-downloaded copy "
                    "with --file <path to public_phase_profile_50000.h5>."
                )
                return 1
            url = _drive_url(file_id)

        if not download_bg_profiles(dest, url, force=args.force_download):
            return 1

    _normalize_permissions(dest)

    if verify_bg_profiles(dest):
        logger.info("Background-profile library setup completed successfully!")
        return 0

    logger.error("Setup verification failed")
    return 1


if __name__ == "__main__":
    sys.exit(main())
