#!/usr/bin/env python3
"""
COD Database Setup Script

This script automates the download and setup of the COD (Crystallography Open Database) crystal structure database
for use with GALAXI. It downloads the database from Google Drive and extracts it
to the proper location in the installed package directory.
"""

import hashlib
import os
import sys
import tarfile
import tempfile
import shutil
import zipfile
from pathlib import Path
import subprocess
import logging
import argparse

from galaxi.log_config import configure_cli_logging
from galaxi.paths import get_default_cod_dir

logger = logging.getLogger(__name__)


def install_gdown():
    """Install gdown if not available."""
    try:
        import gdown
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


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_cod_database(download_path: Path, force_download: bool = False, expected_sha256: str = None):
    """
    Download the COD database from Google Drive.

    Args:
        download_path: Path where to save the downloaded file
        force_download: Whether to redownload if file already exists
        expected_sha256: If given, verify the downloaded file's SHA-256 matches
            this value and abort otherwise. The archive is hosted on Google
            Drive, which is a mutable, third-party-controlled source with no
            built-in integrity guarantee, so this is the only way to detect
            if the served content silently changed or was tampered with.
    """
    cod_file = download_path / "FilteredCIFs.tar.gz"

    if cod_file.exists() and not force_download:
        logger.info(f"COD database file already exists: {cod_file}")
        return cod_file

    if not install_gdown():
        logger.error("Could not install gdown. Please install manually: pip install gdown")
        return None

    # Import gdown after installation
    import gdown

    google_drive_url = "https://drive.google.com/uc?id=116oGxY5Slclr3mC7s4jy6MDN0ZSimsLf"

    logger.info("Downloading COD database from Google Drive...")
    logger.info("This may take several minutes depending on your internet connection.")

    try:
        gdown.download(google_drive_url, str(cod_file), quiet=False)
        digest = _sha256_of(cod_file)
        logger.info(f"Successfully downloaded COD database to: {cod_file} (sha256={digest})")
        if expected_sha256 is not None and digest != expected_sha256:
            cod_file.unlink()
            logger.error(
                f"SHA-256 mismatch: expected {expected_sha256}, got {digest}. "
                "The downloaded file was deleted. This can mean the Google Drive "
                "content changed or the download was corrupted/tampered with."
            )
            return None
        return cod_file
    except Exception as e:
        logger.error(f"Failed to download COD database: {e}")
        logger.error("Please try downloading manually from:")
        logger.error("https://drive.google.com/file/d/116oGxY5Slclr3mC7s4jy6MDN0ZSimsLf/view?usp=sharing")
        return None


def _safe_tar_members(tar: tarfile.TarFile, dest: Path):
    """Yield only members whose extraction path stays within `dest`, rejecting
    path-traversal entries (e.g. `../../etc/passwd` or absolute paths) that a
    tampered or malicious archive could use to write outside the target
    directory.
    """
    dest_resolved = dest.resolve()
    for member in tar.getmembers():
        member_path = (dest_resolved / member.name).resolve()
        if member_path == dest_resolved or dest_resolved in member_path.parents:
            yield member
        else:
            logger.warning(f"Skipping tar member with unsafe path: {member.name}")


def extract_cod_database(tar_file: Path, extract_to: Path):
    """
    Extract the COD database tar.gz file.

    Args:
        tar_file: Path to the tar.gz file
        extract_to: Directory to extract to
    """
    logger.info(f"Extracting COD database to: {extract_to}")

    try:
        with tarfile.open(tar_file, 'r:gz') as tar:
            # Extract to temporary directory first
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                tar.extractall(temp_path, members=_safe_tar_members(tar, temp_path))

                # Find the FilteredCIFs directory
                filtered_cifs_dirs = list(temp_path.glob('**/FilteredCIFs'))

                if not filtered_cifs_dirs:
                    raise RuntimeError("No FilteredCIFs directory found in extracted files")

                filtered_cifs_source = filtered_cifs_dirs[0]
                logger.info(f"Found FilteredCIFs directory: {filtered_cifs_source}")

                # Create target directory and copy files
                extract_to.mkdir(parents=True, exist_ok=True)

                # Copy all CIF files
                cif_files = list(filtered_cifs_source.glob('*.cif'))
                logger.info(f"Copying {len(cif_files)} CIF files...")

                for cif_file in cif_files:
                    shutil.copy2(cif_file, extract_to / cif_file.name)

                logger.info(f"Successfully extracted {len(cif_files)} CIF files to {extract_to}")

    except Exception as e:
        logger.error(f"Failed to extract COD database: {e}")
        raise


def verify_cod_setup(cod_path: Path):
    """
    Verify that the COD database is properly set up.

    Args:
        cod_path: Path to the COD directory

    Returns:
        bool: True if setup is valid
    """
    if not cod_path.exists():
        logger.error(f"COD directory does not exist: {cod_path}")
        return False

    cif_files = list(cod_path.glob('*.cif'))

    if len(cif_files) == 0:
        logger.error(f"No CIF files found in COD directory: {cod_path}")
        return False

    logger.info(f"COD setup verified: {len(cif_files)} CIF files found in {cod_path}")

    # Test loading a few files to make sure they're valid
    from galaxi.cod_query import CODQuery

    try:
        querier = CODQuery(str(cod_path))
        stats = querier.get_database_stats()
        logger.info(f"Database contains {stats['total_files']} files with {stats['unique_elements']} unique elements")
        return True
    except Exception as e:
        logger.error(f"Failed to load COD database: {e}")
        return False


def main():
    """Main function for COD setup."""
    parser = argparse.ArgumentParser(description="Set up COD database for GALAXI")
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force redownload even if file exists"
    )
    parser.add_argument(
        "--temp-dir",
        type=str,
        help="Temporary directory for downloads (default: system temp)"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing installation"
    )
    parser.add_argument(
        "--sha256",
        type=str,
        default=None,
        help="Expected SHA-256 of the downloaded archive; abort if it doesn't match"
    )

    args = parser.parse_args()

    configure_cli_logging()

    try:
        # Resolve the COD target directory (user data dir by default, or
        # GALAXI_COD_DIR if set -- see galaxi.paths.get_default_cod_dir).
        cod_target_path = Path(get_default_cod_dir())

        # If only verifying, check and exit
        if args.verify_only:
            if verify_cod_setup(cod_target_path):
                logger.info("COD database is properly set up and functional")
                return 0
            else:
                logger.error("COD database is not properly set up")
                return 1

        # Check if already set up
        if cod_target_path.exists() and list(cod_target_path.glob('*.cif')) and not args.force_download:
            logger.info("COD database appears to already be set up")
            if verify_cod_setup(cod_target_path):
                logger.info("Setup verified. Use --force-download to reinstall.")
                return 0

        # Set up temporary directory
        if args.temp_dir:
            temp_dir = Path(args.temp_dir)
            temp_dir.mkdir(parents=True, exist_ok=True)
        else:
            temp_dir = Path(tempfile.gettempdir()) / "galaxi_cod_setup"
            temp_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Using temporary directory: {temp_dir}")

        # Download database
        tar_file = download_cod_database(temp_dir, args.force_download, expected_sha256=args.sha256)
        if not tar_file:
            return 1

        # Extract database
        extract_cod_database(tar_file, cod_target_path)

        # Verify setup
        if verify_cod_setup(cod_target_path):
            logger.info("COD database setup completed successfully!")

            # Clean up temporary files
            try:
                if tar_file.exists():
                    tar_file.unlink()
                    logger.info("Cleaned up temporary download file")
            except Exception as e:
                logger.warning(f"Could not clean up temporary file: {e}")

            return 0
        else:
            logger.error("Setup verification failed")
            return 1

    except Exception as e:
        logger.error(f"Setup failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
