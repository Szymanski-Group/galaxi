#!/usr/bin/env python3
"""Download and extract the pretrained model weights for examples/pretrained_catalog/pretrained_models/.

The 365 `.pth` checkpoints are not tracked in git (~640MB total, and too many individual
files for a single figshare upload), so they are distributed as one tarball on figshare
instead. This script downloads that tarball and extracts it into place.

Usage:
    python examples/pretrained_catalog/fetch_model_weights.py --url https://figshare.com/.../download
"""

import argparse
import hashlib
import sys
import tarfile
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# https://doi.org/10.6084/m9.figshare.33360183
DEFAULT_WEIGHTS_URL = "https://ndownloader.figshare.com/files/67931634"

# sha256 of pretrained_model_weights.tar.gz (365 .pth files, paths rooted at the repo root).
EXPECTED_SHA256 = "6a9a008d2022e0b80871d4c490121aa44e00837c29858a16e1a123964d31f83c"


def download(url: str, dest: Path) -> None:
    print(f"Downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)


def verify_checksum(path: Path, expected: str) -> None:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise SystemExit(
            f"Checksum mismatch for {path}:\n  expected {expected}\n  got      {actual}\n"
            "The download may be incomplete or corrupted -- try again before extracting."
        )
    print("Checksum OK.")


def extract(archive: Path, root: Path) -> None:
    # The archive was published when the catalog lived directly under examples/,
    # so its members are still rooted there. Remap that prefix on the fly rather
    # than re-uploading a new archive.
    print(f"Extracting {archive} -> {root}")
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        for member in members:
            if member.name.startswith("examples/"):
                member.name = "examples/pretrained_catalog/" + member.name[len("examples/"):]
        tar.extractall(root, members=members, filter="data")
    print("Done. Weights are now in examples/pretrained_catalog/pretrained_models/models_<phase>/.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_WEIGHTS_URL, help="Direct download URL for pretrained_model_weights.tar.gz")
    parser.add_argument("--archive", default=None, help="Use an already-downloaded tarball instead of fetching it")
    parser.add_argument("--skip-checksum", action="store_true", help="Skip sha256 verification")
    args = parser.parse_args()

    if args.archive:
        archive = Path(args.archive)
    else:
        if not args.url:
            sys.exit("No URL given: pass --url <figshare download link>, or --archive <local tar.gz>.")
        archive = REPO_ROOT / "pretrained_model_weights.tar.gz"
        download(args.url, archive)

    if not args.skip_checksum:
        verify_checksum(archive, EXPECTED_SHA256)

    extract(archive, REPO_ROOT)


if __name__ == "__main__":
    main()
