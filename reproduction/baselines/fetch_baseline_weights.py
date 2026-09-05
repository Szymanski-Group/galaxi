#!/usr/bin/env python3
"""Download and extract pretrained weights for the ML baseline adapters in
`galaxi.baselines` (xca, autoanalyzer, xqueryer).

None of these weights are tracked in git -- xca's is ~17MB, autoanalyzer's
is ~72MB, and xqueryer's is ~3.5GB (inference-only; optimizer state was
stripped from the ~11GB training checkpoint) -- so each is distributed as
its own tarball on figshare and downloaded on demand, mirroring
examples/pretrained_catalog/fetch_model_weights.py's pattern.

Usage:
    python -m galaxi.baselines.fetch_baseline_weights xca
    python -m galaxi.baselines.fetch_baseline_weights autoanalyzer
    python -m galaxi.baselines.fetch_baseline_weights xqueryer
    python -m galaxi.baselines.fetch_baseline_weights all
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tarfile
import urllib.request
from pathlib import Path

DEST_DIR = Path(__file__).resolve().parent / "weights"

# Fill these in once each baseline's weights tarball has been uploaded to
# figshare (see the module docstring for what each tarball contains).
BASELINES = {
    "xca": {
        # https://doi.org/10.6084/m9.figshare.33420511
        "url": "https://ndownloader.figshare.com/files/68153059",
        "sha256": "c93ee2c2e87436538e4c4fc0f2e2c0a598b0d8baff61fdede73edeae51dc8c20",
        "members": ["xca_saved_model.keras", "xca_phase_mapping.json"],
    },
    "autoanalyzer": {
        # https://doi.org/10.6084/m9.figshare.33420508
        "url": "https://ndownloader.figshare.com/files/68153056",
        "sha256": "40e3d2d8ef9ec9d7c02e19b41fd3886973bf092e9363e73a677daacec270a33c",
        "members": ["autoanalyzer_Model.h5"],
    },
    "xqueryer": {
        # https://doi.org/10.6084/m9.figshare.33420550
        "url": "https://ndownloader.figshare.com/files/68153107",
        "sha256": "8789eed1107f81ebff70f0df593c20cc66d9ebb6459df5192919d02d6756b416",
        "members": [
            "xqueryer_model_best_inference_only.pth",
            "xqueryer_labels.json",
            "xqueryer_CGCNN_atom_emb.json",
        ],
    },
}


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


def fetch(name: str, *, url: str | None = None, archive: str | Path | None = None, skip_checksum: bool = False) -> Path:
    """Download (or use a local `archive`) and extract one baseline's weights
    tarball into `galaxi/baselines/weights/`. Returns that directory.
    """
    if name not in BASELINES:
        raise ValueError(f"Unknown baseline {name!r}; choose from {sorted(BASELINES)}")
    spec = BASELINES[name]

    DEST_DIR.mkdir(parents=True, exist_ok=True)

    if archive is not None:
        archive_path = Path(archive)
    else:
        resolved_url = url or spec["url"]
        if not resolved_url:
            raise SystemExit(
                f"No download URL configured for {name!r} yet. Pass --url explicitly, "
                f"or --archive <local tar.gz>."
            )
        archive_path = DEST_DIR / f"{name}_weights.tar.gz"
        download(resolved_url, archive_path)

    if not skip_checksum and spec["sha256"]:
        verify_checksum(archive_path, spec["sha256"])

    print(f"Extracting {archive_path} -> {DEST_DIR}")
    with tarfile.open(archive_path) as tar:
        tar.extractall(DEST_DIR, filter="data")

    missing = [m for m in spec["members"] if not (DEST_DIR / m).exists()]
    if missing:
        raise SystemExit(f"Extraction finished but expected file(s) missing: {missing}")

    print(f"Done. {name} weights are now in {DEST_DIR}/")
    return DEST_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("baseline", choices=sorted(BASELINES) + ["all"])
    parser.add_argument("--url", default=None, help="Override the default figshare download URL")
    parser.add_argument("--archive", default=None, help="Use an already-downloaded tarball instead of fetching it")
    parser.add_argument("--skip-checksum", action="store_true", help="Skip sha256 verification")
    args = parser.parse_args()

    names = list(BASELINES) if args.baseline == "all" else [args.baseline]
    for name in names:
        fetch(name, url=args.url, archive=args.archive, skip_checksum=args.skip_checksum)


if __name__ == "__main__":
    main()
