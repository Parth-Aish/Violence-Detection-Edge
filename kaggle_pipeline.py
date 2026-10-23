"""Download the Real Life Violence Situations dataset from Kaggle."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


DATASET = "mohamedmustafa/real-life-violence-situations-dataset"
ROOT = Path(__file__).resolve().parent


def _has_kaggle_credentials() -> bool:
    if os.environ.get("KAGGLE_API_TOKEN"):
        return True
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True

    config_dir = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle"))
    return any(
        path.is_file()
        for path in (config_dir / "kaggle.json", config_dir / "access_token")
    )


def download_dataset(destination: Path, *, force: bool = False) -> None:
    if not _has_kaggle_credentials():
        raise RuntimeError(
            "Kaggle credentials are missing. Configure KAGGLE_USERNAME and "
            "KAGGLE_KEY, or place kaggle.json in the Kaggle config directory."
        )

    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()) and not force:
        raise FileExistsError(
            f"{destination} is not empty; pass --force to replace/download into it."
        )

    command = [
        sys.executable,
        "-m",
        "kaggle",
        "datasets",
        "download",
        "-d",
        DATASET,
        "-p",
        str(destination),
        "--unzip",
    ]
    if force:
        command.append("--force")

    subprocess.run(command, check=True)
    print(f"Dataset extracted to {destination}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination",
        type=Path,
        default=ROOT / "data" / "raw",
        help="Directory for the extracted dataset (default: ./data/raw)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow download into a non-empty directory and replace matching files.",
    )
    args = parser.parse_args()

    try:
        download_dataset(args.destination.resolve(), force=args.force)
    except (FileExistsError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"Dataset download failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
