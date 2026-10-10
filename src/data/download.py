"""
download.py - Automated & Resumable Dataset Downloader with Fallback
--------------------------------------------------------------------
Downloads the SHWD dataset from Google Drive using gdown, unpacks it to data/raw/,
and verifies extracted image and annotation counts against repository claims.
"""

import os
import sys
import zipfile
import tarfile
from pathlib import Path


GOOGLE_DRIVE_ID = "1qWm7rrwvjAWs1slymbrLaCf7Q-wnGLEX"
EXPECTED_IMAGE_COUNT = 7581


def download_dataset(
    dest_dir: str = "data/raw",
    file_id: str = GOOGLE_DRIVE_ID,
    force_download: bool = False
) -> Path:
    """
    Downloads and extracts the SHWD dataset archive.
    """
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)
    
    # Check if already extracted
    voc_dir = find_voc_root(dest_path)
    if voc_dir and not force_download:
        print(f"[INFO] Dataset already extracted at: {voc_dir}")
        verify_dataset(voc_dir)
        return voc_dir

    archive_path = dest_path / "SHWD.zip"
    
    # Check if archive already downloaded
    if not archive_path.exists() or force_download:
        print(f"[INFO] Downloading SHWD from Google Drive (ID: {file_id})...")
        try:
            import gdown
            output = str(archive_path)
            gdown.download(id=file_id, output=output, quiet=False, resume=True)
        except Exception as e:
            print("\n" + "=" * 70)
            print("[ERROR] Automated download via gdown encountered an error:")
            print(f"        {e}")
            print("\nFALLBACK MANUAL DOWNLOAD INSTRUCTIONS:")
            print("1. Open the official Google Drive link in your browser:")
            print(f"   https://drive.google.com/file/d/{file_id}/view?usp=sharing")
            print("   or alternative Baidu Drive link: https://pan.baidu.com/s/1UbFkGm4EppdAU660Vu7SdQ")
            print(f"2. Download the zip archive and place it at: {archive_path.resolve()}")
            print("3. Re-run this script to extract and verify.")
            print("=" * 70 + "\n")
            if not archive_path.exists():
                sys.exit(1)

    # Extract archive
    print(f"[INFO] Extracting {archive_path.name} to {dest_path}...")
    extract_archive(archive_path, dest_path)
    
    voc_dir = find_voc_root(dest_path)
    if not voc_dir:
        raise FileNotFoundError(f"Could not locate VOC annotations/images in {dest_path}")
        
    verify_dataset(voc_dir)
    return voc_dir


def extract_archive(archive_path: Path, extract_to: Path) -> None:
    if str(archive_path).endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
    elif str(archive_path).endswith((".tar.gz", ".tar")):
        with tarfile.open(archive_path, "r:*") as tar_ref:
            tar_ref.extractall(extract_to)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")


def find_voc_root(base_dir: Path) -> Path | None:
    for candidate in base_dir.rglob("Annotations"):
        if candidate.is_dir():
            parent = candidate.parent
            if (parent / "JPEGImages").is_dir() or (parent / "images").is_dir():
                return parent
    return None


def verify_dataset(voc_root: Path) -> None:
    img_dir = voc_root / "JPEGImages" if (voc_root / "JPEGImages").is_dir() else voc_root / "images"
    ann_dir = voc_root / "Annotations"
    
    img_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    ann_files = list(ann_dir.glob("*.xml"))
    
    print("\n--- Dataset Count Verification ---")
    print(f"Found images:      {len(img_files)} (Claimed: {EXPECTED_IMAGE_COUNT})")
    print(f"Found annotations: {len(ann_files)}")
    
    if len(img_files) == EXPECTED_IMAGE_COUNT:
        print("[SUCCESS] Image count exactly matches the README claim (7,581 images).")
    else:
        print(f"[NOTE] Discrepancy observed: {len(img_files)} found vs {EXPECTED_IMAGE_COUNT} claimed.")


if __name__ == "__main__":
    download_dataset()
