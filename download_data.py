import os
import random
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Optional
from urllib.request import urlretrieve

import kagglehub

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# Where raw downloads / extracted archives are stored
OUTPUT = "./datasets"

# Where the final train/val/test structure is written
SPLIT_OUTPUT = "./data"

# Kaggle dataset handles ("owner/dataset-name")
KAGGLE_DATASETS = [
    "joebeachcapital/realwaste",
    "sumn2u/garbage-classification-v2",
]

# Direct links to .zip archives
REMOTE_DATASETS = [
    "https://github.com/garythung/trashnet/raw/master/data/dataset-resized.zip",
]

# Which class folders to keep, and what to call them in the final dataset.
#   key   = source folder name, normalised (lower-case, spaces -> underscores)
#   value = class name used in data/<split>/<value>/
# Folders NOT listed here are skipped. Several keys may point to the same
# value to merge classes. Leave the dict empty ({}) to keep every folder.
CLASS_MAP: Dict[str, str] = {
    "cardboard": "cardboard",
    "glass": "glass",
    "metal": "metal",
    "paper": "paper",
    "plastic": "plastic",
    "trash": "trash",
#    "food_organics": "organic",   # example of renaming
#    "vegetation": "organic",      # example of merging into one class
}

# File types treated as images when splitting
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# --------------------------------------------------------------------------- #
# Downloading
# --------------------------------------------------------------------------- #

def download_kaggle_datasets(kaggle_datasets: List[str]) -> List[Path]:
    """
    Download datasets from Kaggle using kagglehub.

    Each dataset is stored in its own sub-folder of OUTPUT.
    Requires Kaggle credentials (~/.kaggle/kaggle.json or env variables).

    Returns:
        List of local directories, one per dataset.
    """
    paths = []
    for dataset in kaggle_datasets:
        print(f"Downloading Kaggle dataset: {dataset}")
        target = Path(OUTPUT) / dataset.split("/")[-1]
        kagglehub.dataset_download(handle=dataset, output_dir=str(target))
        paths.append(target)

    print("\n✓ Downloaded all Kaggle datasets")
    return paths


def _reporthook(count: int, block_size: int, total_size: int) -> None:
    """Progress callback for urlretrieve: prints a live percentage."""
    if total_size > 0:
        percent = min(100, int(count * block_size * 100 / total_size))
        print(f"\rDownloading: {percent}%", end="", flush=True)


def download_remote_datasets(remote_datasets: List[str]) -> List[Path]:
    """
    Download zip archives from direct URLs and extract them.

    Each archive is extracted into OUTPUT/<archive-name>/ and the .zip file
    is deleted afterwards.

    Returns:
        List of extraction directories, one per archive.
    """
    os.makedirs(OUTPUT, exist_ok=True)
    paths = []

    for dataset_url in remote_datasets:
        file_name = dataset_url.split("/")[-1]
        zip_path = Path(OUTPUT) / file_name
        extract_dir = Path(OUTPUT) / zip_path.stem

        print(f"Downloading: {dataset_url}")
        urlretrieve(dataset_url, zip_path, _reporthook)
        print("\n✓ Download complete!")

        print("Extracting archive...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
        os.remove(zip_path)  # the archive is no longer needed
        print("✓ Extraction complete!")

        paths.append(extract_dir)

    return paths


# --------------------------------------------------------------------------- #
# Dataset discovery & splitting
# --------------------------------------------------------------------------- #

def _list_images(folder: Path) -> List[Path]:
    """Return all image files directly inside `folder` (case-insensitive)."""
    return [
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ]


def find_class_root(dataset_dir: Path) -> Optional[Path]:
    """
    Find the directory that contains the class folders.

    Downloaded archives often nest the real data several levels deep
    (e.g. `dataset-resized/dataset-resized/glass/*.jpg`), so we walk the tree
    and return the shallowest folder that has at least two sub-folders which
    directly contain images.
    """
    candidates = [dataset_dir] + sorted(p for p in dataset_dir.rglob("*") if p.is_dir())

    for candidate in candidates:
        class_folders = [
            d for d in candidate.iterdir()
            if d.is_dir() and _list_images(d)
        ]
        if len(class_folders) >= 2:
            return candidate

    return None


def split_dataset(
    source_dir: Path,
    output_dir: str = SPLIT_OUTPUT,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> None:
    """
    Split a dataset into train/val/test sets.

    The test ratio is whatever is left: 1 - train_ratio - val_ratio.
    Class folder names are normalised (lower-case, spaces -> underscores) so
    that e.g. "Glass" and "glass" from different datasets end up in the same
    class. Filenames are prefixed with the dataset name to avoid collisions.

    Args:
        source_dir:  Directory containing one folder per class.
        output_dir:  Destination root for the train/val/test folders.
        train_ratio: Proportion of images used for training.
        val_ratio:   Proportion of images used for validation.
    """
    source_dir = Path(source_dir)
    dataset_name = source_dir.parent.name if source_dir.name == "" else source_dir.name
    print(f"\nSplitting '{dataset_name}' into train/val/test...")

    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)

    class_dirs = [d for d in source_dir.iterdir() if d.is_dir()]

    for class_dir in class_dirs:
        class_name = class_dir.name.strip().lower().replace(" ", "_")

        # Keep only the requested classes (and rename them if mapped)
        if CLASS_MAP:
            if class_name not in CLASS_MAP:
                print(f"  {class_name}: skipped (not in CLASS_MAP)")
                continue
            class_name = CLASS_MAP[class_name]

        images = _list_images(class_dir)
        if not images:
            continue  # skip folders without images
        random.shuffle(images)

        # Compute cut points for the three splits
        total = len(images)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        splits = {
            "train": images[:train_end],
            "val": images[train_end:val_end],
            "test": images[val_end:],
        }

        print(
            f"  {class_name}: "
            f"train={len(splits['train'])}, "
            f"val={len(splits['val'])}, "
            f"test={len(splits['test'])}"
        )

        # Copy the files into data/<split>/<class>/
        for split, split_images in splits.items():
            split_class_dir = os.path.join(output_dir, split, class_name)
            os.makedirs(split_class_dir, exist_ok=True)

            for img_path in split_images:
                new_name = f"{dataset_name}_{img_path.name}"
                shutil.copy2(img_path, os.path.join(split_class_dir, new_name))

    print("✓ Dataset split complete!")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main() -> None:
    print("=" * 60)
    print("Waste Dataset Preparation")
    print("=" * 60)

    # Protect existing results from being overwritten by accident
    if os.path.exists(SPLIT_OUTPUT) and any(os.listdir(SPLIT_OUTPUT)):
        response = input(f"\n'{SPLIT_OUTPUT}' already exists. Overwrite? (y/N): ")
        if response.lower() != "y":
            print("Aborted.")
            return
        shutil.rmtree(SPLIT_OUTPUT)

    print("\nOptions:")
    print("  1. Download datasets (Kaggle + remote)  (recommended)")
    print("  2. I already have a dataset downloaded")
    choice = input("\nChoose option (1 or 2): ").strip()

    if choice == "1":
        try:
            dataset_dirs = download_kaggle_datasets(KAGGLE_DATASETS)
            dataset_dirs += download_remote_datasets(REMOTE_DATASETS)
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("\nIf the download failed, you can:")
            print("  1. Download the datasets manually")
            print("  2. Run this script again and choose option 2")
            return

        # Locate the class folders of each dataset and split them
        for dataset_dir in dataset_dirs:
            class_root = find_class_root(dataset_dir)
            if class_root is None:
                print(f"\n⚠ Could not find class folders in {dataset_dir}, skipping.")
                continue
            split_dataset(class_root)

    elif choice == "2":
        dataset_path = Path(input("\nEnter path to dataset directory: ").strip())
        if not dataset_path.exists():
            print(f"❌ Path not found: {dataset_path}")
            return

        class_root = find_class_root(dataset_path)
        if class_root is None:
            print(f"❌ No class folders with images found in {dataset_path}")
            return
        split_dataset(class_root)

    else:
        print("Invalid choice. Aborted.")
        return

    # Summary
    print("\n" + "=" * 60)
    print("Dataset Ready!")
    print("=" * 60)
    print("\nData directory structure:")
    print(f"  {SPLIT_OUTPUT}/")
    print("  ├── train/")
    print("  ├── val/")
    print("  └── test/")

    for split in ("train", "val", "test"):
        split_path = Path(SPLIT_OUTPUT) / split
        if split_path.exists():
            total = sum(
                1 for f in split_path.rglob("*")
                if f.suffix.lower() in IMAGE_EXTENSIONS
            )
            print(f"\n{split.capitalize()}: {total} images")

    print("\n✓ You can now run: uv run model_comparison.py")


if __name__ == "__main__":
    random.seed(42)  # reproducible splits
    main()
