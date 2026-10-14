import os
import shutil
import random
import zipfile
from pathlib import Path
from urllib.request import urlretrieve


def download_trashnet():
    """Download TrashNet dataset from GitHub."""
    print("Downloading TrashNet dataset...")

    url = "https://github.com/garythung/trashnet/raw/master/data/dataset-resized.zip"
    zip_path = "trashnet_data.zip"

    def reporthook(count, block_size, total_size):
        if total_size > 0:
            percent = int(count * block_size * 100 / total_size)
            print(f"\rDownloading: {percent}%", end="", flush=True)

    urlretrieve(url, zip_path, reporthook)
    print("\n✓ Download complete!")

    return zip_path


def split_dataset(source_dir, output_dir="data", train_ratio=0.7, val_ratio=0.15):
    """
    Split dataset into train/val/test sets.

    Args:
        source_dir: Directory containing class folders
        output_dir: Output directory for train/val/test splits
        train_ratio: Proportion for training set (default 0.7)
        val_ratio: Proportion for validation set (default 0.15)
        test_ratio is automatically 1 - train_ratio - val_ratio
    """
    print(f"\nSplitting dataset into train/val/test...")

    splits = ["train", "val", "test"]
    for split in splits:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)

    class_dirs = [d for d in Path(source_dir).iterdir() if d.is_dir()]

    for class_dir in class_dirs:
        class_name = class_dir.name
        print(f"  Processing class: {class_name}")

        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        random.shuffle(images)

        total = len(images)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        train_images = images[:train_end]
        val_images = images[train_end:val_end]
        test_images = images[val_end:]

        print(
            f"    Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)}"
        )

        for split, split_images in [
            ("train", train_images),
            ("val", val_images),
            ("test", test_images),
        ]:
            split_class_dir = os.path.join(output_dir, split, class_name)
            os.makedirs(split_class_dir, exist_ok=True)

            for img_path in split_images:
                shutil.copy2(img_path, os.path.join(split_class_dir, img_path.name))

    print("✓ Dataset split complete!")


def main():
    print("=" * 60)
    print("TrashNet Dataset Preparation")
    print("=" * 60)

    if os.path.exists("data") and any(os.listdir("data")):
        response = input("\n'data' directory already exists. Overwrite? (y/N): ")
        if response.lower() != "y":
            print("Aborted.")
            return
        shutil.rmtree("data")

    print("\nOptions:")
    print("  1. Download dataset from GitHub (recommended)")
    print("  2. I already have the dataset downloaded")

    choice = input("\nChoose option (1 or 2): ").strip()

    if choice == "1":
        try:
            zip_path = download_trashnet()

            print("\nExtracting dataset...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall("temp_trashnet")
            print("✓ Extraction complete!")

            temp_dir = Path("temp_trashnet")
            dataset_dirs = list(temp_dir.rglob("*"))

            source_dir = None
            for d in dataset_dirs:
                if d.is_dir() and any(
                    c.name
                    in ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
                    for c in d.iterdir()
                    if c.is_dir()
                ):
                    source_dir = d
                    break

            if source_dir is None:
                possible_paths = [
                    temp_dir / "dataset-resized",
                    temp_dir / "data" / "dataset-resized",
                    temp_dir,
                ]
                for path in possible_paths:
                    if path.exists():
                        source_dir = path
                        break

            if source_dir is None:
                print(
                    "\n❌ Could not find dataset classes. Please check the downloaded files."
                )
                return

            split_dataset(source_dir)

            print("\nCleaning up temporary files...")
            os.remove(zip_path)
            shutil.rmtree("temp_trashnet")

        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("\nIf download failed, you can:")
            print("  1. Manually download from: https://github.com/garythung/trashnet")
            print("  2. Run this script again and choose option 2")
            return

    elif choice == "2":
        dataset_path = input("\nEnter path to dataset directory: ").strip()
        if not os.path.exists(dataset_path):
            print(f"❌ Path not found: {dataset_path}")
            return

        split_dataset(dataset_path)

    else:
        print("Invalid choice. Aborted.")
        return

    print("\n" + "=" * 60)
    print("Dataset Ready!")
    print("=" * 60)
    print(f"\nData directory structure:")
    print(f"  data/")
    print(f"  ├── train/")
    print(f"  ├── val/")
    print(f"  └── test/")

    for split in ["train", "val", "test"]:
        split_path = Path("data") / split
        if split_path.exists():
            total = sum(1 for _ in split_path.rglob("*.jpg")) + sum(
                1 for _ in split_path.rglob("*.png")
            )
            print(f"\n{split.capitalize()}: {total} images")

    print("\n✓ You can now run: uv run model_comparison.py")


if __name__ == "__main__":
    random.seed(42)
    main()
