import os
import subprocess
import shutil


def count_images_per_class(data_dir: str):
    class_counts = {}

    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)

        if not os.path.isdir(class_path):
            continue

        count = sum(
            1 for f in os.listdir(class_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )

        class_counts[class_name] = count

    return class_counts


def is_dataset_downloaded(output_dir: str) -> bool:
    return (
        os.path.exists(os.path.join(output_dir, "yes")) and
        os.path.exists(os.path.join(output_dir, "no"))
    )


def clean_duplicate_dataset(output_dir: str):
    nested_path = os.path.join(output_dir, "brain_tumor_dataset")

    if os.path.exists(nested_path):
        print("[INFO] Removing duplicated nested dataset...")

        shutil.rmtree(nested_path)


def download_kaggle_dataset(dataset_name: str, output_dir="data/raw", force=False):

    if os.path.exists(output_dir) and force:
        print("[INFO] Cleaning existing dataset...")
        shutil.rmtree(output_dir)

    if is_dataset_downloaded(output_dir) and not force:
        print(f"[INFO] Dataset already exists in {output_dir}. Skipping download.")
        return

    print("[INFO] Downloading dataset from Kaggle...")

    os.makedirs(output_dir, exist_ok=True)

    subprocess.run([
        "kaggle", "datasets", "download",
        "-d", dataset_name,
        "-p", output_dir,
        "--unzip"
    ], check=True)

    # FIX DUPLICATION
    clean_duplicate_dataset(output_dir)

    print(f"[INFO] Dataset downloaded to {output_dir}")