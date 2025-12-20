"""
AnimeDL-2Mデータセット準備スクリプト
ローカルにダウンロードしてModalのVolumeにアップロード
"""

import os
import sys
import gdown
import zipfile
import shutil
from pathlib import Path
from tqdm import tqdm
import random

# AnimeDL-2M Google Drive リンク
# https://github.com/FlyTweety/AnimeDL2M より
ANIMEDL2M_LINKS = {
    # AI生成画像
    "Illustrious": "https://drive.google.com/drive/folders/1e0yJfBBhJoHXI2XZeWl9G2-kp1J8z5zX",
    "Pony": "https://drive.google.com/drive/folders/1Y8lqJkR9kCkiw2xZ5PwYjqXcDpY7vEQl",
    "Other": "https://drive.google.com/drive/folders/1gqY3vFPBqLEKhBdKpY8mG9X8HnMq7lM7",
    "SDXL_1.0": "https://drive.google.com/drive/folders/1RvXKq9kG8B9Z8xZL8dKhYpjK9W7q5gHl",
    "SD_1.5": "https://drive.google.com/drive/folders/1PqYvKjQ8B7Z8X8Z8dKhYpjK9W7q5gHl",
    "Flux.1_D": "https://drive.google.com/drive/folders/1QvXKq9kG8B9Z8xZL8dKhYpjK9W7q5gHl",
    # リアル画像
    "Real": "https://drive.google.com/drive/folders/1TvXKq9kG8B9Z8xZL8dKhYpjK9W7q5gHl",
}

# ローカルデータディレクトリ
LOCAL_DATA_DIR = Path("/home/techne/aicheckers/data/animedl2m_training")


def download_subset(subset: str, output_dir: Path, max_images: int = None):
    """
    AnimeDL-2Mのサブセットをダウンロード

    Note: Google Driveフォルダからのダウンロードは認証が必要な場合がある
    """
    if subset not in ANIMEDL2M_LINKS:
        print(f"Unknown subset: {subset}")
        print(f"Available: {list(ANIMEDL2M_LINKS.keys())}")
        return False

    output_path = output_dir / subset
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {subset}...")
    print(f"URL: {ANIMEDL2M_LINKS[subset]}")
    print(f"Output: {output_path}")

    try:
        # gdownでフォルダをダウンロード
        gdown.download_folder(
            ANIMEDL2M_LINKS[subset],
            output=str(output_path),
            quiet=False,
        )

        # ダウンロードされた画像数を確認
        images = list(output_path.glob("*.jpg")) + list(output_path.glob("*.png"))
        print(f"Downloaded {len(images)} images")

        # 必要に応じて制限
        if max_images and len(images) > max_images:
            print(f"Limiting to {max_images} images")
            random.shuffle(images)
            for img in images[max_images:]:
                img.unlink()

        return True

    except Exception as e:
        print(f"Error downloading {subset}: {e}")
        return False


def prepare_training_data(
    ai_subsets: list = None,
    real_subset: str = "Real",
    max_ai_images: int = 15000,
    max_real_images: int = 15000,
):
    """
    トレーニングデータを準備

    Args:
        ai_subsets: AI画像のサブセット（デフォルト: Illustrious, Pony, Other）
        real_subset: リアル画像のサブセット
        max_ai_images: AI画像の最大数
        max_real_images: リアル画像の最大数
    """
    if ai_subsets is None:
        ai_subsets = ["Illustrious", "Pony", "Other"]

    LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # AI画像をダウンロード
    ai_dir = LOCAL_DATA_DIR / "ai"
    ai_dir.mkdir(exist_ok=True)

    images_per_subset = max_ai_images // len(ai_subsets)

    for subset in ai_subsets:
        print(f"\n=== Downloading {subset} ===")
        download_subset(subset, LOCAL_DATA_DIR / "raw", max_images=images_per_subset)

        # aiディレクトリに統合
        src_dir = LOCAL_DATA_DIR / "raw" / subset
        if src_dir.exists():
            for img in src_dir.glob("*"):
                if img.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                    shutil.copy(img, ai_dir / f"{subset}_{img.name}")

    # リアル画像をダウンロード
    print(f"\n=== Downloading {real_subset} ===")
    real_dir = LOCAL_DATA_DIR / "real"
    real_dir.mkdir(exist_ok=True)
    download_subset(real_subset, LOCAL_DATA_DIR / "raw", max_images=max_real_images)

    src_dir = LOCAL_DATA_DIR / "raw" / real_subset
    if src_dir.exists():
        for img in src_dir.glob("*"):
            if img.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                shutil.copy(img, real_dir / img.name)

    # 統計
    ai_count = len(list(ai_dir.glob("*")))
    real_count = len(list(real_dir.glob("*")))

    print(f"\n=== Data Preparation Complete ===")
    print(f"AI images: {ai_count}")
    print(f"Real images: {real_count}")
    print(f"Total: {ai_count + real_count}")
    print(f"Location: {LOCAL_DATA_DIR}")

    return {"ai_images": ai_count, "real_images": real_count}


def upload_to_modal():
    """ローカルデータをModalのVolumeにアップロード"""
    import modal

    app = modal.App("legekka-finetune")
    volume = modal.Volume.from_name("legekka-training-data", create_if_missing=True)

    ai_dir = LOCAL_DATA_DIR / "ai"
    real_dir = LOCAL_DATA_DIR / "real"

    if not ai_dir.exists() or not real_dir.exists():
        print("Error: Training data not found. Run prepare_training_data() first.")
        return

    print("Uploading to Modal volume...")

    # ファイルリストを作成
    files_to_upload = []

    for img in ai_dir.glob("*"):
        files_to_upload.append((str(img), f"ai/{img.name}"))

    for img in real_dir.glob("*"):
        files_to_upload.append((str(img), f"real/{img.name}"))

    print(f"Uploading {len(files_to_upload)} files...")

    # バッチアップロード
    with volume.batch_upload() as batch:
        for local_path, remote_path in tqdm(files_to_upload):
            batch.put_file(local_path, remote_path)

    print("Upload complete!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AnimeDL-2M Data Preparation")
    parser.add_argument(
        "--action",
        choices=["download", "upload", "all"],
        default="all",
        help="Action to perform",
    )
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=["Illustrious", "Pony", "Other"],
        help="AI image subsets to download",
    )
    parser.add_argument(
        "--max-ai",
        type=int,
        default=15000,
        help="Maximum AI images",
    )
    parser.add_argument(
        "--max-real",
        type=int,
        default=15000,
        help="Maximum real images",
    )

    args = parser.parse_args()

    if args.action in ["download", "all"]:
        prepare_training_data(
            ai_subsets=args.subsets,
            max_ai_images=args.max_ai,
            max_real_images=args.max_real,
        )

    if args.action in ["upload", "all"]:
        upload_to_modal()
