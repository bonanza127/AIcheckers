#!/usr/bin/env python3
"""
Civitai API を使って特定モデルの画像を収集
レート制限を守りながら安全にダウンロード
"""
import os
import time
import random
import requests
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv("/home/techne/aicheckers/.env")

API_KEY = os.getenv("CIVITAI_API_KEY", "d9bb5b07a777d1eabb4d634295ed94f1")
BASE_URL = "https://civitai.com/api/v1"
OUTPUT_DIR = Path("/home/techne/aicheckers/data/civitai_new")

# レート制限設定（bot扱いされないように）
MIN_DELAY = 2.0  # 最小待機秒数
MAX_DELAY = 5.0  # 最大待機秒数

def get_headers():
    return {
        "Authorization": f"Bearer {API_KEY}",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

def search_images(model_name: str = None, model_id: int = None, limit: int = 100, cursor: str = None):
    """
    画像を検索
    model_id: Civitaiのモデル/バージョンID
    """
    params = {
        "limit": min(limit, 200),  # API上限は200
        "sort": "Newest",
        "nsfw": "None",  # SFWのみ
    }

    if model_id:
        params["modelVersionId"] = model_id

    if cursor:
        params["cursor"] = cursor

    try:
        response = requests.get(
            f"{BASE_URL}/images",
            params=params,
            headers=get_headers(),
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"API Error: {e}")
        return None

def download_image(url: str, save_path: Path):
    """画像をダウンロード"""
    try:
        response = requests.get(url, timeout=30, stream=True)
        response.raise_for_status()

        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"Download error: {e}")
        return False

def collect_images(model_name: str, model_version_id: int, target_count: int = 1000):
    """
    特定モデルの画像を収集

    Args:
        model_name: 保存フォルダ名
        model_version_id: CivitaiのモデルバージョンID
        target_count: 目標枚数
    """
    output_dir = OUTPUT_DIR / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    existing = len(list(output_dir.glob("*.jpg"))) + len(list(output_dir.glob("*.png")))
    print(f"Existing images: {existing}")

    if existing >= target_count:
        print(f"Already have {existing} images, skipping.")
        return

    cursor = None
    downloaded = existing

    with tqdm(total=target_count, initial=existing, desc=model_name) as pbar:
        while downloaded < target_count:
            # API呼び出し
            data = search_images(model_id=model_version_id, cursor=cursor)

            if not data or "items" not in data:
                print("No more data or API error")
                break

            items = data["items"]
            if not items:
                print("No more images")
                break

            for item in items:
                if downloaded >= target_count:
                    break

                image_url = item.get("url")
                image_id = item.get("id")

                if not image_url:
                    continue

                # 拡張子を判定
                ext = ".jpg"
                if "png" in image_url.lower():
                    ext = ".png"
                elif "webp" in image_url.lower():
                    ext = ".webp"

                save_path = output_dir / f"{image_id}{ext}"

                if save_path.exists():
                    continue

                # ダウンロード
                if download_image(image_url, save_path):
                    downloaded += 1
                    pbar.update(1)

                # ランダム待機（人間らしく）
                time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))

            # 次のページ
            metadata = data.get("metadata", {})
            cursor = metadata.get("nextCursor")

            if not cursor:
                print("No more pages")
                break

            # ページ間でも待機
            time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))

    print(f"Collected {downloaded} images for {model_name}")

def get_model_version_id(model_name: str):
    """
    モデル名からバージョンIDを取得
    """
    # よく使われるモデルのバージョンID（手動で調べたもの）
    known_models = {
        # SDXL系
        "sdxl_1.0": None,  # 公式SDXLはCivitaiにない
        "pony_v6": 290640,
        "illustrious_xl": 1215460,  # Illustrious XL v0.1

        # FLUX系
        "flux_dev": None,  # Civitaiで検索が必要
        "flux_schnell": None,

        # NovelAI系（Civitaiにはない）
        "nai_v3": None,
    }

    return known_models.get(model_name.lower())

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                       help="Model name (e.g., pony_v6, illustrious_xl)")
    parser.add_argument("--version-id", type=int, default=None,
                       help="Civitai model version ID (overrides --model lookup)")
    parser.add_argument("--count", type=int, default=1000,
                       help="Target number of images")
    args = parser.parse_args()

    version_id = args.version_id or get_model_version_id(args.model)

    if not version_id:
        print(f"Unknown model: {args.model}")
        print("Please provide --version-id manually")
        print("Find it at: https://civitai.com/models/XXXX (check URL of specific version)")
        return

    print(f"Collecting {args.count} images for {args.model} (version ID: {version_id})")
    collect_images(args.model, version_id, args.count)

if __name__ == "__main__":
    main()
