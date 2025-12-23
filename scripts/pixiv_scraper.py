#!/usr/bin/env python3
"""
Pixiv NovelAI画像スクレイパー
pixivpy3を使用してNovelAIタグ付き画像を収集
"""

import argparse
import os
import time
import random
import json
from pathlib import Path
from datetime import datetime

try:
    from pixivpy3 import AppPixivAPI
except ImportError:
    print("pixivpy3 is required. Install with: pip install pixivpy3")
    raise

from tqdm import tqdm

# 設定
DEFAULT_OUTPUT_DIR = "/home/techne/aicheckers/data/pixiv_novelai"
DEFAULT_TAG = "NovelAI"
DEFAULT_MIN_BOOKMARKS = 100
DEFAULT_DELAY_MIN = 2.0
DEFAULT_DELAY_MAX = 5.0


def random_delay(min_sec: float, max_sec: float):
    """ランダムな遅延"""
    delay = random.uniform(min_sec, max_sec)
    # 10%の確率で追加遅延
    if random.random() < 0.1:
        delay += random.uniform(3, 8)
    time.sleep(delay)


def load_existing_ids(output_path: Path) -> set:
    """既存のダウンロード済みIDを読み込み"""
    existing_ids = set()

    # ファイル名からID抽出
    for ext in ["*.jpg", "*.png", "*.gif", "*.webp"]:
        for f in output_path.glob(ext):
            try:
                # ファイル名形式: {id}_{title}.{ext}
                file_id = int(f.stem.split("_")[0])
                existing_ids.add(file_id)
            except (ValueError, IndexError):
                pass

    # metadata.jsonからも読み込み
    metadata_file = output_path / "metadata.json"
    if metadata_file.exists():
        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
                for item in metadata:
                    existing_ids.add(item.get("id", 0))
        except Exception:
            pass

    return existing_ids


def save_metadata(metadata_file: Path, metadata_list: list):
    """メタデータを保存"""
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, ensure_ascii=False, indent=2)


def scrape_pixiv_novelai(
    refresh_token: str,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    max_images: int = 1000,
    min_bookmarks: int = DEFAULT_MIN_BOOKMARKS,
    tag: str = DEFAULT_TAG,
    delay_min: float = DEFAULT_DELAY_MIN,
    delay_max: float = DEFAULT_DELAY_MAX,
    sort: str = "popular_desc",  # date_desc, popular_desc
):
    """Pixiv NovelAI画像をスクレイピング"""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metadata_file = output_path / "metadata.json"

    print(f"[INFO] Output directory: {output_path}")
    print(f"[INFO] Target: {max_images} images")
    print(f"[INFO] Tag: {tag}")
    print(f"[INFO] Min bookmarks: {min_bookmarks}")
    print(f"[INFO] Sort: {sort}")
    print(f"[INFO] Delay: {delay_min}-{delay_max} seconds")
    print()

    # 認証
    print("[INFO] Authenticating with Pixiv...")
    api = AppPixivAPI()
    try:
        api.auth(refresh_token=refresh_token)
        print("[INFO] Authentication successful")
    except Exception as e:
        print(f"[ERROR] Authentication failed: {e}")
        print("[HINT] Refresh token may be expired. Get a new one from browser.")
        return 0

    # 既存ファイル確認
    existing_ids = load_existing_ids(output_path)
    print(f"[INFO] Existing files: {len(existing_ids)}")

    # メタデータ読み込み
    metadata_list = []
    if metadata_file.exists():
        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata_list = json.load(f)
        except Exception:
            metadata_list = []

    downloaded = 0
    skipped_low_bookmarks = 0
    skipped_existing = 0
    errors = 0

    # 検索パラメータ
    search_params = {
        "word": tag,
        "search_target": "partial_match_for_tags",
        "sort": sort,
    }

    print()
    with tqdm(total=max_images, desc="Downloading", unit="img") as pbar:
        next_qs = search_params

        while downloaded < max_images and next_qs:
            random_delay(delay_min, delay_max)

            try:
                res = api.search_illust(**next_qs)
            except Exception as e:
                print(f"\n[ERROR] Search failed: {e}")
                errors += 1
                if errors > 5:
                    print("[ERROR] Too many errors, stopping")
                    break
                time.sleep(30)
                continue

            if not res.illusts:
                print("\n[INFO] No more results")
                break

            for illust in res.illusts:
                if downloaded >= max_images:
                    break

                illust_id = illust.id

                # 既存チェック
                if illust_id in existing_ids:
                    skipped_existing += 1
                    continue

                # ブックマーク数チェック
                if illust.total_bookmarks < min_bookmarks:
                    skipped_low_bookmarks += 1
                    continue

                # 画像URL取得
                if illust.meta_single_page.get("original_image_url"):
                    img_url = illust.meta_single_page["original_image_url"]
                elif illust.meta_pages:
                    img_url = illust.meta_pages[0]["image_urls"]["original"]
                else:
                    img_url = illust.image_urls.get("large") or illust.image_urls.get("medium")

                if not img_url:
                    continue

                # ファイル名（タイトルから危険な文字を除去）
                safe_title = "".join(c for c in illust.title[:20] if c.isalnum() or c in " _-").strip()
                ext = img_url.split(".")[-1].split("?")[0]
                filename = f"{illust_id}_{safe_title}.{ext}"
                filepath = output_path / filename

                # ダウンロード
                random_delay(delay_min / 2, delay_max / 2)

                try:
                    api.download(img_url, path=str(output_path), fname=filename)

                    # メタデータ記録
                    metadata_list.append({
                        "id": illust_id,
                        "title": illust.title,
                        "user_id": illust.user.id,
                        "user_name": illust.user.name,
                        "bookmarks": illust.total_bookmarks,
                        "tags": [t.name for t in illust.tags],
                        "create_date": illust.create_date,
                        "downloaded_at": datetime.now().isoformat(),
                    })

                    existing_ids.add(illust_id)
                    downloaded += 1
                    pbar.update(1)

                    # 定期的にメタデータ保存
                    if downloaded % 10 == 0:
                        save_metadata(metadata_file, metadata_list)

                except Exception as e:
                    print(f"\n[ERROR] Download failed for {illust_id}: {e}")
                    errors += 1

            # 次ページ
            next_qs = api.parse_qs(res.next_url)

            # 定期休憩
            if downloaded > 0 and downloaded % 50 == 0:
                print(f"\n[INFO] Taking a break at {downloaded} images...")
                time.sleep(random.uniform(15, 30))

    # 最終保存
    save_metadata(metadata_file, metadata_list)

    print()
    print(f"[DONE] Downloaded: {downloaded}")
    print(f"[INFO] Skipped (existing): {skipped_existing}")
    print(f"[INFO] Skipped (low bookmarks): {skipped_low_bookmarks}")
    print(f"[INFO] Errors: {errors}")

    return downloaded


def main():
    parser = argparse.ArgumentParser(description="Pixiv NovelAI Scraper")
    parser.add_argument("--token", "-t", required=True, help="Pixiv refresh token")
    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--count", "-n", type=int, default=1000, help="Number of images to download")
    parser.add_argument("--min-bookmarks", "-b", type=int, default=DEFAULT_MIN_BOOKMARKS,
                        help="Minimum bookmark count (quality filter)")
    parser.add_argument("--tag", default=DEFAULT_TAG, help="Tag to search")
    parser.add_argument("--delay-min", type=float, default=DEFAULT_DELAY_MIN, help="Minimum delay")
    parser.add_argument("--delay-max", type=float, default=DEFAULT_DELAY_MAX, help="Maximum delay")
    parser.add_argument("--sort", choices=["date_desc", "popular_desc"], default="popular_desc",
                        help="Sort order (popular_desc recommended for quality)")

    args = parser.parse_args()

    scrape_pixiv_novelai(
        refresh_token=args.token,
        output_dir=args.output,
        max_images=args.count,
        min_bookmarks=args.min_bookmarks,
        tag=args.tag,
        delay_min=args.delay_min,
        delay_max=args.delay_max,
        sort=args.sort,
    )


if __name__ == "__main__":
    main()
