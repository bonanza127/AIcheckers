#!/usr/bin/env python3
"""
AIBooru NovelAI画像スクレイパー
curl_cffiを使用してCloudflare回避
"""

import json
import random
import time
import argparse
from pathlib import Path

try:
    from curl_cffi import requests
    from curl_cffi.requests import Session
except ImportError:
    print("curl_cffi is required. Install with: pip install curl_cffi")
    raise

from tqdm import tqdm


# 設定
BASE_URL = "https://aibooru.online"
API_URL = f"{BASE_URL}/posts.json"
DEFAULT_OUTPUT_DIR = "/home/techne/aicheckers/data/novelai"
DEFAULT_DELAY_MIN = 5
DEFAULT_DELAY_MAX = 12

# リアルなヘッダー
HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,ja;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "cross-site",
    "Sec-Fetch-User": "?1",
    "Referer": "https://www.google.com/",
}


def random_delay(min_sec: float, max_sec: float):
    """ランダムな遅延"""
    delay = random.uniform(min_sec, max_sec)
    if random.random() < 0.1:
        delay += random.uniform(5, 10)
    time.sleep(delay)


def get_posts(session: Session, tags: str, page: int = 1, limit: int = 20) -> list:
    """APIから投稿一覧を取得"""
    # スペースを+に変換（URLエンコードではなく）
    encoded_tags = tags.replace(" ", "+")
    url = f"{API_URL}?tags={encoded_tags}&page={page}&limit={limit}"

    try:
        headers = HEADERS.copy()
        headers["Accept"] = "application/json"
        headers["Referer"] = BASE_URL

        response = session.get(url, headers=headers, timeout=30)

        if response.status_code == 200:
            return response.json()
        else:
            print(f"[ERROR] API returned {response.status_code}")
            return []
    except Exception as e:
        print(f"[ERROR] API request failed: {e}")
        return []


def download_image(session: Session, url: str, output_path: Path) -> bool:
    """画像をダウンロード"""
    if output_path.exists():
        return True

    temp_path = output_path.with_suffix(".tmp")
    try:
        response = session.get(url, headers=HEADERS, timeout=60)

        if response.status_code == 200:
            with open(temp_path, "wb") as f:
                f.write(response.content)
            temp_path.rename(output_path)
            return True
        else:
            print(f"[ERROR] Download failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"[ERROR] Download exception: {e}")
        if temp_path.exists():
            temp_path.unlink()
        return False


def scrape_novelai(
    output_dir: str = DEFAULT_OUTPUT_DIR,
    max_images: int = 1000,
    start_page: int = 1,
    delay_min: float = DEFAULT_DELAY_MIN,
    delay_max: float = DEFAULT_DELAY_MAX,
    rating: str = "g,s",
    skip: int = 1,  # N個おきに取得（1=全部、10=10個飛ばし）
    min_score: int = 0,  # 最低スコア（0=フィルターなし）
):
    """NovelAI画像をスクレイピング"""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metadata_file = output_path / "metadata.txt"

    print(f"[INFO] Output directory: {output_path}")
    print(f"[INFO] Target: {max_images} images")
    print(f"[INFO] Delay: {delay_min}-{delay_max} seconds")
    print(f"[INFO] Rating filter: {rating}")
    print(f"[INFO] Min score: {min_score}" if min_score > 0 else "[INFO] Score filter: disabled")
    print(f"[INFO] Skip: every {skip} posts (diversity mode)")
    print(f"[INFO] Using curl_cffi with Chrome impersonation")
    print()

    # curl_cffiセッション作成
    session = Session(impersonate="chrome")

    downloaded = 0
    page = start_page
    errors_in_a_row = 0
    max_errors = 5

    # 既存ファイル
    existing_files = list(output_path.glob("*.png")) + list(output_path.glob("*.jpg")) + list(output_path.glob("*.webp"))
    print(f"[INFO] Existing files: {len(existing_files)}")

    existing_ids = set()
    for f in existing_files:
        try:
            existing_ids.add(int(f.stem))
        except ValueError:
            pass

    with open(metadata_file, "a", encoding="utf-8") as meta_f:
        with tqdm(total=max_images, desc="Downloading", unit="img") as pbar:
            while downloaded < max_images:
                if page > start_page:
                    random_delay(delay_min, delay_max)

                # タグ構築（単一ratingのみサポート、複数は最初の1つを使用）
                first_rating = rating.split(",")[0] if rating else "s"
                tags = f"novelai rating:{first_rating}"
                if min_score > 0:
                    tags += f" score:>={min_score}"
                posts = get_posts(session, tags, page=page, limit=20)

                if not posts:
                    errors_in_a_row += 1
                    print(f"\n[WARN] Empty response on page {page} (error {errors_in_a_row}/{max_errors})")

                    if errors_in_a_row >= max_errors:
                        print("[ERROR] Too many errors, stopping")
                        break

                    time.sleep(30 + random.uniform(0, 30))
                    page += 1
                    continue

                errors_in_a_row = 0
                skip_counter = 0

                for post in posts:
                    if downloaded >= max_images:
                        break

                    # N個おきに取得
                    skip_counter += 1
                    if skip_counter % skip != 0:
                        continue

                    post_id = post.get("id")
                    if post_id in existing_ids:
                        continue

                    image_url = post.get("large_file_url") or post.get("file_url")
                    if not image_url:
                        continue

                    file_ext = post.get("file_ext", "png")
                    filename = f"{post_id}.{file_ext}"
                    filepath = output_path / filename

                    random_delay(delay_min / 2, delay_max / 2)

                    if download_image(session, image_url, filepath):
                        tags = post.get("tag_string", "")
                        source = post.get("source", "")
                        meta_f.write(f"{filename}\t{post_id}\t{source}\t{tags}\n")
                        meta_f.flush()

                        existing_ids.add(post_id)
                        downloaded += 1
                        pbar.update(1)

                page += 1

                if page % 10 == 0:
                    print(f"\n[INFO] Page {page} reached, taking a break...")
                    time.sleep(random.uniform(30, 60))

    print(f"\n[DONE] Downloaded {downloaded} images to {output_path}")
    return downloaded


def main():
    parser = argparse.ArgumentParser(description="AIBooru NovelAI Scraper (Cloudflare Bypass)")
    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--count", "-n", type=int, default=1000, help="Number of images to download")
    parser.add_argument("--start-page", type=int, default=1, help="Starting page number")
    parser.add_argument("--delay-min", type=float, default=DEFAULT_DELAY_MIN, help="Minimum delay")
    parser.add_argument("--delay-max", type=float, default=DEFAULT_DELAY_MAX, help="Maximum delay")
    parser.add_argument("--rating", default="g,s", help="Rating filter (g,s,q,e)")
    parser.add_argument("--skip", type=int, default=1, help="Take every Nth post (1=all, 10=every 10th)")
    parser.add_argument("--score", type=int, default=0, help="Minimum score filter (0=disabled, 20=recommended)")

    args = parser.parse_args()

    scrape_novelai(
        output_dir=args.output,
        max_images=args.count,
        start_page=args.start_page,
        delay_min=args.delay_min,
        delay_max=args.delay_max,
        rating=args.rating,
        skip=args.skip,
        min_score=args.score,
    )


if __name__ == "__main__":
    main()
