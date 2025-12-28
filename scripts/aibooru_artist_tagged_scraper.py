#!/usr/bin/env python3
"""
AIBooruからアーティストタグ付きNovelAI画像を収集
"""
import json
import random
import time
import argparse
from pathlib import Path
from datetime import datetime

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

# アーティストタグ付きNovelAI画像のクエリ
# クライアント側でアーティストタグあり画像のみフィルタ
SEARCH_TAGS = "novelai"  # NovelAI画像を全て取得

DEFAULT_OUTPUT_DIR = "/home/techne/aicheckers/data/novelai_artist_tagged"
DEFAULT_DELAY_MIN = 5
DEFAULT_DELAY_MAX = 12

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


def extract_artist_tags(post: dict) -> list:
    """投稿からアーティストタグを抽出"""
    tag_string = post.get("tag_string_artist", "")
    if tag_string:
        return tag_string.split()
    return []


def load_checkpoint(checkpoint_file: Path) -> dict:
    """チェックポイントを読み込み"""
    if not checkpoint_file.exists():
        return {
            "last_page": 0,
            "downloaded_count": 0,
            "downloaded_ids": set(),
            "last_updated": None
        }

    try:
        with open(checkpoint_file, "r") as f:
            data = json.load(f)
            # downloaded_ids を set に変換
            data["downloaded_ids"] = set(data.get("downloaded_ids", []))
            return data
    except Exception as e:
        print(f"[WARNING] Failed to load checkpoint: {e}")
        print("[WARNING] Starting from beginning")
        return {
            "last_page": 0,
            "downloaded_count": 0,
            "downloaded_ids": set(),
            "last_updated": None
        }


def save_checkpoint(checkpoint_file: Path, checkpoint: dict):
    """チェックポイントを保存"""
    try:
        # set を list に変換して保存
        data = checkpoint.copy()
        data["downloaded_ids"] = list(data["downloaded_ids"])
        data["last_updated"] = datetime.now().isoformat()

        # 一時ファイルに書き込んでからリネーム（原子性）
        temp_file = checkpoint_file.with_suffix(".tmp")
        with open(temp_file, "w") as f:
            json.dump(data, f, indent=2)
        temp_file.rename(checkpoint_file)
    except Exception as e:
        print(f"[WARNING] Failed to save checkpoint: {e}")


def main():
    parser = argparse.ArgumentParser(description="AIBooruからアーティストタグ付きNovelAI画像を収集")
    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT_DIR, help="出力ディレクトリ")
    parser.add_argument("--limit", "-l", type=int, default=5000, help="最大収集数")
    parser.add_argument("--delay-min", type=float, default=DEFAULT_DELAY_MIN, help="最小遅延(秒)")
    parser.add_argument("--delay-max", type=float, default=DEFAULT_DELAY_MAX, help="最大遅延(秒)")
    parser.add_argument("--dry-run", action="store_true", help="ドライラン（ダウンロードしない）")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # メタデータ保存用
    metadata_file = output_dir / "artist_metadata.jsonl"
    checkpoint_file = output_dir / "checkpoint.json"

    # チェックポイント読み込み
    checkpoint = load_checkpoint(checkpoint_file)

    print(f"{'='*60}")
    print(f"AIBooru Artist-Tagged NovelAI Scraper")
    print(f"{'='*60}")
    print(f"Search tags: {SEARCH_TAGS}")
    print(f"Output directory: {output_dir}")
    print(f"Target limit: {args.limit}")
    print(f"Delay: {args.delay_min}-{args.delay_max}s")
    if args.dry_run:
        print("[DRY RUN MODE]")

    # リジューム情報
    if checkpoint["last_page"] > 0:
        print(f"\n{'='*60}")
        print(f"RESUMING FROM CHECKPOINT")
        print(f"{'='*60}")
        print(f"Last page: {checkpoint['last_page']}")
        print(f"Already downloaded: {checkpoint['downloaded_count']} images")
        print(f"Last updated: {checkpoint['last_updated']}")
        print(f"Starting from page: {checkpoint['last_page'] + 1}")
    print()

    # セッション作成
    session = Session(impersonate="chrome110")

    # まず総数を確認
    print("Checking total available posts...")
    first_page = get_posts(session, SEARCH_TAGS, page=1, limit=1)
    if not first_page:
        print("Failed to fetch data. Exiting.")
        return

    print(f"✓ API accessible\n")

    # 収集開始
    downloaded = checkpoint["downloaded_count"]
    skipped = 0
    failed = 0
    page = checkpoint["last_page"] + 1  # 前回の続きから
    posts_per_page = 100

    with open(metadata_file, "a", encoding="utf-8") as meta_f:
        while downloaded < args.limit:
            print(f"\nFetching page {page}...")
            posts = get_posts(session, SEARCH_TAGS, page=page, limit=posts_per_page)

            if not posts:
                print("No more posts available.")
                break

            for post in tqdm(posts, desc=f"Page {page}"):
                if downloaded >= args.limit:
                    break

                post_id = post.get("id")
                file_url = post.get("file_url")
                file_ext = post.get("file_ext", "jpg")

                if not file_url:
                    continue

                # 既にダウンロード済みならスキップ
                if post_id in checkpoint["downloaded_ids"]:
                    skipped += 1
                    continue

                # アーティストタグを抽出
                artist_tags = extract_artist_tags(post)

                # アーティストタグがない場合はスキップ
                if not artist_tags:
                    skipped += 1
                    continue

                output_path = output_dir / f"{post_id}.{file_ext}"

                if output_path.exists():
                    skipped += 1
                    checkpoint["downloaded_ids"].add(post_id)
                    continue

                # メタデータ保存
                metadata = {
                    "id": post_id,
                    "file_url": file_url,
                    "artist_tags": artist_tags,
                    "tag_string": post.get("tag_string", ""),
                    "score": post.get("score", 0),
                }
                meta_f.write(json.dumps(metadata, ensure_ascii=False) + "\n")
                meta_f.flush()

                if args.dry_run:
                    print(f"[DRY RUN] Would download: {post_id} (artists: {', '.join(artist_tags)})")
                    downloaded += 1
                    checkpoint["downloaded_ids"].add(post_id)
                    checkpoint["downloaded_count"] = downloaded
                    continue

                # ダウンロード
                random_delay(args.delay_min, args.delay_max)

                if download_image(session, file_url, output_path):
                    downloaded += 1
                    checkpoint["downloaded_ids"].add(post_id)
                    checkpoint["downloaded_count"] = downloaded
                    if artist_tags:
                        print(f"✓ {post_id}.{file_ext} (artists: {', '.join(artist_tags[:3])})")
                else:
                    failed += 1

            # ページ完了後、チェックポイント保存
            checkpoint["last_page"] = page
            save_checkpoint(checkpoint_file, checkpoint)
            print(f"[CHECKPOINT] Saved (page {page}, {downloaded} downloaded)")

            page += 1

    print(f"\n{'='*60}")
    print(f"COMPLETE")
    print(f"{'='*60}")
    print(f"Downloaded: {downloaded}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")
    print(f"Output: {output_dir}")
    print(f"Metadata: {metadata_file}")


if __name__ == "__main__":
    main()
