#!/usr/bin/env python3
"""
PixAI Model Gallery Extractor
指定したmodelIdに関連付けられたパブリックな生成物を一括取得
"""
import requests
import os
import time
import argparse
from pathlib import Path

# デフォルト設定
DEFAULT_OUTPUT_DIR = "/home/techne/aicheckers/data/pixai"
LIMIT_PER_PAGE = 30

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Content-Type": "application/json",
}


def fetch_model_gallery(model_id: str, cursor: str = None, use_lora_filter: bool = False):
    """モデルに関連付けられた生成物を取得

    Args:
        model_id: PixAIのモデル/LoRA ID
        cursor: ページネーション用カーソル
        use_lora_filter: Trueの場合loraIdでフィルタ（そのLoRAを使った作品のみ）
                        Falseの場合modelIdでフィルタ（モデルページの作品）
    """
    url = "https://api.pixai.art/graphql"

    if use_lora_filter:
        query = """
        query GetLoraArtworks($id: ID!, $first: Int!, $after: String) {
          artworks(first: $first, loraId: $id, after: $after) {
            edges {
              node {
                id
                media {
                  urls {
                    variant
                    url
                  }
                }
              }
            }
            pageInfo {
              hasNextPage
              endCursor
            }
          }
        }
        """
    else:
        query = """
        query GetModelArtworks($id: ID!, $first: Int!, $after: String) {
          artworks(first: $first, modelId: $id, after: $after) {
            edges {
              node {
                id
                media {
                  urls {
                    variant
                    url
                  }
                }
              }
            }
            pageInfo {
              hasNextPage
              endCursor
            }
          }
        }
        """

    variables = {
        "id": model_id,
        "first": LIMIT_PER_PAGE,
        "after": cursor
    }

    try:
        response = requests.post(url, json={'query': query, 'variables': variables}, headers=HEADERS, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"[ERROR] API request failed: {e}")
        return None


def download_pixai_images(model_id: str, output_dir: str, max_images: int = None, delay: float = 0.5, use_lora_filter: bool = False):
    """PixAI モデルギャラリーから画像をダウンロード

    Args:
        model_id: PixAIのモデル/LoRA ID
        output_dir: 出力ディレクトリ
        max_images: 最大ダウンロード数
        delay: ダウンロード間隔（秒）
        use_lora_filter: Trueの場合loraIdでフィルタ（そのLoRAを使った作品のみ）
    """
    prefix = "lora" if use_lora_filter else "model"
    save_dir = Path(output_dir) / f"{prefix}_{model_id}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # 既存ファイルをチェック
    existing = {f.stem for f in save_dir.glob("*.jpg")}
    existing.update({f.stem for f in save_dir.glob("*.png")})
    existing.update({f.stem for f in save_dir.glob("*.webp")})
    print(f"[INFO] Output: {save_dir}")
    print(f"[INFO] Existing files: {len(existing)}")
    print(f"[INFO] Filter mode: {'loraId' if use_lora_filter else 'modelId'}")

    cursor = None
    count = 0
    skipped = 0

    while True:
        print(f"[INFO] Fetching page (cursor: {cursor[:20] + '...' if cursor else 'None'})")
        res = fetch_model_gallery(model_id, cursor, use_lora_filter)

        if not res or 'data' not in res or not res['data']['artworks']:
            print("[INFO] Finished or Error.")
            break

        data = res['data']['artworks']
        edges = data.get('edges', [])

        if not edges:
            print("[INFO] No more artworks.")
            break

        for edge in edges:
            if max_images and count >= max_images:
                print(f"[INFO] Reached max_images limit: {max_images}")
                return count

            try:
                art = edge['node']
                img_id = art['id']

                # 既に存在する場合はスキップ
                if img_id in existing:
                    skipped += 1
                    continue

                # 'PUBLIC' (フルサイズ) URLを選択、なければ最初のURL
                media_urls = art['media']['urls']
                img_url = next(
                    (u['url'] for u in media_urls if u['variant'] == 'PUBLIC'),
                    media_urls[0]['url'] if media_urls else None
                )

                if not img_url:
                    print(f"[SKIP] {img_id}: No URL found")
                    continue

                # URLから拡張子を推測
                ext = ".jpg"  # デフォルト
                if ".png" in img_url.lower():
                    ext = ".png"
                elif ".webp" in img_url.lower():
                    ext = ".webp"

                file_path = save_dir / f"{img_id}{ext}"

                img_data = requests.get(img_url, timeout=30).content
                with open(file_path, 'wb') as f:
                    f.write(img_data)

                count += 1
                print(f"[{count}] Downloaded {img_id}")
                time.sleep(delay)

            except Exception as e:
                print(f"[SKIP] {edge.get('node', {}).get('id', 'unknown')}: {e}")

        # ページネーション
        page_info = data.get('pageInfo', {})
        if not page_info.get('hasNextPage'):
            print("[INFO] No more pages.")
            break
        cursor = page_info.get('endCursor')

    print(f"\n[DONE] Downloaded: {count}, Skipped (existing): {skipped}")
    return count


def main():
    parser = argparse.ArgumentParser(description="PixAI Model Gallery Extractor")
    parser.add_argument("model_id", help="PixAI model ID (e.g., 1815523442272197668)")
    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--max", "-n", type=int, default=None, help="Maximum images to download")
    parser.add_argument("--delay", "-d", type=float, default=0.3, help="Delay between downloads (seconds)")
    parser.add_argument("--lora", "-l", action="store_true", help="Use loraId filter instead of modelId (gets artworks using this LoRA)")

    args = parser.parse_args()

    print(f"[INFO] Model ID: {args.model_id}")
    print(f"[INFO] Max images: {args.max or 'unlimited'}")
    print(f"[INFO] Mode: {'loraId' if args.lora else 'modelId'}")

    download_pixai_images(
        model_id=args.model_id,
        output_dir=args.output,
        max_images=args.max,
        delay=args.delay,
        use_lora_filter=args.lora
    )


if __name__ == "__main__":
    main()
