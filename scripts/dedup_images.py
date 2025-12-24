#!/usr/bin/env python3
"""
pHashで類似画像を検出し、重複を削除/移動するスクリプト

Usage:
  python scripts/dedup_images.py --dir /path/to/images --threshold 16
  python scripts/dedup_images.py --dir /path/to/images --threshold 16 --move ./duplicates  # 重複を移動
  python scripts/dedup_images.py --dir /path/to/images --threshold 16 --delete  # 実際に削除
"""

import argparse
import os
import shutil
from pathlib import Path
from collections import defaultdict
from PIL import Image
import imagehash
from tqdm import tqdm


def get_image_files(directory: str) -> list[Path]:
    """画像ファイルを再帰的に取得"""
    extensions = {'.jpg', '.jpeg', '.png', '.webp', '.gif'}
    files = []
    for ext in extensions:
        files.extend(Path(directory).rglob(f'*{ext}'))
        files.extend(Path(directory).rglob(f'*{ext.upper()}'))
    return sorted(set(files))


def compute_phash(image_path: Path) -> imagehash.ImageHash | None:
    """pHashを計算"""
    try:
        with Image.open(image_path) as img:
            return imagehash.phash(img)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def find_duplicates(image_files: list[Path], threshold: int = 16) -> list[tuple[Path, Path, int]]:
    """
    類似画像のペアを検出
    threshold: ハッシュ差分の閾値（小さいほど厳しい）
      - 0: 完全一致
      - 8: ほぼ同一
      - 16: 差分画像も検出
      - 24: かなり緩い
    """
    print(f"Computing pHash for {len(image_files)} images...")

    # ハッシュを計算
    hashes = {}
    for f in tqdm(image_files, desc="Hashing"):
        h = compute_phash(f)
        if h is not None:
            hashes[f] = h

    print(f"Successfully hashed {len(hashes)} images")
    print(f"Finding duplicates with threshold={threshold}...")

    # 類似ペアを検出
    duplicates = []
    files = list(hashes.keys())

    for i in tqdm(range(len(files)), desc="Comparing"):
        for j in range(i + 1, len(files)):
            diff = hashes[files[i]] - hashes[files[j]]
            if diff <= threshold:
                duplicates.append((files[i], files[j], diff))

    return duplicates


def select_files_to_delete(duplicates: list[tuple[Path, Path, int]]) -> set[Path]:
    """
    重複グループから削除するファイルを選択
    各グループで1つだけ残す（ファイルサイズが大きい方を残す）
    """
    # Union-Find でグループ化
    parent = {}

    def find(x):
        if x not in parent:
            parent[x] = x
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for f1, f2, _ in duplicates:
        union(f1, f2)

    # グループごとにファイルを集める
    groups = defaultdict(list)
    for f in parent.keys():
        groups[find(f)].append(f)

    # 各グループで最大サイズのファイル以外を削除候補に
    to_delete = set()
    for group_files in groups.values():
        if len(group_files) > 1:
            # ファイルサイズでソート（大きい順）
            sorted_files = sorted(group_files, key=lambda f: f.stat().st_size, reverse=True)
            # 最大サイズ以外を削除候補に
            to_delete.update(sorted_files[1:])

    return to_delete


def main():
    parser = argparse.ArgumentParser(description="pHashで類似画像を検出・削除/移動")
    parser.add_argument("--dir", required=True, help="画像ディレクトリ")
    parser.add_argument("--threshold", type=int, default=16,
                        help="ハッシュ差分の閾値 (default: 16, 厳しめ)")
    parser.add_argument("--delete", action="store_true", help="実際に削除する")
    parser.add_argument("--move", help="重複ファイルを移動する先のディレクトリ")
    parser.add_argument("--output", help="削除候補リストを出力するファイル")
    args = parser.parse_args()

    # 画像ファイルを取得
    image_files = get_image_files(args.dir)
    print(f"Found {len(image_files)} image files in {args.dir}")

    if len(image_files) == 0:
        print("No images found.")
        return

    # 重複を検出
    duplicates = find_duplicates(image_files, args.threshold)
    print(f"\nFound {len(duplicates)} similar pairs")

    if len(duplicates) == 0:
        print("No duplicates found.")
        return

    # 削除候補を選択
    to_delete = select_files_to_delete(duplicates)
    print(f"\nFiles to delete: {len(to_delete)}")
    print(f"Files to keep: {len(image_files) - len(to_delete)}")

    # 削除候補リストを出力
    if args.output:
        with open(args.output, 'w') as f:
            for path in sorted(to_delete):
                f.write(f"{path}\n")
        print(f"\nDelete list saved to: {args.output}")

    # サンプル表示
    print("\n=== Sample duplicates (first 10) ===")
    for f1, f2, diff in duplicates[:10]:
        print(f"  diff={diff}: {f1.name} <-> {f2.name}")

    # 移動または削除
    if args.move:
        move_dir = Path(args.move)
        move_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nMoving {len(to_delete)} files to {move_dir}...")
        moved = 0
        for f in tqdm(to_delete, desc="Moving"):
            try:
                dest = move_dir / f.name
                # 同名ファイルがある場合はリネーム
                if dest.exists():
                    stem = dest.stem
                    suffix = dest.suffix
                    counter = 1
                    while dest.exists():
                        dest = move_dir / f"{stem}_{counter}{suffix}"
                        counter += 1
                shutil.move(str(f), str(dest))
                moved += 1
            except Exception as e:
                print(f"Failed to move {f}: {e}")
        print(f"Moved {moved} files to {move_dir}")
    elif args.delete:
        print(f"\nDeleting {len(to_delete)} files...")
        deleted = 0
        for f in tqdm(to_delete, desc="Deleting"):
            try:
                f.unlink()
                deleted += 1
            except Exception as e:
                print(f"Failed to delete {f}: {e}")
        print(f"Deleted {deleted} files")
    else:
        print(f"\n[DRY RUN] Would delete/move {len(to_delete)} files")
        print("Run with --move <dir> to move, or --delete to delete")


if __name__ == "__main__":
    main()
