#!/usr/bin/env python3
"""
Restore embeddings/{category}_files.txt from embeddings/{category}_image_list.txt.

This is needed when files.txt was overwritten or truncated after raw_patches
were created. We require an exact length match by default.
"""
import argparse
from pathlib import Path

import numpy as np

EMBEDDINGS_DIR = Path("/home/techne/aicheckers/embeddings")


def load_list(path: Path) -> list[str]:
    with path.open("r") as f:
        return [line.strip() for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", required=True)
    parser.add_argument(
        "--allow-mismatch",
        action="store_true",
        help="Allow truncating to min length when counts differ (unsafe)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cat = args.category
    image_list_path = EMBEDDINGS_DIR / f"{cat}_image_list.txt"
    files_path = EMBEDDINGS_DIR / f"{cat}_files.txt"
    patches_path = EMBEDDINGS_DIR / f"{cat}_mid_patches.npy"

    if not image_list_path.exists():
        raise SystemExit(f"[ERROR] missing {image_list_path}")
    if not patches_path.exists():
        raise SystemExit(f"[ERROR] missing {patches_path}")

    image_list = load_list(image_list_path)
    patch_count = len(np.load(patches_path, mmap_mode="r"))

    if len(image_list) != patch_count:
        msg = f"[ERROR] {cat}: image_list={len(image_list)} vs patches={patch_count}"
        if not args.allow_mismatch:
            raise SystemExit(msg + " (use --allow-mismatch to truncate)")
        min_len = min(len(image_list), patch_count)
        print(msg)
        print(f"[WARN] truncating to {min_len} (unsafe)")
        image_list = image_list[:min_len]

    if args.dry_run:
        print(f"[DRY] would write {files_path} with {len(image_list)} lines")
        return

    files_path.write_text("\n".join(image_list) + "\n")
    print(f"[OK] wrote {files_path} ({len(image_list)} lines)")


if __name__ == "__main__":
    main()
