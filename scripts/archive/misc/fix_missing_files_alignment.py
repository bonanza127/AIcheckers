#!/usr/bin/env python3
"""
Fix files list alignment when some files are missing on disk.

Logic:
  - If patches_count == existing_count: rewrite files.txt to only existing files.
  - If patches_count == files_count: drop missing entries from files.txt AND
    filter patches to match (unsafe but consistent).
  - Otherwise: abort.
"""
import argparse
from pathlib import Path

import numpy as np

EMBEDDINGS_DIR = Path("/home/techne/aicheckers/embeddings")
DATA_ROOT = Path("/home/techne/aicheckers/data")
ANIMEDL_ROOT = DATA_ROOT / "animedl2m_dataset_release"

CATEGORY_PATHS = {
    "illustrious_ai": ANIMEDL_ROOT / "civitai_subset/image/Illustrious",
    "pony_ai": ANIMEDL_ROOT / "civitai_subset/image/Pony",
    "sdxl10_ai": ANIMEDL_ROOT / "civitai_subset/image/SDXL 1.0",
    "sd15_ai": ANIMEDL_ROOT / "civitai_subset/image/SD 1.5",
    "other_ai": ANIMEDL_ROOT / "civitai_subset/image/Other",
    "flux1d_ai": ANIMEDL_ROOT / "civitai_subset/image/Flux.1 D",
    "novelai_ai": DATA_ROOT / "novelai",
    "pixai_ai": DATA_ROOT / "pixai",
    "novelai_combined_ai": DATA_ROOT / "novelai_combined",
    "novelai_artist_tagged_ai": DATA_ROOT / "novelai_artist_tagged",
    "danbooru_real": ANIMEDL_ROOT / "real_images/images",
}


def load_lines(path: Path) -> list[str]:
    with path.open("r") as f:
        return [line.strip() for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cat = args.category
    files_path = EMBEDDINGS_DIR / f"{cat}_files.txt"
    patches_path = EMBEDDINGS_DIR / f"{cat}_mid_patches.npy"

    if not files_path.exists():
        raise SystemExit(f"[ERROR] missing {files_path}")
    if not patches_path.exists():
        raise SystemExit(f"[ERROR] missing {patches_path}")

    files = load_lines(files_path)
    patches = np.load(patches_path, mmap_mode="r")
    patches_count = len(patches)

    base_dir = CATEGORY_PATHS.get(cat)
    if base_dir is None:
        raise SystemExit(f"[ERROR] unknown category: {cat}")

    file_paths = []
    for p in files:
        path = Path(p)
        if not path.is_absolute():
            path = base_dir / path
        file_paths.append(path)

    keep_mask = [p.exists() for p in file_paths]
    existing_files = [f for f, keep in zip(files, keep_mask) if keep]

    files_count = len(files)
    existing_count = len(existing_files)

    print(f"[INFO] {cat}: files={files_count}, existing={existing_count}, patches={patches_count}")
    if existing_count == files_count:
        print("[OK] no missing files detected")
        return

    if patches_count == existing_count:
        if args.dry_run:
            print(f"[DRY] would rewrite {files_path} to {existing_count} lines")
            return
        backup = files_path.with_suffix(".txt.bak")
        files_path.rename(backup)
        files_path.write_text("\n".join(existing_files) + "\n")
        print(f"[OK] rewrote files list (backup: {backup.name})")
        return

    if patches_count == files_count:
        if args.dry_run:
            print(f"[DRY] would filter patches to {existing_count} and rewrite files list")
            return
        backup = patches_path.with_suffix(".npy.bak")
        files_backup = files_path.with_suffix(".txt.bak")
        patches_path.rename(backup)
        filtered = np.asarray(patches)[keep_mask]
        np.save(patches_path, filtered.astype(np.float16))
        files_path.rename(files_backup)
        files_path.write_text("\n".join(existing_files) + "\n")
        print(f"[OK] filtered patches and files (backup: {backup.name})")
        return

    raise SystemExit(
        f"[ERROR] {cat}: incompatible counts (files={files_count}, "
        f"existing={existing_count}, patches={patches_count})"
    )


if __name__ == "__main__":
    main()
