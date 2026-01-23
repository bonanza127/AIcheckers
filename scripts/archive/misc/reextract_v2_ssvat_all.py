#!/usr/bin/env python3
"""
SS-VAT向け: 中間層パッチ埋め込みを含めて全embeddingを再抽出

- extract_embeddings_v2_ssvat.py を使用
- mid-layer=6（デフォルト）
"""
import os
import subprocess
from pathlib import Path

EMBEDDINGS_DIR = Path("/home/techne/aicheckers/embeddings")
CIVITAI_BASE = "/home/techne/aicheckers/data/animedl2m_dataset_release/civitai_subset/image"

CATEGORY_SOURCES = {
    # civitai_subset からのAIデータ
    "pony_ai": f"{CIVITAI_BASE}/Pony",
    "illustrious_ai": f"{CIVITAI_BASE}/Illustrious",
    "sdxl10_ai": f"{CIVITAI_BASE}/SDXL 1.0",
    "sd15_ai": f"{CIVITAI_BASE}/SD 1.5",
    "flux1d_ai": f"{CIVITAI_BASE}/Flux.1 D",
    "other_ai": f"{CIVITAI_BASE}/Other",
    # NovelAI関連
    "novelai_ai": "/home/techne/aicheckers/data/novelai",
    "novelai_combined_ai": "/home/techne/aicheckers/data/novelai_combined",
    "novelai_artist_tagged_ai": "/home/techne/aicheckers/data/novelai_artist_tagged",
    "pixai_ai": "/home/techne/aicheckers/data/pixai",
    # Real データ
    "danbooru_real": "/home/techne/aicheckers/data/animedl2m_dataset_release/real_images/images",
}


def main() -> None:
    print("=== SS-VAT Re-extraction (mid patches) ===")
    print("Using mid-layer 6, saving mid patches (default float16)")
    print()

    lock_path = EMBEDDINGS_DIR / "ssvat_reextract.lock"
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        print(f"Lock exists: {lock_path}. Another reextract may be running.")
        return

    for name, source_dir in CATEGORY_SOURCES.items():
        files_txt = EMBEDDINGS_DIR / f"{name}_files.txt"

        if not Path(source_dir).exists():
            print(f"❌ {name}: Source directory not found: {source_dir}")
            continue

        if files_txt.exists():
            with open(files_txt) as f:
                expected_count = len(f.readlines())
            print(f"📁 {name}: {expected_count} images from {source_dir}")
        else:
            print(f"📁 {name}: (new) from {source_dir}")

        cmd = [
            "python3", "scripts/extract_embeddings_v2_ssvat.py",
            "--dir", source_dir,
            "--name", name,
            "--batch-size", "4",
            "--mid-layer", "6",
            "--mid-dtype", "float16",
            "--seed", "42",
        ]
        if name == "sd15_ai":
            cmd += ["--limit", "10000"]
        print("   Running...")
        result = subprocess.run(cmd, text=True, cwd="/home/techne/aicheckers")
        if result.returncode == 0:
            print("   ✅ Done")
        else:
            print("   ❌ Failed")
        print()
    if lock_path.exists():
        try:
            lock_path.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    main()
