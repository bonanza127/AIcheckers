#!/usr/bin/env python3
"""
v2パッチ統計で全embeddingを再抽出

元のファイルリスト(_files.txt)を使って、同じ画像セットを再抽出する。
CLSトークン(768d)は最終層から、パッチ統計(7d)は中間層(Block 8)から取得。
"""
import sys
import subprocess
from pathlib import Path

EMBEDDINGS_DIR = Path("/home/techne/aicheckers/embeddings")
CIVITAI_BASE = "/home/techne/aicheckers/data/animedl2m_dataset_release/civitai_subset/image"

# カテゴリとソースディレクトリのマッピング
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

def main():
    print("=== v2 Patch Stats Re-extraction ===")
    print("Using mid-layer (Block 8) for unsupervised statistics")
    print()
    
    for name, source_dir in CATEGORY_SOURCES.items():
        files_txt = EMBEDDINGS_DIR / f"{name}_files.txt"
        
        if not Path(source_dir).exists():
            print(f"❌ {name}: Source directory not found: {source_dir}")
            continue
        
        # 画像数を確認
        if files_txt.exists():
            with open(files_txt) as f:
                expected_count = len(f.readlines())
            print(f"📁 {name}: {expected_count} images from {source_dir}")
        else:
            print(f"📁 {name}: (new) from {source_dir}")
        
        # 再抽出コマンド
        cmd = [
            "python3", "scripts/extract_embeddings_v2.py",
            "--dir", source_dir,
            "--name", name,
            "--batch-size", "32"
        ]
        print(f"   Running...")
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd="/home/techne/aicheckers")
        if result.returncode == 0:
            # 結果の最後の数行を表示
            lines = result.stdout.strip().split('\n')
            for line in lines[-4:]:
                print(f"   {line}")
            print(f"   ✅ Done")
        else:
            print(f"   ❌ Failed: {result.stderr[:300]}")
        print()

if __name__ == "__main__":
    main()
