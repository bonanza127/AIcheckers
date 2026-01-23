#!/usr/bin/env python3
"""
v2パッチ統計で漏れていたカテゴリのembeddingを追加抽出
"""
import sys
import subprocess
from pathlib import Path

EMBEDDINGS_DIR = Path("/home/techne/aicheckers/embeddings")
CIVITAI_BASE = "/home/techne/aicheckers/data/animedl2m_dataset_release/civitai_subset/image"

# 漏れていたカテゴリ
CATEGORY_SOURCES = {
    "flux1s_ai": f"{CIVITAI_BASE}/Flux.1 S",
    "sd14_ai": f"{CIVITAI_BASE}/SD 1.4",
    "sd15_hyper_ai": f"{CIVITAI_BASE}/SD 1.5 Hyper",
    "sd15_lcm_ai": f"{CIVITAI_BASE}/SD 1.5 LCM",
    "sd20_ai": f"{CIVITAI_BASE}/SD 2.0",
    "sd20_768_ai": f"{CIVITAI_BASE}/SD 2.0 768",
    "sd21_ai": f"{CIVITAI_BASE}/SD 2.1",
    "sd21_768_ai": f"{CIVITAI_BASE}/SD 2.1 768",
    "sdxl09_ai": f"{CIVITAI_BASE}/SDXL 0.9",
    "sdxl10_lcm_ai": f"{CIVITAI_BASE}/SDXL 1.0 LCM",
    "sdxl_hyper_ai": f"{CIVITAI_BASE}/SDXL Hyper",
    "sdxl_lightning_ai": f"{CIVITAI_BASE}/SDXL Lightning",
    "sdxl_turbo_ai": f"{CIVITAI_BASE}/SDXL Turbo",
}

def main():
    print("=== v2 Patch Stats Supplement Re-extraction ===")
    print("Extracting missing categories...")
    print()
    
    for name, source_dir in CATEGORY_SOURCES.items():
        if not Path(source_dir).exists():
            print(f"❌ {name}: Source directory not found: {source_dir}")
            continue
            
        print(f"📁 {name}: from {source_dir}")
        
        # 再抽出コマンド
        cmd = [
            "python3", "scripts/extract_embeddings_v2.py",
            "--dir", source_dir,
            "--name", name,
            "--batch-size", "32"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd="/home/techne/aicheckers")
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in lines[-4:]:
                print(f"   {line}")
            print(f"   ✅ Done")
        else:
            print(f"   ❌ Failed: {result.stderr[:300]}")
        print()

if __name__ == "__main__":
    main()
