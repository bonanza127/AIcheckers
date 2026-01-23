#!/usr/bin/env python3
"""
CPU Stats v3 20d 抽出（unified→20d変換、並列処理対応）

28dで使用: CPU20_11D_IDX = [0,1,2,3,4,5,8,10,15,16,17]
  0: histogram_modality
  1: color_palette_entropy
  2: luminance_layer_count
  3: luminance_skewness
  4: value_bimodality
  5: multiscale_variance_ratio
  8: luminance_mean
  10: saturation_mean
  15: radial_spectrum_slope_patch_gap
  16: color_banding_score
  17: compression_artifact_pattern
"""
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from tqdm import tqdm

REPO_ROOT = Path("/home/techne/aicheckers")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.extract_cpu_stats_v3_all import load_image, extract_unified

EMB_DIR = REPO_ROOT / "embeddings"
DATA_DIR = REPO_ROOT / "data/aibooru_new"
CATEGORY = "aibooru_new_ai"

# unified(27d) → 20d mapping (lib/cpu_stats.pyと同一)
UNIFIED_TO_20D_IDX = [1, 2, 3, 6, 8, 9, 10, 11, 12, 13, 14, 16, 18, 19, 20, 22, 23, 24, 25, 26]

def process_single(path):
    try:
        img = load_image(path)
        unified = extract_unified(img)
        return unified[UNIFIED_TO_20D_IDX]
    except:
        return np.zeros(20, dtype=np.float32)

def main():
    files_path = EMB_DIR / f"{CATEGORY}_files.txt"
    with open(files_path) as f:
        files = [line.strip() for line in f if line.strip()]

    paths = [DATA_DIR / f for f in files]
    print(f"Processing {len(paths)} files with 16 workers (unified→20d)...")

    with ThreadPoolExecutor(max_workers=16) as executor:
        results = list(tqdm(executor.map(process_single, paths), total=len(paths)))

    arr = np.stack(results)
    np.save(EMB_DIR / f"{CATEGORY}_cpu_stats_v3_20d.npy", arr)
    print(f"Saved: {CATEGORY}_cpu_stats_v3_20d.npy ({arr.shape})")

if __name__ == "__main__":
    main()
