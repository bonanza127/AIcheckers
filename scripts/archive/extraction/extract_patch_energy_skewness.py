#!/usr/bin/env python3
"""
patch_energy_skewness を既存の mid_patches から計算して保存

Usage:
    python3 scripts/extract_patch_energy_skewness.py
"""
import numpy as np
from pathlib import Path
from scipy import stats as scipy_stats
from tqdm import tqdm

EMBEDDINGS_DIR = Path("/home/techne/aicheckers/embeddings")

# All categories
AI_CATS = [
    "illustrious_ai", "pony_ai", "sdxl10_ai", "sd15_ai", "other_ai",
    "flux1d_ai", "novelai_ai", "pixai_ai", "novelai_combined_ai",
    "novelai_artist_tagged_ai", "pixiv_novelai_v2_ai", "twitter_novelai_v2_ai",
    "novelai_aibooru_ai"
]
REAL_CATS = ["danbooru_real"]

ALL_CATS = AI_CATS + REAL_CATS


def compute_patch_energy_skewness(patches: np.ndarray) -> np.ndarray:
    """
    パッチのエネルギー分布の歪度を計算

    Args:
        patches: (N, 196, 768) array

    Returns:
        (N,) array of skewness values
    """
    # Cast to float64 to avoid overflow
    patches_f64 = patches.astype(np.float64)

    # パッチごとのエネルギー（ノルムの2乗）
    patch_energy = np.sum(patches_f64 ** 2, axis=-1)  # (N, 196)

    # Log1p transform to reduce dynamic range
    patch_energy = np.log1p(patch_energy)

    # 各サンプルの歪度を計算
    skewness = scipy_stats.skew(patch_energy, axis=1)  # (N,)

    return skewness.astype(np.float32)


def main():
    for cat in ALL_CATS:
        patches_path = EMBEDDINGS_DIR / f"{cat}_mid_patches.npy"
        output_path = EMBEDDINGS_DIR / f"{cat}_energy_skewness.npy"

        if not patches_path.exists():
            print(f"Skip {cat}: patches not found")
            continue

        # Overwrite existing files (re-extraction with fixed calculation)
        # if output_path.exists():
        #     print(f"Skip {cat}: already exists")
        #     continue

        print(f"Processing {cat}...")
        patches = np.load(patches_path, mmap_mode='r')

        # バッチ処理
        batch_size = 1000
        n_samples = len(patches)
        all_skewness = []

        for i in tqdm(range(0, n_samples, batch_size), desc=cat):
            batch = patches[i:i+batch_size]
            skewness = compute_patch_energy_skewness(batch)
            all_skewness.append(skewness)

        result = np.concatenate(all_skewness).astype(np.float32)
        np.save(output_path, result)
        print(f"  Saved: {output_path} ({len(result)} samples)")


if __name__ == "__main__":
    main()
