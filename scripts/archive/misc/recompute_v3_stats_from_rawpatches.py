#!/usr/bin/env python3
"""
rawpatches (_mid_patches.npy) から v3 統計量を再計算

KNN_K=4 に変更したので全カテゴリ再計算
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from tqdm import tqdm
from lib.patch_stats import compute_patch_stats_v3, KNN_K

EMBEDDINGS_DIR = Path("/home/techne/aicheckers/embeddings")

CATEGORIES = [
    "illustrious_ai", "pony_ai", "sdxl10_ai", "sd15_ai", "other_ai",
    "flux1d_ai", "novelai_ai", "pixai_ai", "novelai_combined_ai",
    "novelai_artist_tagged_ai", "danbooru_real"
]

BATCH_SIZE = 256  # GPU batch size

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"KNN_K: {KNN_K}")

for cat in CATEGORIES:
    patches_path = EMBEDDINGS_DIR / f"{cat}_mid_patches.npy"
    output_path = EMBEDDINGS_DIR / f"{cat}_patch_stats_v3.npy"

    if not patches_path.exists():
        print(f"Skip {cat}: no mid_patches")
        continue

    print(f"\nProcessing {cat}...")
    patches = np.load(patches_path, mmap_mode='r')
    n_samples = len(patches)
    print(f"  {n_samples} samples")

    all_stats = []
    n_batches = (n_samples + BATCH_SIZE - 1) // BATCH_SIZE

    for i in tqdm(range(n_batches), desc=cat):
        start = i * BATCH_SIZE
        end = min(start + BATCH_SIZE, n_samples)
        batch = torch.tensor(patches[start:end], device=device, dtype=torch.float32)

        with torch.no_grad():
            stats = compute_patch_stats_v3(batch)  # (batch, 33)

        all_stats.append(stats.cpu().numpy())

        # GPU memory cleanup
        del batch, stats
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    result = np.concatenate(all_stats, axis=0).astype(np.float32)
    np.save(output_path, result)
    print(f"  Saved: {output_path} ({result.shape})")

print("\nDone!")
