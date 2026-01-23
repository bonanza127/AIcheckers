
import numpy as np
from pathlib import Path

def check_stats_distribution():
    real_path = Path("/home/techne/aicheckers/embeddings/danbooru_real_patch_stats.npy")
    test_real_path = Path("/home/techne/aicheckers/embeddings/danbooru_real_test_v2_patch_stats.npy")
    ai_path = Path("/home/techne/aicheckers/embeddings/illustrious_ai_patch_stats.npy")
    
    # Inference stats from debug log (Step 285 - Noise Image)
    # [DEBUG] Stats: [0.8203, 0.0086, 0.2060, 5.3308, 0.0078, 34.8393, 29.3195]
    inf_noise = np.array([0.8203, 0.0086, 0.2060, 5.3308, 0.0078, 34.8393, 29.3195])
    
    # Inference stats from debug log (Step 275 - AI Image)
    # [DEBUG] Stats: [0.6814, 0.0313, 0.0824, 15.5360, 0.0508, 255.8752, 84.0135]
    inf_ai = np.array([0.6814, 0.0313, 0.0824, 15.5360, 0.0508, 255.8752, 84.0135])

    print("=== Training Data Stats (Real: danbooru) ===")
    if real_path.exists():
        real_stats = np.load(real_path)
        print(f"Shape: {real_stats.shape}")
        print(f"Mean: {real_stats.mean(axis=0)}")
        print(f"Std:  {real_stats.std(axis=0)}")
        print(f"Max:  {real_stats.max(axis=0)}")
    else:
        print("Real stats file not found.")

    print("\n=== Test Data Stats (Real: 100 images) ===")
    if test_real_path.exists():
        test_stats = np.load(test_real_path)
        print(f"Shape: {test_stats.shape}")
        print(f"Mean: {test_stats.mean(axis=0)}")
        print(f"Std:  {test_stats.std(axis=0)}")
    else:
        print("Test stats file not found.")

    print("\n=== Training Data Stats (AI: illustrious) ===")
    if ai_path.exists():
        ai_stats = np.load(ai_path)
        print(f"Shape: {ai_stats.shape}")
        print(f"Mean: {ai_stats.mean(axis=0)}")
    else:
        print("AI stats file not found.")

    print("\n=== Inference Stats (Captured) ===")
    print(f"Noise Input: {inf_noise}")
    print(f"AI Input:    {inf_ai}")

if __name__ == "__main__":
    check_stats_distribution()
