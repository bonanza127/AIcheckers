
import numpy as np
from pathlib import Path

def compare_stats():
    # Old corrupted stats
    old_path = Path("/home/techne/aicheckers/embeddings/danbooru_real_patch_stats.npy")
    # New test stats
    new_path = Path("/home/techne/aicheckers/embeddings/danbooru_real_test_patch_stats.npy")
    
    print("=== Old Stats (Dec 20) ===")
    if old_path.exists():
        old = np.load(old_path)
        print(f"Mean: {old.mean(axis=0)}")
        print(f"Var:  {old.var(axis=0)}")
        print(f"Variance of Feature[1] (adj_sim_var): {old[:, 1].var():.6f}") # Wait, I was checking column 1 variance
        # Actually in Step 295 I saw:
        # Mean: [... 0.9807 ...]
        # Std:  [... 0.0426 ...]
        # The mean itself was 0.98. Let's look at the mean of column 1.
    
    print("\n=== New Stats (Limit 100) ===")
    if new_path.exists():
        new = np.load(new_path)
        print(f"Mean: {new.mean(axis=0)}")
        print(f"Var:  {new.var(axis=0)}")
    else:
        print("New stats file not found!")

if __name__ == "__main__":
    compare_stats()
