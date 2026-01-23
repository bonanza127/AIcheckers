#!/usr/bin/env python3
"""
Extract embeddings for hard_negatives dataset.

Outputs (to embeddings/):
  hard_negatives_ai.npy              - CLS embeddings (N, 768)
  hard_negatives_ai_patch_stats_v3.npy - Patch statistics (N, 34)
  hard_negatives_ai_cpu_stats_v2.npy   - CPU v2 features (N, 18)
  hard_negatives_ai_cpu_stats_v3_20d.npy - CPU v3 20d features (N, 20)
  hard_negatives_ai_mid_adj_sim_var.npy - Mid adjacency sim var (N,)
  hard_negatives_ai_files.txt          - File paths

Usage:
    python scripts/extract_hard_negatives_embeddings.py           # Full extraction
    python scripts/extract_hard_negatives_embeddings.py --incremental  # Only new files
    python scripts/extract_hard_negatives_embeddings.py --workers 8
"""
import sys
from pathlib import Path
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.patch_stats import compute_patch_stats_v3_batch
from lib.cpu_stats import compute_cpu_stats

# Config
DATA_DIR = Path("/home/techne/aicheckers/data/hard_negatives")
EMB_DIR = Path("/home/techne/aicheckers/embeddings")
DINOV3_PATH = Path("/home/techne/aicheckers/models/dinov3-vitb16")
MID_LAYER = 6
CATEGORY_NAME = "hard_negatives_ai"


def compute_mid_adj_sim_var(patches):
    """Compute adjacency similarity variance from mid-layer patches."""
    B, N, D = patches.shape
    grid = patches.reshape(B, 14, 14, D)
    h_sim = F.cosine_similarity(
        grid[:, :, :-1].reshape(-1, D),
        grid[:, :, 1:].reshape(-1, D),
        dim=1
    ).reshape(B, 14, 13)
    v_sim = F.cosine_similarity(
        grid[:, :-1, :].reshape(-1, D),
        grid[:, 1:, :].reshape(-1, D),
        dim=1
    ).reshape(B, 13, 14)
    all_sim = torch.cat([h_sim.reshape(B, -1), v_sim.reshape(B, -1)], dim=1)
    return all_sim.var(dim=1).cpu().numpy()


def extract_cpu_single(fp):
    """Extract CPU features for a single image."""
    try:
        cpu_v2, cpu_v3_20d = compute_cpu_stats(fp)
        return str(fp), cpu_v2, cpu_v3_20d, None
    except Exception as e:
        return str(fp), None, None, str(e)


def load_existing_files():
    """Load list of already processed files."""
    files_path = EMB_DIR / f"{CATEGORY_NAME}_files.txt"
    if files_path.exists():
        with open(files_path) as f:
            return set(line.strip() for line in f if line.strip())
    return set()


def load_existing_embeddings():
    """Load existing embeddings for incremental mode."""
    try:
        cls = np.load(EMB_DIR / f"{CATEGORY_NAME}.npy")
        patch = np.load(EMB_DIR / f"{CATEGORY_NAME}_patch_stats_v3.npy")
        mid_adj = np.load(EMB_DIR / f"{CATEGORY_NAME}_mid_adj_sim_var.npy")
        cpu_v2 = np.load(EMB_DIR / f"{CATEGORY_NAME}_cpu_stats_v2.npy")
        cpu_v3 = np.load(EMB_DIR / f"{CATEGORY_NAME}_cpu_stats_v3_20d.npy")
        with open(EMB_DIR / f"{CATEGORY_NAME}_files.txt") as f:
            files = [line.strip() for line in f if line.strip()]
        return cls, patch, mid_adj, cpu_v2, cpu_v3, files
    except FileNotFoundError:
        return None, None, None, None, None, []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=32, help='Batch size for GPU')
    parser.add_argument('--workers', type=int, default=4, help='CPU workers for parallel extraction')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of images')
    parser.add_argument('--incremental', action='store_true', help='Only process new files')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch}, Workers: {args.workers}")
    print(f"Incremental mode: {args.incremental}")

    # Check for incremental mode
    existing_files = set()
    existing_cls, existing_patch, existing_mid_adj, existing_cpu_v2, existing_cpu_v3, existing_file_list = None, None, None, None, None, []

    if args.incremental:
        existing_cls, existing_patch, existing_mid_adj, existing_cpu_v2, existing_cpu_v3, existing_file_list = load_existing_embeddings()
        if existing_cls is not None:
            existing_files = set(existing_file_list)
            print(f"Found {len(existing_files)} existing embeddings")
        else:
            print("No existing embeddings found, will do full extraction")

    # Load DINOv3
    print("Loading DINOv3...")
    from transformers import AutoImageProcessor, AutoModel
    processor = AutoImageProcessor.from_pretrained(str(DINOV3_PATH))
    dino = AutoModel.from_pretrained(str(DINOV3_PATH)).to(device)
    dino.eval()

    # Get images
    all_images = sorted([
        p for p in DATA_DIR.glob("*")
        if p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.webp']
    ])

    # Filter for new images only in incremental mode
    if args.incremental and existing_files:
        images = [p for p in all_images if str(p) not in existing_files]
        print(f"Found {len(images)} NEW images (skipping {len(existing_files)} existing)")
    else:
        images = all_images

    if args.limit:
        images = images[:args.limit]

    if not images:
        print("No new images to process!")
        return

    print(f"Processing {len(images)} images from {DATA_DIR}")
    print("=" * 60)

    # Storage
    all_cls = []
    all_patch_stats = []
    all_mid_adj = []
    all_cpu_v2 = []
    all_cpu_v3_20d = []
    all_files = []
    errors = []

    # Process in batches
    for batch_start in tqdm(range(0, len(images), args.batch), desc="Extracting"):
        batch_files = images[batch_start:batch_start + args.batch]

        # Load images for GPU
        batch_images = []
        batch_valid_files = []
        for fp in batch_files:
            try:
                img = Image.open(fp).convert('RGB')
                batch_images.append(img)
                batch_valid_files.append(fp)
            except Exception as e:
                errors.append((str(fp), str(e)))

        if not batch_images:
            continue

        # GPU: DINOv3 inference
        inputs = processor(images=batch_images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = dino(**inputs, output_hidden_states=True)
            cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            mid_hidden = outputs.hidden_states[MID_LAYER + 1]
            mid_patches = mid_hidden[:, 5:5+196, :]

            patch_stats = compute_patch_stats_v3_batch(mid_patches)
            mid_adj_var = compute_mid_adj_sim_var(mid_patches)

        all_cls.append(cls_emb)
        all_patch_stats.append(patch_stats)
        all_mid_adj.append(mid_adj_var)

        # CPU: Parallel feature extraction
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(extract_cpu_single, fp): i
                      for i, fp in enumerate(batch_valid_files)}
            results = [None] * len(batch_valid_files)

            for future in as_completed(futures):
                idx = futures[future]
                fp_str, cpu_v2, cpu_v3_20d, error = future.result()
                if error:
                    errors.append((fp_str, error))
                    results[idx] = (np.zeros(18, dtype=np.float32),
                                   np.zeros(20, dtype=np.float32),
                                   fp_str)
                else:
                    results[idx] = (cpu_v2, cpu_v3_20d, fp_str)

        for cpu_v2, cpu_v3_20d, fp_str in results:
            all_cpu_v2.append(cpu_v2)
            all_cpu_v3_20d.append(cpu_v3_20d)
            all_files.append(fp_str)

    # Concatenate new data
    new_cls_arr = np.vstack(all_cls).astype(np.float32)
    new_patch_arr = np.vstack(all_patch_stats).astype(np.float32)
    new_mid_adj_arr = np.concatenate(all_mid_adj).astype(np.float32)
    new_cpu_v2_arr = np.vstack(all_cpu_v2).astype(np.float32)
    new_cpu_v3_arr = np.vstack(all_cpu_v3_20d).astype(np.float32)

    # NaN handling for new data
    new_cls_arr = np.nan_to_num(new_cls_arr, nan=0.0)
    new_patch_arr = np.nan_to_num(new_patch_arr, nan=0.0)
    new_mid_adj_arr = np.nan_to_num(new_mid_adj_arr, nan=0.0)
    new_cpu_v2_arr = np.nan_to_num(new_cpu_v2_arr, nan=0.0)
    new_cpu_v3_arr = np.nan_to_num(new_cpu_v3_arr, nan=0.0)

    # Merge with existing in incremental mode
    if args.incremental and existing_cls is not None:
        cls_arr = np.vstack([existing_cls, new_cls_arr])
        patch_stats_arr = np.vstack([existing_patch, new_patch_arr])
        mid_adj_arr = np.concatenate([existing_mid_adj, new_mid_adj_arr])
        cpu_v2_arr = np.vstack([existing_cpu_v2, new_cpu_v2_arr])
        cpu_v3_20d_arr = np.vstack([existing_cpu_v3, new_cpu_v3_arr])
        all_files = existing_file_list + all_files
        print(f"\nMerged with existing: {len(existing_file_list)} + {len(all_files) - len(existing_file_list)} = {len(all_files)}")
    else:
        cls_arr = new_cls_arr
        patch_stats_arr = new_patch_arr
        mid_adj_arr = new_mid_adj_arr
        cpu_v2_arr = new_cpu_v2_arr
        cpu_v3_20d_arr = new_cpu_v3_arr

    print(f"\n{'=' * 60}")
    print(f"Extraction complete!")
    print(f"Total: {len(all_files)} | Errors: {len(errors)}")
    print(f"\nShapes:")
    print(f"  CLS: {cls_arr.shape}")
    print(f"  Patch stats: {patch_stats_arr.shape}")
    print(f"  Mid adj var: {mid_adj_arr.shape}")
    print(f"  CPU v2: {cpu_v2_arr.shape}")
    print(f"  CPU v3 20d: {cpu_v3_20d_arr.shape}")

    # Save
    EMB_DIR.mkdir(exist_ok=True)

    np.save(EMB_DIR / f"{CATEGORY_NAME}.npy", cls_arr)
    np.save(EMB_DIR / f"{CATEGORY_NAME}_patch_stats_v3.npy", patch_stats_arr)
    np.save(EMB_DIR / f"{CATEGORY_NAME}_mid_adj_sim_var.npy", mid_adj_arr)
    np.save(EMB_DIR / f"{CATEGORY_NAME}_cpu_stats_v2.npy", cpu_v2_arr)
    np.save(EMB_DIR / f"{CATEGORY_NAME}_cpu_stats_v3_20d.npy", cpu_v3_20d_arr)

    with open(EMB_DIR / f"{CATEGORY_NAME}_files.txt", 'w') as f:
        for fp in all_files:
            f.write(f"{fp}\n")

    print(f"\nSaved to {EMB_DIR}/{CATEGORY_NAME}_*.npy")

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for fp, err in errors[:5]:
            print(f"  {fp}: {err}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")


if __name__ == "__main__":
    main()
