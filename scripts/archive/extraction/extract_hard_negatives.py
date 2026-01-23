#!/usr/bin/env python3
"""
Extract CLS + mid_adj_sim_var for hard_negatives.
Uses DINOv3-vitb16 (same as training data extraction).
Memory-efficient version - calculates mid_adj_sim_var on-the-fly.
"""
import os
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from transformers import AutoImageProcessor, AutoModel

# Config
EMBED_DIR = Path("embeddings")
IMG_DIR = Path("/home/techne/aicheckers/data/hard_negatives")
CAT_NAME = "hard_negatives_ai"
DINOV3_MODEL_PATH = Path("/home/techne/aicheckers/models/dinov3-vitb16")
MID_LAYER = 6
BATCH_SIZE = 32

def compute_adj_sim_var_batch(patches):
    """Compute adjacency similarity variance from patches (B, 196, 768)
    Returns (B,) array of variances
    """
    B, N, D = patches.shape
    # 14x14 grid for vitb16 with 224x224 input
    grid = patches.reshape(B, 14, 14, D)

    h_sim = torch.nn.functional.cosine_similarity(
        grid[:, :, :-1].reshape(-1, D),
        grid[:, :, 1:].reshape(-1, D),
        dim=1
    ).reshape(B, 14, 13)

    v_sim = torch.nn.functional.cosine_similarity(
        grid[:, :-1, :].reshape(-1, D),
        grid[:, 1:, :].reshape(-1, D),
        dim=1
    ).reshape(B, 13, 14)

    all_sim = torch.cat([h_sim.reshape(B, -1), v_sim.reshape(B, -1)], dim=1)
    adj_sim_var = all_sim.var(dim=1).cpu().numpy()

    return adj_sim_var

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load DINOv3-vitb16
    print(f"Loading DINOv3 from: {DINOV3_MODEL_PATH}")
    processor = AutoImageProcessor.from_pretrained(str(DINOV3_MODEL_PATH))
    model = AutoModel.from_pretrained(str(DINOV3_MODEL_PATH))
    model.to(device)
    model.eval()

    # Get file list
    files = sorted([
        str(p) for p in IMG_DIR.glob("*")
        if p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.webp']
    ])
    print(f"Found {len(files)} images")

    all_cls = []
    all_adj_sim_var = []
    valid_files = []

    # Process in batches
    num_batches = (len(files) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in tqdm(range(num_batches), desc="Extracting"):
        batch_files = files[batch_idx * BATCH_SIZE : (batch_idx + 1) * BATCH_SIZE]

        # Load images
        batch_images = []
        batch_valid = []
        for fp in batch_files:
            try:
                img = Image.open(fp).convert("RGB")
                batch_images.append(img)
                batch_valid.append(fp)
            except Exception as e:
                print(f"Error loading {fp}: {e}")

        if not batch_images:
            continue

        # Process with HF processor
        inputs = processor(images=batch_images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

            # CLS from final layer
            cls_tokens = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_cls.append(cls_tokens)

            # Mid-layer patches (skip CLS and 4 registers)
            # DINOv3: [CLS(0), REG1-4(1-4), PATCH1-196(5-200)] = 201 tokens
            mid_hidden = outputs.hidden_states[MID_LAYER + 1]  # +1 because hidden_states[0] is embeddings
            mid_patches = mid_hidden[:, 5:5+196, :]  # Skip CLS+4regs, keep 196 patches

            # Compute adj_sim_var immediately
            adj_sim_var = compute_adj_sim_var_batch(mid_patches)
            all_adj_sim_var.append(adj_sim_var)

            valid_files.extend(batch_valid)

        # Clear CUDA cache periodically
        if batch_idx % 20 == 0:
            torch.cuda.empty_cache()

    # Concatenate
    all_cls = np.concatenate(all_cls, axis=0)
    all_adj_sim_var = np.concatenate(all_adj_sim_var, axis=0)

    print(f"CLS shape: {all_cls.shape}")
    print(f"adj_sim_var shape: {all_adj_sim_var.shape}")

    # Save CLS
    cls_path = EMBED_DIR / f"{CAT_NAME}.npy"
    np.save(cls_path, all_cls)
    print(f"Saved: {cls_path}")

    # Save mid_adj_sim_var
    adj_var_path = EMBED_DIR / f"{CAT_NAME}_mid_adj_sim_var.npy"
    np.save(adj_var_path, all_adj_sim_var.astype(np.float32))
    print(f"Saved: {adj_var_path}")

    # Save files list
    files_path = EMBED_DIR / f"{CAT_NAME}_files.txt"
    with open(files_path, 'w') as f:
        f.write('\n'.join([os.path.basename(fp) for fp in valid_files]))
    print(f"Saved files list: {files_path}")

    print(f"\nTotal extracted: {len(valid_files)} images")

if __name__ == "__main__":
    main()
