#!/usr/bin/env python3
"""
Extract mid-layer patches only for categories that already have CLS embeddings.
Also computes mid_adj_sim_var from the extracted patches.
"""
import os
import sys
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path

# Config
EMBED_DIR = Path("embeddings")
MID_LAYER = 6
BATCH_SIZE = 16  # GTX 1660 friendly

# Categories to process
CATEGORIES = [
    ("pixiv_novelai_v2_ai", "/mnt/d/images/pixiv_novelai_v2", True),      # has CLS
    ("twitter_novelai_v2_ai", "/mnt/d/images/twitter_novelai_v2", True),  # has CLS
    ("novelai_aibooru_ai", "/mnt/d/images/novelai_aibooru", True),        # has CLS
    ("hard_negatives_ai", "/mnt/d/images/hard_negatives", False),         # no CLS
]

transform = transforms.Compose([
    transforms.Resize(518, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(518),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_image(path):
    try:
        img = Image.open(path).convert("RGB")
        return transform(img)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def compute_adj_sim_var(patches):
    """Compute adjacency similarity variance from patches (N, 196, 768)"""
    # Reshape to 14x14 grid
    B, N, D = patches.shape
    grid = patches.reshape(B, 14, 14, D)

    # Calculate horizontal and vertical adjacency similarities
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

    # Combine and compute variance per sample
    all_sim = torch.cat([h_sim.reshape(B, -1), v_sim.reshape(B, -1)], dim=1)
    adj_sim_var = all_sim.var(dim=1).cpu().numpy()

    return adj_sim_var

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load DINOv3
    print("Loading DINOv3...")
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg', pretrained=True)
    model = model.to(device)
    model.eval()

    # Hook for mid-layer
    mid_features = {}
    def get_mid_hook(name):
        def hook(module, input, output):
            mid_features[name] = output
        return hook

    model.blocks[MID_LAYER].register_forward_hook(get_mid_hook('mid'))

    for cat_name, img_dir, has_cls in CATEGORIES:
        mid_path = EMBED_DIR / f"{cat_name}_mid_patches.npy"
        adj_var_path = EMBED_DIR / f"{cat_name}_mid_adj_sim_var.npy"
        cls_path = EMBED_DIR / f"{cat_name}.npy"

        # Skip if already done
        if mid_path.exists() and adj_var_path.exists():
            print(f"\n{cat_name}: Already extracted, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"Processing: {cat_name}")
        print(f"Directory: {img_dir}")
        print(f"Has CLS: {has_cls}")
        print(f"{'='*60}")

        # Get file list
        if has_cls:
            # Use existing files list to ensure alignment
            files_path = EMBED_DIR / f"{cat_name}_files.txt"
            if files_path.exists():
                with open(files_path) as f:
                    raw_files = [line.strip() for line in f if line.strip()]
                # Check if paths are absolute or relative
                if raw_files and not os.path.isabs(raw_files[0]):
                    # Prepend base directory
                    files = [os.path.join(img_dir, f) for f in raw_files]
                else:
                    files = raw_files
                print(f"Using existing files list: {len(files)} files")
            else:
                # Fallback: find images
                files = sorted([
                    str(p) for p in Path(img_dir).rglob("*")
                    if p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.webp']
                ])
        else:
            # Find all images
            files = sorted([
                str(p) for p in Path(img_dir).rglob("*")
                if p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.webp']
            ])
            print(f"Found {len(files)} images")

        if not files:
            print(f"No files found, skipping")
            continue

        all_mid_patches = []
        all_cls = [] if not has_cls else None

        # Process in batches
        num_batches = (len(files) + BATCH_SIZE - 1) // BATCH_SIZE

        for batch_idx in tqdm(range(num_batches), desc="Extracting"):
            batch_files = files[batch_idx * BATCH_SIZE : (batch_idx + 1) * BATCH_SIZE]

            # Load images
            tensors = []
            for fp in batch_files:
                t = load_image(fp)
                if t is not None:
                    tensors.append(t)

            if not tensors:
                continue

            batch = torch.stack(tensors).to(device)

            with torch.no_grad():
                output = model(batch)

                # Extract mid-layer patches (skip CLS token at position 0)
                mid_out = mid_features['mid']
                # mid_out shape: (B, 197, 768) - CLS + 196 patches
                mid_patches = mid_out[:, 1:, :].cpu()  # (B, 196, 768)
                all_mid_patches.append(mid_patches)

                # Extract CLS if needed
                if all_cls is not None:
                    cls_tokens = output.cpu()  # Final CLS
                    all_cls.append(cls_tokens)

        # Concatenate
        all_mid_patches = torch.cat(all_mid_patches, dim=0)
        print(f"Mid patches shape: {all_mid_patches.shape}")

        # Save mid patches
        np.save(mid_path, all_mid_patches.numpy().astype(np.float16))
        print(f"Saved: {mid_path}")

        # Compute and save mid_adj_sim_var
        adj_sim_var = compute_adj_sim_var(all_mid_patches)
        np.save(adj_var_path, adj_sim_var.astype(np.float32))
        print(f"Saved: {adj_var_path}")

        # Save CLS if extracted
        if all_cls is not None:
            all_cls = torch.cat(all_cls, dim=0)
            np.save(cls_path, all_cls.numpy())
            print(f"Saved CLS: {cls_path} ({all_cls.shape})")

            # Save files list
            files_path = EMBED_DIR / f"{cat_name}_files.txt"
            with open(files_path, 'w') as f:
                f.write('\n'.join(files))
            print(f"Saved files list: {files_path}")

        # Clear cache
        torch.cuda.empty_cache()

    print("\n" + "="*60)
    print("All extractions complete!")
    print("="*60)

if __name__ == "__main__":
    main()
