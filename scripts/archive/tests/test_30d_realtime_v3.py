#!/usr/bin/env python3
"""
30d Model Real-time Scan Test v3

Uses correct CPU feature extraction from lib/cpu_stats.py
Includes std floor safety to prevent normalization explosion.
"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.patch_stats import compute_patch_stats_v3_batch
from lib.cpu_stats import compute_cpu_stats

# Config
MODEL_DIR = Path("/home/techne/aicheckers/models/two_head_30d")
DINOV3_PATH = Path("/home/techne/aicheckers/models/dinov3-vitb16")
MID_LAYER = 6

# Feature indices
# GPU_5D_IDX: From patch_stats_v3 (34d), selecting 4 features + mid_adj_sim_var
# According to V3_STAT_NAMES in lib/patch_stats.py:
#   [1] = adj_sim_var, [3] = patch_var, [5] = norm_var, [6] = norm_range
GPU_5D_IDX = [1, 3, 5, 6]

# CPU16_13D_IDX: From cpu_stats_v2 (18d), selecting 13 features
CPU16_13D_IDX = [0, 1, 2, 4, 5, 7, 8, 9, 11, 12, 13, 14, 15]

# CPU20_12D_IDX: From cpu_stats_v3_20d (20d), selecting 12 features
CPU20_12D_IDX = [0, 1, 2, 3, 4, 5, 8, 10, 15, 16, 17, 18]

# Safety: minimum std floor to prevent normalization explosion
STD_FLOOR = 1e-3


class TwoHeadClassifier(nn.Module):
    def __init__(self, cls_dim=768, gpu_dim=5, cpu_dim=25, hidden_dim=256):
        super().__init__()
        total_dim = cls_dim + gpu_dim + cpu_dim
        self.bn_input = nn.BatchNorm1d(total_dim)
        self.fc1 = nn.Linear(total_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        self.register_buffer('cls_mean', torch.zeros(cls_dim))
        self.register_buffer('cls_std', torch.ones(cls_dim))
        self.register_buffer('gpu_mean', torch.zeros(gpu_dim))
        self.register_buffer('gpu_std', torch.ones(gpu_dim))
        self.register_buffer('cpu_mean', torch.zeros(cpu_dim))
        self.register_buffer('cpu_std', torch.ones(cpu_dim))

    def forward(self, cls_feat, gpu_feat, cpu_feat):
        # Apply std floor to prevent explosion
        cls_std_safe = torch.clamp(self.cls_std, min=STD_FLOOR)
        gpu_std_safe = torch.clamp(self.gpu_std, min=STD_FLOOR)
        cpu_std_safe = torch.clamp(self.cpu_std, min=STD_FLOOR)

        cls_norm = (cls_feat - self.cls_mean) / (cls_std_safe + 1e-8)
        gpu_norm = (gpu_feat - self.gpu_mean) / (gpu_std_safe + 1e-8)
        cpu_norm = (cpu_feat - self.cpu_mean) / (cpu_std_safe + 1e-8)

        x = torch.cat([cls_norm, gpu_norm, cpu_norm], dim=-1)
        x = self.bn_input(x)
        x = F.gelu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.gelu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        return self.fc3(x)


def compute_mid_adj_sim_var(patches, device):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='/home/techne/aicheckers/data/hard_negatives')
    parser.add_argument('--limit', type=int, default=100)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print("Loading 30d model...")
    model = TwoHeadClassifier(cls_dim=768, gpu_dim=5, cpu_dim=25).to(device)
    state_dict = torch.load(MODEL_DIR / "model.pt", map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()

    # Check for dangerous std values
    print("\nChecking normalization stats...")
    dangerous_indices = []
    for name, buffer in [('cls_std', model.cls_std), ('gpu_std', model.gpu_std), ('cpu_std', model.cpu_std)]:
        for i, v in enumerate(buffer.cpu().numpy()):
            if v < STD_FLOOR:
                dangerous_indices.append((name, i, v))
                print(f"  WARNING: {name}[{i}] = {v:.2e} < {STD_FLOOR} (will be floored)")
    if dangerous_indices:
        print(f"  Applied std floor of {STD_FLOOR} to {len(dangerous_indices)} values")
    else:
        print("  All std values are safe.")

    # Load DINOv3
    print("Loading DINOv3...")
    from transformers import AutoImageProcessor, AutoModel
    processor = AutoImageProcessor.from_pretrained(str(DINOV3_PATH))
    dino = AutoModel.from_pretrained(str(DINOV3_PATH)).to(device)
    dino.eval()

    # Get images
    img_dir = Path(args.dir)
    images = sorted([
        p for p in img_dir.glob("*")
        if p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.webp']
    ])[:args.limit]
    print(f"Testing {len(images)} images from {img_dir}")

    scores = []
    logits_list = []
    errors = []

    for i in tqdm(range(0, len(images), args.batch), desc="Scanning"):
        batch_files = images[i:i+args.batch]
        batch_images = []
        batch_valid_idx = []

        for j, fp in enumerate(batch_files):
            try:
                img = Image.open(fp).convert('RGB')
                batch_images.append(img)
                batch_valid_idx.append(j)
            except Exception as e:
                errors.append(str(fp))

        if not batch_images:
            continue

        # DINOv3
        inputs = processor(images=batch_images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = dino(**inputs, output_hidden_states=True)
            cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            mid_hidden = outputs.hidden_states[MID_LAYER + 1]
            mid_patches = mid_hidden[:, 5:5+196, :]

            patch_stats = compute_patch_stats_v3_batch(mid_patches)
            gpu_4d = patch_stats[:, GPU_5D_IDX]
            mid_adj_var = compute_mid_adj_sim_var(mid_patches, device)
            gpu_5d = np.hstack([gpu_4d, mid_adj_var.reshape(-1, 1)])

        # CPU features using correct extraction
        cpu_v2_list = []
        cpu_v3_list = []
        valid_idx = 0
        for j, fp in enumerate(batch_files):
            if j in batch_valid_idx:
                cpu_v2, cpu_v3_20d = compute_cpu_stats(fp)
                valid_idx += 1
            else:
                cpu_v2 = np.zeros(18, dtype=np.float32)
                cpu_v3_20d = np.zeros(20, dtype=np.float32)
            cpu_v2_list.append(cpu_v2)
            cpu_v3_list.append(cpu_v3_20d)

        cpu_v2_arr = np.array(cpu_v2_list[:len(batch_images)])
        cpu_v3_arr = np.array(cpu_v3_list[:len(batch_images)])

        cpu16_13d = cpu_v2_arr[:, CPU16_13D_IDX]
        cpu20_12d = cpu_v3_arr[:, CPU20_12D_IDX]
        cpu_25d = np.hstack([cpu16_13d, cpu20_12d])

        # NaN handling
        cls_emb = np.nan_to_num(cls_emb, nan=0.0)
        gpu_5d = np.nan_to_num(gpu_5d, nan=0.0)
        cpu_25d = np.nan_to_num(cpu_25d, nan=0.0)

        # Inference
        cls_t = torch.tensor(cls_emb, dtype=torch.float32).to(device)
        gpu_t = torch.tensor(gpu_5d, dtype=torch.float32).to(device)
        cpu_t = torch.tensor(cpu_25d, dtype=torch.float32).to(device)

        with torch.no_grad():
            logits = model(cls_t, gpu_t, cpu_t)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            scores.extend(probs.tolist())
            logits_list.extend(logits.cpu().numpy().flatten().tolist())

        # Verbose output
        if args.verbose:
            for j, (fp, prob, logit) in enumerate(zip(batch_files, probs, logits.cpu().numpy().flatten())):
                verdict = "AI" if prob >= 0.5 else "Human"
                print(f"  {fp.name}: {prob*100:.1f}% -> {verdict} [logit={logit:.2f}]")

    scores = np.array(scores)
    logits_arr = np.array(logits_list)

    print("\n" + "="*60)
    print(f"30d Model Real-time Scan Results v3")
    print("="*60)
    print(f"Total: {len(scores)} | Errors: {len(errors)}")
    print()

    # Logit sanity check
    print("Logit range check:")
    print(f"  Min: {logits_arr.min():.2f} | Max: {logits_arr.max():.2f}")
    if abs(logits_arr.max()) > 20 or abs(logits_arr.min()) > 20:
        print("  WARNING: Logits outside normal range [-20, 20]!")
    else:
        print("  OK: Logits are within normal range.")
    print()

    thresholds = [0.3, 0.5, 0.7, 0.9]
    for t in thresholds:
        detected = (scores >= t).sum()
        rate = detected / len(scores) * 100
        print(f"Threshold {t}: {detected}/{len(scores)} ({rate:.1f}%)")

    print()
    print(f"Mean: {scores.mean():.4f} | Median: {np.median(scores):.4f}")
    print(f"Min: {scores.min():.4f} | Max: {scores.max():.4f}")

    # Distribution
    print("\nScore distribution:")
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    hist, _ = np.histogram(scores, bins=bins)
    max_count = max(hist) if max(hist) > 0 else 1
    for j in range(len(hist)):
        bar_len = int(hist[j] * 40 / max_count)
        bar = "█" * bar_len
        print(f"  {bins[j]:.1f}-{bins[j+1]:.1f}: {hist[j]:4d} {bar}")


if __name__ == "__main__":
    main()
