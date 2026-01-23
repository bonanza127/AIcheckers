#!/usr/bin/env python3
"""
Fast model testing using pre-extracted embeddings.

Tests model on pre-extracted embeddings (instant - no image loading needed).

Usage:
    python scripts/test_from_embeddings.py --category hard_negatives_ai --model 29d
    python scripts/test_from_embeddings.py --category hard_negatives_ai --model 28d
"""
import sys
from pathlib import Path
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

EMB_DIR = Path("/home/techne/aicheckers/embeddings")
MODEL_DIR = Path("/home/techne/aicheckers/models")

STD_FLOOR = 1e-3


class TwoHeadClassifier(nn.Module):
    def __init__(self, cls_dim=768, gpu_dim=5, cpu_dim=24, hidden_dim=256):
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


# Model configurations
MODEL_CONFIGS = {
    '29d': {
        'path': MODEL_DIR / 'two_head_29d',
        'gpu_dim': 5,
        'cpu_dim': 24,
        'gpu_idx': [1, 3, 5, 6],
        'cpu16_idx': [0, 1, 2, 4, 5, 7, 8, 9, 11, 12, 13, 14, 15],  # 13d
        'cpu20_idx': [0, 1, 2, 3, 4, 5, 8, 10, 15, 16, 17],  # 11d
    },
    '28d': {
        'path': MODEL_DIR / 'two_head_28d',
        'gpu_dim': 5,
        'cpu_dim': 23,
        'gpu_idx': [1, 3, 5, 6],
        'cpu16_idx': [0, 1, 2, 4, 7, 8, 9, 11, 12, 13, 14, 15],  # 12d (no patchwise_edge_density)
        'cpu20_idx': [0, 1, 2, 3, 4, 5, 8, 10, 15, 16, 17],  # 11d
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--category', type=str, required=True, help='Category name (e.g., hard_negatives_ai)')
    parser.add_argument('--model', type=str, default='29d', choices=['29d', '28d'], help='Model to test')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = MODEL_CONFIGS[args.model]

    print(f"Testing {args.model} model on {args.category}")
    print(f"Device: {device}")
    print("=" * 60)

    # Load model
    print(f"Loading {args.model} model...")
    model = TwoHeadClassifier(
        cls_dim=768,
        gpu_dim=config['gpu_dim'],
        cpu_dim=config['cpu_dim']
    ).to(device)
    state_dict = torch.load(config['path'] / "model.pt", map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()

    # Load embeddings
    print("Loading embeddings...")
    cls_emb = np.load(EMB_DIR / f"{args.category}.npy")
    patch_stats = np.load(EMB_DIR / f"{args.category}_patch_stats_v3.npy")
    mid_adj = np.load(EMB_DIR / f"{args.category}_mid_adj_sim_var.npy")
    cpu_v2 = np.load(EMB_DIR / f"{args.category}_cpu_stats_v2.npy")
    cpu_v3_20d = np.load(EMB_DIR / f"{args.category}_cpu_stats_v3_20d.npy")

    print(f"Loaded {len(cls_emb)} samples")

    # Prepare features
    gpu_4d = patch_stats[:, config['gpu_idx']]
    gpu_5d = np.hstack([gpu_4d, mid_adj.reshape(-1, 1)])

    cpu16 = cpu_v2[:, config['cpu16_idx']]
    cpu20 = cpu_v3_20d[:, config['cpu20_idx']]
    cpu_full = np.hstack([cpu16, cpu20])

    # NaN handling
    cls_emb = np.nan_to_num(cls_emb, nan=0.0)
    gpu_5d = np.nan_to_num(gpu_5d, nan=0.0)
    cpu_full = np.nan_to_num(cpu_full, nan=0.0)

    # To tensors
    cls_t = torch.tensor(cls_emb, dtype=torch.float32).to(device)
    gpu_t = torch.tensor(gpu_5d, dtype=torch.float32).to(device)
    cpu_t = torch.tensor(cpu_full, dtype=torch.float32).to(device)

    # Inference
    print("Running inference...")
    with torch.no_grad():
        logits = model(cls_t, gpu_t, cpu_t)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        logits_np = logits.cpu().numpy().flatten()

    # Results
    print("\n" + "=" * 60)
    print(f"{args.model} Model - {args.category} Test Results")
    print("=" * 60)
    print(f"Total: {len(probs)}")
    print()

    print("Logit range check:")
    print(f"  Min: {logits_np.min():.2f} | Max: {logits_np.max():.2f}")
    if abs(logits_np.max()) > 20 or abs(logits_np.min()) > 20:
        print("  WARNING: Logits outside normal range [-20, 20]!")
    else:
        print("  OK: Logits are within normal range.")
    print()

    thresholds = [0.3, 0.5, 0.7, 0.9]
    for t in thresholds:
        detected = (probs >= t).sum()
        rate = detected / len(probs) * 100
        print(f"Threshold {t}: {detected}/{len(probs)} ({rate:.1f}%)")

    print()
    print(f"Mean: {probs.mean():.4f} | Median: {np.median(probs):.4f}")
    print(f"Min: {probs.min():.4f} | Max: {probs.max():.4f}")

    # Distribution
    print("\nScore distribution:")
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    hist, _ = np.histogram(probs, bins=bins)
    max_count = max(hist) if max(hist) > 0 else 1
    for j in range(len(hist)):
        bar_len = int(hist[j] * 40 / max_count)
        bar = "█" * bar_len
        print(f"  {bins[j]:.1f}-{bins[j+1]:.1f}: {hist[j]:4d} {bar}")


if __name__ == "__main__":
    main()
