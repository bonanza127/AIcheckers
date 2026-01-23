#!/usr/bin/env python3
"""
Test 30d Two-Head model on hard negatives using pre-computed embeddings.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# Config
MODEL_DIR = Path("/home/techne/aicheckers/models/two_head_30d")
EMB_DIR = Path("/home/techne/aicheckers/embeddings")
CAT_NAME = "hard_negatives_ai"

# Feature indices (from train_30d.py)
GPU_5D_IDX = [1, 3, 5, 6]  # patch_var, degree_centrality, local_efficiency, edge_interior_gap
CPU16_13D_IDX = [0, 1, 2, 4, 5, 7, 8, 9, 11, 12, 13, 14, 15]
CPU20_12D_IDX = [0, 1, 2, 3, 4, 5, 8, 10, 15, 16, 17, 18]


class TwoHeadClassifier(nn.Module):
    """Two-Head 30d分類器 (CLS 768d + GPU 5d + CPU 25d = 798d)"""

    def __init__(self, cls_dim=768, gpu_dim=5, cpu_dim=25, hidden_dim=256):
        super().__init__()
        total_dim = cls_dim + gpu_dim + cpu_dim  # 798d

        self.bn_input = nn.BatchNorm1d(total_dim)

        self.fc1 = nn.Linear(total_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(hidden_dim // 2, 1)

        # 統計量保存用
        self.register_buffer('cls_mean', torch.zeros(cls_dim))
        self.register_buffer('cls_std', torch.ones(cls_dim))
        self.register_buffer('gpu_mean', torch.zeros(gpu_dim))
        self.register_buffer('gpu_std', torch.ones(gpu_dim))
        self.register_buffer('cpu_mean', torch.zeros(cpu_dim))
        self.register_buffer('cpu_std', torch.ones(cpu_dim))

    def forward(self, cls_feat, gpu_feat, cpu_feat):
        # Z-score正規化
        cls_norm = (cls_feat - self.cls_mean) / (self.cls_std + 1e-8)
        gpu_norm = (gpu_feat - self.gpu_mean) / (self.gpu_std + 1e-8)
        cpu_norm = (cpu_feat - self.cpu_mean) / (self.cpu_std + 1e-8)

        x = torch.cat([cls_norm, gpu_norm, cpu_norm], dim=-1)
        x = self.bn_input(x)

        x = F.gelu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)

        x = F.gelu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)

        return self.fc3(x)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model_path = MODEL_DIR / "model.pt"
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return

    model = TwoHeadClassifier(cls_dim=768, gpu_dim=5, cpu_dim=25).to(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"Loaded model from: {model_path}")

    # Load embeddings
    cls_emb = np.load(EMB_DIR / f"{CAT_NAME}.npy")
    patch_stats = np.load(EMB_DIR / f"{CAT_NAME}_patch_stats_v3.npy")
    cpu_stats_v2 = np.load(EMB_DIR / f"{CAT_NAME}_cpu_stats_v2.npy")
    cpu_stats_v3 = np.load(EMB_DIR / f"{CAT_NAME}_cpu_stats_v3_20d.npy")
    mid_adj_var = np.load(EMB_DIR / f"{CAT_NAME}_mid_adj_sim_var.npy")

    print(f"\nLoaded embeddings:")
    print(f"  CLS: {cls_emb.shape}")
    print(f"  patch_stats_v3: {patch_stats.shape}")
    print(f"  cpu_stats_v2: {cpu_stats_v2.shape}")
    print(f"  cpu_stats_v3_20d: {cpu_stats_v3.shape}")
    print(f"  mid_adj_sim_var: {mid_adj_var.shape}")

    # Prepare features
    # GPU 5d: 4 from patch_stats + mid_adj_sim_var
    gpu_4d = patch_stats[:, GPU_5D_IDX]
    gpu_5d = np.hstack([gpu_4d, mid_adj_var.reshape(-1, 1)])

    # CPU 25d: 13 from cpu_stats_v2 + 12 from cpu_stats_v3
    cpu16_13d = cpu_stats_v2[:, CPU16_13D_IDX]
    cpu20_12d = cpu_stats_v3[:, CPU20_12D_IDX]
    cpu_25d = np.hstack([cpu16_13d, cpu20_12d])

    print(f"\nPrepared features:")
    print(f"  GPU 5d: {gpu_5d.shape}")
    print(f"  CPU 25d: {cpu_25d.shape}")

    # Handle NaN
    cls_emb = np.nan_to_num(cls_emb, nan=0.0, posinf=0.0, neginf=0.0)
    gpu_5d = np.nan_to_num(gpu_5d, nan=0.0, posinf=0.0, neginf=0.0)
    cpu_25d = np.nan_to_num(cpu_25d, nan=0.0, posinf=0.0, neginf=0.0)

    # Convert to tensors
    cls_t = torch.tensor(cls_emb, dtype=torch.float32).to(device)
    gpu_t = torch.tensor(gpu_5d, dtype=torch.float32).to(device)
    cpu_t = torch.tensor(cpu_25d, dtype=torch.float32).to(device)

    # Inference
    with torch.no_grad():
        logits = model(cls_t, gpu_t, cpu_t)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()

    # Statistics
    print("\n" + "="*60)
    print("Hard Negatives Test Results (30d Model)")
    print("="*60)
    print(f"Total samples: {len(probs)}")
    print()

    # Detection rates at various thresholds
    thresholds = [0.3, 0.5, 0.7, 0.9]
    for t in thresholds:
        detected = (probs >= t).sum()
        rate = detected / len(probs) * 100
        print(f"Threshold {t}: {detected}/{len(probs)} ({rate:.1f}%)")

    print()
    print(f"Mean score: {probs.mean():.4f}")
    print(f"Median score: {np.median(probs):.4f}")
    print(f"Std: {probs.std():.4f}")
    print(f"Min/Max: {probs.min():.4f} / {probs.max():.4f}")

    # Score distribution
    print("\nScore distribution:")
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    hist, _ = np.histogram(probs, bins=bins)
    for i in range(len(hist)):
        bar = "█" * (hist[i] // 50)
        print(f"  {bins[i]:.1f}-{bins[i+1]:.1f}: {hist[i]:4d} {bar}")

    # Save results
    output_path = Path("logs/hardneg_30d_test.csv")
    files_path = EMB_DIR / f"{CAT_NAME}_files.txt"
    if files_path.exists():
        with open(files_path) as f:
            files = [l.strip() for l in f if l.strip()]
    else:
        files = [f"sample_{i}" for i in range(len(probs))]

    import pandas as pd
    df = pd.DataFrame({
        'file': files[:len(probs)],
        'score': probs
    })
    df.to_csv(output_path, index=False)
    print(f"\nSaved detailed results to: {output_path}")


if __name__ == "__main__":
    main()
