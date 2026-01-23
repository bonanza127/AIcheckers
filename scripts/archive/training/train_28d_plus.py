#!/usr/bin/env python3
"""
28d+ Two-Head モデル学習スクリプト
777dの学習データ + niji7_twitter_ai を追加

構成:
- GPU 4d: adj_sim_var[1], patch_var[3], norm_var[5], norm_range[6]
- CPU 24d: CPU16 13d + CPU20 11d
"""

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from datetime import datetime
from pathlib import Path
import json

# 設定
MAX_EPOCHS = 30  # 固定
BATCH_SIZE = 512
LR = 1e-3
SEED = 42
VAL_RATIO = 0.1
STD_FLOOR = 1e-3

# 特徴量インデックス (28d: GPU 4d + CPU 24d)
GPU_4D_IDX = [1, 3, 5, 6]  # adj_sim_var, patch_var, norm_var, norm_range
CPU16_13D_IDX = [0, 1, 2, 4, 5, 7, 8, 9, 11, 12, 13, 14, 15]
CPU20_11D_IDX = [0, 1, 2, 3, 4, 5, 8, 10, 15, 16, 17]

# 777dのAIカテゴリ + niji7
AI_CATEGORIES = [
    'illustrious_ai', 'pony_ai', 'sdxl10_ai', 'sd15_ai', 'other_ai',
    'flux1d_ai', 'novelai_ai', 'pixai_ai', 'novelai_aibooru_ai',
    'novelai_combined_ai', 'pixiv_novelai_v2_ai', 'twitter_novelai_v2_ai',
    'novelai_artist_tagged_ai',
    'niji7_twitter_ai',  # 追加
]
REAL_CATEGORIES = ['danbooru_real']

EMB_DIR = Path('embeddings')
MODEL_DIR = Path('models/two_head_28d_plus')


def sanitize_array(arr, name="array"):
    """inf/nan を 0 に置換し、極端な値もclip"""
    arr = arr.astype(np.float32)
    mask = ~np.isfinite(arr)
    if mask.any():
        count = mask.sum()
        print(f"  Warning: {name} has {count} inf/nan values, replacing with 0")
        arr = np.where(mask, 0.0, arr)
    # 極端な値をclip
    arr = np.clip(arr, -1e6, 1e6)
    return arr


def load_category_data(cat):
    cls_path = EMB_DIR / f'{cat}.npy'
    gpu_path = EMB_DIR / f'{cat}_patch_stats_v3.npy'
    cpu16_path = EMB_DIR / f'{cat}_cpu_stats_v2.npy'
    cpu20_path = EMB_DIR / f'{cat}_cpu_stats_v3_20d.npy'

    for p in [cls_path, gpu_path, cpu16_path, cpu20_path]:
        if not p.exists():
            print(f"  [SKIP] {cat}: {p.name} not found")
            return None

    cls = np.load(cls_path)
    gpu = np.load(gpu_path)[:, GPU_4D_IDX]
    cpu16 = np.load(cpu16_path)[:, CPU16_13D_IDX]
    cpu20 = np.load(cpu20_path)[:, CPU20_11D_IDX]

    min_len = min(len(cls), len(gpu), len(cpu16), len(cpu20))
    return cls[:min_len], gpu[:min_len], np.hstack([cpu16[:min_len], cpu20[:min_len]])


class TwoHeadClassifier(nn.Module):
    def __init__(self, cls_dim=768, gpu_dim=4, cpu_dim=24, hidden_dim=256):
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
        cls_norm = (cls_feat - self.cls_mean) / (torch.clamp(self.cls_std, min=STD_FLOOR) + 1e-8)
        gpu_norm = (gpu_feat - self.gpu_mean) / (torch.clamp(self.gpu_std, min=STD_FLOOR) + 1e-8)
        cpu_norm = (cpu_feat - self.cpu_mean) / (torch.clamp(self.cpu_std, min=STD_FLOOR) + 1e-8)
        # 正規化後の値をclamp
        cls_norm = torch.clamp(cls_norm, -100, 100)
        gpu_norm = torch.clamp(gpu_norm, -100, 100)
        cpu_norm = torch.clamp(cpu_norm, -100, 100)
        x = torch.cat([cls_norm, gpu_norm, cpu_norm], dim=-1)
        x = self.bn_input(x)
        x = F.gelu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.gelu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        logits = self.fc3(x)
        return torch.clamp(logits, -50, 50)


def main():
    print("=" * 60)
    print("Two-Head 28d+ Model Training")
    print("777d categories + niji7_twitter_ai")
    print(f"Fixed epochs: {MAX_EPOCHS}")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    ai_cls, ai_gpu, ai_cpu = [], [], []
    real_cls, real_gpu, real_cpu = [], [], []
    skipped_categories = []

    for cat in AI_CATEGORIES:
        result = load_category_data(cat)
        if result:
            c, g, p = result
            ai_cls.append(c); ai_gpu.append(g); ai_cpu.append(p)
            print(f"  ✅ {cat}: {len(c)} samples")
        else:
            skipped_categories.append(cat)
            print(f"  ❌ {cat}: SKIPPED (missing files)")

    for cat in REAL_CATEGORIES:
        result = load_category_data(cat)
        if result:
            c, g, p = result
            real_cls.append(c); real_gpu.append(g); real_cpu.append(p)
            print(f"  ✅ {cat}: {len(c)} samples")
        else:
            skipped_categories.append(cat)
            print(f"  ❌ {cat}: SKIPPED (missing files)")

    if skipped_categories:
        print(f"\n⚠️  WARNING: {len(skipped_categories)} categories skipped!")
        print(f"   Skipped: {', '.join(skipped_categories)}")
        print("   Check if embeddings extraction is complete.")

    ai_cls = sanitize_array(np.vstack(ai_cls).astype(np.float32), "ai_cls")
    ai_gpu = sanitize_array(np.vstack(ai_gpu).astype(np.float32), "ai_gpu")
    ai_cpu = sanitize_array(np.vstack(ai_cpu).astype(np.float32), "ai_cpu")
    real_cls = sanitize_array(np.vstack(real_cls).astype(np.float32), "real_cls")
    real_gpu = sanitize_array(np.vstack(real_gpu).astype(np.float32), "real_gpu")
    real_cpu = sanitize_array(np.vstack(real_cpu).astype(np.float32), "real_cpu")

    print(f"\nBefore balancing: AI={len(ai_cls)}, Real={len(real_cls)}")

    # Balance
    n_ai = len(ai_cls)
    np.random.seed(SEED)
    idx = np.random.choice(len(real_cls), n_ai, replace=False)
    real_cls, real_gpu, real_cpu = real_cls[idx], real_gpu[idx], real_cpu[idx]
    print(f"After balancing: AI={len(ai_cls)}, Real={len(real_cls)}")

    # Combine
    X_cls = np.vstack([ai_cls, real_cls])
    X_gpu = np.vstack([ai_gpu, real_gpu])
    X_cpu = np.vstack([ai_cpu, real_cpu])
    y = np.array([1]*len(ai_cls) + [0]*len(real_cls), dtype=np.float32)

    # Split
    indices = np.arange(len(y))
    train_idx, val_idx = train_test_split(indices, test_size=VAL_RATIO, random_state=SEED, stratify=y)
    print(f"\nTrain: {len(train_idx)}, Val: {len(val_idx)}")

    device = torch.device("cuda")
    model = TwoHeadClassifier().to(device)

    # Normalization stats (float32を明示)
    model.cls_mean.copy_(torch.tensor(X_cls[train_idx].mean(0), dtype=torch.float32))
    model.cls_std.copy_(torch.tensor(X_cls[train_idx].std(0), dtype=torch.float32))
    model.gpu_mean.copy_(torch.tensor(X_gpu[train_idx].mean(0), dtype=torch.float32))
    model.gpu_std.copy_(torch.tensor(X_gpu[train_idx].std(0), dtype=torch.float32))
    model.cpu_mean.copy_(torch.tensor(X_cpu[train_idx].mean(0), dtype=torch.float32))
    model.cpu_std.copy_(torch.tensor(X_cpu[train_idx].std(0), dtype=torch.float32))

    # Loaders (float32を明示)
    train_ds = TensorDataset(
        torch.tensor(X_cls[train_idx], dtype=torch.float32),
        torch.tensor(X_gpu[train_idx], dtype=torch.float32),
        torch.tensor(X_cpu[train_idx], dtype=torch.float32),
        torch.tensor(y[train_idx], dtype=torch.float32)
    )
    val_ds = TensorDataset(
        torch.tensor(X_cls[val_idx], dtype=torch.float32),
        torch.tensor(X_gpu[val_idx], dtype=torch.float32),
        torch.tensor(X_cpu[val_idx], dtype=torch.float32),
        torch.tensor(y[val_idx], dtype=torch.float32)
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    criterion = nn.BCEWithLogitsLoss()

    best_pr_auc = -np.inf
    best_epoch = 0
    best_state = None

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for cls_b, gpu_b, cpu_b, y_b in train_loader:
            cls_b, gpu_b, cpu_b, y_b = cls_b.to(device), gpu_b.to(device), cpu_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(cls_b, gpu_b, cpu_b).squeeze(), y_b)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for cls_b, gpu_b, cpu_b, y_b in val_loader:
                probs = torch.sigmoid(model(cls_b.to(device), gpu_b.to(device), cpu_b.to(device)).squeeze()).cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(y_b.numpy())

        pr_auc = average_precision_score(all_labels, all_probs)
        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} - PR-AUC: {pr_auc:.4f} (best: {best_pr_auc:.4f} @ ep{best_epoch})")

    print(f"\nBest PR-AUC: {best_pr_auc:.4f} at epoch {best_epoch}")
    print(f"Using epoch {MAX_EPOCHS} (fixed)")

    MODEL_DIR.mkdir(exist_ok=True, parents=True)
    # epoch 30固定で保存（best_stateではなく最終状態）
    torch.save(model.state_dict(), MODEL_DIR / "model.pt")

    config = {
        "cls_dim": 768,
        "gpu_dim": 4,
        "cpu_dim": 24,
        "gpu_indices": GPU_4D_IDX,
        "cpu16_indices": CPU16_13D_IDX,
        "cpu20_indices": CPU20_11D_IDX,
        "use_mid_adj": False,
        "pr_auc": float(pr_auc),  # 最終エポックのPR-AUC
        "best_epoch": MAX_EPOCHS,  # 固定エポック
        "seed": SEED,
        "training_categories": AI_CATEGORIES,
    }
    np.save(MODEL_DIR / "config.npy", config)

    print(f"Model saved to {MODEL_DIR}")
    print(f"AI categories: {len(AI_CATEGORIES)}")
    print(f"Total AI samples: {len(ai_cls)}")


if __name__ == "__main__":
    main()
