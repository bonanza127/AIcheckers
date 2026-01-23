#!/usr/bin/env python3
"""
28d + extra (niji7以外) Two-Head モデル学習スクリプト
28dと同じ設定で、28d_plusの追加カテゴリからniji7を除いたもの
"""

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from pathlib import Path
import json

# 設定（28dと同じ）
MAX_EPOCHS = 60
BATCH_SIZE = 512
LR = 1e-3
SEED = 42
VAL_RATIO = 0.1
PATIENCE = 8
MIN_DELTA = 1e-4
STD_FLOOR = 1e-3

# 特徴量インデックス
GPU_4D_IDX = [1, 3, 5, 6]
CPU16_13D_IDX = [0, 1, 2, 4, 5, 7, 8, 9, 11, 12, 13, 14, 15]
CPU20_11D_IDX = [0, 1, 2, 3, 4, 5, 8, 10, 15, 16, 17]

# カテゴリ（28d + 28d_plusの追加分からniji7を除く）
AI_CATEGORIES = [
    # 28dオリジナル
    'illustrious_ai', 'pony_ai', 'sdxl10_ai', 'sd15_ai', 'flux1d_ai',
    'novelai_combined_ai', 'other_ai', 'novelai_ai', 'novelai_artist_tagged_ai', 'pixai_ai',
    # 28d_plusの追加分（niji7以外）
    'novelai_aibooru_ai', 'pixiv_novelai_v2_ai', 'twitter_novelai_v2_ai',
]
REAL_CATEGORIES = ['danbooru_real']

EMB_DIR = Path('embeddings')
MODEL_DIR = Path('models/two_head_28d_extra')


def sanitize_array(arr, name="array"):
    arr = arr.astype(np.float32)
    mask = ~np.isfinite(arr)
    if mask.any():
        print(f"  Warning: {name} has {mask.sum()} inf/nan values, replacing with 0")
        arr = np.where(mask, 0.0, arr)
    arr = np.clip(arr, -1e6, 1e6)
    return arr


def load_category_data(cat):
    cls_path = EMB_DIR / f'{cat}.npy'
    gpu_path = EMB_DIR / f'{cat}_patch_stats_v3.npy'
    cpu16_path = EMB_DIR / f'{cat}_cpu_stats_v2.npy'
    cpu20_path = EMB_DIR / f'{cat}_cpu_stats_v3_20d.npy'
    for p in [cls_path, gpu_path, cpu16_path, cpu20_path]:
        if not p.exists():
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
        x = torch.cat([cls_norm, gpu_norm, cpu_norm], dim=-1)
        x = self.bn_input(x)
        x = F.gelu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.gelu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        return self.fc3(x)


class EarlyStopping:
    def __init__(self, patience=PATIENCE, min_delta=MIN_DELTA):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = -np.inf
        self.best_epoch = 0
        self.counter = 0
        self.best_state = None

    def __call__(self, score, epoch, model):
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return False
        self.counter += 1
        return self.counter >= self.patience


def main():
    print("=" * 60)
    print("Two-Head 28d + extra (no niji7) Model Training")
    print(f"Early stopping: patience={PATIENCE}")
    print("=" * 60)

    ai_cls, ai_gpu, ai_cpu = [], [], []
    real_cls, real_gpu, real_cpu = [], [], []

    print("Loading data...")
    for cat in AI_CATEGORIES:
        result = load_category_data(cat)
        if result:
            c, g, p = result
            ai_cls.append(c); ai_gpu.append(g); ai_cpu.append(p)
            print(f"  {cat}: {len(c)} samples")

    for cat in REAL_CATEGORIES:
        result = load_category_data(cat)
        if result:
            c, g, p = result
            real_cls.append(c); real_gpu.append(g); real_cpu.append(p)
            print(f"  {cat}: {len(c)} samples")

    ai_cls = sanitize_array(np.vstack(ai_cls), "ai_cls")
    ai_gpu = sanitize_array(np.vstack(ai_gpu), "ai_gpu")
    ai_cpu = sanitize_array(np.vstack(ai_cpu), "ai_cpu")
    real_cls = sanitize_array(np.vstack(real_cls), "real_cls")
    real_gpu = sanitize_array(np.vstack(real_gpu), "real_gpu")
    real_cpu = sanitize_array(np.vstack(real_cpu), "real_cpu")

    print(f"\nBefore balancing: AI={len(ai_cls)}, Real={len(real_cls)}")
    n_ai = len(ai_cls)
    np.random.seed(SEED)
    idx = np.random.choice(len(real_cls), n_ai, replace=False)
    real_cls, real_gpu, real_cpu = real_cls[idx], real_gpu[idx], real_cpu[idx]
    print(f"After balancing: AI={len(ai_cls)}, Real={len(real_cls)}")

    X_cls = np.vstack([ai_cls, real_cls])
    X_gpu = np.vstack([ai_gpu, real_gpu])
    X_cpu = np.vstack([ai_cpu, real_cpu])
    y = np.array([1]*len(ai_cls) + [0]*len(real_cls), dtype=np.float32)

    indices = np.arange(len(y))
    train_idx, val_idx = train_test_split(indices, test_size=VAL_RATIO, random_state=SEED, stratify=y)
    print(f"\nTrain: {len(train_idx)}, Val: {len(val_idx)}")

    device = torch.device("cuda")
    model = TwoHeadClassifier().to(device)

    model.cls_mean.copy_(torch.tensor(X_cls[train_idx].mean(0)))
    model.cls_std.copy_(torch.tensor(X_cls[train_idx].std(0)))
    model.gpu_mean.copy_(torch.tensor(X_gpu[train_idx].mean(0)))
    model.gpu_std.copy_(torch.tensor(X_gpu[train_idx].std(0)))
    model.cpu_mean.copy_(torch.tensor(X_cpu[train_idx].mean(0)))
    model.cpu_std.copy_(torch.tensor(X_cpu[train_idx].std(0)))

    train_ds = TensorDataset(torch.tensor(X_cls[train_idx]), torch.tensor(X_gpu[train_idx]),
                             torch.tensor(X_cpu[train_idx]), torch.tensor(y[train_idx]))
    val_ds = TensorDataset(torch.tensor(X_cls[val_idx]), torch.tensor(X_gpu[val_idx]),
                           torch.tensor(X_cpu[val_idx]), torch.tensor(y[val_idx]))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    criterion = nn.BCEWithLogitsLoss()
    early_stopping = EarlyStopping()

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
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} - PR-AUC: {pr_auc:.4f} (best: {early_stopping.best_score:.4f} @ ep{early_stopping.best_epoch}) [{early_stopping.counter}/{PATIENCE}]")

        if early_stopping(pr_auc, epoch, model):
            print(f"\n*** Early stopping at epoch {epoch} ***")
            break

    print(f"\nBest PR-AUC: {early_stopping.best_score:.4f} at epoch {early_stopping.best_epoch}")

    MODEL_DIR.mkdir(exist_ok=True, parents=True)
    model.load_state_dict(early_stopping.best_state)
    torch.save(model.state_dict(), MODEL_DIR / "model.pt")

    config = {
        "version": "28d_extra", "gpu_dim": 4, "cpu_dim": 24,
        "training_categories": AI_CATEGORIES,
        "pr_auc": float(early_stopping.best_score), "best_epoch": early_stopping.best_epoch,
        "notes": "28d + extra categories (no niji7)"
    }
    with open(MODEL_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"Model saved to {MODEL_DIR}")


if __name__ == "__main__":
    main()
