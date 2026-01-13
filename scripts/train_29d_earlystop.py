#!/usr/bin/env python3
"""
29d Two-Head モデル学習スクリプト (Early Stopping版)

特徴:
- PR-AUC based early stopping (patience=8, min_delta=1e-4)
- std floor 1e-3 for normalization safety
- patchwise_edge_density を含む (CPU 13d + 11d = 24d)

構成:
- GPU 5d: adj_sim_var[1], patch_var[3], norm_var[5], norm_range[6], mid_adj_sim_var
- CPU16 13d: banding_score, radial_spectrum_slope, patchwise_edge_density等
- CPU20 11d: histogram_modality等（edge_continuity_ratio除外）
"""

import os
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
import shutil
from concurrent.futures import ThreadPoolExecutor

# 設定
MAX_EPOCHS = 100  # 上限（早期終了で切り上げ）
BATCH_SIZE = 512
LR = 1e-3
SEED = 42
VAL_RATIO = 0.1

# Early stopping
PATIENCE = 8
MIN_DELTA = 1e-4

# Normalization safety
STD_FLOOR = 1e-3

# 特徴量インデックス
GPU_4D_IDX = [1, 3, 5, 6]
CPU16_13D_IDX = [0, 1, 2, 4, 5, 7, 8, 9, 11, 12, 13, 14, 15]  # patchwise_edge_density含む
CPU20_11D_IDX = [0, 1, 2, 3, 4, 5, 8, 10, 15, 16, 17]  # edge_continuity_ratio除外

# カテゴリ
AI_CATEGORIES = [
    'illustrious_ai', 'pony_ai', 'sdxl10_ai', 'sd15_ai', 'flux1d_ai',
    'novelai_combined_ai', 'other_ai', 'novelai_ai', 'novelai_artist_tagged_ai', 'pixai_ai'
]
REAL_CATEGORIES = ['danbooru_real']

EMB_DIR = Path('embeddings')
MODEL_DIR = Path('models')


def load_category_data(cat: str) -> tuple:
    """カテゴリのデータを読み込み"""
    cls_path = EMB_DIR / f'{cat}.npy'
    gpu_path = EMB_DIR / f'{cat}_patch_stats_v3.npy'
    cpu16_path = EMB_DIR / f'{cat}_cpu_stats_v2.npy'
    cpu20_path = EMB_DIR / f'{cat}_cpu_stats_v3_20d.npy'
    mid_adj_path = EMB_DIR / f'{cat}_mid_adj_sim_var.npy'

    required = [cls_path, gpu_path, cpu16_path, cpu20_path, mid_adj_path]
    for p in required:
        if not p.exists():
            return None

    cls = np.load(cls_path)
    gpu = np.load(gpu_path)[:, GPU_4D_IDX]
    cpu16 = np.load(cpu16_path)[:, CPU16_13D_IDX]
    cpu20 = np.load(cpu20_path)[:, CPU20_11D_IDX]
    mid_adj = np.load(mid_adj_path).reshape(-1, 1)

    min_len = min(len(cls), len(gpu), len(cpu16), len(cpu20), len(mid_adj))
    cls = cls[:min_len]
    gpu = gpu[:min_len]
    cpu16 = cpu16[:min_len]
    cpu20 = cpu20[:min_len]
    mid_adj = mid_adj[:min_len]

    gpu_full = np.hstack([gpu, mid_adj])
    cpu_full = np.hstack([cpu16, cpu20])

    return cls, gpu_full, cpu_full


def load_all_data():
    """全データを並列読み込み"""
    print("Loading data...")

    ai_cls_list, ai_gpu_list, ai_cpu_list = [], [], []
    real_cls_list, real_gpu_list, real_cpu_list = [], [], []

    def load_cat(cat):
        return cat, load_category_data(cat)

    with ThreadPoolExecutor(max_workers=8) as executor:
        ai_futures = [executor.submit(load_cat, cat) for cat in AI_CATEGORIES]
        for future in ai_futures:
            cat, data = future.result()
            if data is not None:
                cls, gpu, cpu = data
                ai_cls_list.append(cls)
                ai_gpu_list.append(gpu)
                ai_cpu_list.append(cpu)
                print(f"  {cat}: {len(cls)} samples")

        real_futures = [executor.submit(load_cat, cat) for cat in REAL_CATEGORIES]
        for future in real_futures:
            cat, data = future.result()
            if data is not None:
                cls, gpu, cpu = data
                real_cls_list.append(cls)
                real_gpu_list.append(gpu)
                real_cpu_list.append(cpu)
                print(f"  {cat}: {len(cls)} samples")

    ai_cls = np.vstack(ai_cls_list)
    ai_gpu = np.vstack(ai_gpu_list)
    ai_cpu = np.vstack(ai_cpu_list)

    real_cls = np.vstack(real_cls_list)
    real_gpu = np.vstack(real_gpu_list)
    real_cpu = np.vstack(real_cpu_list)

    print(f"\nBefore balancing: AI={len(ai_cls)}, Real={len(real_cls)}")

    n_samples = min(len(ai_cls), len(real_cls))
    np.random.seed(SEED)

    if len(ai_cls) > n_samples:
        idx = np.random.choice(len(ai_cls), n_samples, replace=False)
        ai_cls, ai_gpu, ai_cpu = ai_cls[idx], ai_gpu[idx], ai_cpu[idx]

    if len(real_cls) > n_samples:
        idx = np.random.choice(len(real_cls), n_samples, replace=False)
        real_cls, real_gpu, real_cpu = real_cls[idx], real_gpu[idx], real_cpu[idx]

    print(f"After balancing: AI={len(ai_cls)}, Real={len(real_cls)}")

    X_cls = np.vstack([ai_cls, real_cls])
    X_gpu = np.vstack([ai_gpu, real_gpu])
    X_cpu = np.vstack([ai_cpu, real_cpu])
    y = np.array([1] * len(ai_cls) + [0] * len(real_cls))

    X_cls = np.nan_to_num(X_cls, nan=0.0, posinf=0.0, neginf=0.0)
    X_gpu = np.nan_to_num(X_gpu, nan=0.0, posinf=0.0, neginf=0.0)
    X_cpu = np.nan_to_num(X_cpu, nan=0.0, posinf=0.0, neginf=0.0)

    return X_cls, X_gpu, X_cpu, y


class TwoHeadClassifier(nn.Module):
    """Two-Head 29d分類器 (CLS 768d + GPU 5d + CPU 24d)"""

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


class EarlyStopping:
    """PR-AUC based early stopping"""

    def __init__(self, patience=PATIENCE, min_delta=MIN_DELTA):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = -np.inf
        self.best_epoch = 0
        self.counter = 0
        self.best_state = None
        self.should_stop = False

    def __call__(self, score, epoch, model):
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.best_epoch = epoch
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


def train_model(X_cls, X_gpu, X_cpu, y):
    """モデル学習 with early stopping"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nTraining on {device}")
    print(f"Feature dims: CLS={X_cls.shape[1]}, GPU={X_gpu.shape[1]}, CPU={X_cpu.shape[1]}")
    print(f"Early stopping: patience={PATIENCE}, min_delta={MIN_DELTA}")

    (X_cls_train, X_cls_val, X_gpu_train, X_gpu_val,
     X_cpu_train, X_cpu_val, y_train, y_val) = train_test_split(
        X_cls, X_gpu, X_cpu, y, test_size=VAL_RATIO, random_state=SEED, stratify=y
    )

    print(f"Train: {len(y_train)}, Val: {len(y_val)}")

    cls_mean = X_cls_train.mean(axis=0)
    cls_std = X_cls_train.std(axis=0)
    gpu_mean = X_gpu_train.mean(axis=0)
    gpu_std = X_gpu_train.std(axis=0)
    cpu_mean = X_cpu_train.mean(axis=0)
    cpu_std = X_cpu_train.std(axis=0)

    print("\nChecking std values...")
    danger = False
    for i, s in enumerate(cpu_std):
        if s < STD_FLOOR:
            print(f"  WARNING: cpu_std[{i}] = {s:.6f} < {STD_FLOOR} (will be floored)")
            danger = True
    if not danger:
        print("  All std values are safe!")

    X_cls_train_t = torch.from_numpy(X_cls_train).float()
    X_gpu_train_t = torch.from_numpy(X_gpu_train).float()
    X_cpu_train_t = torch.from_numpy(X_cpu_train).float()
    y_train_t = torch.from_numpy(y_train).float().unsqueeze(1)

    X_cls_val_t = torch.from_numpy(X_cls_val).float()
    X_gpu_val_t = torch.from_numpy(X_gpu_val).float()
    X_cpu_val_t = torch.from_numpy(X_cpu_val).float()

    train_dataset = TensorDataset(X_cls_train_t, X_gpu_train_t, X_cpu_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)

    model = TwoHeadClassifier(
        cls_dim=X_cls.shape[1],
        gpu_dim=X_gpu.shape[1],
        cpu_dim=X_cpu.shape[1]
    ).to(device)

    model.cls_mean.copy_(torch.from_numpy(cls_mean).float())
    model.cls_std.copy_(torch.from_numpy(cls_std).float())
    model.gpu_mean.copy_(torch.from_numpy(gpu_mean).float())
    model.gpu_std.copy_(torch.from_numpy(gpu_std).float())
    model.cpu_mean.copy_(torch.from_numpy(cpu_mean).float())
    model.cpu_std.copy_(torch.from_numpy(cpu_std).float())

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    criterion = nn.BCEWithLogitsLoss()

    X_cls_val_t = X_cls_val_t.to(device)
    X_gpu_val_t = X_gpu_val_t.to(device)
    X_cpu_val_t = X_cpu_val_t.to(device)

    early_stopping = EarlyStopping(patience=PATIENCE, min_delta=MIN_DELTA)

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss = 0

        for cls_b, gpu_b, cpu_b, y_b in train_loader:
            cls_b = cls_b.to(device)
            gpu_b = gpu_b.to(device)
            cpu_b = cpu_b.to(device)
            y_b = y_b.to(device)

            optimizer.zero_grad()
            logits = model(cls_b, gpu_b, cpu_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_cls_val_t, X_gpu_val_t, X_cpu_val_t)
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            val_preds = (val_probs > 0.5).astype(int)
            val_acc = (val_preds == y_val.reshape(-1, 1)).mean()
            pr_auc = average_precision_score(y_val, val_probs)

        # Early stopping check
        stopped = early_stopping(pr_auc, epoch, model)

        if epoch % 5 == 0 or epoch == 1 or stopped:
            status = f"(best: {early_stopping.best_score:.4f} @ ep{early_stopping.best_epoch})"
            patience_str = f"[{early_stopping.counter}/{PATIENCE}]"
            print(f"Epoch {epoch:3d} - Loss: {total_loss/len(train_loader):.4f} - "
                  f"Val Acc: {val_acc:.2%} - PR-AUC: {pr_auc:.4f} {status} {patience_str}")

        if stopped:
            print(f"\n*** Early stopping at epoch {epoch} ***")
            break

    print(f"\nBest PR-AUC: {early_stopping.best_score:.4f} at epoch {early_stopping.best_epoch}")

    model.load_state_dict(early_stopping.best_state)

    return model, early_stopping.best_score, early_stopping.best_epoch


def main():
    print("=" * 60)
    print("Two-Head 29d Model Training (Early Stopping)")
    print(f"GPU 5d + CPU 24d (13d + 11d)")
    print(f"Removed: edge_continuity_ratio only")
    print(f"Early stopping: patience={PATIENCE}, min_delta={MIN_DELTA}")
    print(f"Std floor: {STD_FLOOR}")
    print("=" * 60)

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    X_cls, X_gpu, X_cpu, y = load_all_data()

    model, pr_auc, best_epoch = train_model(X_cls, X_gpu, X_cpu, y)

    MODEL_DIR.mkdir(exist_ok=True)
    out_path = MODEL_DIR / 'two_head_29d'
    out_path.mkdir(exist_ok=True)

    if (out_path / 'model.pt').exists():
        backup_name = f"model_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        shutil.copy(out_path / 'model.pt', out_path / backup_name)
        print(f"Backup: {backup_name}")

    torch.save(model.state_dict(), out_path / 'model.pt')

    config = {
        'cls_dim': 768,
        'gpu_dim': 5,
        'cpu_dim': 24,
        'gpu_indices': GPU_4D_IDX,
        'cpu16_indices': CPU16_13D_IDX,
        'cpu20_indices': CPU20_11D_IDX,
        'pr_auc': pr_auc,
        'best_epoch': best_epoch,
        'patience': PATIENCE,
        'min_delta': MIN_DELTA,
        'std_floor': STD_FLOOR,
        'seed': SEED,
        'removed_features': ['edge_continuity_ratio (20d[18], std=6.7e-05, d=+0.31)'],
    }
    np.save(out_path / 'config.npy', config)

    print(f"\nModel saved to {out_path}")
    print(f"Final PR-AUC: {pr_auc:.4f} (epoch {best_epoch})")


if __name__ == '__main__':
    main()
