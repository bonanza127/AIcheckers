#!/usr/bin/env python3
"""
28d_plus_60: スクラッチから60エポック学習
- 重複カテゴリ除外済み
- patch_stats_v3形式
- 最終エポック（60）の重みを保存
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from pathlib import Path
import json

# 設定
START_EPOCH = 0  # スクラッチから
MAX_EPOCHS = 100  # Early Stoppingで自動停止
BATCH_SIZE = 512
LR = 1e-3
WEIGHT_DECAY = 1e-5
SEED = 42
VAL_RATIO = 0.1
STD_FLOOR = 1e-3
PATIENCE = 15  # Early Stopping patience

# 特徴量インデックス
GPU_4D_IDX = [1, 3, 5, 6]
CPU16_13D_IDX = [0, 1, 2, 4, 5, 7, 8, 9, 11, 12, 13, 14, 15]
CPU20_11D_IDX = [0, 1, 2, 3, 4, 5, 8, 10, 15, 16, 17]

# カテゴリ（重複除外済み）
# - novelai_aibooru_ai: novelai_aiと重複
# - pixiv_novelai_v2_ai, twitter_novelai_v2_ai: novelai_combined_aiに統合済み
AI_CATEGORIES = [
    'illustrious_ai', 'pony_ai', 'sdxl10_ai', 'sd15_ai', 'flux1d_ai',
    'novelai_combined_ai', 'other_ai', 'novelai_ai', 'novelai_artist_tagged_ai', 'pixai_ai',
    'niji7_twitter_ai',
]
REAL_CATEGORIES = ['danbooru_real']

CATEGORY_CAP = {
    'pony_ai': 10000,
    'novelai_combined_ai': 10000,
}

EMB_DIR = Path('embeddings')
MODEL_DIR = Path('models/two_head_28d_plus_60')


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

    def forward(self, cls_feat, gpu_feat, cpu_feat):
        x = torch.cat([cls_feat, gpu_feat, cpu_feat], dim=-1)
        x = self.bn_input(x)
        x = F.gelu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.gelu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        return self.fc3(x)


def main():
    print("=" * 60)
    print("Two-Head 28d_plus_60 Training (Scratch)")
    print(f"Training from ep1 to ep{MAX_EPOCHS}")
    print("=" * 60)

    ai_cls, ai_gpu, ai_cpu = [], [], []
    real_cls, real_gpu, real_cpu = [], [], []

    print("Loading data...")
    for cat in AI_CATEGORIES:
        result = load_category_data(cat)
        if result:
            c, g, p = result
            if cat in CATEGORY_CAP and len(c) > CATEGORY_CAP[cat]:
                cap = CATEGORY_CAP[cat]
                idx = np.random.choice(len(c), cap, replace=False)
                c, g, p = c[idx], g[idx], p[idx]
                print(f"  {cat}: {len(c)} samples (capped from original)")
            else:
                print(f"  {cat}: {len(c)} samples")
            ai_cls.append(c); ai_gpu.append(g); ai_cpu.append(p)

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
    np.random.seed(SEED)
    # 少ない方に合わせてバランシング
    n_samples = min(len(ai_cls), len(real_cls))
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
    y = np.array([1]*len(ai_cls) + [0]*len(real_cls), dtype=np.float32)

    indices = np.arange(len(y))
    train_idx, val_idx = train_test_split(indices, test_size=VAL_RATIO, random_state=SEED, stratify=y)
    print(f"\nTrain: {len(train_idx)}, Val: {len(val_idx)}")

    device = torch.device("cuda")
    model = TwoHeadClassifier().to(device)

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
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    criterion = nn.BCEWithLogitsLoss()

    best_pr_auc = 0
    best_epoch = 0
    best_state_dict = None
    no_improve_count = 0

    for epoch in range(START_EPOCH + 1, MAX_EPOCHS + 1):
        model.train()
        for cls_b, gpu_b, cpu_b, y_b in train_loader:
            cls_b, gpu_b, cpu_b, y_b = cls_b.to(device), gpu_b.to(device), cpu_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(cls_b, gpu_b, cpu_b).squeeze(), y_b)
            loss.backward()
            optimizer.step()

        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for cls_b, gpu_b, cpu_b, y_b in val_loader:
                probs = torch.sigmoid(model(cls_b.to(device), gpu_b.to(device), cpu_b.to(device)).squeeze()).cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(y_b.numpy())

        pr_auc = average_precision_score(all_labels, all_probs)
        scheduler.step(pr_auc)

        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            best_epoch = epoch
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve_count = 0
        else:
            no_improve_count += 1

        if epoch % 5 == 0 or epoch == 1:
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:3d} - PR-AUC: {pr_auc:.4f} (best: {best_pr_auc:.4f} @ ep{best_epoch}) lr={lr:.1e}")

        # Early Stopping
        if no_improve_count >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
            break

    print(f"\nFinal PR-AUC: {pr_auc:.4f} (best was {best_pr_auc:.4f} @ ep{best_epoch})")

    # Save BEST epoch weights
    MODEL_DIR.mkdir(exist_ok=True, parents=True)
    if best_state_dict is None:
        print("[WARN] best_state_dict is None, saving current model state instead")
        best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        best_epoch = epoch
    torch.save(best_state_dict, MODEL_DIR / "model.pt")

    config = {
        "version": "28d_plus_v5",
        "gpu_dim": 4, "cpu_dim": 24,
        "training_categories": AI_CATEGORIES,
        "final_pr_auc": float(pr_auc),
        "best_pr_auc": float(best_pr_auc),
        "best_epoch": best_epoch,
        "saved_epoch": best_epoch,
        "notes": "ReduceLROnPlateau + EarlyStopping, best epoch saved"
    }
    with open(MODEL_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"Model saved to {MODEL_DIR} (best epoch {best_epoch} weights)")


if __name__ == "__main__":
    main()
