#!/usr/bin/env python3
"""
30d Two-Head モデル学習スクリプト
- GPU 5d: patch_var, degree_centrality, local_efficiency, edge_interior_gap, mid_adj_sim_var
- CPU16 13d: banding_score, radial_spectrum_slope, text_area_ratio, patchwise_edge_density,
             st_aniso_mean, st_aniso_spatial_gradient, flat_boundary_peri_area, flat_hole_ratio,
             flat_ratio, patch_vs_global_st_aniso_gap, cbcr_autocorr, edge_length_mean, rank_entropy
- CPU20 12d: histogram_modality, color_palette_entropy, luminance_layer_count, luminance_skewness,
             value_bimodality, multiscale_variance_ratio, luminance_mean, saturation_mean,
             radial_spectrum_slope_patch_gap, color_banding_score, compression_artifact_pattern,
             edge_continuity_ratio
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
EPOCHS = 60
BATCH_SIZE = 512
LR = 1e-3
SEED = 42
VAL_RATIO = 0.1

# 特徴量インデックス
# GPU: patch_stats_v3から (34dのうち5d選択)
GPU_5D_IDX = [1, 3, 5, 6]  # patch_var, degree_centrality, local_efficiency, edge_interior_gap
# mid_adj_sim_varは別ファイルから読み込む

# CPU16: cpu_stats_v2から (18dのうち13d選択、fractal_dim_edge_512除外)
CPU16_13D_IDX = [0, 1, 2, 4, 5, 7, 8, 9, 11, 12, 13, 14, 15]

# CPU20: cpu_stats_v3_20dから (20dのうち12d選択、quantization_step_count, noise_floor_variance除外)
CPU20_12D_IDX = [0, 1, 2, 3, 4, 5, 8, 10, 15, 16, 17, 18]

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

    # 必須ファイルチェック
    required = [cls_path, gpu_path, cpu16_path, cpu20_path, mid_adj_path]
    for p in required:
        if not p.exists():
            return None

    cls = np.load(cls_path)
    gpu = np.load(gpu_path)[:, GPU_5D_IDX]
    cpu16 = np.load(cpu16_path)[:, CPU16_13D_IDX]
    cpu20 = np.load(cpu20_path)[:, CPU20_12D_IDX]
    mid_adj = np.load(mid_adj_path).reshape(-1, 1)

    # サンプル数を揃える
    min_len = min(len(cls), len(gpu), len(cpu16), len(cpu20), len(mid_adj))
    cls = cls[:min_len]
    gpu = gpu[:min_len]
    cpu16 = cpu16[:min_len]
    cpu20 = cpu20[:min_len]
    mid_adj = mid_adj[:min_len]

    # GPU特徴量にmid_adj_sim_varを追加 (4d + 1d = 5d)
    gpu_full = np.hstack([gpu, mid_adj])

    # CPU特徴量を結合 (13d + 12d = 25d)
    cpu_full = np.hstack([cpu16, cpu20])

    return cls, gpu_full, cpu_full


def load_all_data():
    """全データを並列読み込み"""
    print("Loading data...")

    ai_cls_list, ai_gpu_list, ai_cpu_list = [], [], []
    real_cls_list, real_gpu_list, real_cpu_list = [], [], []

    def load_cat(cat):
        return cat, load_category_data(cat)

    # 並列読み込み
    with ThreadPoolExecutor(max_workers=8) as executor:
        # AI
        ai_futures = [executor.submit(load_cat, cat) for cat in AI_CATEGORIES]
        for future in ai_futures:
            cat, data = future.result()
            if data is not None:
                cls, gpu, cpu = data
                ai_cls_list.append(cls)
                ai_gpu_list.append(gpu)
                ai_cpu_list.append(cpu)
                print(f"  {cat}: {len(cls)} samples")

        # Real
        real_futures = [executor.submit(load_cat, cat) for cat in REAL_CATEGORIES]
        for future in real_futures:
            cat, data = future.result()
            if data is not None:
                cls, gpu, cpu = data
                real_cls_list.append(cls)
                real_gpu_list.append(gpu)
                real_cpu_list.append(cpu)
                print(f"  {cat}: {len(cls)} samples")

    # 結合
    ai_cls = np.vstack(ai_cls_list)
    ai_gpu = np.vstack(ai_gpu_list)
    ai_cpu = np.vstack(ai_cpu_list)

    real_cls = np.vstack(real_cls_list)
    real_gpu = np.vstack(real_gpu_list)
    real_cpu = np.vstack(real_cpu_list)

    print(f"\nBefore balancing: AI={len(ai_cls)}, Real={len(real_cls)}")

    # バランシング（小さい方に合わせる）
    n_samples = min(len(ai_cls), len(real_cls))
    np.random.seed(SEED)

    if len(ai_cls) > n_samples:
        idx = np.random.choice(len(ai_cls), n_samples, replace=False)
        ai_cls, ai_gpu, ai_cpu = ai_cls[idx], ai_gpu[idx], ai_cpu[idx]

    if len(real_cls) > n_samples:
        idx = np.random.choice(len(real_cls), n_samples, replace=False)
        real_cls, real_gpu, real_cpu = real_cls[idx], real_gpu[idx], real_cpu[idx]

    print(f"After balancing: AI={len(ai_cls)}, Real={len(real_cls)}")

    # 結合
    X_cls = np.vstack([ai_cls, real_cls])
    X_gpu = np.vstack([ai_gpu, real_gpu])
    X_cpu = np.vstack([ai_cpu, real_cpu])
    y = np.array([1] * len(ai_cls) + [0] * len(real_cls))

    # NaN処理
    X_cls = np.nan_to_num(X_cls, nan=0.0, posinf=0.0, neginf=0.0)
    X_gpu = np.nan_to_num(X_gpu, nan=0.0, posinf=0.0, neginf=0.0)
    X_cpu = np.nan_to_num(X_cpu, nan=0.0, posinf=0.0, neginf=0.0)

    return X_cls, X_gpu, X_cpu, y


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


def train_model(X_cls, X_gpu, X_cpu, y):
    """モデル学習"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nTraining on {device}")
    print(f"Feature dims: CLS={X_cls.shape[1]}, GPU={X_gpu.shape[1]}, CPU={X_cpu.shape[1]}")
    print(f"Total: {X_cls.shape[1] + X_gpu.shape[1] + X_cpu.shape[1]}d")

    # Train/Val分割
    (X_cls_train, X_cls_val, X_gpu_train, X_gpu_val,
     X_cpu_train, X_cpu_val, y_train, y_val) = train_test_split(
        X_cls, X_gpu, X_cpu, y, test_size=VAL_RATIO, random_state=SEED, stratify=y
    )

    print(f"Train: {len(y_train)}, Val: {len(y_val)}")

    # Z-score統計量計算
    cls_mean = X_cls_train.mean(axis=0)
    cls_std = X_cls_train.std(axis=0)
    gpu_mean = X_gpu_train.mean(axis=0)
    gpu_std = X_gpu_train.std(axis=0)
    cpu_mean = X_cpu_train.mean(axis=0)
    cpu_std = X_cpu_train.std(axis=0)

    # Tensor変換
    X_cls_train_t = torch.from_numpy(X_cls_train).float()
    X_gpu_train_t = torch.from_numpy(X_gpu_train).float()
    X_cpu_train_t = torch.from_numpy(X_cpu_train).float()
    y_train_t = torch.from_numpy(y_train).float().unsqueeze(1)

    X_cls_val_t = torch.from_numpy(X_cls_val).float()
    X_gpu_val_t = torch.from_numpy(X_gpu_val).float()
    X_cpu_val_t = torch.from_numpy(X_cpu_val).float()
    y_val_t = torch.from_numpy(y_val).float().unsqueeze(1)

    # DataLoader
    train_dataset = TensorDataset(X_cls_train_t, X_gpu_train_t, X_cpu_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)

    # モデル
    model = TwoHeadClassifier(
        cls_dim=X_cls.shape[1],
        gpu_dim=X_gpu.shape[1],
        cpu_dim=X_cpu.shape[1]
    ).to(device)

    # 統計量設定
    model.cls_mean.copy_(torch.from_numpy(cls_mean).float())
    model.cls_std.copy_(torch.from_numpy(cls_std).float())
    model.gpu_mean.copy_(torch.from_numpy(gpu_mean).float())
    model.gpu_std.copy_(torch.from_numpy(gpu_std).float())
    model.cpu_mean.copy_(torch.from_numpy(cpu_mean).float())
    model.cpu_std.copy_(torch.from_numpy(cpu_std).float())

    # 最適化
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.BCEWithLogitsLoss()

    # Validation data to device
    X_cls_val_t = X_cls_val_t.to(device)
    X_gpu_val_t = X_gpu_val_t.to(device)
    X_cpu_val_t = X_cpu_val_t.to(device)
    y_val_t = y_val_t.to(device)

    best_pr_auc = 0
    best_epoch = 0

    for epoch in range(1, EPOCHS + 1):
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

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_cls_val_t, X_gpu_val_t, X_cpu_val_t)
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            val_preds = (val_probs > 0.5).astype(int)
            val_acc = (val_preds == y_val.reshape(-1, 1)).mean()
            pr_auc = average_precision_score(y_val, val_probs)

        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            best_epoch = epoch
            best_state = model.state_dict().copy()

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:2d}/{EPOCHS} - Loss: {total_loss/len(train_loader):.4f} - "
                  f"Val Acc: {val_acc:.2%} - PR-AUC: {pr_auc:.4f}")

    print(f"\nBest PR-AUC: {best_pr_auc:.4f} at epoch {best_epoch}")

    # Best modelをロード
    model.load_state_dict(best_state)

    return model, best_pr_auc


def main():
    print("=" * 60)
    print("Two-Head 30d Model Training")
    print(f"GPU 5d + CPU 25d (13d + 12d)")
    print(f"Epochs: {EPOCHS}, Seed: {SEED}")
    print("=" * 60)

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # データ読み込み
    X_cls, X_gpu, X_cpu, y = load_all_data()

    # 学習
    model, pr_auc = train_model(X_cls, X_gpu, X_cpu, y)

    # 保存
    MODEL_DIR.mkdir(exist_ok=True)
    out_path = MODEL_DIR / 'two_head_30d'
    out_path.mkdir(exist_ok=True)

    # バックアップ
    if (out_path / 'model.pt').exists():
        backup_name = f"model_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        shutil.copy(out_path / 'model.pt', out_path / backup_name)
        print(f"Backup: {backup_name}")

    torch.save(model.state_dict(), out_path / 'model.pt')

    # 設定保存
    config = {
        'cls_dim': 768,
        'gpu_dim': 5,
        'cpu_dim': 25,
        'gpu_indices': GPU_5D_IDX,
        'cpu16_indices': CPU16_13D_IDX,
        'cpu20_indices': CPU20_12D_IDX,
        'pr_auc': pr_auc,
        'epochs': EPOCHS,
        'seed': SEED,
    }
    np.save(out_path / 'config.npy', config)

    print(f"\nModel saved to {out_path}")
    print(f"Final PR-AUC: {pr_auc:.4f}")


if __name__ == '__main__':
    main()
