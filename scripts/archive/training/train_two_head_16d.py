#!/usr/bin/env python3
"""
Train 15d 2-head classifier: CLS (768d) + Stats (15d)

15d構成:
- 既存11d (v3から): adj_sim_var, high_sim_ratio, patch_var, norm_var, cls_sim_mean,
                    cls_angle_disp, knn_sim_var, top3_ratio, degree_cent,
                    clustering_coef, center_edge_hi
- 新規4d (計算):    local_efficiency, corner_coherence,
                    edge_interior_gap, cls_sim_center_bias

Total input: 768 + 15 = 783d
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import average_precision_score, roc_auc_score


EMBEDDINGS_DIR = Path("/home/techne/aicheckers/embeddings")
DEFAULT_OUT_DIR = Path("/home/techne/aicheckers/models/two_head_15d")

# 既存11d: v3統計量からのインデックス
KEEP_IDX_11D = [1, 2, 3, 5, 9, 12, 14, 17, 18, 19, 32]
FEAT_NAMES_11D = [
    'adj_sim_var', 'high_sim_ratio', 'patch_var', 'norm_var', 'cls_sim_mean',
    'cls_angle_disp', 'knn_sim_var', 'top3_ratio', 'degree_cent',
    'clustering_coef', 'center_edge_hi'
]

# 新規4d: 特徴量名
NEW_FEAT_NAMES = [
    'local_efficiency', 'corner_coherence',
    'edge_interior_gap', 'cls_sim_center_bias'
]


def discover_categories(emb_dir: Path):
    """カテゴリを自動検出"""
    cats = []
    for stats_path in emb_dir.glob("*_patch_stats_v3.npy"):
        name = stats_path.name.replace("_patch_stats_v3.npy", "")
        cls_path = emb_dir / f"{name}.npy"
        patches_path = emb_dir / f"{name}_mid_patches.npy"
        mid_cls_path = emb_dir / f"{name}_mid_cls.npy"

        if not all(p.exists() for p in [cls_path, patches_path, mid_cls_path]):
            continue
        if name.endswith("_ai"):
            label = 1
        elif name.endswith("_real"):
            label = 0
        else:
            continue
        cats.append((name, label))
    return sorted(cats)


def compute_new_4d_features(patches: np.ndarray, mid_cls: np.ndarray,
                            batch_size: int = 32) -> np.ndarray:
    """新規4d特徴量を計算

    Args:
        patches: (N, 196, 768) mid_patches
        mid_cls: (N, 768) mid_cls
        batch_size: バッチサイズ

    Returns:
        (N, 4) 新規4d特徴量
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_samples = patches.shape[0]

    results = {name: [] for name in NEW_FEAT_NAMES}

    for i in range(0, n_samples, batch_size):
        batch_p = torch.tensor(patches[i:i+batch_size], dtype=torch.float32, device=device)
        batch_c = torch.tensor(mid_cls[i:i+batch_size], dtype=torch.float32, device=device)
        B, N, D = batch_p.shape

        pn = F.normalize(batch_p, dim=-1)
        cn = F.normalize(batch_c, dim=-1)
        sim = torch.bmm(pn, pn.transpose(1, 2))

        # グラフ構築（閾値0.7）
        adj = ((sim > 0.7).float() * (1 - torch.eye(N, device=device)))
        degree = adj.sum(dim=-1)

        # 1. local_efficiency（三角形密度）
        adj_sq = torch.bmm(adj, adj)
        triangles = (adj_sq * adj).sum(dim=(1, 2)) / 6
        possible = (degree * (degree - 1) / 2).sum(dim=-1)
        local_eff = triangles / (possible + 1e-8)
        results['local_efficiency'].extend(local_eff.cpu().tolist())

        # 2. corner_coherence（四隅の類似度）
        corners = [0, 13, 182, 195]
        corner_sims = []
        for c1 in range(len(corners)):
            for c2 in range(c1 + 1, len(corners)):
                corner_sims.append(sim[:, corners[c1], corners[c2]])
        corner_mean = torch.stack(corner_sims, dim=-1).mean(dim=-1)
        results['corner_coherence'].extend(corner_mean.cpu().tolist())

        # 3. edge_interior_gap（エッジと内部の差）
        edge_idx = list(range(14)) + list(range(14, 182, 14)) + \
                   list(range(27, 196, 14)) + list(range(182, 196))
        edge_idx = list(set(edge_idx))
        interior_idx = [i for i in range(196) if i not in edge_idx]

        edge_mean = sim[:, edge_idx, :][:, :, edge_idx].mean(dim=(1, 2))
        interior_mean = sim[:, interior_idx, :][:, :, interior_idx].mean(dim=(1, 2))
        results['edge_interior_gap'].extend((interior_mean - edge_mean).cpu().tolist())

        # 4. cls_sim_center_bias（中心バイアス）
        cls_sims = torch.bmm(pn, cn.unsqueeze(-1)).squeeze(-1)
        cls_grid = cls_sims.view(B, 14, 14)

        coords = torch.stack(torch.meshgrid(
            torch.arange(14, device=device, dtype=torch.float32),
            torch.arange(14, device=device, dtype=torch.float32),
            indexing='ij'
        ), dim=-1)
        center = torch.tensor([6.5, 6.5], device=device)
        dist_from_center = ((coords - center) ** 2).sum(dim=-1).sqrt()
        dist_flat = dist_from_center.flatten()

        center_corr = []
        for b in range(B):
            cls_flat = cls_grid[b].flatten()
            r = torch.corrcoef(torch.stack([dist_flat, cls_flat]))[0, 1]
            center_corr.append(r.item() if not torch.isnan(r) else 0)
        results['cls_sim_center_bias'].extend(center_corr)

        del batch_p, batch_c, pn, cn, sim, adj, adj_sq
        torch.cuda.empty_cache()

    # (N, 4) に結合
    return np.column_stack([np.array(results[name]) for name in NEW_FEAT_NAMES])


def load_category(name: str, emb_dir: Path, max_samples: int = 30000):
    """カテゴリのデータを読み込み"""
    cls_path = emb_dir / f"{name}.npy"
    stats_path = emb_dir / f"{name}_patch_stats_v3.npy"
    patches_path = emb_dir / f"{name}_mid_patches.npy"
    mid_cls_path = emb_dir / f"{name}_mid_cls.npy"

    cls = np.load(cls_path)
    stats_v3 = np.load(stats_path)
    patches = np.load(patches_path, mmap_mode='r')
    mid_cls = np.load(mid_cls_path)

    # サンプル数を揃える（メモリ節約のため上限設定）
    n_min = min(cls.shape[0], stats_v3.shape[0], patches.shape[0], mid_cls.shape[0], max_samples)

    cls = cls[:n_min]
    stats_11d = stats_v3[:n_min, KEEP_IDX_11D]
    patches = np.array(patches[:n_min])  # memmapから実データに
    mid_cls = mid_cls[:n_min]

    return cls, stats_11d, patches, mid_cls


def train_head(features, labels, epochs=40, lr=1e-3, weight_decay=1e-2,
               batch_size=64, seed=42):
    """線形ヘッドを学習"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.tensor(features, dtype=torch.float32, device=device)
    y = torch.tensor(labels, dtype=torch.float32, device=device).unsqueeze(1)

    model = nn.Linear(x.shape[1], 1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for bx, by in loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()

    return model


def grid_search_alpha(logits_a, logits_b, labels):
    """最適なαをグリッドサーチ"""
    best = {"alpha": 0.0, "pr_auc": 0.0, "roc_auc": 0.0}
    for a in np.linspace(0.0, 2.0, 41):
        logits = logits_a + a * logits_b
        probs = 1 / (1 + np.exp(-logits))
        pr = average_precision_score(labels, probs)
        roc = roc_auc_score(labels, probs)
        if pr > best["pr_auc"]:
            best = {"alpha": float(a), "pr_auc": float(pr), "roc_auc": float(roc)}
    return best


def main():
    parser = argparse.ArgumentParser(description="Train 15d 2-head classifier")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    np.random.seed(args.seed)
    cats = discover_categories(EMBEDDINGS_DIR)
    print(f"Found {len(cats)} categories")

    ai_cls, ai_stats, real_cls, real_stats = [], [], [], []

    for name, label in cats:
        print(f"  Loading {name}...", end=" ", flush=True)
        cls, stats_11d, patches, mid_cls = load_category(name, EMBEDDINGS_DIR)

        # 新規4d計算
        print("computing new features...", end=" ", flush=True)
        new_4d = compute_new_4d_features(patches, mid_cls)
        del patches, mid_cls

        # 15d = 11d + 4d
        stats_15d = np.concatenate([stats_11d, new_4d], axis=1)

        print(f"{cls.shape[0]} samples")

        if label == 1:
            ai_cls.append(cls)
            ai_stats.append(stats_15d)
        else:
            real_cls.append(cls)
            real_stats.append(stats_15d)

    ai_cls = np.concatenate(ai_cls)
    ai_stats = np.concatenate(ai_stats)
    real_cls = np.concatenate(real_cls)
    real_stats = np.concatenate(real_stats)

    print(f"\nBefore balancing: AI={ai_cls.shape[0]}, Real={real_cls.shape[0]}")
    print(f"Stats dim: {ai_stats.shape[1]} (expected 15)")

    # 1:1バランシング
    n_target = min(ai_cls.shape[0], real_cls.shape[0])
    if ai_cls.shape[0] > n_target:
        idx = np.random.choice(ai_cls.shape[0], n_target, replace=False)
        ai_cls, ai_stats = ai_cls[idx], ai_stats[idx]
    if real_cls.shape[0] > n_target:
        idx = np.random.choice(real_cls.shape[0], n_target, replace=False)
        real_cls, real_stats = real_cls[idx], real_stats[idx]

    print(f"After balancing: AI={n_target}, Real={n_target} (1:1)")

    X_cls = np.concatenate([ai_cls, real_cls])
    X_stats = np.concatenate([ai_stats, real_stats])
    y = np.concatenate([np.ones(n_target), np.zeros(n_target)])

    # シャッフル
    perm = np.random.permutation(len(y))
    X_cls, X_stats, y = X_cls[perm], X_stats[perm], y[perm]

    # Train/Val分割
    split = int(len(y) * 0.9)
    X_cls_train, X_cls_val = X_cls[:split], X_cls[split:]
    X_stats_train, X_stats_val = X_stats[:split], X_stats[split:]
    y_train, y_val = y[:split], y[split:]

    print(f"Train: {split}, Val: {len(y) - split}")

    # 正規化
    cls_mean, cls_std = X_cls_train.mean(0), X_cls_train.std(0) + 1e-6
    stats_mean, stats_std = X_stats_train.mean(0), X_stats_train.std(0) + 1e-6

    X_cls_train = (X_cls_train - cls_mean) / cls_std
    X_cls_val = (X_cls_val - cls_mean) / cls_std
    X_stats_train = (X_stats_train - stats_mean) / stats_std
    X_stats_val = (X_stats_val - stats_mean) / stats_std

    print("\nTraining Head A (CLS 768d)...")
    head_a = train_head(X_cls_train, y_train, epochs=args.epochs, lr=args.lr,
                        weight_decay=args.weight_decay, batch_size=args.batch_size,
                        seed=args.seed)

    print("Training Head B (Stats 15d)...")
    head_b = train_head(X_stats_train, y_train, epochs=args.epochs, lr=args.lr,
                        weight_decay=args.weight_decay, batch_size=args.batch_size,
                        seed=args.seed + 1)

    # 最適α探索
    device = next(head_a.parameters()).device
    with torch.no_grad():
        logits_a = head_a(torch.tensor(X_cls_val, dtype=torch.float32, device=device)).cpu().numpy().reshape(-1)
        logits_b = head_b(torch.tensor(X_stats_val, dtype=torch.float32, device=device)).cpu().numpy().reshape(-1)
    best = grid_search_alpha(logits_a, logits_b, y_val)

    # 保存
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(head_a.state_dict(), args.out_dir / "head_a.pt")
    torch.save(head_b.state_dict(), args.out_dir / "head_b.pt")
    torch.save({
        "cls_mean": cls_mean.astype(np.float32),
        "cls_std": cls_std.astype(np.float32),
        "stats_mean": stats_mean.astype(np.float32),
        "stats_std": stats_std.astype(np.float32),
        "keep_idx_11d": KEEP_IDX_11D,
        "feat_names_11d": FEAT_NAMES_11D,
        "new_feat_names": NEW_FEAT_NAMES,
    }, args.out_dir / "norm_stats.pt")

    with open(args.out_dir / "best_alpha.json", "w") as f:
        json.dump(best, f, indent=2)

    # 特徴量情報
    feature_info = {
        "total_dim": 783,
        "cls_dim": 768,
        "stats_dim": 15,
        "existing_11d": FEAT_NAMES_11D,
        "new_4d": NEW_FEAT_NAMES,
        "keep_idx_11d": KEEP_IDX_11D,
    }
    with open(args.out_dir / "feature_info.json", "w") as f:
        json.dump(feature_info, f, indent=2)

    print("\n" + "=" * 60)
    print("=== Results ===")
    print(f"PR-AUC:  {best['pr_auc']:.4f}")
    print(f"ROC-AUC: {best['roc_auc']:.4f}")
    print(f"Best α:  {best['alpha']:.2f}")
    print(f"Total dim: 768 + 15 = 783")
    print("=" * 60)
    print(f"\nFeatures (15d):")
    print(f"  Existing 11d: {', '.join(FEAT_NAMES_11D)}")
    print(f"  New 4d: {', '.join(NEW_FEAT_NAMES)}")
    print(f"\nSaved to: {args.out_dir}")


if __name__ == "__main__":
    main()
