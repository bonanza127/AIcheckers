#!/usr/bin/env python3
"""
Train Two-Head 20d classifier

構成:
- Head A: CLS (768d)
- Head B: cpu_stats_v2 (16d) + extra (2d) + boundary (2d) = 20d

cpu_stats_v2から (16d, Inf列除外):
  banding_score, radial_spectrum_slope, text_area_ratio, fractal_dim_edge_512,
  patchwise_edge_density, st_aniso_mean, st_aniso_var, st_aniso_spatial_gradient,
  flat_boundary_peri_area, flat_hole_ratio, highfreq_spatial_autocorr,
  patch_vs_global_rank_entropy_gap, flat_ratio, flat_ratio_variance_across_tiles,
  patch_vs_global_st_aniso_gap, patch_vs_global_spectrum_slope_gap

extra_statsから (2d):
  cbcr_autocorr [5], edge_length_mean [9]

boundary_statsから (2d):
  rank_entropy [3], curvature_var [4]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import average_precision_score, roc_auc_score
from scipy import stats as scipy_stats


EMBEDDINGS_DIR = Path("/home/techne/aicheckers/embeddings")
DEFAULT_OUT_DIR = Path("/home/techne/aicheckers/models/two_head_18d")

# cpu_stats_v2: 列2,10はInf、列17は効果薄のため除外
CPU_V2_ALL_NAMES = [
    "banding_score",             # 0
    "radial_spectrum_slope",     # 1
    "stroke_width_proxy",        # 2 EXCLUDE (Inf)
    "text_area_ratio",           # 3 ★将来の文字対策で残す
    "fractal_dim_edge_512",      # 4
    "patchwise_edge_density",    # 5
    "st_aniso_mean",             # 6
    "st_aniso_var",              # 7
    "st_aniso_spatial_gradient", # 8
    "flat_boundary_peri_area",   # 9
    "stroke_p90",                # 10 EXCLUDE (Inf)
    "flat_hole_ratio",           # 11
    "highfreq_spatial_autocorr", # 12
    "patch_vs_global_rank_entropy_gap",   # 13
    "flat_ratio",                # 14
    "flat_ratio_variance_across_tiles",   # 15
    "patch_vs_global_st_aniso_gap",       # 16
    "patch_vs_global_spectrum_slope_gap", # 17 EXCLUDE (効果薄)
]
CPU_V2_SELECT_IDX = [0, 1, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16]  # 17除外
CPU_V2_NAMES = [CPU_V2_ALL_NAMES[i] for i in CPU_V2_SELECT_IDX]

# extra_stats: [5]=cbcr_autocorr, [9]=edge_length_mean
EXTRA_SELECT_IDX = [5, 9]
EXTRA_NAMES = ["cbcr_autocorr", "edge_length_mean"]

# boundary_stats: [3]=rank_entropy のみ（curvature_var除外）
BOUNDARY_SELECT_IDX = [3]
BOUNDARY_NAMES = ["rank_entropy"]

ALL_FEAT_NAMES = CPU_V2_NAMES + EXTRA_NAMES + BOUNDARY_NAMES


def discover_categories(emb_dir: Path):
    """カテゴリを自動検出"""
    cats = []
    for v2_path in emb_dir.glob("*_cpu_stats_v2.npy"):
        name = v2_path.name.replace("_cpu_stats_v2.npy", "")
        cls_path = emb_dir / f"{name}.npy"
        extra_path = emb_dir / f"{name}_extra_stats.npy"
        boundary_path = emb_dir / f"{name}_boundary_stats.npy"

        if not all(p.exists() for p in [cls_path, extra_path, boundary_path]):
            print(f"  [SKIP] {name}: missing files")
            continue

        if name.endswith("_ai"):
            label = 1
        elif name.endswith("_real"):
            label = 0
        else:
            continue
        cats.append((name, label))
    return sorted(cats)


def compute_cohens_d(ai_data, real_data):
    """Cohen's d"""
    n1, n2 = len(ai_data), len(real_data)
    var1, var2 = np.var(ai_data, ddof=1), np.var(real_data, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(ai_data) - np.mean(real_data)) / (pooled_std + 1e-8)


def load_category(name: str, emb_dir: Path, max_samples: int = 30000):
    """カテゴリのデータを読み込み"""
    cls = np.load(emb_dir / f"{name}.npy")
    v2 = np.load(emb_dir / f"{name}_cpu_stats_v2.npy")
    extra = np.load(emb_dir / f"{name}_extra_stats.npy")
    boundary = np.load(emb_dir / f"{name}_boundary_stats.npy")

    n_min = min(cls.shape[0], v2.shape[0], extra.shape[0], boundary.shape[0], max_samples)

    cls = cls[:n_min].astype(np.float32)
    v2_sel = v2[:n_min, CPU_V2_SELECT_IDX].astype(np.float32)
    extra_sel = extra[:n_min, EXTRA_SELECT_IDX].astype(np.float32)
    boundary_sel = boundary[:n_min, BOUNDARY_SELECT_IDX].astype(np.float32)

    # NaN/Inf置換
    v2_sel = np.nan_to_num(v2_sel, nan=0.0, posinf=0.0, neginf=0.0)
    extra_sel = np.nan_to_num(extra_sel, nan=0.0, posinf=0.0, neginf=0.0)
    boundary_sel = np.nan_to_num(boundary_sel, nan=0.0, posinf=0.0, neginf=0.0)

    # 統合: 16 + 2 + 2 = 20d
    stats = np.concatenate([v2_sel, extra_sel, boundary_sel], axis=1)
    return cls, stats


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
    """最適なα"""
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
    parser = argparse.ArgumentParser(description="Train Two-Head 20d")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    np.random.seed(args.seed)

    print(f"Features: {len(ALL_FEAT_NAMES)}d")
    print(f"  cpu_stats_v2: {len(CPU_V2_NAMES)}d")
    print(f"  extra_stats:  {len(EXTRA_NAMES)}d")
    print(f"  boundary:     {len(BOUNDARY_NAMES)}d")

    cats = discover_categories(EMBEDDINGS_DIR)
    print(f"\nFound {len(cats)} categories")

    ai_cls, ai_stats, real_cls, real_stats = [], [], [], []

    for name, label in cats:
        print(f"  Loading {name}...", end=" ", flush=True)
        cls, stats = load_category(name, EMBEDDINGS_DIR)
        print(f"{cls.shape[0]} samples")

        if label == 1:
            ai_cls.append(cls)
            ai_stats.append(stats)
        else:
            real_cls.append(cls)
            real_stats.append(stats)

    ai_cls = np.concatenate(ai_cls)
    ai_stats = np.concatenate(ai_stats)
    real_cls = np.concatenate(real_cls)
    real_stats = np.concatenate(real_stats)

    print(f"\nBefore balancing: AI={ai_cls.shape[0]}, Real={real_cls.shape[0]}")

    # Cohen's d
    print("\n" + "=" * 80)
    print("=== Cohen's d ===")
    print("=" * 80)
    cohens_d = {}
    for i, name in enumerate(ALL_FEAT_NAMES):
        d = compute_cohens_d(ai_stats[:, i], real_stats[:, i])
        cohens_d[name] = float(d)
        mag = "Large" if abs(d) > 0.8 else "Medium" if abs(d) > 0.5 else "Small"
        print(f"  {i:2d}. {name:45s} {d:+.3f} [{mag}]")

    # 1:1バランシング
    n_target = min(ai_cls.shape[0], real_cls.shape[0])
    if ai_cls.shape[0] > n_target:
        idx = np.random.choice(ai_cls.shape[0], n_target, replace=False)
        ai_cls, ai_stats = ai_cls[idx], ai_stats[idx]
    if real_cls.shape[0] > n_target:
        idx = np.random.choice(real_cls.shape[0], n_target, replace=False)
        real_cls, real_stats = real_cls[idx], real_stats[idx]

    print(f"\nAfter balancing: {n_target} each")

    X_cls = np.concatenate([ai_cls, real_cls])
    X_stats = np.concatenate([ai_stats, real_stats])
    y = np.concatenate([np.ones(n_target), np.zeros(n_target)])

    perm = np.random.permutation(len(y))
    X_cls, X_stats, y = X_cls[perm], X_stats[perm], y[perm]

    split = int(len(y) * 0.9)
    X_cls_train, X_cls_val = X_cls[:split], X_cls[split:]
    X_stats_train, X_stats_val = X_stats[:split], X_stats[split:]
    y_train, y_val = y[:split], y[split:]

    # 正規化
    cls_mean, cls_std = X_cls_train.mean(0), X_cls_train.std(0) + 1e-6
    stats_mean, stats_std = X_stats_train.mean(0), X_stats_train.std(0) + 1e-6

    X_cls_train_n = (X_cls_train - cls_mean) / cls_std
    X_cls_val_n = (X_cls_val - cls_mean) / cls_std
    X_stats_train_n = (X_stats_train - stats_mean) / stats_std
    X_stats_val_n = (X_stats_val - stats_mean) / stats_std

    print(f"Train: {split}, Val: {len(y) - split}")

    print("\nTraining Head A (CLS 768d)...")
    head_a = train_head(X_cls_train_n, y_train, epochs=args.epochs, lr=args.lr,
                        weight_decay=args.weight_decay, batch_size=args.batch_size,
                        seed=args.seed)

    print(f"Training Head B (Stats 20d)...")
    head_b = train_head(X_stats_train_n, y_train, epochs=args.epochs, lr=args.lr,
                        weight_decay=args.weight_decay, batch_size=args.batch_size,
                        seed=args.seed + 1)

    # Head B Weights
    print("\n" + "=" * 80)
    print("=== Head B Weights ===")
    print("=" * 80)
    weights = head_b.weight.detach().cpu().numpy().flatten()
    bias = head_b.bias.detach().cpu().numpy().item()

    weight_dict = {}
    for i, name in enumerate(ALL_FEAT_NAMES):
        weight_dict[name] = float(weights[i])
        print(f"  {i:2d}. {name:45s} w={weights[i]:+.4f}")
    print(f"  Bias: {bias:+.4f}")

    # CLS相関
    print("\n" + "=" * 80)
    print("=== CLS Correlation ===")
    print("=" * 80)
    all_cls = np.concatenate([X_cls_train, X_cls_val])
    all_stats = np.concatenate([X_stats_train, X_stats_val])
    cls_mean_per_sample = all_cls.mean(axis=1)

    cls_corr = {}
    for i, name in enumerate(ALL_FEAT_NAMES):
        r, _ = scipy_stats.pearsonr(cls_mean_per_sample, all_stats[:, i])
        cls_corr[name] = float(r) if not np.isnan(r) else 0.0
        print(f"  {i:2d}. {name:45s} r={r:+.3f}")

    # 特徴間相関
    print("\n" + "=" * 80)
    print("=== Inter-feature Correlation (|r| > 0.7) ===")
    print("=" * 80)
    corr_matrix = np.corrcoef(all_stats.T)
    high_corr = []
    for i in range(len(ALL_FEAT_NAMES)):
        for j in range(i + 1, len(ALL_FEAT_NAMES)):
            r = corr_matrix[i, j]
            if abs(r) > 0.7:
                high_corr.append((ALL_FEAT_NAMES[i], ALL_FEAT_NAMES[j], r))
    high_corr.sort(key=lambda x: abs(x[2]), reverse=True)
    for n1, n2, r in high_corr[:10]:
        print(f"  {n1:30s} <-> {n2:30s} r={r:+.3f}")

    # 最適α
    device = next(head_a.parameters()).device
    with torch.no_grad():
        logits_a = head_a(torch.tensor(X_cls_val_n, dtype=torch.float32, device=device)).cpu().numpy().reshape(-1)
        logits_b = head_b(torch.tensor(X_stats_val_n, dtype=torch.float32, device=device)).cpu().numpy().reshape(-1)
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
        "feature_names": ALL_FEAT_NAMES,
        "cpu_v2_select_idx": CPU_V2_SELECT_IDX,
        "extra_select_idx": EXTRA_SELECT_IDX,
        "boundary_select_idx": BOUNDARY_SELECT_IDX,
    }, args.out_dir / "norm_stats.pt")

    with open(args.out_dir / "best_alpha.json", "w") as f:
        json.dump(best, f, indent=2)

    metrics = {
        "cohens_d": cohens_d,
        "weights": weight_dict,
        "cls_correlation": cls_corr,
        "high_corr_pairs": [(n1, n2, float(r)) for n1, n2, r in high_corr],
        "best_alpha": best,
    }
    with open(args.out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # 全特徴量一覧
    print("\n" + "=" * 80)
    print("=== 全特徴量一覧 ===")
    print("=" * 80)
    sorted_d = sorted(cohens_d.items(), key=lambda x: abs(x[1]), reverse=True)
    sorted_w = sorted(weight_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    d_rank = {k: i+1 for i, (k, _) in enumerate(sorted_d)}
    w_rank = {k: i+1 for i, (k, _) in enumerate(sorted_w)}

    print(f"{'#':>2} | {'Feature':<45} | {'Cohen d':>9} | {'Weight':>9} | {'|d|':>4} | {'|w|':>4}")
    print("-" * 90)
    for i, name in enumerate(ALL_FEAT_NAMES):
        d = cohens_d[name]
        w = weight_dict[name]
        print(f"{i:2d} | {name:<45} | {d:+9.3f} | {w:+9.4f} | {d_rank[name]:>4} | {w_rank[name]:>4}")

    print("\n" + "=" * 80)
    print("=== Final Results ===")
    print("=" * 80)
    print(f"PR-AUC:  {best['pr_auc']:.4f}")
    print(f"ROC-AUC: {best['roc_auc']:.4f}")
    print(f"Best α:  {best['alpha']:.2f}")
    print(f"Total:   768 + 20 = 788d")
    print(f"\nSaved to: {args.out_dir}")


if __name__ == "__main__":
    main()
