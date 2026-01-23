#!/usr/bin/env python3
"""
Train Two-Head classifier: cpu_stats_v2 (画像2) + cpu_stats_v3 (画像1 Batch系)

構成:
- Head A: CLS (768d)
- Head B: cpu_stats_v2 (16d, Inf列除外) + cpu_stats_v3 (13d) = 29d

Output:
- Cohen's d, Weight, CLS相関, 特徴間相関
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
DEFAULT_OUT_DIR = Path("/home/techne/aicheckers/models/two_head_unified")

# ========================================
# 画像2: cpu_stats_v2 (18d → 16d, 列2,10除外)
# ========================================
CPU_V2_ALL_NAMES = [
    "banding_score",             # 0
    "radial_spectrum_slope",     # 1
    "stroke_width_proxy",        # 2 EXCLUDE (Inf)
    "text_area_ratio",           # 3
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
    "patch_vs_global_spectrum_slope_gap", # 17
]
CPU_V2_SELECT_IDX = [0, 1, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17]  # 16d
CPU_V2_NAMES = [CPU_V2_ALL_NAMES[i] for i in CPU_V2_SELECT_IDX]

# ========================================
# 画像1: cpu_stats_v3 (Batch系 13d)
# ========================================
CPU_V3_NAMES = [
    "histogram_flatness",         # Cohen's d: -1.361
    "histogram_modality",         # +1.153
    "color_palette_entropy",      # +1.124
    "luminance_layer_count",      # +0.892
    "edge_sharpness",             # -0.867
    "chroma_spatial_entropy",     # +0.846
    "lbp_uniformity",             # -0.802
    "luminance_skewness",         # +0.776
    "frequency_band_ratio_var",   # -0.728
    "value_bimodality",           # -0.705
    "multiscale_variance_ratio",  # -0.684
    "gradient_magnitude_entropy", # +0.680
    "noise_spectrum_slope",       # -0.631
]


def discover_categories(emb_dir: Path):
    """カテゴリを自動検出（両方のcpu_statsが存在するもの）"""
    cats = []
    for v2_path in emb_dir.glob("*_cpu_stats_v2.npy"):
        name = v2_path.name.replace("_cpu_stats_v2.npy", "")
        cls_path = emb_dir / f"{name}.npy"
        v3_path = emb_dir / f"{name}_cpu_stats_v3.npy"

        if not cls_path.exists():
            continue
        # v3が存在しない場合はスキップ
        if not v3_path.exists():
            print(f"  [SKIP] {name}: cpu_stats_v3 not found")
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
    """Cohen's d を計算"""
    n1, n2 = len(ai_data), len(real_data)
    var1, var2 = np.var(ai_data, ddof=1), np.var(real_data, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(ai_data) - np.mean(real_data)) / (pooled_std + 1e-8)


def load_category(name: str, emb_dir: Path, max_samples: int = 30000):
    """カテゴリのデータを読み込み"""
    cls_path = emb_dir / f"{name}.npy"
    v2_path = emb_dir / f"{name}_cpu_stats_v2.npy"
    v3_path = emb_dir / f"{name}_cpu_stats_v3.npy"

    cls = np.load(cls_path)
    v2 = np.load(v2_path)
    v3 = np.load(v3_path)

    # サンプル数を揃える
    n_min = min(cls.shape[0], v2.shape[0], v3.shape[0], max_samples)

    cls = cls[:n_min].astype(np.float32)
    v2_selected = v2[:n_min, CPU_V2_SELECT_IDX].astype(np.float32)
    v3_all = v3[:n_min].astype(np.float32)

    # NaN/Infを0に置換
    v2_selected = np.nan_to_num(v2_selected, nan=0.0, posinf=0.0, neginf=0.0)
    v3_all = np.nan_to_num(v3_all, nan=0.0, posinf=0.0, neginf=0.0)

    # 統合: v2 (16d) + v3 (13d) = 29d
    stats = np.concatenate([v2_selected, v3_all], axis=1)

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
    parser = argparse.ArgumentParser(description="Train Two-Head: v2 + v3 unified")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    np.random.seed(args.seed)

    # 特徴量名（29d = 16 + 13）
    all_feat_names = CPU_V2_NAMES + CPU_V3_NAMES
    print(f"Total features: {len(all_feat_names)}d")
    print(f"  - cpu_stats_v2: {len(CPU_V2_NAMES)}d (画像2)")
    print(f"  - cpu_stats_v3: {len(CPU_V3_NAMES)}d (画像1 Batch系)")

    cats = discover_categories(EMBEDDINGS_DIR)
    print(f"\nFound {len(cats)} categories with both v2 and v3")

    if len(cats) == 0:
        print("ERROR: No categories found. cpu_stats_v3 extraction may not be complete.")
        return

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
    print(f"Stats dim: {ai_stats.shape[1]} (expected 29)")

    # ========================================
    # Cohen's d 計算
    # ========================================
    print("\n" + "=" * 80)
    print("=== Cohen's d (per feature) ===")
    print("=" * 80)
    cohens_d = {}
    for i, name in enumerate(all_feat_names):
        d = compute_cohens_d(ai_stats[:, i], real_stats[:, i])
        cohens_d[name] = float(d)
        sign = "+" if d > 0 else ""
        magnitude = "Large" if abs(d) > 0.8 else "Medium" if abs(d) > 0.5 else "Small"
        source = "v2" if i < len(CPU_V2_NAMES) else "v3"
        print(f"  {i:2d}. [{source}] {name:40s} {sign}{d:+.3f}  [{magnitude}]")

    # Top 10
    print("\n--- Top 10 by |Cohen's d| ---")
    sorted_d = sorted(cohens_d.items(), key=lambda x: abs(x[1]), reverse=True)
    for i, (name, d) in enumerate(sorted_d[:10]):
        print(f"  {i+1:2d}. {name:40s} {d:+.3f}")

    # ========================================
    # CLS相関
    # ========================================
    print("\n" + "=" * 80)
    print("=== CLS Correlation ===")
    print("=" * 80)
    all_cls = np.concatenate([ai_cls, real_cls])
    all_stats = np.concatenate([ai_stats, real_stats])
    cls_mean_per_sample = all_cls.mean(axis=1)

    cls_correlations = {}
    for i, name in enumerate(all_feat_names):
        corr, _ = scipy_stats.pearsonr(cls_mean_per_sample, all_stats[:, i])
        cls_correlations[name] = float(corr) if not np.isnan(corr) else 0.0
        print(f"  {i:2d}. {name:40s} r={corr:+.3f}")

    # ========================================
    # 特徴間相関
    # ========================================
    print("\n" + "=" * 80)
    print("=== Inter-feature Correlation (|r| > 0.7) ===")
    print("=" * 80)
    corr_matrix = np.corrcoef(all_stats.T)

    high_corr_pairs = []
    for i in range(len(all_feat_names)):
        for j in range(i + 1, len(all_feat_names)):
            r = corr_matrix[i, j]
            if abs(r) > 0.7:
                high_corr_pairs.append((all_feat_names[i], all_feat_names[j], r))
    high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    for n1, n2, r in high_corr_pairs[:15]:
        print(f"  {n1:35s} <-> {n2:35s}  r={r:+.3f}")

    # ========================================
    # 1:1バランシング・学習
    # ========================================
    n_target = min(ai_cls.shape[0], real_cls.shape[0])
    if ai_cls.shape[0] > n_target:
        idx = np.random.choice(ai_cls.shape[0], n_target, replace=False)
        ai_cls, ai_stats = ai_cls[idx], ai_stats[idx]
    if real_cls.shape[0] > n_target:
        idx = np.random.choice(real_cls.shape[0], n_target, replace=False)
        real_cls, real_stats = real_cls[idx], real_stats[idx]

    print(f"\nAfter balancing: AI={n_target}, Real={n_target}")

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

    print(f"Training Head B (Stats {len(all_feat_names)}d)...")
    head_b = train_head(X_stats_train_n, y_train, epochs=args.epochs, lr=args.lr,
                        weight_decay=args.weight_decay, batch_size=args.batch_size,
                        seed=args.seed + 1)

    # ========================================
    # Head B Weights
    # ========================================
    print("\n" + "=" * 80)
    print("=== Head B Weights ===")
    print("=" * 80)
    weights = head_b.weight.detach().cpu().numpy().flatten()
    bias = head_b.bias.detach().cpu().numpy().item()

    weight_dict = {}
    for i, name in enumerate(all_feat_names):
        weight_dict[name] = float(weights[i])
        source = "v2" if i < len(CPU_V2_NAMES) else "v3"
        print(f"  {i:2d}. [{source}] {name:40s} w={weights[i]:+.4f}")
    print(f"  Bias: {bias:+.4f}")

    # Top 10
    print("\n--- Top 10 by |weight| ---")
    sorted_w = sorted(weight_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    for i, (name, w) in enumerate(sorted_w[:10]):
        print(f"  {i+1:2d}. {name:40s} w={w:+.4f}")

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
        "feature_names": all_feat_names,
        "cpu_v2_names": CPU_V2_NAMES,
        "cpu_v3_names": CPU_V3_NAMES,
        "cpu_v2_select_idx": CPU_V2_SELECT_IDX,
    }, args.out_dir / "norm_stats.pt")

    with open(args.out_dir / "best_alpha.json", "w") as f:
        json.dump(best, f, indent=2)

    metrics = {
        "cohens_d": cohens_d,
        "cls_correlation": cls_correlations,
        "weights": weight_dict,
        "bias": bias,
        "high_corr_pairs": [(n1, n2, float(r)) for n1, n2, r in high_corr_pairs],
        "best_alpha": best,
    }
    with open(args.out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    np.save(args.out_dir / "corr_matrix.npy", corr_matrix)

    # ========================================
    # 全特徴量一覧表
    # ========================================
    print("\n" + "=" * 80)
    print("=== 全特徴量一覧 (Cohen's d / Weight) ===")
    print("=" * 80)
    print(f"{'#':>2} | {'Source':>4} | {'Feature':<40} | {'Cohen d':>10} | {'Weight':>10} | {'|d|順位':>6} | {'|w|順位':>6}")
    print("-" * 100)

    d_rank = {k: i+1 for i, (k, _) in enumerate(sorted_d)}
    w_rank = {k: i+1 for i, (k, _) in enumerate(sorted_w)}

    for i, name in enumerate(all_feat_names):
        d = cohens_d[name]
        w = weight_dict[name]
        source = "v2" if i < len(CPU_V2_NAMES) else "v3"
        print(f"{i:2d} | {source:>4} | {name:<40} | {d:+10.3f} | {w:+10.4f} | {d_rank[name]:>6} | {w_rank[name]:>6}")

    print("\n" + "=" * 80)
    print("=== Final Results ===")
    print("=" * 80)
    print(f"PR-AUC:  {best['pr_auc']:.4f}")
    print(f"ROC-AUC: {best['roc_auc']:.4f}")
    print(f"Best α:  {best['alpha']:.2f}")
    print(f"Total dim: 768 + {len(all_feat_names)} = {768 + len(all_feat_names)}")
    print(f"\nSaved to: {args.out_dir}")


if __name__ == "__main__":
    main()
