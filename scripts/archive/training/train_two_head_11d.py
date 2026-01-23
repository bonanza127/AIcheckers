#!/usr/bin/env python3
"""
Train 11d 2-head classifier: CLS (768d) + Stats (11d)

11d = 14d から高相関グループ(eigenvalue_ratio, spectral_gap, attn_spectral_ratio)を除外
      top3_ratioのみ残す
Total input: 768 + 11 = 779d
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import average_precision_score, roc_auc_score


EMBEDDINGS_DIR = Path("/home/techne/aicheckers/embeddings")
DEFAULT_OUT_DIR = Path("/home/techne/aicheckers/models/two_head_11d")

# 11d: 高相関グループ除外後
# adj_sim_var, high_sim_ratio, patch_var, norm_var, cls_sim_mean, cls_angle_dispersion,
# knn_sim_var, top3_ratio, degree_centrality, clustering_coef, center_edge_high_ratio
KEEP_IDX_11D = [1, 2, 3, 5, 9, 12, 14, 17, 18, 19, 32]


def discover_categories(emb_dir: Path):
    cats = []
    for stats_path in emb_dir.glob("*_patch_stats_v3.npy"):
        name = stats_path.name.replace("_patch_stats_v3.npy", "")
        cls_path = emb_dir / f"{name}.npy"
        if not cls_path.exists():
            continue
        if name.endswith("_ai"):
            label = 1
        elif name.endswith("_real"):
            label = 0
        else:
            continue
        cats.append((name, label))
    return sorted(cats)


def load_category(name: str, truncate: bool = True):
    cls_path = EMBEDDINGS_DIR / f"{name}.npy"
    stats_path = EMBEDDINGS_DIR / f"{name}_patch_stats_v3.npy"
    cls = np.load(cls_path)
    stats_v3 = np.load(stats_path)
    n_min = min(cls.shape[0], stats_v3.shape[0])
    if cls.shape[0] != stats_v3.shape[0]:
        if not truncate:
            raise ValueError(f"{name}: count mismatch")
        print(f"[WARN] {name}: count mismatch cls={cls.shape[0]}, v3={stats_v3.shape[0]} -> truncate {n_min}")
        cls = cls[:n_min]
        stats_v3 = stats_v3[:n_min]
    stats = stats_v3[:, KEEP_IDX_11D]
    return cls, stats


def train_head(features, labels, epochs=40, lr=1e-3, weight_decay=1e-2, batch_size=64, seed=42):
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
    parser = argparse.ArgumentParser(description="Train 11d 2-head classifier")
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
        cls, stats = load_category(name)
        print(f"  {name}: {cls.shape[0]} samples")
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
    print(f"Stats dim: {ai_stats.shape[1]} (expected 11)")

    n_target = min(ai_cls.shape[0], real_cls.shape[0])
    if ai_cls.shape[0] > n_target:
        idx = np.random.choice(ai_cls.shape[0], n_target, replace=False)
        ai_cls, ai_stats = ai_cls[idx], ai_stats[idx]
    if real_cls.shape[0] > n_target:
        idx = np.random.choice(real_cls.shape[0], n_target, replace=False)
        real_cls, real_stats = real_cls[idx], real_stats[idx]

    print(f"After balancing:  AI={n_target}, Real={n_target} (1:1)")

    X_cls = np.concatenate([ai_cls, real_cls])
    X_stats = np.concatenate([ai_stats, real_stats])
    y = np.concatenate([np.ones(n_target), np.zeros(n_target)])

    perm = np.random.permutation(len(y))
    X_cls, X_stats, y = X_cls[perm], X_stats[perm], y[perm]

    split = int(len(y) * 0.9)
    X_cls_train, X_cls_val = X_cls[:split], X_cls[split:]
    X_stats_train, X_stats_val = X_stats[:split], X_stats[split:]
    y_train, y_val = y[:split], y[split:]

    print(f"Train: {split}, Val: {len(y) - split}")

    cls_mean, cls_std = X_cls_train.mean(0), X_cls_train.std(0) + 1e-6
    stats_mean, stats_std = X_stats_train.mean(0), X_stats_train.std(0) + 1e-6

    X_cls_train = (X_cls_train - cls_mean) / cls_std
    X_cls_val = (X_cls_val - cls_mean) / cls_std
    X_stats_train = (X_stats_train - stats_mean) / stats_std
    X_stats_val = (X_stats_val - stats_mean) / stats_std

    print("\nTraining Head A (CLS 768d)...")
    head_a = train_head(X_cls_train, y_train, epochs=args.epochs, lr=args.lr,
                        weight_decay=args.weight_decay, batch_size=args.batch_size, seed=args.seed)

    print("Training Head B (Stats 11d)...")
    head_b = train_head(X_stats_train, y_train, epochs=args.epochs, lr=args.lr,
                        weight_decay=args.weight_decay, batch_size=args.batch_size, seed=args.seed + 1)

    device = next(head_a.parameters()).device
    with torch.no_grad():
        logits_a = head_a(torch.tensor(X_cls_val, dtype=torch.float32, device=device)).cpu().numpy().reshape(-1)
        logits_b = head_b(torch.tensor(X_stats_val, dtype=torch.float32, device=device)).cpu().numpy().reshape(-1)
    best = grid_search_alpha(logits_a, logits_b, y_val)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(head_a.state_dict(), args.out_dir / "head_a.pt")
    torch.save(head_b.state_dict(), args.out_dir / "head_b.pt")
    torch.save({
        "cls_mean": cls_mean.astype(np.float32),
        "cls_std": cls_std.astype(np.float32),
        "stats_mean": stats_mean.astype(np.float32),
        "stats_std": stats_std.astype(np.float32),
        "keep_idx": KEEP_IDX_11D,
    }, args.out_dir / "norm_stats.pt")

    with open(args.out_dir / "best_alpha.json", "w") as f:
        json.dump(best, f, indent=2)

    print("\n" + "=" * 60)
    print("=== Results ===")
    print(f"PR-AUC:  {best['pr_auc']:.4f}")
    print(f"ROC-AUC: {best['roc_auc']:.4f}")
    print(f"Best α:  {best['alpha']:.2f}")
    print(f"Total dim: 768 + 11 = 779")
    print("=" * 60)
    print(f"\nSaved to: {args.out_dir}")


if __name__ == "__main__":
    main()
