#!/usr/bin/env python3
"""
Train 25d 2-head classifier: 24d (filtered from v3) + patch_energy_skewness

Based on train_two_head_classifier.py, with:
  - Use 24d keep_idx from previous best model
  - Add patch_energy_skewness as 25th feature

Outputs:
  - head_a.pt (CLS head, 768d)
  - head_b.pt (Stats head, 25d)
  - norm_stats.pt (z-score mean/std + keep_idx)
  - best_alpha.json (alpha grid search result)
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
DEFAULT_OUT_DIR = Path("/home/techne/aicheckers/models/two_head_25d")

# 24d keep indices from the previous best model (9 useless features removed)
KEEP_IDX_24D = [0, 1, 2, 3, 5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 18, 19, 20, 23, 24, 26, 28, 30, 31, 32]


def discover_categories(emb_dir: Path):
    """Find categories with CLS, v3 patch stats, and energy skewness."""
    cats = []
    for stats_path in emb_dir.glob("*_patch_stats_v3.npy"):
        name = stats_path.name.replace("_patch_stats_v3.npy", "")
        cls_path = emb_dir / f"{name}.npy"
        skew_path = emb_dir / f"{name}_energy_skewness.npy"

        if not cls_path.exists():
            continue
        if not skew_path.exists():
            print(f"[Skip] {name}: no energy_skewness")
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
    """Load CLS, filtered v3 stats (24d), and energy_skewness arrays."""
    cls_path = EMBEDDINGS_DIR / f"{name}.npy"
    stats_path = EMBEDDINGS_DIR / f"{name}_patch_stats_v3.npy"
    skew_path = EMBEDDINGS_DIR / f"{name}_energy_skewness.npy"

    cls = np.load(cls_path)  # (N, 768)
    stats_v3 = np.load(stats_path)  # (N, 33)
    skew = np.load(skew_path)  # (N,)

    # Find minimum size across all three sources
    n_cls = cls.shape[0]
    n_v3 = stats_v3.shape[0]
    n_skew = skew.shape[0]
    n_min = min(n_cls, n_v3, n_skew)

    if n_cls != n_v3 or n_v3 != n_skew:
        if not truncate:
            raise ValueError(f"{name}: count mismatch cls={n_cls}, v3={n_v3}, skew={n_skew}")
        print(f"[WARN] {name}: count mismatch cls={n_cls}, v3={n_v3}, skew={n_skew} -> truncate {n_min}")
        cls = cls[:n_min]
        stats_v3 = stats_v3[:n_min]
        skew = skew[:n_min]

    # Filter to 24d
    stats_24d = stats_v3[:, KEEP_IDX_24D]  # (N, 24)

    # Combine: 24d + 1d = 25d
    stats = np.concatenate([stats_24d, skew.reshape(-1, 1)], axis=1)  # (N, 25)

    return cls, stats


def train_head(features, labels, epochs=40, lr=1e-3, weight_decay=1e-2, batch_size=64, seed=42):
    """Train a single linear head with BCEWithLogitsLoss."""
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
    for epoch in range(epochs):
        epoch_loss = 0.0
        for bx, by in loader:
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
    return model


def grid_search_alpha(logits_a, logits_b, labels):
    """Grid search alpha in [0, 2] for best PR-AUC."""
    best = {"alpha": 0.0, "pr_auc": 0.0, "roc_auc": 0.0}
    alphas = np.linspace(0.0, 2.0, 41)
    labels_np = labels.astype(np.float32)

    for a in alphas:
        logits = logits_a + a * logits_b
        probs = 1 / (1 + np.exp(-logits))
        pr = average_precision_score(labels_np, probs)
        roc = roc_auc_score(labels_np, probs)

        if pr > best["pr_auc"]:
            best = {"alpha": float(a), "pr_auc": float(pr), "roc_auc": float(roc)}
    return best


def main():
    parser = argparse.ArgumentParser(description="Train 25d 2-head classifier")
    parser.add_argument("--epochs", type=int, default=40, help="Epochs per head")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="Weight decay for AdamW")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-per-class", type=int, default=0, help="Limit samples per class (0=auto balance)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    np.random.seed(args.seed)

    cats = discover_categories(EMBEDDINGS_DIR)
    if not cats:
        raise SystemExit("No categories found with CLS, v3 stats, and energy_skewness.")

    print(f"Found {len(cats)} categories")

    # Load data
    ai_cls, ai_stats = [], []
    real_cls, real_stats = [], []
    for name, label in cats:
        cls, stats = load_category(name, truncate=True)
        print(f"  {name}: {cls.shape[0]} samples")
        if label == 1:
            ai_cls.append(cls)
            ai_stats.append(stats)
        else:
            real_cls.append(cls)
            real_stats.append(stats)

    if not ai_cls or not real_cls:
        raise SystemExit("No AI/Real data loaded.")

    ai_cls = np.concatenate(ai_cls, axis=0)
    ai_stats = np.concatenate(ai_stats, axis=0)
    real_cls = np.concatenate(real_cls, axis=0)
    real_stats = np.concatenate(real_stats, axis=0)

    print(f"\nTotal: AI={ai_cls.shape[0]}, Real={real_cls.shape[0]}")
    print(f"Stats dim: {ai_stats.shape[1]} (expected 25)")

    # Balance classes
    n_ai = ai_cls.shape[0]
    n_real = real_cls.shape[0]
    n_target = min(n_ai, n_real) if args.max_per_class == 0 else min(args.max_per_class, n_ai, n_real)

    if n_ai > n_target:
        idx = np.random.choice(n_ai, n_target, replace=False)
        ai_cls, ai_stats = ai_cls[idx], ai_stats[idx]
    if n_real > n_target:
        idx = np.random.choice(n_real, n_target, replace=False)
        real_cls, real_stats = real_cls[idx], real_stats[idx]

    print(f"Balanced: {n_target} each")

    X_cls = np.concatenate([ai_cls, real_cls], axis=0)
    X_stats = np.concatenate([ai_stats, real_stats], axis=0)
    y = np.concatenate([np.ones(n_target), np.zeros(n_target)], axis=0)

    # Handle NaN/Inf in energy_skewness (from overflow)
    nan_mask = ~np.isfinite(X_stats[:, -1])
    if nan_mask.any():
        print(f"[WARN] Found {nan_mask.sum()} NaN/Inf in energy_skewness, replacing with median")
        finite_vals = X_stats[~nan_mask, -1]
        median_val = np.median(finite_vals)
        X_stats[nan_mask, -1] = median_val

    # Shuffle
    perm = np.random.permutation(len(y))
    X_cls, X_stats, y = X_cls[perm], X_stats[perm], y[perm]

    # Train/val split
    split = int(len(y) * 0.9)
    X_cls_train, X_cls_val = X_cls[:split], X_cls[split:]
    X_stats_train, X_stats_val = X_stats[:split], X_stats[split:]
    y_train, y_val = y[:split], y[split:]

    print(f"Train: {split}, Val: {len(y) - split}")

    # Z-score normalization (train stats)
    cls_mean = X_cls_train.mean(axis=0)
    cls_std = X_cls_train.std(axis=0) + 1e-6
    stats_mean = X_stats_train.mean(axis=0)
    stats_std = X_stats_train.std(axis=0) + 1e-6

    X_cls_train = (X_cls_train - cls_mean) / cls_std
    X_cls_val = (X_cls_val - cls_mean) / cls_std
    X_stats_train = (X_stats_train - stats_mean) / stats_std
    X_stats_val = (X_stats_val - stats_mean) / stats_std

    # Train heads
    print("\nTraining Head A (CLS 768d)...")
    head_a = train_head(
        X_cls_train,
        y_train,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    print("Training Head B (Stats 25d)...")
    head_b = train_head(
        X_stats_train,
        y_train,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        seed=args.seed + 1,
    )

    # Alpha search on validation
    device = next(head_a.parameters()).device
    with torch.no_grad():
        logits_a = head_a(torch.tensor(X_cls_val, dtype=torch.float32, device=device)).cpu().numpy().reshape(-1)
        logits_b = head_b(torch.tensor(X_stats_val, dtype=torch.float32, device=device)).cpu().numpy().reshape(-1)
    best = grid_search_alpha(logits_a, logits_b, y_val)

    # Save
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(head_a.state_dict(), args.out_dir / "head_a.pt")
    torch.save(head_b.state_dict(), args.out_dir / "head_b.pt")

    # keep_idx for 25d: 24d indices + index 33 (energy_skewness, appended as last feature)
    # Note: The actual source indices are KEEP_IDX_24D from v3 stats + "energy_skewness" as separate file
    torch.save({
        "cls_mean": cls_mean.astype(np.float32),
        "cls_std": cls_std.astype(np.float32),
        "stats_mean": stats_mean.astype(np.float32),
        "stats_std": stats_std.astype(np.float32),
        "keep_idx_v3": KEEP_IDX_24D,  # Reference to v3 indices (24d portion)
        "has_energy_skewness": True,   # Flag for the 25th feature
    }, args.out_dir / "norm_stats.pt")

    with open(args.out_dir / "best_alpha.json", "w") as f:
        json.dump(best, f, indent=2)

    print("\n" + "=" * 60)
    print("=== Results ===")
    print(f"PR-AUC:  {best['pr_auc']:.4f}")
    print(f"ROC-AUC: {best['roc_auc']:.4f}")
    print(f"Best α:  {best['alpha']:.2f}")
    print("=" * 60)

    print(f"\nSaved to: {args.out_dir}")
    print(f"  head_a.pt (CLS 768d)")
    print(f"  head_b.pt (Stats 25d)")
    print(f"  norm_stats.pt")
    print(f"  best_alpha.json")

    # Weight analysis for new feature
    with torch.no_grad():
        weights_b = head_b.weight.data.cpu().numpy().flatten()
        bias_b = head_b.bias.data.cpu().numpy().item()

    print(f"\n=== Head B Weight Analysis ===")
    print(f"Energy skewness (idx 24) weight: {weights_b[24]:.4f}")
    print(f"Top 5 weights by |w|:")
    sorted_idx = np.argsort(-np.abs(weights_b))
    for i, idx in enumerate(sorted_idx[:5]):
        print(f"  {i+1}. idx={idx}: w={weights_b[idx]:.4f}")


if __name__ == "__main__":
    main()
