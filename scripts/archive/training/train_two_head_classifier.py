#!/usr/bin/env python3
"""
Train a 2-head classifier using CLS (768d) and patch_stats_v3 (33d).

Outputs:
  - head_a.pt (CLS head)
  - head_b.pt (Stats head)
  - norm_stats.pt (z-score mean/std)
  - best_alpha.json (alpha grid search result)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


EMBEDDINGS_DIR = Path("/home/techne/aicheckers/embeddings")
DEFAULT_OUT_DIR = Path("/home/techne/aicheckers/models/two_head")


def discover_categories(emb_dir: Path):
    """Find categories with both CLS and v3 patch stats."""
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
    """Load CLS and v3 stats arrays for a category."""
    cls_path = EMBEDDINGS_DIR / f"{name}.npy"
    stats_path = EMBEDDINGS_DIR / f"{name}_patch_stats_v3.npy"
    cls = np.load(cls_path)  # (N, 768)
    stats = np.load(stats_path)  # (N, 33)
    if cls.shape[1] != 768 or stats.shape[1] != 33:
        raise ValueError(f"{name}: unexpected dims cls={cls.shape}, stats={stats.shape}")
    if cls.shape[0] != stats.shape[0]:
        if not truncate:
            raise ValueError(f"{name}: count mismatch cls={cls.shape[0]}, stats={stats.shape[0]}")
        n = min(cls.shape[0], stats.shape[0])
        print(f"[WARN] {name}: count mismatch cls={cls.shape[0]}, stats={stats.shape[0]} -> truncate {n}")
        cls = cls[:n]
        stats = stats[:n]
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
    for _ in range(epochs):
        for bx, by in loader:
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
    return model


def grid_search_alpha(logits_a, logits_b, labels):
    """Grid search alpha in [0, 2] for best PR-AUC (fallback to ROC-AUC)."""
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
    except Exception:
        average_precision_score = None
        roc_auc_score = None

    best = {"alpha": 0.0, "pr_auc": None, "roc_auc": None}
    alphas = np.linspace(0.0, 2.0, 41)
    labels_np = labels.astype(np.float32)

    for a in alphas:
        logits = logits_a + a * logits_b
        probs = 1 / (1 + np.exp(-logits))
        if average_precision_score is not None:
            pr = average_precision_score(labels_np, probs)
        else:
            pr = None
        if roc_auc_score is not None:
            roc = roc_auc_score(labels_np, probs)
        else:
            roc = None

        score = pr if pr is not None else roc
        best_score = best["pr_auc"] if pr is not None else best["roc_auc"]
        if best_score is None or (score is not None and score > best_score):
            best = {"alpha": float(a), "pr_auc": pr, "roc_auc": roc}
    return best


def main():
    parser = argparse.ArgumentParser(description="Train 2-head classifier (CLS + v3 stats)")
    parser.add_argument("--epochs", type=int, default=40, help="Epochs per head")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="Weight decay for AdamW")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-per-class", type=int, default=0, help="Limit samples per class (0=auto balance)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--strict", action="store_true", help="Fail on count mismatch")
    args = parser.parse_args()

    cats = discover_categories(EMBEDDINGS_DIR)
    if not cats:
        raise SystemExit("No categories found with CLS and v3 stats.")

    # Load data
    ai_cls, ai_stats = [], []
    real_cls, real_stats = [], []
    for name, label in cats:
        cls, stats = load_category(name, truncate=not args.strict)
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

    X_cls = np.concatenate([ai_cls, real_cls], axis=0)
    X_stats = np.concatenate([ai_stats, real_stats], axis=0)
    y = np.concatenate([np.ones(n_target), np.zeros(n_target)], axis=0)

    # Shuffle
    perm = np.random.permutation(len(y))
    X_cls, X_stats, y = X_cls[perm], X_stats[perm], y[perm]

    # Train/val split
    split = int(len(y) * 0.9)
    X_cls_train, X_cls_val = X_cls[:split], X_cls[split:]
    X_stats_train, X_stats_val = X_stats[:split], X_stats[split:]
    y_train, y_val = y[:split], y[split:]

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
    head_a = train_head(
        X_cls_train,
        y_train,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        seed=args.seed,
    )
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
    torch.save({
        "cls_mean": cls_mean.astype(np.float32),
        "cls_std": cls_std.astype(np.float32),
        "stats_mean": stats_mean.astype(np.float32),
        "stats_std": stats_std.astype(np.float32),
    }, args.out_dir / "norm_stats.pt")
    with open(args.out_dir / "best_alpha.json", "w") as f:
        json.dump(best, f, indent=2)

    print("\n=== Saved ===")
    print(f"Head A: {args.out_dir / 'head_a.pt'}")
    print(f"Head B: {args.out_dir / 'head_b.pt'}")
    print(f"Norm stats: {args.out_dir / 'norm_stats.pt'}")
    print(f"Best alpha: {best['alpha']} (PR-AUC={best['pr_auc']}, ROC-AUC={best['roc_auc']})")


if __name__ == "__main__":
    main()
