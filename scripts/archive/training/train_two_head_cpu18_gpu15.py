#!/usr/bin/env python3
"""
Train CPU18 + GPU11 2-head classifier: CLS (768d) + Stats (29d)

Stats 구성:
- CPU18d: cpu_stats_v2(15d) + extra(2d) + boundary(1d)
- GPU11d: v3 7d + new 4d (from mid_patches/mid_cls)

Total input: 768 + 29 = 797d
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
DEFAULT_OUT_DIR = Path("/home/techne/aicheckers/models/two_head_cpu18_gpu11")

# CPU 18d 구성: cpu_stats_v2 15d + extra 2d + boundary 1d
CPU_V2_SELECT_IDX = [0, 1, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16]
EXTRA_SELECT_IDX = [5, 9]       # cbcr_autocorr, edge_length_mean
BOUNDARY_SELECT_IDX = [3]       # rank_entropy

# GPU 11d (v3 stats indices)
V3_NAMES = [
    "adj_sim_mean", "adj_sim_var", "high_sim_ratio", "patch_var",
    "anisotropy", "norm_var", "norm_range", "low_sim_ratio_v",
    "low_sim_ratio_h", "cls_sim_mean", "cls_sim_var", "cls_angle_dispersion",
    "knn_sim_mean", "knn_sim_var", "eigenvalue_ratio", "spectral_gap",
    "top3_ratio", "degree_centrality", "clustering_coef", "low_freq_energy_2x2",
    "high_freq_energy_2x2", "freq_ratio_2x2", "attn_entropy",
    "attn_spectral_ratio", "attn_spectral_gap", "band_energy_spatial_var_low",
    "band_energy_spatial_var_mid", "band_energy_spatial_var_high",
    "band_adj_sim_var_2x2", "band_adj_sim_var_4x4", "band_entropy",
    "band_transition_entropy", "center_edge_high_ratio", "center_edge_low_ratio"
]
GPU_V3_IDX = [1, 2, 3, 5, 13, 17, 32]
GPU_V3_NAMES = [V3_NAMES[i] for i in GPU_V3_IDX]

# 新規 4d
NEW_FEAT_NAMES = ["local_efficiency", "corner_coherence", "edge_interior_gap", "cls_sim_center_bias"]


def discover_categories(emb_dir: Path):
    """Find categories with required inputs for CPU18+GPU15."""
    cats = []
    for cls_path in emb_dir.glob("*.npy"):
        name = cls_path.name.replace(".npy", "")
        if name.endswith("_patch_stats_v3") or name.endswith("_mid_patches") or name.endswith("_mid_cls"):
            continue
        v2_path = emb_dir / f"{name}_cpu_stats_v2.npy"
        extra_path = emb_dir / f"{name}_extra_stats.npy"
        boundary_path = emb_dir / f"{name}_boundary_stats.npy"
        v3_path = emb_dir / f"{name}_patch_stats_v3.npy"
        mid_p = emb_dir / f"{name}_mid_patches.npy"
        mid_c = emb_dir / f"{name}_mid_cls.npy"

        if not all(p.exists() for p in [v2_path, extra_path, boundary_path, v3_path, mid_p, mid_c]):
            continue

        if name.endswith("_ai"):
            label = 1
        elif name.endswith("_real"):
            label = 0
        else:
            continue
        cats.append((name, label))
    return sorted(cats)


def compute_new_4d_features(patches, mid_cls, batch_size: int = 32) -> np.ndarray:
    """Compute 4d graph features."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_samples = patches.shape[0]
    results = [[] for _ in range(4)]

    for i in range(0, n_samples, batch_size):
        batch_p = torch.tensor(patches[i:i + batch_size], dtype=torch.float32, device=device)
        batch_c = torch.tensor(mid_cls[i:i + batch_size], dtype=torch.float32, device=device)
        B, N, _ = batch_p.shape

        pn = F.normalize(batch_p, dim=-1)
        cn = F.normalize(batch_c, dim=-1)
        sim = torch.bmm(pn, pn.transpose(1, 2))

        adj = ((sim > 0.7).float() * (1 - torch.eye(N, device=device)))
        degree = adj.sum(dim=-1)

        # local_efficiency
        adj_sq = torch.bmm(adj, adj)
        triangles = (adj_sq * adj).sum(dim=(1, 2)) / 6
        possible = (degree * (degree - 1) / 2).sum(dim=-1)
        local_eff = (triangles / (possible + 1e-8)).cpu().numpy()

        # corner_coherence
        corners = [0, 13, 182, 195]
        corner_sims = []
        for c1 in range(len(corners)):
            for c2 in range(c1 + 1, len(corners)):
                corner_sims.append(sim[:, corners[c1], corners[c2]])
        corner_mean = torch.stack(corner_sims, dim=-1).mean(dim=-1).cpu().numpy()

        # edge_interior_gap
        edge_idx = list(range(14)) + list(range(14, 182, 14)) + \
                   list(range(27, 196, 14)) + list(range(182, 196))
        edge_idx = list(set(edge_idx))
        interior_idx = [i for i in range(196) if i not in edge_idx]
        edge_mean = sim[:, edge_idx, :][:, :, edge_idx].mean(dim=(1, 2))
        interior_mean = sim[:, interior_idx, :][:, :, interior_idx].mean(dim=(1, 2))
        edge_gap = (interior_mean - edge_mean).cpu().numpy()

        # cls_sim_center_bias
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
            center_corr.append(r.item() if not torch.isnan(r) else 0.0)
        center_corr = np.array(center_corr)

        for dst, arr in zip(results, [local_eff, corner_mean, edge_gap, center_corr]):
            dst.extend(arr.tolist())

        del batch_p, batch_c, pn, cn, sim, adj, adj_sq
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return np.column_stack([np.array(r) for r in results])


def load_category(name: str, truncate: bool = True):
    """Load CLS + CPU18 + GPU15."""
    cls_path = EMBEDDINGS_DIR / f"{name}.npy"
    v2_path = EMBEDDINGS_DIR / f"{name}_cpu_stats_v2.npy"
    extra_path = EMBEDDINGS_DIR / f"{name}_extra_stats.npy"
    boundary_path = EMBEDDINGS_DIR / f"{name}_boundary_stats.npy"
    v3_path = EMBEDDINGS_DIR / f"{name}_patch_stats_v3.npy"
    mid_p_path = EMBEDDINGS_DIR / f"{name}_mid_patches.npy"
    mid_c_path = EMBEDDINGS_DIR / f"{name}_mid_cls.npy"

    cls = np.load(cls_path)
    v2 = np.load(v2_path)
    extra = np.load(extra_path)
    boundary = np.load(boundary_path)
    v3 = np.load(v3_path)
    mid_p = np.load(mid_p_path, mmap_mode="r")
    mid_c = np.load(mid_c_path, mmap_mode="r")

    n_min = min(
        cls.shape[0], v2.shape[0], extra.shape[0], boundary.shape[0],
        v3.shape[0], mid_p.shape[0], mid_c.shape[0]
    )

    if truncate and n_min < cls.shape[0]:
        print(f"[WARN] {name}: count mismatch -> truncate {n_min}")

    cls = cls[:n_min]

    cpu = np.concatenate([
        v2[:n_min, CPU_V2_SELECT_IDX],
        extra[:n_min, EXTRA_SELECT_IDX],
        boundary[:n_min, BOUNDARY_SELECT_IDX],
    ], axis=1).astype(np.float32)
    cpu = np.nan_to_num(cpu, nan=0.0, posinf=0.0, neginf=0.0)

    gpu_11d = v3[:n_min, GPU_V3_IDX].astype(np.float32)
    gpu_11d = np.nan_to_num(gpu_11d, nan=0.0, posinf=0.0, neginf=0.0)

    new4 = compute_new_4d_features(mid_p[:n_min], mid_c[:n_min])
    gpu = np.concatenate([gpu_11d, new4], axis=1).astype(np.float32)

    stats = np.concatenate([cpu, gpu], axis=1).astype(np.float32)
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
    parser = argparse.ArgumentParser(description="Train CPU18 + GPU15 2-head classifier")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-per-class", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    np.random.seed(args.seed)

    cats = discover_categories(EMBEDDINGS_DIR)
    print(f"Found {len(cats)} categories")

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

    ai_cls = np.concatenate(ai_cls, axis=0)
    ai_stats = np.concatenate(ai_stats, axis=0)
    real_cls = np.concatenate(real_cls, axis=0)
    real_stats = np.concatenate(real_stats, axis=0)

    print(f"\nBefore balancing: AI={ai_cls.shape[0]}, Real={real_cls.shape[0]}")
    print(f"Stats dim: {ai_stats.shape[1]} (expected 29)")

    n_ai, n_real = ai_cls.shape[0], real_cls.shape[0]
    n_target = min(n_ai, n_real) if args.max_per_class == 0 else min(args.max_per_class, n_ai, n_real)

    # 1:1 バランシング（少ない方に合わせる）
    if n_ai > n_target:
        idx = np.random.choice(n_ai, n_target, replace=False)
        ai_cls, ai_stats = ai_cls[idx], ai_stats[idx]
    if n_real > n_target:
        idx = np.random.choice(n_real, n_target, replace=False)
        real_cls, real_stats = real_cls[idx], real_stats[idx]

    print(f"After balancing:  AI={n_target}, Real={n_target} (1:1)")

    X_cls = np.concatenate([ai_cls, real_cls], axis=0)
    X_stats = np.concatenate([ai_stats, real_stats], axis=0)
    y = np.concatenate([np.ones(n_target), np.zeros(n_target)], axis=0)

    perm = np.random.permutation(len(y))
    X_cls, X_stats, y = X_cls[perm], X_stats[perm], y[perm]

    split = int(len(y) * 0.9)
    X_cls_train, X_cls_val = X_cls[:split], X_cls[split:]
    X_stats_train, X_stats_val = X_stats[:split], X_stats[split:]
    y_train, y_val = y[:split], y[split:]

    print(f"Train: {split}, Val: {len(y) - split}")

    cls_mean = X_cls_train.mean(axis=0)
    cls_std = X_cls_train.std(axis=0) + 1e-6
    stats_mean = X_stats_train.mean(axis=0)
    stats_std = X_stats_train.std(axis=0) + 1e-6

    X_cls_train = (X_cls_train - cls_mean) / cls_std
    X_cls_val = (X_cls_val - cls_mean) / cls_std
    X_stats_train = (X_stats_train - stats_mean) / stats_std
    X_stats_val = (X_stats_val - stats_mean) / stats_std

    print("\nTraining Head A (CLS 768d)...")
    head_a = train_head(X_cls_train, y_train, epochs=args.epochs, lr=args.lr,
                        weight_decay=args.weight_decay, batch_size=args.batch_size, seed=args.seed)

    print("Training Head B (Stats 29d)...")
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
        "cpu_v2_idx": CPU_V2_SELECT_IDX,
        "extra_idx": EXTRA_SELECT_IDX,
        "boundary_idx": BOUNDARY_SELECT_IDX,
        "gpu_v3_idx": GPU_V3_IDX,
        "new_feat_names": NEW_FEAT_NAMES,
    }, args.out_dir / "norm_stats.pt")

    with open(args.out_dir / "best_alpha.json", "w") as f:
        json.dump(best, f, indent=2)

    print("\n" + "=" * 60)
    print("=== Results ===")
    print(f"PR-AUC:  {best['pr_auc']:.4f}")
    print(f"ROC-AUC: {best['roc_auc']:.4f}")
    print(f"Best α:  {best['alpha']:.2f}")
    print(f"Total dim: 768 + 29 = 797")
    print("=" * 60)
    print(f"\nSaved to: {args.out_dir}")


if __name__ == "__main__":
    main()
