#!/usr/bin/env python3
"""
Train a stacking Logistic Regression meta-classifier.

Inputs:
  - Base model outputs from 2-head DINO classifier
  - CPU extra stats (cpu_stats_v2)

Outputs:
  - artifacts/meta_logistic.joblib
  - artifacts/meta_scaler_stats.npz
  - artifacts/meta_feature_order.json
  - artifacts/meta_splits.json
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

EMBEDDINGS_DIR = Path("/home/techne/aicheckers/embeddings")
ARTIFACTS_DIR = Path("/home/techne/aicheckers/artifacts")


def load_list(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def discover_categories():
    cats = []
    for stats_path in EMBEDDINGS_DIR.glob("*_patch_stats_v3.npy"):
        name = stats_path.name.replace("_patch_stats_v3.npy", "")
        cls_path = EMBEDDINGS_DIR / f"{name}.npy"
        cpu_path = EMBEDDINGS_DIR / f"{name}_cpu_stats_v2.npy"
        if not cls_path.exists() or not cpu_path.exists():
            continue
        if name.endswith("_ai"):
            label = 1
        elif name.endswith("_real"):
            label = 0
        else:
            continue
        cats.append((name, label))
    return sorted(cats)


def _get_keep_idx(norm_stats: dict):
    keep_idx = norm_stats.get("keep_idx")
    if keep_idx is None:
        keep_idx = norm_stats.get("keep_idx_11d")
    if keep_idx is None:
        keep_idx = list(range(len(norm_stats["stats_mean"])))
    return keep_idx


def compute_new_4d_features(patches: np.ndarray, mid_cls: np.ndarray, batch_size: int = 32) -> np.ndarray:
    """Compute 4d graph features used by the 15d base model."""
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


def compute_base_logits(
    cls: np.ndarray,
    stats: np.ndarray,
    head_a,
    head_b,
    norm_stats: dict,
    alpha: float,
    batch_size: int = 256,
):
    stats_mean = norm_stats["stats_mean"]
    stats_std = norm_stats["stats_std"]
    expected_dim = len(stats_mean)
    if stats.shape[1] != expected_dim:
        keep_idx = _get_keep_idx(norm_stats)
        stats = stats[:, keep_idx]
    cls_mean = norm_stats["cls_mean"]
    cls_std = norm_stats["cls_std"]

    cls = (cls - cls_mean) / (cls_std + 1e-6)
    stats = (stats - stats_mean) / (stats_std + 1e-6)

    device = next(head_a.parameters()).device
    logits = []
    with torch.no_grad():
        for i in range(0, len(cls), batch_size):
            cls_batch = torch.tensor(cls[i:i + batch_size], dtype=torch.float32, device=device)
            stats_batch = torch.tensor(stats[i:i + batch_size], dtype=torch.float32, device=device)
            la = head_a(cls_batch).squeeze(1)
            lb = head_b(stats_batch).squeeze(1)
            logits.append((la + alpha * lb).cpu().numpy())
    return np.concatenate(logits, axis=0)


def align_by_files(base_files, base_vals, cpu_files, cpu_vals):
    cpu_map = {p: i for i, p in enumerate(cpu_files)}
    keep_base = []
    keep_cpu = []
    ids = []
    missing = 0
    for i, p in enumerate(base_files):
        j = cpu_map.get(p)
        if j is None:
            missing += 1
            continue
        keep_base.append(base_vals[i])
        keep_cpu.append(cpu_vals[j])
        ids.append(p)
    return np.array(keep_base), np.array(keep_cpu), ids, missing


def split_train_val_test(y, seed=42, test_ratio=0.1, val_ratio=0.1):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    train_idx, test_idx = next(sss.split(np.zeros_like(y), y))

    y_train = y[train_idx]
    val_ratio_adj = val_ratio / (1.0 - test_ratio)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio_adj, random_state=seed)
    train_idx2, val_idx = next(sss2.split(np.zeros_like(y_train), y_train))
    train_idx = train_idx[train_idx2]

    return train_idx, val_idx, test_idx


def zscore_fit(x: np.ndarray):
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    return mean, std


def zscore_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray):
    std_safe = np.where(std < 1e-6, 1e-6, std)
    return (x - mean) / std_safe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-per-class", type=int, default=0)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--class-weight-balanced", action="store_true")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--hard-neg-files", type=Path, default=None)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cats = discover_categories()
    if not cats:
        raise SystemExit("No matching categories found.")

    # Load base model
    head_a = torch.nn.Linear(768, 1)
    head_b = None
    norm_stats = torch.load(args.base_model_dir / "norm_stats.pt", map_location="cpu", weights_only=False)
    keep_idx = norm_stats.get("keep_idx")
    if keep_idx is None:
        keep_idx = norm_stats.get("keep_idx_11d")
    if keep_idx is None:
        keep_idx = list(range(len(norm_stats["stats_mean"])))
    stats_dim = len(keep_idx)
    head_b = torch.nn.Linear(len(norm_stats["stats_mean"]), 1)
    head_a.load_state_dict(torch.load(args.base_model_dir / "head_a.pt", map_location="cpu"))
    head_b.load_state_dict(torch.load(args.base_model_dir / "head_b.pt", map_location="cpu"))
    head_a.eval()
    head_b.eval()

    alpha_path = args.base_model_dir / "best_alpha.json"
    if alpha_path.exists():
        alpha = json.loads(alpha_path.read_text())["alpha"]
    else:
        alpha = 1.0

    all_base = []
    all_cpu = []
    all_labels = []
    all_ids = []
    cpu_feature_names = None

    for name, label in cats:
        cls_path = EMBEDDINGS_DIR / f"{name}.npy"
        stats_path = EMBEDDINGS_DIR / f"{name}_patch_stats_v3.npy"
        patches_path = EMBEDDINGS_DIR / f"{name}_mid_patches.npy"
        mid_cls_path = EMBEDDINGS_DIR / f"{name}_mid_cls.npy"
        base_files_path = EMBEDDINGS_DIR / f"{name}_files.txt"
        cpu_path = EMBEDDINGS_DIR / f"{name}_cpu_stats_v2.npy"
        cpu_files_path = EMBEDDINGS_DIR / f"{name}_cpu_stats_v2_files.txt"
        cpu_meta_path = EMBEDDINGS_DIR / f"{name}_cpu_stats_v2_meta.json"

        if not cls_path.exists() or not stats_path.exists() or not cpu_path.exists():
            continue

        cls = np.load(cls_path, mmap_mode="r")
        stats_v3 = np.load(stats_path, mmap_mode="r")
        n_min = min(len(cls), len(stats_v3))
        if len(cls) != len(stats_v3):
            cls = cls[:n_min]
            stats_v3 = stats_v3[:n_min]

        base_files = load_list(base_files_path)
        if not base_files:
            base_files = [str(p) for p in range(n_min)]
        else:
            base_files = base_files[:n_min]

        stats_b = stats_v3
        if "new_feat_names" in norm_stats:
            if not patches_path.exists() or not mid_cls_path.exists():
                raise SystemExit(f"{name}: missing mid_patches or mid_cls for 15d base model")
            patches = np.load(patches_path, mmap_mode="r")
            mid_cls = np.load(mid_cls_path, mmap_mode="r")
            n_min = min(n_min, len(patches), len(mid_cls))
            cls = cls[:n_min]
            stats_v3 = stats_v3[:n_min]
            patches = patches[:n_min]
            mid_cls = mid_cls[:n_min]

            keep_idx = _get_keep_idx(norm_stats)
            stats_11d = stats_v3[:, keep_idx]
            new4_batch = min(32, args.batch_size)
            new4 = compute_new_4d_features(patches, mid_cls, batch_size=new4_batch)
            stats_b = np.concatenate([stats_11d, new4], axis=1)

        base_logits = compute_base_logits(
            cls,
            stats_b,
            head_a,
            head_b,
            norm_stats,
            alpha,
            batch_size=args.batch_size,
        )

        cpu = np.load(cpu_path, mmap_mode="r")
        cpu_files = load_list(cpu_files_path)
        if not cpu_files:
            cpu_files = base_files[: len(cpu)]
        if cpu_feature_names is None and cpu_meta_path.exists():
            cpu_feature_names = json.loads(cpu_meta_path.read_text()).get("features", None)

        base_logits, cpu, ids, missing = align_by_files(base_files, base_logits, cpu_files, cpu)
        if missing:
            print(f"[WARN] {name}: {missing} missing CPU entries")

        all_base.append(base_logits.reshape(-1, 1))
        all_cpu.append(cpu)
        all_labels.append(np.full(len(base_logits), label, dtype=np.int64))
        all_ids.extend(ids)

    X_base = np.concatenate(all_base, axis=0).astype(np.float32)
    X_cpu = np.concatenate(all_cpu, axis=0).astype(np.float32)
    y = np.concatenate(all_labels, axis=0)

    # Balance per class if requested
    if args.max_per_class > 0:
        rng = np.random.default_rng(args.seed)
        idx_ai = np.where(y == 1)[0]
        idx_real = np.where(y == 0)[0]
        n = min(args.max_per_class, len(idx_ai), len(idx_real))
        idx_ai = rng.choice(idx_ai, n, replace=False)
        idx_real = rng.choice(idx_real, n, replace=False)
        idx = np.concatenate([idx_ai, idx_real])
        X_base = X_base[idx]
        X_cpu = X_cpu[idx]
        y = y[idx]
        all_ids = [all_ids[i] for i in idx]

    train_idx, val_idx, test_idx = split_train_val_test(
        y, seed=args.seed, test_ratio=args.test_ratio, val_ratio=args.val_ratio
    )

    # Drop near-constant CPU features (train only)
    cpu_mean, cpu_std = zscore_fit(X_cpu[train_idx])
    keep_mask = cpu_std >= 1e-6
    X_cpu = X_cpu[:, keep_mask]
    cpu_mean = cpu_mean[keep_mask]
    cpu_std = cpu_std[keep_mask]
    if cpu_feature_names:
        cpu_feature_names = [n for n, k in zip(cpu_feature_names, keep_mask) if k]

    # OOF CV for C
    cs = [0.05, 0.1, 0.2, 0.5, 1.0]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    best = {"C": None, "logloss": 1e9, "auc": 0.0}

    for c in cs:
        oof_pred = np.zeros(len(train_idx), dtype=np.float32)
        for fold, (tr, va) in enumerate(skf.split(X_cpu[train_idx], y[train_idx])):
            tr_idx = train_idx[tr]
            va_idx = train_idx[va]

            mean_f, std_f = zscore_fit(X_cpu[tr_idx])
            X_tr = zscore_apply(X_cpu[tr_idx], mean_f, std_f)
            X_va = zscore_apply(X_cpu[va_idx], mean_f, std_f)
            X_tr_meta = np.concatenate([X_tr, X_base[tr_idx]], axis=1)
            X_va_meta = np.concatenate([X_va, X_base[va_idx]], axis=1)

            clf = LogisticRegression(
                penalty="l2",
                C=c,
                solver="liblinear",
                max_iter=5000,
                class_weight="balanced" if args.class_weight_balanced else None,
            )
            clf.fit(X_tr_meta, y[tr_idx])
            oof_pred[va] = clf.predict_proba(X_va_meta)[:, 1]

        loss = log_loss(y[train_idx], oof_pred, labels=[0, 1])
        auc = roc_auc_score(y[train_idx], oof_pred)
        print(f"[OOF] C={c}: logloss={loss:.5f}, auc={auc:.5f}")
        if loss < best["logloss"]:
            best = {"C": c, "logloss": loss, "auc": auc}

    # Train final model on full train
    X_cpu_z = zscore_apply(X_cpu, cpu_mean, cpu_std)
    X_cpu_train = X_cpu_z[train_idx]
    X_cpu_val = X_cpu_z[val_idx]
    X_cpu_test = X_cpu_z[test_idx]
    X_train = np.concatenate([X_cpu_train, X_base[train_idx]], axis=1)
    X_val = np.concatenate([X_cpu_val, X_base[val_idx]], axis=1)
    X_test = np.concatenate([X_cpu_test, X_base[test_idx]], axis=1)

    clf = LogisticRegression(
        penalty="l2",
        C=best["C"],
        solver="liblinear",
        max_iter=5000,
        class_weight="balanced" if args.class_weight_balanced else None,
    )
    clf.fit(X_train, y[train_idx])

    val_pred = clf.predict_proba(X_val)[:, 1]
    test_pred = clf.predict_proba(X_test)[:, 1]
    print(f"[VAL] logloss={log_loss(y[val_idx], val_pred):.5f}, auc={roc_auc_score(y[val_idx], val_pred):.5f}")
    print(f"[TEST] logloss={log_loss(y[test_idx], test_pred):.5f}, auc={roc_auc_score(y[test_idx], test_pred):.5f}")

    final_model = clf
    if args.calibrate:
        cal = CalibratedClassifierCV(clf, method="sigmoid", cv=5)
        cal.fit(X_train, y[train_idx])
        cal_val = cal.predict_proba(X_val)[:, 1]
        print(f"[CAL] val logloss={log_loss(y[val_idx], cal_val):.5f}")
        final_model = cal

    # Coefficient ranking
    if hasattr(clf, "coef_"):
        coefs = clf.coef_.reshape(-1)
        names = (cpu_feature_names or [f"cpu_{i}" for i in range(X_cpu.shape[1])]) + ["base_logit"]
        order = np.argsort(np.abs(coefs))[::-1]
        print("\nTop coefficients:")
        for i in order[:20]:
            print(f"  {names[i]:<35} {coefs[i]:+.4f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, args.out_dir / "meta_logistic.joblib")
    np.savez(
        args.out_dir / "meta_scaler_stats.npz",
        mean=cpu_mean.astype(np.float32),
        std=cpu_std.astype(np.float32),
        keep_mask=keep_mask,
    )
    feature_order = {
        "cpu_features": cpu_feature_names or [f"cpu_{i}" for i in range(X_cpu.shape[1])],
        "base_feature": "base_logit",
    }
    (args.out_dir / "meta_feature_order.json").write_text(json.dumps(feature_order, indent=2))
    splits = {
        "train_ids": [all_ids[i] for i in train_idx],
        "val_ids": [all_ids[i] for i in val_idx],
        "test_ids": [all_ids[i] for i in test_idx],
    }
    (args.out_dir / "meta_splits.json").write_text(json.dumps(splits, indent=2))

    # Hard negatives quick check if IDs exist
    hard_path = args.hard_neg_files
    if hard_path and hard_path.exists():
        hard_list = set(load_list(hard_path))
        hard_scores = []
        for idx, img_id in enumerate(all_ids):
            if img_id in hard_list:
                hard_scores.append(
                    final_model.predict_proba(
                        np.concatenate([X_cpu_z[idx:idx + 1], X_base[idx:idx + 1]], axis=1)
                    )[:, 1][0]
                )
        if hard_scores:
            arr = np.array(hard_scores)
            print(f"[HARD] n={len(arr)} min={arr.min():.4f} p50={np.percentile(arr,50):.4f} p90={np.percentile(arr,90):.4f} max={arr.max():.4f}")


if __name__ == "__main__":
    main()
