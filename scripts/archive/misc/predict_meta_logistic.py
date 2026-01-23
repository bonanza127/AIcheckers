#!/usr/bin/env python3
"""
Predict AI probability using meta logistic model.

Outputs:
  - CSV with path, ai_prob
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch

EMBEDDINGS_DIR = Path("/home/techne/aicheckers/embeddings")


def load_list(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def compute_base_logits(cls, stats_v3, head_a, head_b, norm_stats, alpha, batch_size=256):
    keep_idx = norm_stats["keep_idx"]
    stats = stats_v3[:, keep_idx]
    cls_mean = norm_stats["cls_mean"]
    cls_std = norm_stats["cls_std"]
    stats_mean = norm_stats["stats_mean"]
    stats_std = norm_stats["stats_std"]

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
    for i, p in enumerate(base_files):
        j = cpu_map.get(p)
        if j is None:
            continue
        keep_base.append(base_vals[i])
        keep_cpu.append(cpu_vals[j])
        ids.append(p)
    return np.array(keep_base), np.array(keep_cpu), ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-dir", type=Path, required=True)
    parser.add_argument("--meta-dir", type=Path, required=True)
    parser.add_argument("--category", type=str)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--out-csv", type=Path, default=Path("meta_predictions.csv"))
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    if not args.category and not args.all:
        raise SystemExit("Use --category or --all")

    # Load meta artifacts
    model = joblib.load(args.meta_dir / "meta_logistic.joblib")
    scaler = np.load(args.meta_dir / "meta_scaler_stats.npz")
    mean = scaler["mean"]
    std = scaler["std"]
    keep_mask = scaler["keep_mask"].astype(bool)

    # Load base model
    norm_stats = torch.load(args.base_model_dir / "norm_stats.pt", map_location="cpu")
    keep_idx = norm_stats["keep_idx"]
    head_a = torch.nn.Linear(768, 1)
    head_b = torch.nn.Linear(len(keep_idx), 1)
    head_a.load_state_dict(torch.load(args.base_model_dir / "head_a.pt", map_location="cpu"))
    head_b.load_state_dict(torch.load(args.base_model_dir / "head_b.pt", map_location="cpu"))
    head_a.eval()
    head_b.eval()

    alpha_path = args.base_model_dir / "best_alpha.json"
    if alpha_path.exists():
        alpha = json.loads(alpha_path.read_text())["alpha"]
    else:
        alpha = 1.0

    cats = []
    if args.all:
        for stats_path in EMBEDDINGS_DIR.glob("*_patch_stats_v3.npy"):
            name = stats_path.name.replace("_patch_stats_v3.npy", "")
            if (EMBEDDINGS_DIR / f"{name}.npy").exists():
                cats.append(name)
    else:
        cats = [args.category]

    rows = []
    for name in sorted(cats):
        cls_path = EMBEDDINGS_DIR / f"{name}.npy"
        stats_path = EMBEDDINGS_DIR / f"{name}_patch_stats_v3.npy"
        cpu_path = EMBEDDINGS_DIR / f"{name}_cpu_stats_v2.npy"
        base_files_path = EMBEDDINGS_DIR / f"{name}_files.txt"
        cpu_files_path = EMBEDDINGS_DIR / f"{name}_cpu_stats_v2_files.txt"

        if not cls_path.exists() or not stats_path.exists() or not cpu_path.exists():
            continue

        cls = np.load(cls_path, mmap_mode="r")
        stats_v3 = np.load(stats_path, mmap_mode="r")
        n_min = min(len(cls), len(stats_v3))
        cls = cls[:n_min]
        stats_v3 = stats_v3[:n_min]

        base_files = load_list(base_files_path)
        if not base_files:
            base_files = [str(p) for p in range(n_min)]
        else:
            base_files = base_files[:n_min]

        base_logits = compute_base_logits(
            cls, stats_v3, head_a, head_b, norm_stats, alpha, batch_size=args.batch_size
        )

        cpu = np.load(cpu_path, mmap_mode="r")
        cpu_files = load_list(cpu_files_path)
        if not cpu_files:
            cpu_files = base_files[: len(cpu)]

        base_logits, cpu, ids = align_by_files(base_files, base_logits, cpu_files, cpu)
        cpu = cpu[:, keep_mask]
        cpu_z = (cpu - mean) / np.where(std < 1e-6, 1e-6, std)
        X_meta = np.concatenate([cpu_z, base_logits.reshape(-1, 1)], axis=1)
        probs = model.predict_proba(X_meta)[:, 1]

        for img_id, p in zip(ids, probs):
            rows.append((img_id, float(p)))

    args.out_csv.write_text("path,ai_prob\n" + "\n".join(f"{p},{s:.6f}" for p, s in rows) + "\n")
    print(f"[DONE] wrote {args.out_csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
