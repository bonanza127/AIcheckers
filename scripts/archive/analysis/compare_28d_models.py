#!/usr/bin/env python3
"""
28d系モデル比較スクリプト
28d, 28d_niji7, 28d_extra, 28d_plusの検出率を比較
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


EMB_DIR = Path("/home/techne/aicheckers/embeddings")

# All 28d models to compare
MODELS = {
    '28d': Path("/home/techne/aicheckers/models/two_head_28d"),
    '28d_plus': Path("/home/techne/aicheckers/models/two_head_28d_plus"),
    '28d_plus_b': Path("/home/techne/aicheckers/models/two_head_28d_plus_beta"),
    '28d_60': Path("/home/techne/aicheckers/models/two_head_28d_plus_60"),
}

# 28d indices (same for all models)
GPU_4D_IDX = [1, 3, 5, 6]  # adj_sim_var, patch_var, norm_var, norm_range
CPU16_13D_IDX = [0, 1, 2, 4, 5, 7, 8, 9, 11, 12, 13, 14, 15]
CPU20_11D_IDX = [0, 1, 2, 3, 4, 5, 8, 10, 15, 16, 17]

STD_FLOOR = 1e-3


class TwoHeadClassifier(nn.Module):
    def __init__(self, cls_dim=768, gpu_dim=4, cpu_dim=24, hidden_dim=256):
        super().__init__()
        total_dim = cls_dim + gpu_dim + cpu_dim
        self.bn_input = nn.BatchNorm1d(total_dim)
        self.fc1 = nn.Linear(total_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)

        self.register_buffer("cls_mean", torch.zeros(cls_dim))
        self.register_buffer("cls_std", torch.ones(cls_dim))
        self.register_buffer("gpu_mean", torch.zeros(gpu_dim))
        self.register_buffer("gpu_std", torch.ones(gpu_dim))
        self.register_buffer("cpu_mean", torch.zeros(cpu_dim))
        self.register_buffer("cpu_std", torch.ones(cpu_dim))

    def forward(self, cls_feat, gpu_feat, cpu_feat):
        cls_std = torch.clamp(self.cls_std, min=STD_FLOOR)
        gpu_std = torch.clamp(self.gpu_std, min=STD_FLOOR)
        cpu_std = torch.clamp(self.cpu_std, min=STD_FLOOR)

        cls_norm = (cls_feat - self.cls_mean) / (cls_std + 1e-8)
        gpu_norm = (gpu_feat - self.gpu_mean) / (gpu_std + 1e-8)
        cpu_norm = (cpu_feat - self.cpu_mean) / (cpu_std + 1e-8)

        x = torch.cat([cls_norm, gpu_norm, cpu_norm], dim=-1)
        x = self.bn_input(x)
        x = F.gelu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.gelu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        return self.fc3(x)


def _load_required(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(f"Missing {name}: {path}")
    arr = np.load(path).astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)
    arr = np.clip(arr, -1e6, 1e6)
    return arr


def load_features_28d(cat: str):
    """Load features for 28d model (GPU 4d + CPU 24d)"""
    cls = _load_required(EMB_DIR / f"{cat}.npy", "cls")
    patch = _load_required(EMB_DIR / f"{cat}_patch_stats_v3.npy", "patch_stats_v3")
    cpu16 = _load_required(EMB_DIR / f"{cat}_cpu_stats_v2.npy", "cpu_stats_v2")
    cpu20 = _load_required(EMB_DIR / f"{cat}_cpu_stats_v3_20d.npy", "cpu_stats_v3_20d")

    gpu = patch[:, GPU_4D_IDX]
    cpu16 = cpu16[:, CPU16_13D_IDX]
    cpu20 = cpu20[:, CPU20_11D_IDX]
    cpu = np.hstack([cpu16, cpu20])

    min_len = min(len(cls), len(gpu), len(cpu))
    return cls[:min_len], gpu[:min_len], cpu[:min_len]


def load_model(model_dir: Path, device):
    if not (model_dir / "model.pt").exists():
        return None
    model = TwoHeadClassifier(cls_dim=768, gpu_dim=4, cpu_dim=24)
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


def predict(model, cls, gpu, cpu, device, batch_size=1024):
    probs = []
    n = len(cls)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            cls_b = torch.tensor(cls[i:i+batch_size]).to(device)
            gpu_b = torch.tensor(gpu[i:i+batch_size]).to(device)
            cpu_b = torch.tensor(cpu[i:i+batch_size]).to(device)
            logits = model(cls_b, gpu_b, cpu_b).squeeze()
            p = torch.sigmoid(logits).cpu().numpy()
            if p.ndim == 0:
                p = np.array([p.item()])
            probs.extend(p)
    return np.array(probs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories", nargs="+", default=[
        "hard_negatives_ai", "novelai_artist_tagged_ai", "illustrious_ai",
        "niji7_twitter_ai", "danbooru_real"
    ])
    parser.add_argument("--threshold", type=float, default=0.9)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load models
    print("\nLoading models...")
    models = {}
    for name, path in MODELS.items():
        model = load_model(path, device)
        if model:
            models[name] = model
            print(f"  {name}: OK")
        else:
            print(f"  {name}: NOT FOUND")

    print(f"\n{'='*80}")
    print(f"28d系モデル比較 (@{args.threshold:.0%} threshold)")
    print(f"{'='*80}")

    # Header
    print(f"\n{'Category':<28}", end='')
    for name in models:
        print(f"{name:>12}", end='')
    print(f"{'':>6} (Type)")
    print("-" * (28 + 12 * len(models) + 10))

    results = {name: {} for name in models}

    for cat in args.categories:
        try:
            cls, gpu, cpu = load_features_28d(cat)
        except FileNotFoundError as e:
            print(f"{cat:<28} [SKIP] {e}")
            continue

        is_ai = not cat.endswith("_real")
        label = "AI" if is_ai else "Real"

        print(f"{cat:<28}", end='')

        for name, model in models.items():
            probs = predict(model, cls, gpu, cpu, device)
            if is_ai:
                rate = (probs >= args.threshold).mean() * 100
            else:
                rate = (probs < args.threshold).mean() * 100
            results[name][cat] = rate
            print(f"{rate:>11.1f}%", end='')

        print(f"  ({label})")

    print("-" * (28 + 12 * len(models) + 10))

    # Summary
    print(f"\n【分析サマリー】")
    base_model = '28d'
    if base_model in results:
        for name in models:
            if name == base_model:
                continue
            print(f"\n{base_model} → {name}:")
            for cat in args.categories:
                if cat in results[base_model] and cat in results[name]:
                    diff = results[name][cat] - results[base_model][cat]
                    sign = "+" if diff >= 0 else ""
                    indicator = "↑" if diff > 0 else ("↓" if diff < 0 else "=")
                    print(f"  {cat:<28}: {sign}{diff:.1f}% {indicator}")

    # Conclusions
    print(f"\n{'='*80}")
    print("【結論】")

    # Check if niji7 is problematic
    if '28d_niji7' in results and '28d' in results:
        hard_neg_diff = results['28d_niji7'].get('hard_negatives_ai', 0) - results['28d'].get('hard_negatives_ai', 0)
        if hard_neg_diff < -10:
            print(f"  - 28d_niji7: hard_negatives が {hard_neg_diff:.1f}% 低下 → niji7だけ追加すると悪化")

    if '28d_extra' in results and '28d' in results:
        hard_neg_diff = results['28d_extra'].get('hard_negatives_ai', 0) - results['28d'].get('hard_negatives_ai', 0)
        niji7_rate = results['28d_extra'].get('niji7_twitter_ai', 0)
        print(f"  - 28d_extra: hard_negatives {hard_neg_diff:+.1f}%, niji7検出率 {niji7_rate:.1f}%")

    if '28d_plus' in results and '28d' in results:
        hard_neg_diff = results['28d_plus'].get('hard_negatives_ai', 0) - results['28d'].get('hard_negatives_ai', 0)
        niji7_rate = results['28d_plus'].get('niji7_twitter_ai', 0)
        print(f"  - 28d_plus: hard_negatives {hard_neg_diff:+.1f}%, niji7検出率 {niji7_rate:.1f}%")


if __name__ == "__main__":
    main()
