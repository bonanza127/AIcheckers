#!/usr/bin/env python3
"""
Compare 25d vs 29d models on the same embeddings.

Reads precomputed embeddings and applies each model's exact feature indexing.
Outputs detection rates at requested thresholds.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


EMB_DIR = Path("/home/techne/aicheckers/embeddings")

# 25d indices
GPU_3D_IDX_25D = [1, 3, 5]  # adj_sim_var, patch_var, norm_var
CPU16_12D_IDX_25D = [0, 1, 2, 4, 7, 8, 9, 11, 12, 13, 14, 15]
CPU20_10D_IDX_25D = [0, 1, 2, 3, 4, 8, 10, 15, 16, 17]

# 29d indices
GPU_4D_IDX_29D = [1, 3, 5, 6]  # adj_sim_var, patch_var, norm_var, norm_range
CPU16_13D_IDX_29D = [0, 1, 2, 4, 5, 7, 8, 9, 11, 12, 13, 14, 15]
CPU20_11D_IDX_29D = [0, 1, 2, 3, 4, 5, 8, 10, 15, 16, 17]

STD_FLOOR = 1e-3


class TwoHeadClassifier(nn.Module):
    def __init__(self, cls_dim=768, gpu_dim=0, cpu_dim=0, hidden_dim=256):
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
    return np.load(path)


def load_features_25d(cat: str):
    cls = _load_required(EMB_DIR / f"{cat}.npy", "cls")
    patch = _load_required(EMB_DIR / f"{cat}_patch_stats_v3.npy", "patch_stats_v3")
    cpu16 = _load_required(EMB_DIR / f"{cat}_cpu_stats_v2.npy", "cpu_stats_v2")
    cpu20 = _load_required(EMB_DIR / f"{cat}_cpu_stats_v3_20d.npy", "cpu_stats_v3_20d")

    gpu = patch[:, GPU_3D_IDX_25D]
    cpu16 = cpu16[:, CPU16_12D_IDX_25D]
    cpu20 = cpu20[:, CPU20_10D_IDX_25D]
    cpu = np.hstack([cpu16, cpu20])

    min_len = min(len(cls), len(gpu), len(cpu))
    return cls[:min_len], gpu[:min_len], cpu[:min_len]


def load_features_29d(cat: str):
    cls = _load_required(EMB_DIR / f"{cat}.npy", "cls")
    patch = _load_required(EMB_DIR / f"{cat}_patch_stats_v3.npy", "patch_stats_v3")
    cpu16 = _load_required(EMB_DIR / f"{cat}_cpu_stats_v2.npy", "cpu_stats_v2")
    cpu20 = _load_required(EMB_DIR / f"{cat}_cpu_stats_v3_20d.npy", "cpu_stats_v3_20d")
    mid_adj = _load_required(EMB_DIR / f"{cat}_mid_adj_sim_var.npy", "mid_adj_sim_var")

    gpu = patch[:, GPU_4D_IDX_29D]
    cpu16 = cpu16[:, CPU16_13D_IDX_29D]
    cpu20 = cpu20[:, CPU20_11D_IDX_29D]
    cpu = np.hstack([cpu16, cpu20])

    min_len = min(len(cls), len(gpu), len(cpu), len(mid_adj))
    gpu = np.hstack([gpu[:min_len], mid_adj[:min_len].reshape(-1, 1)])
    return cls[:min_len], gpu[:min_len], cpu[:min_len]


def load_model(model_path: Path, gpu_dim: int, cpu_dim: int, device: torch.device):
    model = TwoHeadClassifier(cls_dim=768, gpu_dim=gpu_dim, cpu_dim=cpu_dim).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def run_inference(model, cls, gpu, cpu, device, batch=1024):
    logits_list = []
    for i in range(0, len(cls), batch):
        cls_b = torch.tensor(cls[i:i+batch], dtype=torch.float32, device=device)
        gpu_b = torch.tensor(gpu[i:i+batch], dtype=torch.float32, device=device)
        cpu_b = torch.tensor(cpu[i:i+batch], dtype=torch.float32, device=device)
        with torch.no_grad():
            logits = model(cls_b, gpu_b, cpu_b).cpu().numpy().reshape(-1)
            logits_list.append(logits)
    return np.concatenate(logits_list, axis=0)


def eval_category(cat: str, model_25, model_29, device, thresholds):
    cls25, gpu25, cpu25 = load_features_25d(cat)
    cls29, gpu29, cpu29 = load_features_29d(cat)

    min_len = min(len(cls25), len(cls29))
    cls25, gpu25, cpu25 = cls25[:min_len], gpu25[:min_len], cpu25[:min_len]
    cls29, gpu29, cpu29 = cls29[:min_len], gpu29[:min_len], cpu29[:min_len]

    cls25 = np.nan_to_num(cls25, nan=0.0, posinf=0.0, neginf=0.0)
    gpu25 = np.nan_to_num(gpu25, nan=0.0, posinf=0.0, neginf=0.0)
    cpu25 = np.nan_to_num(cpu25, nan=0.0, posinf=0.0, neginf=0.0)
    cls29 = np.nan_to_num(cls29, nan=0.0, posinf=0.0, neginf=0.0)
    gpu29 = np.nan_to_num(gpu29, nan=0.0, posinf=0.0, neginf=0.0)
    cpu29 = np.nan_to_num(cpu29, nan=0.0, posinf=0.0, neginf=0.0)

    logits25 = run_inference(model_25, cls25, gpu25, cpu25, device)
    logits29 = run_inference(model_29, cls29, gpu29, cpu29, device)
    probs25 = 1 / (1 + np.exp(-logits25))
    probs29 = 1 / (1 + np.exp(-logits29))

    results = []
    for t in thresholds:
        d25 = (probs25 >= t).mean()
        d29 = (probs29 >= t).mean()
        results.append((t, d25, d29))
    return results


def main():
    parser = argparse.ArgumentParser(description="Compare 25d vs 29d models on embeddings")
    parser.add_argument("--model-25d", default="/home/techne/aicheckers/models/two_head_25d/model.pt")
    parser.add_argument("--model-29d", default="/home/techne/aicheckers/models/two_head_29d_ep30/model.pt")
    parser.add_argument("--categories", nargs="*", default=[
        "hard_negatives_ai",
        "novelai_artist_tagged_ai",
        "novelai_combined_ai",
        "illustrious_ai",
        "danbooru_real",
    ])
    parser.add_argument("--thresholds", nargs="*", type=float, default=[0.5, 0.7, 0.9])
    parser.add_argument("--batch", type=int, default=1024)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_25 = load_model(Path(args.model_25d), gpu_dim=3, cpu_dim=22, device=device)
    model_29 = load_model(Path(args.model_29d), gpu_dim=5, cpu_dim=24, device=device)

    print(f"Device: {device}")
    print(f"25d model: {args.model_25d}")
    print(f"29d model: {args.model_29d}")
    print(f"Thresholds: {args.thresholds}")

    for cat in args.categories:
        print(f"\n[{cat}]")
        results = eval_category(cat, model_25, model_29, device, args.thresholds)
        for t, d25, d29 in results:
            print(f"  @ {t:.2f}: 25d={d25*100:.1f}%  29d={d29*100:.1f}%  Δ={(d25-d29)*100:+.1f}%")


if __name__ == "__main__":
    main()
