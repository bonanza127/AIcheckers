#!/usr/bin/env python3
"""
29d Model Real-time Scan Test
"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.patch_stats import compute_patch_stats_v3_batch
from lib.cpu_stats import compute_cpu_stats

# Config
MODEL_DIR = Path("/home/techne/aicheckers/models/two_head_29d")
DINOV3_PATH = Path("/home/techne/aicheckers/models/dinov3-vitb16")
MID_LAYER = 6

# Feature indices (29d model)
GPU_4D_IDX = [1, 3, 5, 6]  # adj_sim_var, patch_var, norm_var, norm_range
CPU16_13D_IDX = [0, 1, 2, 4, 5, 7, 8, 9, 11, 12, 13, 14, 15]
CPU20_11D_IDX = [0, 1, 2, 3, 4, 5, 8, 10, 15, 16, 17]  # 18 (edge_continuity_ratio) removed

STD_FLOOR = 1e-3


class TwoHeadClassifier(nn.Module):
    def __init__(self, cls_dim=768, gpu_dim=5, cpu_dim=24, hidden_dim=256):
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
        self.register_buffer('cls_mean', torch.zeros(cls_dim))
        self.register_buffer('cls_std', torch.ones(cls_dim))
        self.register_buffer('gpu_mean', torch.zeros(gpu_dim))
        self.register_buffer('gpu_std', torch.ones(gpu_dim))
        self.register_buffer('cpu_mean', torch.zeros(cpu_dim))
        self.register_buffer('cpu_std', torch.ones(cpu_dim))

    def forward(self, cls_feat, gpu_feat, cpu_feat):
        cls_std_safe = torch.clamp(self.cls_std, min=STD_FLOOR)
        gpu_std_safe = torch.clamp(self.gpu_std, min=STD_FLOOR)
        cpu_std_safe = torch.clamp(self.cpu_std, min=STD_FLOOR)

        cls_norm = (cls_feat - self.cls_mean) / (cls_std_safe + 1e-8)
        gpu_norm = (gpu_feat - self.gpu_mean) / (gpu_std_safe + 1e-8)
        cpu_norm = (cpu_feat - self.cpu_mean) / (cpu_std_safe + 1e-8)

        x = torch.cat([cls_norm, gpu_norm, cpu_norm], dim=-1)
        x = self.bn_input(x)
        x = F.gelu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.gelu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        return self.fc3(x)


def compute_mid_adj_sim_var(patches, device):
    B, N, D = patches.shape
    grid = patches.reshape(B, 14, 14, D)
    h_sim = F.cosine_similarity(grid[:, :, :-1].reshape(-1, D), grid[:, :, 1:].reshape(-1, D), dim=1).reshape(B, 14, 13)
    v_sim = F.cosine_similarity(grid[:, :-1, :].reshape(-1, D), grid[:, 1:, :].reshape(-1, D), dim=1).reshape(B, 13, 14)
    all_sim = torch.cat([h_sim.reshape(B, -1), v_sim.reshape(B, -1)], dim=1)
    return all_sim.var(dim=1).cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='/home/techne/Downloads')
    parser.add_argument('--limit', type=int, default=100)
    parser.add_argument('--verbose', '-v', action='store_true', default=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print("Loading 29d model...")
    model = TwoHeadClassifier(cls_dim=768, gpu_dim=5, cpu_dim=24).to(device)
    state_dict = torch.load(MODEL_DIR / "model.pt", map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()

    # Check std values
    print("\nChecking cpu_std values...")
    cpu_std = model.cpu_std.cpu().numpy()
    danger_count = 0
    for i, v in enumerate(cpu_std):
        if v < 0.001:
            print(f"  WARNING: cpu_std[{i}] = {v:.6f}")
            danger_count += 1
    if danger_count == 0:
        print("  All std values are safe!")

    # Load DINOv3
    print("\nLoading DINOv3...")
    from transformers import AutoImageProcessor, AutoModel
    processor = AutoImageProcessor.from_pretrained(str(DINOV3_PATH))
    dino = AutoModel.from_pretrained(str(DINOV3_PATH)).to(device)
    dino.eval()

    # Get images
    img_dir = Path(args.dir)
    images = sorted([p for p in img_dir.glob("*") if p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.webp']])[:args.limit]
    print(f"\nTesting {len(images)} images from {img_dir}")
    print("="*60)

    for fp in images:
        try:
            img = Image.open(fp).convert('RGB')
        except:
            continue

        # DINOv3
        inputs = processor(images=img, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = dino(**inputs, output_hidden_states=True)
            cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            mid_hidden = outputs.hidden_states[MID_LAYER + 1]
            mid_patches = mid_hidden[:, 5:5+196, :]

            patch_stats = compute_patch_stats_v3_batch(mid_patches)
            gpu_4d = patch_stats[:, GPU_4D_IDX]
            mid_adj_var = compute_mid_adj_sim_var(mid_patches, device)
            gpu_5d = np.hstack([gpu_4d, mid_adj_var.reshape(-1, 1)])

        # CPU features
        cpu_v2, cpu_v3_20d = compute_cpu_stats(fp)
        cpu16_13d = cpu_v2[CPU16_13D_IDX].reshape(1, -1)
        cpu20_11d = cpu_v3_20d[CPU20_11D_IDX].reshape(1, -1)
        cpu_24d = np.hstack([cpu16_13d, cpu20_11d])

        # NaN handling
        cls_emb = np.nan_to_num(cls_emb, nan=0.0)
        gpu_5d = np.nan_to_num(gpu_5d, nan=0.0)
        cpu_24d = np.nan_to_num(cpu_24d, nan=0.0)

        # Inference
        cls_t = torch.tensor(cls_emb, dtype=torch.float32).to(device)
        gpu_t = torch.tensor(gpu_5d, dtype=torch.float32).to(device)
        cpu_t = torch.tensor(cpu_24d, dtype=torch.float32).to(device)

        with torch.no_grad():
            logit = model(cls_t, gpu_t, cpu_t)
            prob = torch.sigmoid(logit).item()

        verdict = "AI" if prob >= 0.5 else "Human"
        logit_val = logit.item()

        # Sanity check
        status = "OK" if -20 < logit_val < 20 else "WARNING"
        print(f"{fp.name}: {prob*100:.1f}% -> {verdict}  [logit={logit_val:.2f}] {status}")


if __name__ == "__main__":
    main()
