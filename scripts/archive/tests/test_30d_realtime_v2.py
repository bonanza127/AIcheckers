#!/usr/bin/env python3
"""
30dモデル リアルタイムスキャンテスト v2
正しいCPU特徴量計算を使用
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
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from lib.patch_stats import compute_patch_stats_v3_batch
from extract_cpu_stats_v2 import extract_features as extract_cpu_v2

# Config
MODEL_DIR = Path("/home/techne/aicheckers/models/two_head_30d")
DINOV3_PATH = Path("/home/techne/aicheckers/models/dinov3-vitb16")
MID_LAYER = 6

# Feature indices
GPU_5D_IDX = [1, 3, 5, 6]
CPU16_13D_IDX = [0, 1, 2, 4, 5, 7, 8, 9, 11, 12, 13, 14, 15]
CPU20_12D_IDX = [0, 1, 2, 3, 4, 5, 8, 10, 15, 16, 17, 18]


class TwoHeadClassifier(nn.Module):
    def __init__(self, cls_dim=768, gpu_dim=5, cpu_dim=25, hidden_dim=256):
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
        cls_norm = (cls_feat - self.cls_mean) / (self.cls_std + 1e-8)
        gpu_norm = (gpu_feat - self.gpu_mean) / (self.gpu_std + 1e-8)
        cpu_norm = (cpu_feat - self.cpu_mean) / (self.cpu_std + 1e-8)
        x = torch.cat([cls_norm, gpu_norm, cpu_norm], dim=-1)
        x = self.bn_input(x)
        x = F.gelu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.gelu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        return self.fc3(x)


def compute_cpu_v2_from_path(img_path):
    """CPU v2特徴量を計算"""
    img = cv2.imread(str(img_path))
    if img is None:
        return np.zeros(18, dtype=np.float32)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    scale = 512 / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = cv2.resize(img_rgb, (new_w, new_h))
    mask = np.ones((new_h, new_w), dtype=bool)

    return extract_cpu_v2(img_resized, mask)


def compute_cpu_v3_20d(img_path):
    """CPU v3 20d特徴量（簡易版 - embeddingsから読み込むのが理想）"""
    # 簡易実装
    img = cv2.imread(str(img_path))
    if img is None:
        return np.zeros(20, dtype=np.float32)

    h, w = img.shape[:2]
    scale = 512 / max(h, w)
    img_resized = cv2.resize(img, (int(w * scale), int(h * scale)))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)

    features = np.zeros(20, dtype=np.float32)

    # Histogram features
    hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256))
    hist_norm = hist / (hist.sum() + 1e-8)
    features[0] = 1.0 / (np.std(hist_norm) + 1e-8)
    features[1] = (hist_norm > 0.01).sum() / 256.0

    # Hue histogram entropy
    h_hist, _ = np.histogram(hsv[:,:,0].flatten(), bins=180, range=(0, 180))
    h_hist = h_hist / (h_hist.sum() + 1e-8)
    h_hist = h_hist[h_hist > 0]
    features[2] = -np.sum(h_hist * np.log2(h_hist + 1e-10))

    features[3] = (gray > 200).sum() / gray.size

    edges = cv2.Canny(gray, 50, 150)
    features[4] = edges.mean() / 255.0
    features[5] = np.std(hsv[:,:,1].flatten()) / 255.0
    features[6] = float(np.mean(gray) - np.median(gray)) / 255.0

    sat = hsv[:,:,1].astype(np.float32)
    edges_f = edges.astype(np.float32)
    if sat.std() > 0 and edges_f.std() > 0:
        features[7] = np.corrcoef(sat.flatten(), edges_f.flatten())[0, 1]

    f = np.fft.fft2(gray.astype(np.float32))
    f_shift = np.fft.fftshift(f)
    mag = np.abs(f_shift)
    features[8] = np.mean(mag[mag.shape[0]//4:3*mag.shape[0]//4, mag.shape[1]//4:3*mag.shape[1]//4])

    cy, cx = mag.shape[0]//2, mag.shape[1]//2
    y, x = np.ogrid[:mag.shape[0], :mag.shape[1]]
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    features[9] = np.sum(r * mag) / (np.sum(mag) + 1e-8)

    low_mask = r < 30
    mid_mask = (r >= 30) & (r < 100)
    high_mask = r >= 100
    band_energies = [mag[low_mask].sum(), mag[mid_mask].sum(), mag[high_mask].sum()]
    features[10] = np.var(band_energies) / (np.mean(band_energies) + 1e-8)

    features[11] = np.std(hsv[:,:,0].flatten()) / 180.0
    features[12] = abs(float(np.mean(hsv[:,:,2])) - 128) / 128.0

    var_orig = np.var(gray)
    if gray.shape[0] > 2 and gray.shape[1] > 2:
        gray_down = cv2.resize(gray, (gray.shape[1]//2, gray.shape[0]//2))
        var_down = np.var(gray_down)
        features[13] = var_orig / (var_down + 1e-8)

    features[15] = float((np.diff(gray.astype(np.int16), axis=1) == 0).mean())
    features[16] = np.var(gray.astype(np.float32) - cv2.GaussianBlur(gray, (5, 5), 0))
    features[17] = (gray > 250).mean()
    features[18] = np.std(band_energies)

    return features


def compute_mid_adj_sim_var(patches, device):
    B, N, D = patches.shape
    grid = patches.reshape(B, 14, 14, D)
    h_sim = F.cosine_similarity(grid[:, :, :-1].reshape(-1, D), grid[:, :, 1:].reshape(-1, D), dim=1).reshape(B, 14, 13)
    v_sim = F.cosine_similarity(grid[:, :-1, :].reshape(-1, D), grid[:, 1:, :].reshape(-1, D), dim=1).reshape(B, 13, 14)
    all_sim = torch.cat([h_sim.reshape(B, -1), v_sim.reshape(B, -1)], dim=1)
    return all_sim.var(dim=1).cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='/home/techne/aicheckers/data/hard_negatives')
    parser.add_argument('--limit', type=int, default=100)
    parser.add_argument('--batch', type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print("Loading 30d model...")
    model = TwoHeadClassifier(cls_dim=768, gpu_dim=5, cpu_dim=25).to(device)
    state_dict = torch.load(MODEL_DIR / "model.pt", map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()

    # Load DINOv3
    print("Loading DINOv3...")
    from transformers import AutoImageProcessor, AutoModel
    processor = AutoImageProcessor.from_pretrained(str(DINOV3_PATH))
    dino = AutoModel.from_pretrained(str(DINOV3_PATH)).to(device)
    dino.eval()

    # Get images
    img_dir = Path(args.dir)
    images = sorted([p for p in img_dir.glob("*") if p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.webp']])[:args.limit]
    print(f"Testing {len(images)} images from {img_dir}")

    scores = []
    errors = []

    for i in tqdm(range(0, len(images), args.batch), desc="Scanning"):
        batch_files = images[i:i+args.batch]
        batch_images = []
        batch_valid_idx = []

        for j, fp in enumerate(batch_files):
            try:
                img = Image.open(fp).convert('RGB')
                batch_images.append(img)
                batch_valid_idx.append(j)
            except:
                errors.append(str(fp))

        if not batch_images:
            continue

        # DINOv3
        inputs = processor(images=batch_images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = dino(**inputs, output_hidden_states=True)
            cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            mid_hidden = outputs.hidden_states[MID_LAYER + 1]
            mid_patches = mid_hidden[:, 5:5+196, :]

            patch_stats = compute_patch_stats_v3_batch(mid_patches)
            gpu_4d = patch_stats[:, GPU_5D_IDX]
            mid_adj_var = compute_mid_adj_sim_var(mid_patches, device)
            gpu_5d = np.hstack([gpu_4d, mid_adj_var.reshape(-1, 1)])

        # CPU features
        cpu_v2_list = []
        cpu_v3_list = []
        valid_idx = 0
        for j, fp in enumerate(batch_files):
            if j in batch_valid_idx:
                cpu_v2 = compute_cpu_v2_from_path(fp)
                cpu_v3 = compute_cpu_v3_20d(fp)
                valid_idx += 1
            else:
                cpu_v2 = np.zeros(18, dtype=np.float32)
                cpu_v3 = np.zeros(20, dtype=np.float32)
            cpu_v2_list.append(cpu_v2)
            cpu_v3_list.append(cpu_v3)

        cpu_v2_arr = np.array(cpu_v2_list[:len(batch_images)])
        cpu_v3_arr = np.array(cpu_v3_list[:len(batch_images)])

        cpu16_13d = cpu_v2_arr[:, CPU16_13D_IDX]
        cpu20_12d = cpu_v3_arr[:, CPU20_12D_IDX]
        cpu_25d = np.hstack([cpu16_13d, cpu20_12d])

        # NaN handling
        cls_emb = np.nan_to_num(cls_emb, nan=0.0)
        gpu_5d = np.nan_to_num(gpu_5d, nan=0.0)
        cpu_25d = np.nan_to_num(cpu_25d, nan=0.0)

        # Inference
        cls_t = torch.tensor(cls_emb, dtype=torch.float32).to(device)
        gpu_t = torch.tensor(gpu_5d, dtype=torch.float32).to(device)
        cpu_t = torch.tensor(cpu_25d, dtype=torch.float32).to(device)

        with torch.no_grad():
            logits = model(cls_t, gpu_t, cpu_t)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            scores.extend(probs.tolist())

    scores = np.array(scores)

    print("\n" + "="*60)
    print(f"30d Model Real-time Scan Results v2")
    print("="*60)
    print(f"Total: {len(scores)} | Errors: {len(errors)}")
    print()

    thresholds = [0.3, 0.5, 0.7, 0.9]
    for t in thresholds:
        detected = (scores >= t).sum()
        rate = detected / len(scores) * 100
        print(f"Threshold {t}: {detected}/{len(scores)} ({rate:.1f}%)")

    print()
    print(f"Mean: {scores.mean():.4f} | Median: {np.median(scores):.4f}")
    print(f"Min: {scores.min():.4f} | Max: {scores.max():.4f}")

    # Distribution
    print("\nScore distribution:")
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    hist, _ = np.histogram(scores, bins=bins)
    for j in range(len(hist)):
        bar = "█" * (hist[j] * 50 // max(len(scores), 1))
        print(f"  {bins[j]:.1f}-{bins[j+1]:.1f}: {hist[j]:4d} {bar}")


if __name__ == "__main__":
    main()
