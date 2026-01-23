#!/usr/bin/env python3
"""
Test 30d Two-Head model on hard negatives
Extracts features on-the-fly for images in data/hard_negatives/
"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from lib.patch_stats import compute_patch_stats_v3
from lib.cpu_stats import compute_cpu_stats

# Config
MODEL_DIR = Path("/home/techne/aicheckers/models/two_head_30d")
HARDNEG_DIR = Path("/home/techne/aicheckers/data/hard_negatives")
MID_LAYER = 6

# Feature indices (from train_30d.py)
GPU_5D_IDX = [1, 3, 5, 6]  # patch_var, degree_centrality, local_efficiency, edge_interior_gap
CPU16_13D_IDX = [0, 1, 2, 4, 5, 7, 8, 9, 11, 12, 13, 14, 15]
CPU20_12D_IDX = [0, 1, 2, 3, 4, 5, 8, 10, 15, 16, 17, 18]


class TwoHeadClassifier(nn.Module):
    """Two-Head Classifier matching train_30d.py architecture"""
    def __init__(self, cls_dim=768, gpu_dim=5, cpu_dim=25):
        super().__init__()
        self.cls_bn = nn.BatchNorm1d(cls_dim)
        self.gpu_bn = nn.BatchNorm1d(gpu_dim)
        self.cpu_bn = nn.BatchNorm1d(cpu_dim)

        combined_dim = cls_dim + gpu_dim + cpu_dim
        self.fc1 = nn.Linear(combined_dim, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, cls_emb, gpu_stats, cpu_stats):
        cls_n = self.cls_bn(cls_emb)
        gpu_n = self.gpu_bn(gpu_stats)
        cpu_n = self.cpu_bn(cpu_stats)

        x = torch.cat([cls_n, gpu_n, cpu_n], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def load_model(device):
    """Load 30d model"""
    model_path = MODEL_DIR / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    model = TwoHeadClassifier(cls_dim=768, gpu_dim=5, cpu_dim=25).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint.get("gpu_mean"), checkpoint.get("gpu_std"), \
           checkpoint.get("cpu_mean"), checkpoint.get("cpu_std")


def load_dinov3(device):
    """Load DINOv3 model"""
    from transformers import AutoImageProcessor, AutoModel

    model_path = Path("/home/techne/aicheckers/models/dinov3-vitb16")
    print(f"Loading DINOv3 from: {model_path}")

    processor = AutoImageProcessor.from_pretrained(str(model_path))
    model = AutoModel.from_pretrained(str(model_path))
    model.to(device)
    model.eval()

    return model, processor


def compute_mid_adj_sim_var(patches):
    """Compute adjacency similarity variance from patches (1, 196, 768)"""
    B, N, D = patches.shape
    grid = patches.reshape(B, 14, 14, D)

    h_sim = F.cosine_similarity(
        grid[:, :, :-1].reshape(-1, D),
        grid[:, :, 1:].reshape(-1, D),
        dim=1
    ).reshape(B, 14, 13)

    v_sim = F.cosine_similarity(
        grid[:, :-1, :].reshape(-1, D),
        grid[:, 1:, :].reshape(-1, D),
        dim=1
    ).reshape(B, 13, 14)

    all_sim = torch.cat([h_sim.reshape(B, -1), v_sim.reshape(B, -1)], dim=1)
    return all_sim.var(dim=1).cpu().numpy()


def extract_features_single(dino_model, processor, img_path, device):
    """Extract all features for a single image"""
    try:
        img = Image.open(img_path).convert('RGB')
    except Exception as e:
        return None

    # DINOv3 features
    inputs = processor(images=img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = dino_model(**inputs, output_hidden_states=True)

        # CLS from final layer
        cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # (1, 768)

        # Mid-layer patches
        mid_hidden = outputs.hidden_states[MID_LAYER + 1]
        mid_patches = mid_hidden[:, 5:5+196, :]  # (1, 196, 768)

    # GPU features from patch_stats_v3
    patch_stats = compute_patch_stats_v3(mid_patches)  # dict with 34 features
    gpu_4d = np.array([patch_stats[k] for k in [
        'patch_var', 'degree_centrality', 'local_efficiency', 'edge_interior_gap'
    ]]).reshape(1, 4)

    # mid_adj_sim_var
    mid_adj_var = compute_mid_adj_sim_var(mid_patches).reshape(1, 1)

    # GPU 5d
    gpu_5d = np.hstack([gpu_4d, mid_adj_var])

    # CPU features from image directly
    cpu_stats_v2, cpu_stats_v3_20d = compute_cpu_stats(img)

    # CPU16 13d
    cpu16_13d = cpu_stats_v2[CPU16_13D_IDX].reshape(1, -1)

    # CPU20 12d
    cpu20_12d = cpu_stats_v3_20d[CPU20_12D_IDX].reshape(1, -1)

    # CPU 25d
    cpu_25d = np.hstack([cpu16_13d, cpu20_12d])

    return cls_emb, gpu_5d, cpu_25d


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load models
    classifier, gpu_mean, gpu_std, cpu_mean, cpu_std = load_model(device)
    print(f"Loaded 30d classifier from: {MODEL_DIR}")

    dino_model, processor = load_dinov3(device)

    # Get hard negative images
    images = sorted([
        p for p in HARDNEG_DIR.glob("*")
        if p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.webp']
    ])
    print(f"Found {len(images)} hard negative images")

    # Test
    scores = []
    errors = []

    for img_path in tqdm(images, desc="Testing"):
        result = extract_features_single(dino_model, processor, img_path, device)
        if result is None:
            errors.append(str(img_path))
            continue

        cls_emb, gpu_5d, cpu_25d = result

        # Normalize
        if gpu_mean is not None:
            gpu_5d = (gpu_5d - gpu_mean) / (gpu_std + 1e-8)
        if cpu_mean is not None:
            cpu_25d = (cpu_25d - cpu_mean) / (cpu_std + 1e-8)

        # Convert to tensors
        cls_t = torch.tensor(cls_emb, dtype=torch.float32).to(device)
        gpu_t = torch.tensor(gpu_5d, dtype=torch.float32).to(device)
        cpu_t = torch.tensor(cpu_25d, dtype=torch.float32).to(device)

        with torch.no_grad():
            logit = classifier(cls_t, gpu_t, cpu_t)
            prob = torch.sigmoid(logit).item()
            scores.append(prob)

    scores = np.array(scores)

    # Statistics
    print("\n" + "="*60)
    print("Hard Negatives Test Results (30d Model)")
    print("="*60)
    print(f"Total images: {len(images)}")
    print(f"Processed: {len(scores)}")
    print(f"Errors: {len(errors)}")
    print()

    # Detection rates at various thresholds
    thresholds = [0.3, 0.5, 0.7, 0.9]
    for t in thresholds:
        detected = (scores >= t).sum()
        rate = detected / len(scores) * 100
        print(f"Threshold {t}: {detected}/{len(scores)} ({rate:.1f}%)")

    print()
    print(f"Mean score: {scores.mean():.4f}")
    print(f"Median score: {np.median(scores):.4f}")
    print(f"Min/Max: {scores.min():.4f} / {scores.max():.4f}")

    # Save detailed results
    output_path = Path("logs/hardneg_30d_test.csv")
    import pandas as pd
    df = pd.DataFrame({
        'file': [p.name for p in images[:len(scores)]],
        'score': scores
    })
    df.to_csv(output_path, index=False)
    print(f"\nSaved detailed results to: {output_path}")


if __name__ == "__main__":
    main()
