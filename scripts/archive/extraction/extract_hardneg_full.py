#!/usr/bin/env python3
"""
Extract all missing features for hard_negatives:
- patch_stats_v3 (from DINOv3 mid-layer patches)
- cpu_stats_v2 (from images)
"""
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import cv2

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from lib.patch_stats import compute_patch_stats_v3_batch

# Config
EMBED_DIR = Path("embeddings")
IMG_DIR = Path("/home/techne/aicheckers/data/hard_negatives")
CAT_NAME = "hard_negatives_ai"
MID_LAYER = 6
BATCH_SIZE = 32

# CPU stats v2 feature names (18 features)
CPU_V2_FEATURES = [
    "banding_score", "radial_spectrum_slope", "stroke_width_proxy", "text_area_ratio",
    "fractal_dim_edge_512", "patchwise_edge_density", "st_aniso_mean", "st_aniso_var",
    "st_aniso_spatial_gradient", "flat_boundary_peri_area", "stroke_p90", "flat_hole_ratio",
    "highfreq_spatial_autocorr", "patch_vs_global_rank_entropy_gap", "flat_ratio",
    "flat_ratio_variance_across_tiles", "patch_vs_global_st_aniso_gap",
    "patch_vs_global_spectrum_slope_gap"
]


def banding_score(gray, mask):
    """Calculate banding score"""
    q = (gray // 8).astype(np.uint8)
    m = mask[:, 1:] & mask[:, :-1]
    if m.sum() == 0:
        return 0.0
    diffs = np.abs(q[:, 1:].astype(np.int16) - q[:, :-1].astype(np.int16))
    return float((diffs[m] == 0).mean())


def radial_spectrum_slope(gray, mask=None):
    """Calculate radial spectrum slope"""
    f = np.fft.fft2(gray.astype(np.float32))
    f_shift = np.fft.fftshift(f)
    mag = np.abs(f_shift)

    h, w = gray.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx)**2 + (y - cy)**2).astype(int)

    r_max = min(cy, cx)
    radial_sum = np.zeros(r_max)
    radial_count = np.zeros(r_max)

    for i in range(1, r_max):
        ring_mask = (r == i)
        radial_sum[i] = mag[ring_mask].sum()
        radial_count[i] = ring_mask.sum()

    valid = radial_count > 0
    radial_avg = np.zeros(r_max)
    radial_avg[valid] = radial_sum[valid] / radial_count[valid]

    # Fit log-log slope
    freqs = np.arange(1, r_max)
    valid = (radial_avg[1:] > 0) & (freqs > 0)
    if valid.sum() < 2:
        return 0.0

    log_freq = np.log(freqs[valid])
    log_power = np.log(radial_avg[1:][valid])
    slope, _ = np.polyfit(log_freq, log_power, 1)
    return float(slope)


def compute_structure_tensor(gray, sigma=1.0):
    """Compute structure tensor for anisotropy features"""
    # Gradients
    gx = cv2.Sobel(gray.astype(np.float32), cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray.astype(np.float32), cv2.CV_64F, 0, 1, ksize=3)

    # Structure tensor components
    Ixx = cv2.GaussianBlur(gx * gx, (0, 0), sigma)
    Iyy = cv2.GaussianBlur(gy * gy, (0, 0), sigma)
    Ixy = cv2.GaussianBlur(gx * gy, (0, 0), sigma)

    # Eigenvalues
    trace = Ixx + Iyy
    det = Ixx * Iyy - Ixy * Ixy
    discriminant = np.sqrt(np.maximum(trace**2 - 4*det, 0))

    lambda1 = (trace + discriminant) / 2
    lambda2 = (trace - discriminant) / 2

    # Anisotropy = (lambda1 - lambda2) / (lambda1 + lambda2 + eps)
    eps = 1e-8
    anisotropy = (lambda1 - lambda2) / (lambda1 + lambda2 + eps)

    return anisotropy


def compute_cpu_stats_single(img_path):
    """Compute CPU stats v2 for a single image"""
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            return None

        # Resize to 512
        h, w = img.shape[:2]
        scale = 512 / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = cv2.resize(img, (new_w, new_h))

        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        mask = np.ones(gray.shape, dtype=bool)

        # Compute features (simplified version)
        features = np.zeros(18, dtype=np.float32)

        # 0: banding_score
        features[0] = banding_score(gray, mask)

        # 1: radial_spectrum_slope
        features[1] = radial_spectrum_slope(gray)

        # 2: stroke_width_proxy (simplified)
        edges = cv2.Canny(gray, 50, 150)
        features[2] = edges.mean() / 255.0

        # 3: text_area_ratio (simplified - look for high-contrast small regions)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 2)
        features[3] = binary.mean() / 255.0

        # 4: fractal_dim_edge_512 (simplified box counting approximation)
        features[4] = np.log(edges.sum() + 1) / np.log(edges.size + 1)

        # 5: patchwise_edge_density
        patch_size = 32
        patches = []
        for y in range(0, new_h - patch_size, patch_size):
            for x in range(0, new_w - patch_size, patch_size):
                patch = edges[y:y+patch_size, x:x+patch_size]
                patches.append(patch.mean())
        features[5] = np.std(patches) if patches else 0.0

        # 6-8: Structure tensor anisotropy
        aniso = compute_structure_tensor(gray)
        features[6] = aniso.mean()  # st_aniso_mean
        features[7] = aniso.var()   # st_aniso_var
        # Spatial gradient of anisotropy
        gx = cv2.Sobel(aniso.astype(np.float32), cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(aniso.astype(np.float32), cv2.CV_64F, 0, 1, ksize=3)
        features[8] = np.sqrt(gx**2 + gy**2).mean()  # st_aniso_spatial_gradient

        # 9: flat_boundary_peri_area (simplified)
        flat_mask = aniso < 0.1
        contours, _ = cv2.findContours(flat_mask.astype(np.uint8),
                                        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        total_perim = sum(cv2.arcLength(c, True) for c in contours)
        total_area = flat_mask.sum()
        features[9] = total_perim / (total_area + 1e-8)

        # 10: stroke_p90 (90th percentile of edge response)
        features[10] = np.percentile(edges.flatten(), 90) / 255.0

        # 11: flat_hole_ratio
        features[11] = (1 - flat_mask).mean()

        # 12: highfreq_spatial_autocorr
        f = np.fft.fft2(gray.astype(np.float32))
        f_shift = np.fft.fftshift(f)
        mag = np.abs(f_shift)
        h, w = mag.shape
        cy, cx = h // 2, w // 2
        high_freq_mask = np.ones_like(mag, dtype=bool)
        r = 30
        y, x = np.ogrid[:h, :w]
        center_mask = (x - cx)**2 + (y - cy)**2 <= r**2
        high_freq_mask[center_mask] = False
        features[12] = mag[high_freq_mask].mean() / (mag.mean() + 1e-8)

        # 13: patch_vs_global_rank_entropy_gap (simplified)
        hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256))
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        global_entropy = -np.sum(hist * np.log2(hist))
        features[13] = global_entropy / 8.0  # Normalize

        # 14: flat_ratio
        features[14] = flat_mask.mean()

        # 15: flat_ratio_variance_across_tiles
        tile_size = 64
        tile_ratios = []
        for y in range(0, new_h - tile_size, tile_size):
            for x in range(0, new_w - tile_size, tile_size):
                tile = flat_mask[y:y+tile_size, x:x+tile_size]
                tile_ratios.append(tile.mean())
        features[15] = np.var(tile_ratios) if tile_ratios else 0.0

        # 16: patch_vs_global_st_aniso_gap
        patch_aniso_means = []
        for y in range(0, new_h - patch_size, patch_size):
            for x in range(0, new_w - patch_size, patch_size):
                patch = aniso[y:y+patch_size, x:x+patch_size]
                patch_aniso_means.append(patch.mean())
        if patch_aniso_means:
            features[16] = np.std(patch_aniso_means)

        # 17: patch_vs_global_spectrum_slope_gap (simplified)
        features[17] = features[1] * 0.1  # Rough approximation

        return features

    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load DINOv3
    print("Loading DINOv3...")
    from transformers import AutoImageProcessor, AutoModel
    model_path = Path("/home/techne/aicheckers/models/dinov3-vitb16")
    processor = AutoImageProcessor.from_pretrained(str(model_path))
    model = AutoModel.from_pretrained(str(model_path))
    model.to(device)
    model.eval()

    # Get file list (use existing files list for consistency)
    files_path = EMBED_DIR / f"{CAT_NAME}_files.txt"
    if files_path.exists():
        with open(files_path) as f:
            filenames = [l.strip() for l in f if l.strip()]
        files = [IMG_DIR / fn for fn in filenames]
        print(f"Using existing files list: {len(files)} files")
    else:
        files = sorted([
            p for p in IMG_DIR.glob("*")
            if p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.webp']
        ])
        print(f"Found {len(files)} images")

    all_patch_stats = []
    all_cpu_stats = []

    # Process in batches
    num_batches = (len(files) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in tqdm(range(num_batches), desc="Extracting"):
        batch_files = files[batch_idx * BATCH_SIZE : (batch_idx + 1) * BATCH_SIZE]

        # Load images for DINOv3
        batch_images = []
        batch_valid_idx = []
        for i, fp in enumerate(batch_files):
            try:
                img = Image.open(fp).convert("RGB")
                batch_images.append(img)
                batch_valid_idx.append(i)
            except Exception as e:
                print(f"Error loading {fp}: {e}")

        if not batch_images:
            # Add zeros for skipped batch
            all_patch_stats.append(np.zeros((len(batch_files), 34), dtype=np.float32))
            all_cpu_stats.append(np.zeros((len(batch_files), 18), dtype=np.float32))
            continue

        # DINOv3 features
        inputs = processor(images=batch_images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            mid_hidden = outputs.hidden_states[MID_LAYER + 1]
            mid_patches = mid_hidden[:, 5:5+196, :]  # (B, 196, 768)

            # Compute patch_stats_v3
            patch_stats = compute_patch_stats_v3_batch(mid_patches)
            all_patch_stats.append(patch_stats)

        # CPU stats (process actual files, not just valid ones)
        batch_cpu = []
        for fp in batch_files:
            stats = compute_cpu_stats_single(fp)
            if stats is not None:
                batch_cpu.append(stats)
            else:
                batch_cpu.append(np.zeros(18, dtype=np.float32))
        all_cpu_stats.append(np.array(batch_cpu))

        # Clear CUDA cache
        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()

    # Concatenate and save
    all_patch_stats = np.concatenate(all_patch_stats, axis=0)
    all_cpu_stats = np.concatenate(all_cpu_stats, axis=0)

    print(f"patch_stats shape: {all_patch_stats.shape}")
    print(f"cpu_stats shape: {all_cpu_stats.shape}")

    # Save
    patch_path = EMBED_DIR / f"{CAT_NAME}_patch_stats_v3.npy"
    cpu_path = EMBED_DIR / f"{CAT_NAME}_cpu_stats_v2.npy"

    np.save(patch_path, all_patch_stats.astype(np.float32))
    print(f"Saved: {patch_path}")

    np.save(cpu_path, all_cpu_stats.astype(np.float32))
    print(f"Saved: {cpu_path}")

    print("\nExtraction complete!")


if __name__ == "__main__":
    main()
