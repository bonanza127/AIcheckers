#!/usr/bin/env python3
"""CPU Stats v3 並列版"""
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2
from PIL import Image
from scipy import ndimage
from scipy.signal import find_peaks
from tqdm import tqdm

REPO_ROOT = Path("/home/techne/aicheckers")
EMB_DIR = REPO_ROOT / "embeddings"
DATA_DIR = REPO_ROOT / "data/aibooru_new"
CATEGORY = "aibooru_new_ai"

def load_image(path):
    img = Image.open(path).convert("RGB")
    img = np.array(img)
    return img

def extract_features(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    features = np.zeros(20, dtype=np.float32)
    
    # 0: histogram_flatness
    hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
    hist = hist.astype(np.float64) / (hist.sum() + 1e-10)
    entropy = -np.sum(hist * np.log2(hist + 1e-10))
    features[0] = entropy / np.log2(256)
    
    # 1: histogram_modality
    hist64, _ = np.histogram(gray.ravel(), bins=64, range=(0, 256))
    hist_smooth = ndimage.gaussian_filter1d(hist64.astype(np.float64), sigma=2)
    peaks, _ = find_peaks(hist_smooth, height=hist_smooth.max() * 0.05, distance=5)
    features[1] = len(peaks)
    
    # 2: color_palette_entropy
    quantized = (img_rgb // 8).astype(np.uint8)
    colors = quantized.reshape(-1, 3)
    color_ids = colors[:, 0].astype(np.int32) * 1024 + colors[:, 1] * 32 + colors[:, 2]
    unique, counts = np.unique(color_ids, return_counts=True)
    probs = counts.astype(np.float64) / counts.sum()
    features[2] = -np.sum(probs * np.log2(probs + 1e-10))
    
    # 3: luminance_layer_count
    features[3] = len(np.unique(gray // 16))
    
    # 4: edge_sharpness
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(sobelx**2 + sobely**2)
    threshold = np.percentile(gradient, 95)
    high_grad = gradient[gradient > threshold]
    features[4] = high_grad.mean() / (gradient.mean() + 1e-6) if len(high_grad) > 0 and gradient.mean() > 1e-6 else 0
    
    # 5: chroma_spatial_entropy
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    a, b = lab[:, :, 1], lab[:, :, 2]
    chroma = np.sqrt(a.astype(np.float64)**2 + b.astype(np.float64)**2)
    features[5] = np.std(chroma)
    
    # 6: lbp_uniformity (simplified)
    features[6] = np.std(gray) / 128.0
    
    # 7: luminance_skewness
    mean_val = gray.mean()
    std_val = gray.std() + 1e-6
    features[7] = np.mean(((gray - mean_val) / std_val) ** 3)
    
    # 8: frequency_band_ratio_var
    f = np.fft.fft2(gray.astype(np.float64))
    f_shift = np.fft.fftshift(f)
    mag = np.abs(f_shift)
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    low = mag[max(0,cy-h//8):cy+h//8, max(0,cx-w//8):cx+w//8].sum()
    features[8] = low / (mag.sum() + 1e-10)
    
    # 9: value_bimodality
    features[9] = ((gray.mean() - 128) ** 2) / (gray.var() + 1e-6)
    
    # 10: multiscale_variance_ratio
    blur1 = cv2.GaussianBlur(gray, (3, 3), 0)
    blur2 = cv2.GaussianBlur(gray, (7, 7), 0)
    features[10] = np.var(blur1) / (np.var(blur2) + 1e-6)
    
    # 11: gradient_magnitude_entropy
    hist_grad, _ = np.histogram(gradient, bins=64)
    hist_grad = hist_grad.astype(np.float64) / (hist_grad.sum() + 1e-10)
    features[11] = -np.sum(hist_grad * np.log2(hist_grad + 1e-10))
    
    # 12: noise_spectrum_slope
    noise = gray.astype(np.float64) - cv2.GaussianBlur(gray.astype(np.float64), (5, 5), 0)
    features[12] = np.std(noise)
    
    # 13-19: additional features
    corners = cv2.cornerHarris(gray.astype(np.float32), 2, 3, 0.04)
    features[13] = np.percentile(corners, 99)
    
    local_var = cv2.GaussianBlur((gray.astype(np.float64) - cv2.GaussianBlur(gray.astype(np.float64), (15, 15), 0))**2, (15, 15), 0)
    features[14] = np.std(np.sqrt(local_var + 1e-6))
    
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    s_ch = hsv[:, :, 1]
    features[15] = np.mean(np.abs(np.diff(s_ch.astype(np.float32), axis=0))) + np.mean(np.abs(np.diff(s_ch.astype(np.float32), axis=1)))
    
    edges = cv2.Canny(gray, 50, 150)
    features[16] = np.sum(edges > 0) / edges.size
    
    v_ch = hsv[:, :, 2]
    features[17] = np.sum(v_ch > 250) / v_ch.size
    features[18] = np.sum(v_ch < 5) / v_ch.size
    midtones = v_ch[(v_ch > 50) & (v_ch < 200)]
    if len(midtones) > 100:
        m = midtones.mean()
        s = midtones.std() + 1e-6
        features[19] = np.mean(((midtones - m) / s) ** 4) - 3
    
    return features

def process_single(path):
    try:
        img = load_image(path)
        return extract_features(img)
    except:
        return np.zeros(20, dtype=np.float32)

def main():
    files_path = EMB_DIR / f"{CATEGORY}_files.txt"
    with open(files_path) as f:
        files = [line.strip() for line in f if line.strip()]
    
    paths = [DATA_DIR / f for f in files]
    print(f"Processing {len(paths)} files with 16 workers...")
    
    with ThreadPoolExecutor(max_workers=16) as executor:
        results = list(tqdm(executor.map(process_single, paths), total=len(paths)))
    
    arr = np.stack(results)
    np.save(EMB_DIR / f"{CATEGORY}_cpu_stats_v3_20d.npy", arr)
    print(f"Saved: {CATEGORY}_cpu_stats_v3_20d.npy ({arr.shape})")

if __name__ == "__main__":
    main()
