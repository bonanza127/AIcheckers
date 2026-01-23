#!/usr/bin/env python3
"""
Unified CPU v3 extractor.

Profiles:
  - batch13: 13d (original batch v3)
  - nolbp12: 12d (batch v3 minus LBP)
  - tier1: 12d (Tier1 set)

Outputs:
  embeddings/{name}_cpu_stats_v3.npy           (batch13)
  embeddings/{name}_cpu_stats_v3_nolbp.npy     (nolbp12)
  embeddings/{name}_cpu_stats_v3_tier1.npy     (tier1)
"""
import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.signal import find_peaks
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

EMBEDDINGS_DIR = Path("/home/techne/aicheckers/embeddings")
DATA_ROOT = Path("/home/techne/aicheckers/data")
ANIMEDL_ROOT = DATA_ROOT / "animedl2m_dataset_release"

# グローバル変数（マルチプロセス用）
_GLOBAL_EXTRACT_FN = None
_GLOBAL_FEATURES = None


def _init_worker(extract_fn, features):
    """ワーカープロセスの初期化"""
    global _GLOBAL_EXTRACT_FN, _GLOBAL_FEATURES
    _GLOBAL_EXTRACT_FN = extract_fn
    _GLOBAL_FEATURES = features


def _process_one(path_str):
    """モジュールレベルのワーカー関数（pickle可能）"""
    global _GLOBAL_EXTRACT_FN, _GLOBAL_FEATURES
    try:
        img = load_image(path_str)
        feats = _GLOBAL_EXTRACT_FN(img)
        return path_str, feats, False
    except Exception:
        return path_str, None, True


CATEGORY_PATHS = {
    "illustrious_ai": ANIMEDL_ROOT / "civitai_subset/image/Illustrious",
    "pony_ai": ANIMEDL_ROOT / "civitai_subset/image/Pony",
    "sdxl10_ai": ANIMEDL_ROOT / "civitai_subset/image/SDXL 1.0",
    "sd15_ai": ANIMEDL_ROOT / "civitai_subset/image/SD 1.5",
    "other_ai": ANIMEDL_ROOT / "civitai_subset/image/Other",
    "flux1d_ai": ANIMEDL_ROOT / "civitai_subset/image/Flux.1 D",
    "novelai_ai": DATA_ROOT / "novelai",
    "pixai_ai": DATA_ROOT / "pixai",
    "novelai_combined_ai": DATA_ROOT / "novelai_combined",
    "novelai_artist_tagged_ai": DATA_ROOT / "novelai_artist_tagged",
    "danbooru_real": ANIMEDL_ROOT / "real_images/images",
    "hard_negatives_ai": DATA_ROOT / "hard_negatives",
}

CHECKPOINT_INTERVAL = 1000


def load_image(path, target_size=512):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = target_size / max(h, w)
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))
    if (nw, nh) != img.size:
        img = img.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGB", (target_size, target_size), (128, 128, 128))
    x0 = (target_size - nw) // 2
    y0 = (target_size - nh) // 2
    canvas.paste(img, (x0, y0))
    return np.array(canvas)


# ===== batch13 / nolbp12 =====
def histogram_flatness(gray):
    hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
    hist = hist.astype(np.float64)
    hist = hist / (hist.sum() + 1e-10)
    entropy = -np.sum(hist * np.log2(hist + 1e-10))
    return float(entropy / np.log2(256))


def histogram_modality(gray):
    hist, _ = np.histogram(gray.ravel(), bins=64, range=(0, 256))
    hist_smooth = ndimage.gaussian_filter1d(hist.astype(np.float64), sigma=2)
    peaks, _ = find_peaks(hist_smooth, height=hist_smooth.max() * 0.05, distance=5)
    return float(len(peaks))


def color_palette_entropy(img_rgb):
    # 最適化版: np.bincount使用 (73倍高速)
    quantized = (img_rgb // 8).astype(np.uint32)
    color_idx = quantized[:,:,0] * 1024 + quantized[:,:,1] * 32 + quantized[:,:,2]
    counts = np.bincount(color_idx.ravel(), minlength=32768)
    counts = counts[counts > 0]
    probs = counts.astype(np.float64) / counts.sum()
    entropy = -np.sum(probs * np.log2(probs))
    return float(entropy)


def luminance_layer_count(gray):
    quantized = (gray // 16).astype(np.uint8)
    return float(len(np.unique(quantized)))


def edge_sharpness(gray):
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(sobelx**2 + sobely**2)
    threshold = np.percentile(gradient, 95)
    high_grad = gradient[gradient > threshold]
    if len(high_grad) == 0 or gradient.mean() < 1e-6:
        return 0.0
    return float(high_grad.mean() / (gradient.mean() + 1e-6))


def chroma_spatial_entropy(img_rgb):
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    a, b = lab[:, :, 1], lab[:, :, 2]
    chroma = np.sqrt(a.astype(np.float64)**2 + b.astype(np.float64)**2)
    h, w = chroma.shape
    tiles = 16
    tile_h, tile_w = h // tiles, w // tiles
    if tile_h == 0 or tile_w == 0:
        return 0.0
    means = []
    for i in range(tiles):
        for j in range(tiles):
            tile = chroma[i*tile_h:(i+1)*tile_h, j*tile_w:(j+1)*tile_w]
            means.append(tile.mean())
    means = np.array(means)
    hist, _ = np.histogram(means, bins=32)
    hist = hist.astype(np.float64) / (hist.sum() + 1e-10)
    entropy = -np.sum(hist * np.log2(hist + 1e-10))
    return float(entropy)


def lbp_uniformity(gray, radius=1, n_points=8):
    h, w = gray.shape
    lbp = np.zeros((h, w), dtype=np.uint8)
    for i in range(n_points):
        angle = 2 * np.pi * i / n_points
        dy = -radius * np.cos(angle)
        dx = radius * np.sin(angle)
        y = np.arange(h).reshape(-1, 1) + dy
        x = np.arange(w).reshape(1, -1) + dx
        y = np.clip(y, 0, h - 1).astype(int)
        x = np.clip(x, 0, w - 1).astype(int)
        neighbor = gray[y, x]
        lbp += ((neighbor >= gray).astype(np.uint8) << i)

    def count_transitions(val):
        bits = [(val >> i) & 1 for i in range(n_points)]
        bits.append(bits[0])
        return sum(1 for i in range(n_points) if bits[i] != bits[i+1])

    uniform_count = 0
    for val in lbp.ravel():
        if count_transitions(val) <= 2:
            uniform_count += 1
    return float(uniform_count / lbp.size)


def luminance_skewness(gray):
    vals = gray.astype(np.float64).ravel()
    mean = vals.mean()
    std = vals.std()
    if std < 1e-6:
        return 0.0
    skewness = ((vals - mean) ** 3).mean() / (std ** 3 + 1e-10)
    return float(skewness)


def frequency_band_ratio_var(gray):
    f = np.fft.fft2(gray.astype(np.float64))
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    max_r = min(cy, cx)
    bands = [0, max_r // 4, max_r // 2, 3 * max_r // 4, max_r]
    band_powers = []
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    for i in range(len(bands) - 1):
        mask = (r >= bands[i]) & (r < bands[i + 1])
        power = mag[mask].mean() if mask.sum() > 0 else 0.0
        band_powers.append(power)
    total = sum(band_powers) + 1e-10
    ratios = [p / total for p in band_powers]
    return float(np.var(ratios))


def value_bimodality(gray):
    vals = gray.astype(np.float64).ravel()
    n = len(vals)
    if n < 3:
        return 0.0
    mean = vals.mean()
    std = vals.std()
    if std < 1e-6:
        return 0.0
    skewness = ((vals - mean) ** 3).mean() / (std ** 3 + 1e-10)
    kurtosis = ((vals - mean) ** 4).mean() / (std ** 4 + 1e-10) - 3
    bc = (skewness ** 2 + 1) / (kurtosis + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)) + 1e-10)
    return float(bc)


def multiscale_variance_ratio(gray):
    variances = []
    for scale in [1, 2, 4, 8]:
        if gray.shape[0] // scale < 8 or gray.shape[1] // scale < 8:
            break
        resized = cv2.resize(gray, (gray.shape[1] // scale, gray.shape[0] // scale))
        variances.append(resized.var())
    if len(variances) < 2:
        return 0.0
    ratios = [variances[i] / (variances[i + 1] + 1e-10) for i in range(len(variances) - 1)]
    return float(np.mean(ratios))


def gradient_magnitude_entropy(gray):
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(sobelx**2 + sobely**2)
    if gradient.max() < 1e-6:
        return 0.0
    gradient_norm = (gradient / (gradient.max() + 1e-10) * 255).astype(np.uint8)
    hist, _ = np.histogram(gradient_norm.ravel(), bins=64, range=(0, 256))
    hist = hist.astype(np.float64) / (hist.sum() + 1e-10)
    entropy = -np.sum(hist * np.log2(hist + 1e-10))
    return float(entropy)


def noise_spectrum_slope(gray):
    low = cv2.GaussianBlur(gray.astype(np.float64), (5, 5), 1.5)
    noise = gray.astype(np.float64) - low
    f = np.fft.fft2(noise)
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift) + 1e-10
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(np.int32)
    max_r = r.max()
    radial_mean = np.bincount(r.ravel(), mag.ravel(), minlength=max_r + 1)
    radial_cnt = np.bincount(r.ravel(), minlength=max_r + 1)
    radial_mean = radial_mean / (radial_cnt + 1e-6)
    start_r = max_r // 2
    xs = np.log(np.arange(start_r, max_r) + 1)
    ys = np.log(radial_mean[start_r:max_r] + 1e-10)
    if len(xs) < 2:
        return 0.0
    slope, _ = np.polyfit(xs, ys, 1)
    return float(slope)


def extract_batch13(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    feats = [
        histogram_flatness(gray),
        histogram_modality(gray),
        color_palette_entropy(img_rgb),
        luminance_layer_count(gray),
        edge_sharpness(gray),
        chroma_spatial_entropy(img_rgb),
        lbp_uniformity(gray),
        luminance_skewness(gray),
        frequency_band_ratio_var(gray),
        value_bimodality(gray),
        multiscale_variance_ratio(gray),
        gradient_magnitude_entropy(gray),
        noise_spectrum_slope(gray),
    ]
    return np.array(feats, dtype=np.float32)


def extract_nolbp12(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    feats = [
        histogram_flatness(gray),
        histogram_modality(gray),
        color_palette_entropy(img_rgb),
        luminance_layer_count(gray),
        edge_sharpness(gray),
        chroma_spatial_entropy(img_rgb),
        luminance_skewness(gray),
        frequency_band_ratio_var(gray),
        value_bimodality(gray),
        multiscale_variance_ratio(gray),
        gradient_magnitude_entropy(gray),
        noise_spectrum_slope(gray),
    ]
    return np.array(feats, dtype=np.float32)


# ===== tier1 =====
def _safe_entropy(probs):
    probs = probs / (probs.sum() + 1e-10)
    return float(-np.sum(probs * np.log2(probs + 1e-10)))


def quantization_step_count(gray):
    vals = np.unique(gray)
    if len(vals) < 2:
        return 0.0
    diffs = np.diff(vals).astype(np.float64)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return 0.0
    step = np.median(diffs)
    if step <= 0:
        return 0.0
    return float(min(255.0 / step, 255.0))


def piecewise_constant_ratio(gray, grad_thresh=2.0):
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(sobelx**2 + sobely**2)
    return float((gradient <= grad_thresh).mean())


def noise_floor_variance(gray):
    low = cv2.GaussianBlur(gray.astype(np.float64), (5, 5), 1.5)
    residual = gray.astype(np.float64) - low
    abs_res = np.abs(residual)
    thr = np.percentile(abs_res, 25)
    mask = abs_res <= thr
    if mask.sum() == 0:
        return 0.0
    return float(residual[mask].var())


def highlight_clipping_ratio(gray, clip_val=250):
    return float((gray >= clip_val).mean())


def _band_powers(gray, bands=4):
    f = np.fft.fft2(gray.astype(np.float64))
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    max_r = min(cy, cx)
    band_edges = np.linspace(0, max_r, bands + 1, dtype=np.int32)
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    powers = []
    for i in range(bands):
        mask = (r >= band_edges[i]) & (r < band_edges[i + 1])
        power = mag[mask].mean() if mask.sum() > 0 else 0.0
        powers.append(power)
    return np.array(powers, dtype=np.float64)


def band_entropy(gray):
    powers = _band_powers(gray, bands=4)
    return _safe_entropy(powers)


def band_energy_gini(gray):
    powers = _band_powers(gray, bands=4)
    if np.all(powers == 0):
        return 0.0
    sorted_p = np.sort(powers)
    n = len(sorted_p)
    cum = np.cumsum(sorted_p)
    gini = (n + 1 - 2 * np.sum(cum) / (cum[-1] + 1e-10)) / n
    return float(gini)


def _radial_spectrum_slope(gray):
    f = np.fft.fft2(gray.astype(np.float64))
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift) + 1e-10
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(np.int32)
    max_r = r.max()
    radial_mean = np.bincount(r.ravel(), mag.ravel(), minlength=max_r + 1)
    radial_cnt = np.bincount(r.ravel(), minlength=max_r + 1)
    radial_mean = radial_mean / (radial_cnt + 1e-6)
    start_r = max_r // 4
    xs = np.log(np.arange(start_r, max_r) + 1)
    ys = np.log(radial_mean[start_r:max_r] + 1e-10)
    if len(xs) < 2:
        return 0.0
    slope, _ = np.polyfit(xs, ys, 1)
    return float(slope)


def radial_spectrum_slope_patch_gap(gray, tiles=4):
    h, w = gray.shape
    if h < tiles or w < tiles:
        return 0.0
    global_slope = _radial_spectrum_slope(gray)
    tile_h = h // tiles
    tile_w = w // tiles
    slopes = []
    for i in range(tiles):
        for j in range(tiles):
            tile = gray[i * tile_h:(i + 1) * tile_h, j * tile_w:(j + 1) * tile_w]
            slopes.append(_radial_spectrum_slope(tile))
    if not slopes:
        return 0.0
    return float(global_slope - np.mean(slopes))


def color_banding_score(img_rgb):
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    a = lab[:, :, 1].astype(np.int16)
    b = lab[:, :, 2].astype(np.int16)
    chroma = np.sqrt(a.astype(np.float64) ** 2 + b.astype(np.float64) ** 2)
    quant = np.clip((chroma / (chroma.max() + 1e-10) * 31).astype(np.uint8), 0, 31)
    unique = np.unique(quant).size
    return float(1.0 - unique / 32.0)


def compression_artifact_pattern(gray):
    gray = gray.astype(np.float64)
    h, w = gray.shape
    if h < 8 or w < 8:
        return 0.0
    v_idx = np.arange(8, w, 8)
    h_idx = np.arange(8, h, 8)
    vert = np.abs(gray[:, v_idx] - gray[:, v_idx - 1]).mean() if v_idx.size else 0.0
    horiz = np.abs(gray[h_idx, :] - gray[h_idx - 1, :]).mean() if h_idx.size else 0.0
    overall = np.mean(np.abs(np.diff(gray, axis=1))) + np.mean(np.abs(np.diff(gray, axis=0)))
    return float((vert + horiz) / (overall + 1e-6))


def edge_continuity_ratio(gray, min_size=10):
    edges = cv2.Canny(gray, 100, 200)
    if edges.sum() == 0:
        return 0.0
    num_labels, labels = cv2.connectedComponents((edges > 0).astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return 0.0
    counts = np.bincount(labels.ravel())
    large = counts >= min_size
    large[0] = False
    return float(counts[large].sum() / (edges.sum() + 1e-10))


def edge_endpoint_density(gray):
    edges = (cv2.Canny(gray, 100, 200) > 0).astype(np.uint8)
    if edges.sum() == 0:
        return 0.0
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    neighbors = cv2.filter2D(edges, -1, kernel)
    endpoints = (edges == 1) & (neighbors == 1)
    return float(endpoints.sum() / (edges.sum() + 1e-10))


# ===== tier2 (new validated features) =====
def color_transition_histogram(img_rgb):
    """色遷移パターンの分布 - AI生成は特定パターンに偏る (d=+0.461)"""
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l_channel = lab[:, :, 0].astype(np.float64)
    diff_h = np.abs(np.diff(l_channel, axis=1)).ravel()
    diff_v = np.abs(np.diff(l_channel, axis=0)).ravel()
    all_diff = np.concatenate([diff_h, diff_v])
    hist, _ = np.histogram(all_diff, bins=32, range=(0, 64))
    hist = hist.astype(np.float64) / (hist.sum() + 1e-10)
    entropy = -np.sum(hist * np.log2(hist + 1e-10))
    return float(entropy / np.log2(32))


def anti_aliasing_sharpness(gray):
    """アンチエイリアシング境界の鮮明度 - AI生成のAAは特徴的 (d=+0.457)"""
    edges = cv2.Canny(gray, 100, 200)
    if edges.sum() == 0:
        return 0.0
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(sobelx**2 + sobely**2)
    edge_gradient = gradient[edges > 0]
    if len(edge_gradient) < 10:
        return 0.0
    return float(np.std(edge_gradient) / (np.mean(edge_gradient) + 1e-6))


def extract_tier2(img_rgb):
    """Tier2: 検証済み新特徴量 (2d)"""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    feats = [
        color_transition_histogram(img_rgb),
        anti_aliasing_sharpness(gray),
    ]
    return np.array(feats, dtype=np.float32)


# ===== tier0 (NovelAI分析発見・最高効果量) =====
def luminance_mean(img_rgb):
    """輝度平均 - AI画像は暗め (d=-0.843)"""
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    return float(lab[:, :, 0].mean())


def color_hue_entropy(img_rgb):
    """色相エントロピー - AIは色相の多様性が高い (d=+0.834)"""
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h = hsv[:, :, 0].ravel()
    hist, _ = np.histogram(h, bins=36, range=(0, 180))
    hist = hist.astype(np.float64) / (hist.sum() + 1e-10)
    entropy = -np.sum(hist * np.log2(hist + 1e-10))
    return float(entropy)


def luminance_peak_count(img_rgb):
    """輝度分布のピーク数 - AIは多峰性 (d=+0.726)"""
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l = lab[:, :, 0].ravel()
    hist, _ = np.histogram(l, bins=32, range=(0, 256))
    hist_smooth = ndimage.gaussian_filter1d(hist.astype(np.float64), sigma=1)
    peaks, _ = find_peaks(hist_smooth, height=hist_smooth.max() * 0.02)
    return float(len(peaks))


def saturation_mean(img_rgb):
    """彩度平均 - AIは彩度が高い (d=+0.688)"""
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    return float(hsv[:, :, 1].mean())


def curvature_var(img_rgb):
    """Flat領域境界の曲率分散 - AIは滑らかな境界 (d=+0.988)"""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    # Flat mask
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    threshold = np.percentile(grad_mag, 15)
    flat_mask = (grad_mag <= threshold).astype(np.uint8)

    # Contour curvature
    contours, _ = cv2.findContours(flat_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    all_curvatures = []
    for contour in contours:
        if len(contour) < 10:
            continue
        area = cv2.contourArea(contour)
        if area < 100:
            continue
        contour = contour.squeeze()
        if len(contour.shape) == 1:
            continue
        for i in range(len(contour)):
            p1 = contour[i - 2]
            p2 = contour[i - 1]
            p3 = contour[i]
            v1 = p2 - p1
            v2 = p3 - p2
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            dot = v1[0] * v2[0] + v1[1] * v2[1]
            angle = np.arctan2(cross, dot)
            all_curvatures.append(abs(angle))
    return float(np.var(all_curvatures)) if all_curvatures else 0.0


def flat_cluster_size(img_rgb):
    """Flat領域の最大クラスタサイズ - AIは大きなFlat領域 (d=-0.976)"""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    # Flat mask
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    threshold = np.percentile(grad_mag, 15)
    flat_mask = (grad_mag <= threshold).astype(np.uint8)

    # Connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(flat_mask, connectivity=4)
    img_size = gray.shape[0] * gray.shape[1]
    if num_labels <= 1:
        return 0.0
    cluster_sizes = stats[1:, cv2.CC_STAT_AREA].astype(np.float32) / img_size
    return float(cluster_sizes.max()) if cluster_sizes.size > 0 else 0.0


def extract_tier0(img_rgb):
    """Tier0: AI共通・最高効果量 (7d)"""
    feats = [
        color_palette_entropy(img_rgb),  # d=+1.128 ◎ 最強
        luminance_mean(img_rgb),          # d=-0.843 ◎
        color_hue_entropy(img_rgb),       # d=+0.796 ○
        saturation_mean(img_rgb),         # d=+0.760 ○
        luminance_peak_count(img_rgb),    # d=+0.501 ○
        curvature_var(img_rgb),           # d=+0.988 ◎ NEW
        flat_cluster_size(img_rgb),       # d=-0.976 ◎ NEW
    ]
    return np.array(feats, dtype=np.float32)


def extract_tier1(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    feats = [
        quantization_step_count(gray),
        piecewise_constant_ratio(gray),
        noise_floor_variance(gray),
        highlight_clipping_ratio(gray),
        band_entropy(gray),
        band_energy_gini(gray),
        radial_spectrum_slope_patch_gap(gray),
        color_banding_score(img_rgb),
        color_palette_entropy(img_rgb),
        compression_artifact_pattern(gray),
        edge_continuity_ratio(gray),
        edge_endpoint_density(gray),
    ]
    return np.array(feats, dtype=np.float32)


def extract_unified(img_rgb):
    """統合抽出: 27d (lbp, curvature_var, flat_cluster_size除外)"""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    feats = [
        # batch13 (lbp除外) - 12d
        histogram_flatness(gray),
        histogram_modality(gray),
        color_palette_entropy(img_rgb),
        luminance_layer_count(gray),
        edge_sharpness(gray),
        chroma_spatial_entropy(img_rgb),
        luminance_skewness(gray),
        frequency_band_ratio_var(gray),
        value_bimodality(gray),
        multiscale_variance_ratio(gray),
        gradient_magnitude_entropy(gray),
        noise_spectrum_slope(gray),
        # tier0追加分 (curvature_var, flat_cluster_size除外) - 4d
        luminance_mean(img_rgb),
        color_hue_entropy(img_rgb),
        saturation_mean(img_rgb),
        luminance_peak_count(img_rgb),
        # tier1追加分 (color_palette_entropy重複除外) - 11d
        quantization_step_count(gray),
        piecewise_constant_ratio(gray),
        noise_floor_variance(gray),
        highlight_clipping_ratio(gray),
        band_entropy(gray),
        band_energy_gini(gray),
        radial_spectrum_slope_patch_gap(gray),
        color_banding_score(img_rgb),
        compression_artifact_pattern(gray),
        edge_continuity_ratio(gray),
        edge_endpoint_density(gray),
    ]
    return np.array(feats, dtype=np.float32)


PROFILES = {
    "unified": {
        "features": [
            "histogram_flatness", "histogram_modality", "color_palette_entropy",
            "luminance_layer_count", "edge_sharpness", "chroma_spatial_entropy",
            "luminance_skewness", "frequency_band_ratio_var", "value_bimodality",
            "multiscale_variance_ratio", "gradient_magnitude_entropy", "noise_spectrum_slope",
            "luminance_mean", "color_hue_entropy", "saturation_mean", "luminance_peak_count",
            "quantization_step_count", "piecewise_constant_ratio", "noise_floor_variance",
            "highlight_clipping_ratio", "band_entropy", "band_energy_gini",
            "radial_spectrum_slope_patch_gap", "color_banding_score",
            "compression_artifact_pattern", "edge_continuity_ratio", "edge_endpoint_density",
        ],
        "extract": extract_unified,
        "suffix": "cpu_stats_v3_unified.npy",
    },
    "batch13": {
        "features": [
            "histogram_flatness", "histogram_modality", "color_palette_entropy",
            "luminance_layer_count", "edge_sharpness", "chroma_spatial_entropy",
            "lbp_uniformity", "luminance_skewness", "frequency_band_ratio_var",
            "value_bimodality", "multiscale_variance_ratio", "gradient_magnitude_entropy",
            "noise_spectrum_slope",
        ],
        "extract": extract_batch13,
        "suffix": "cpu_stats_v3.npy",
    },
    "nolbp12": {
        "features": [
            "histogram_flatness", "histogram_modality", "color_palette_entropy",
            "luminance_layer_count", "edge_sharpness", "chroma_spatial_entropy",
            "luminance_skewness", "frequency_band_ratio_var", "value_bimodality",
            "multiscale_variance_ratio", "gradient_magnitude_entropy",
            "noise_spectrum_slope",
        ],
        "extract": extract_nolbp12,
        "suffix": "cpu_stats_v3_nolbp.npy",
    },
    "tier1": {
        "features": [
            "quantization_step_count", "piecewise_constant_ratio", "noise_floor_variance",
            "highlight_clipping_ratio", "band_entropy", "band_energy_gini",
            "radial_spectrum_slope_patch_gap", "color_banding_score",
            "color_palette_entropy", "compression_artifact_pattern",
            "edge_continuity_ratio", "edge_endpoint_density",
        ],
        "extract": extract_tier1,
        "suffix": "cpu_stats_v3_tier1.npy",
    },
    "tier2": {
        "features": [
            "color_transition_histogram", "anti_aliasing_sharpness",
        ],
        "extract": extract_tier2,
        "suffix": "cpu_stats_v3_tier2.npy",
    },
    "tier0": {
        "features": [
            "color_palette_entropy", "luminance_mean", "color_hue_entropy",
            "saturation_mean", "luminance_peak_count",
            "curvature_var", "flat_cluster_size",  # NEW: d=+0.988, d=-0.976
        ],
        "extract": extract_tier0,
        "suffix": "cpu_stats_v3_tier0.npy",
    },
}


def _save_checkpoint(path, stats_list, files_list, feat_len):
    stats_arr = (
        np.concatenate(stats_list, axis=0).astype(np.float32)
        if stats_list
        else np.zeros((0, feat_len), dtype=np.float32)
    )
    np.savez_compressed(path, stats=stats_arr, files=np.array(files_list, dtype=object))


def _load_checkpoint(path):
    if not path.exists():
        return None, []
    data = np.load(path, allow_pickle=True)
    return data["stats"], data["files"].tolist()


def extract_category(name, img_dir, profile, limit=0, workers=1):
    cfg = PROFILES[profile]
    img_dir = Path(img_dir)
    if not img_dir.exists():
        print(f"[SKIP] {name}: missing dir {img_dir}")
        return

    out_stats = EMBEDDINGS_DIR / f"{name}_{cfg['suffix']}"
    out_files = EMBEDDINGS_DIR / f"{name}_{cfg['suffix'].replace('.npy', '_files.txt')}"
    out_meta = EMBEDDINGS_DIR / f"{name}_{cfg['suffix'].replace('.npy', '_meta.json')}"
    checkpoint_file = EMBEDDINGS_DIR / f"{name}_{cfg['suffix'].replace('.npy', '_ckpt.npz')}"

    files_list_path = EMBEDDINGS_DIR / f"{name}_files.txt"
    paths = []
    if files_list_path.exists():
        raw_lines = [line.strip() for line in files_list_path.read_text().splitlines() if line.strip()]
        for line in raw_lines:
            p = Path(line)
            if not p.is_absolute():
                p = img_dir / line
            paths.append(p)
        print(f"[INFO] {name}: using existing files list ({len(paths)} files)")
    else:
        exts = {".jpg", ".jpeg", ".png", ".webp"}
        paths = sorted([p for p in img_dir.rglob("*") if p.suffix.lower() in exts])
        print(f"[INFO] {name}: scanned dir ({len(paths)} files)")

    if limit > 0:
        paths = paths[:limit]

    stats_list = []
    files_list = []
    start_index = 0

    if checkpoint_file.exists():
        ckpt_stats, ckpt_files = _load_checkpoint(checkpoint_file)
        if ckpt_files:
            if len(ckpt_files) <= len(paths) and all(str(paths[i]) == ckpt_files[i] for i in range(len(ckpt_files))):
                stats_list.append(ckpt_stats)
                files_list.extend(ckpt_files)
                start_index = len(ckpt_files)
                print(f"[RESUME] {name}: {start_index} samples from checkpoint")

    zero_feats = np.zeros(len(cfg["features"]), dtype=np.float32)
    error_count = 0
    remaining = [str(p) for p in paths[start_index:]]

    if workers > 1:
        with mp.Pool(
            processes=workers,
            initializer=_init_worker,
            initargs=(cfg["extract"], cfg["features"]),
        ) as pool:
            for path_str, feats, failed in tqdm(
                pool.imap(_process_one, remaining, chunksize=16),
                total=len(remaining),
                desc=f"{name}:{profile}",
            ):
                files_list.append(path_str)
                if failed or feats is None:
                    stats_list.append(zero_feats[None, :])
                    error_count += 1
                else:
                    stats_list.append(feats[None, :])
                if len(files_list) % CHECKPOINT_INTERVAL == 0:
                    _save_checkpoint(checkpoint_file, stats_list, files_list, len(cfg["features"]))
    else:
        for p in tqdm(remaining, desc=f"{name}:{profile}"):
            try:
                img = load_image(p)
                feats = cfg["extract"](img)
                stats_list.append(feats[None, :])
                files_list.append(str(p))
            except Exception:
                stats_list.append(zero_feats[None, :])
                files_list.append(str(p))
                error_count += 1
            if len(files_list) % CHECKPOINT_INTERVAL == 0:
                _save_checkpoint(checkpoint_file, stats_list, files_list, len(cfg["features"]))

    if not stats_list:
        print(f"[WARN] {name}: no samples")
        return

    stats_arr = np.concatenate(stats_list, axis=0).astype(np.float32)
    np.save(out_stats, stats_arr)
    out_files.write_text("\n".join(files_list) + "\n")
    out_meta.write_text(json.dumps({
        "features": cfg["features"],
        "samples": len(stats_arr),
        "dtype": "float32",
        "profile": profile,
    }, indent=2))

    if error_count:
        print(f"[WARN] {name}: {error_count} failures (zero-filled)")
    if checkpoint_file.exists():
        checkpoint_file.unlink()
    print(f"[DONE] {name}:{profile} {stats_arr.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--profile", type=str, default="batch13", choices=list(PROFILES.keys()) + ["all"])
    args = parser.parse_args()

    profiles = list(PROFILES.keys()) if args.profile == "all" else [args.profile]

    if args.category:
        if args.category not in CATEGORY_PATHS:
            raise SystemExit(f"Unknown category: {args.category}")
        for prof in profiles:
            extract_category(args.category, CATEGORY_PATHS[args.category], prof, limit=args.limit, workers=args.workers)
        return

    if args.all:
        for name, path in CATEGORY_PATHS.items():
            for prof in profiles:
                extract_category(name, path, prof, limit=args.limit, workers=args.workers)
        return

    print("Use --category or --all")


if __name__ == "__main__":
    main()
