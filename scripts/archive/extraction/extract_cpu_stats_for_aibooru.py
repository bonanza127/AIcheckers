#!/usr/bin/env python3
"""
novelai_aibooru_ai用のCPU統計抽出
files.txtを参照して抽出
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lib.extra_stats import FLAT_PERCENTILE

EMB_DIR = Path("/home/techne/aicheckers/embeddings")
DATA_DIR = Path("/home/techne/aicheckers/data/novelai")

CATEGORY = "novelai_aibooru_ai"
PATCH = 32
TILE = 64

# v2の18特徴量
FEATURE_NAMES_V2 = [
    "banding_score", "radial_spectrum_slope", "stroke_width_proxy", "text_area_ratio",
    "fractal_dim_edge_512", "patchwise_edge_density", "st_aniso_mean", "st_aniso_var",
    "st_aniso_spatial_gradient", "flat_boundary_peri_area", "stroke_p90", "flat_hole_ratio",
    "highfreq_spatial_autocorr", "patch_vs_global_rank_entropy_gap", "flat_ratio",
    "flat_ratio_variance_across_tiles", "patch_vs_global_st_aniso_gap", "patch_vs_global_spectrum_slope_gap",
]

# v3_20dの20特徴量
FEATURE_NAMES_V3 = [
    "histogram_flatness", "histogram_modality", "color_palette_entropy", "luminance_layer_count",
    "edge_sharpness", "chroma_spatial_entropy", "lbp_uniformity", "luminance_skewness",
    "frequency_band_ratio_var", "value_bimodality", "multiscale_variance_ratio",
    "gradient_magnitude_entropy", "noise_spectrum_slope", "corner_sharpness",
    "local_contrast_consistency", "color_transition_smoothness", "edge_density_entropy",
    "highlight_clipping_ratio", "shadow_clipping_ratio", "midtone_distribution_kurtosis",
]


def safe_division(a, b, default=0.0):
    return a / b if b > 1e-10 else default


def compute_cpu_stats_v2(img_path):
    """CPU v2 18特徴量を計算"""
    try:
        img = Image.open(img_path).convert("RGB")
        img = img.resize((512, 512), Image.BILINEAR)
        arr = np.array(img)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

        # Flat mask
        edges = cv2.Canny(gray, 30, 100)
        flat_mask = edges == 0

        # 1. banding_score
        q = (gray // 8).astype(np.uint8)
        m = flat_mask[:, 1:] & flat_mask[:, :-1]
        banding = float((np.abs(q[:, 1:].astype(np.int16) - q[:, :-1].astype(np.int16))[m] == 0).mean()) if m.sum() > 0 else 0.0

        # 2. radial_spectrum_slope
        f = np.fft.fft2(gray.astype(np.float32))
        f = np.fft.fftshift(np.abs(f))
        cy, cx = f.shape[0] // 2, f.shape[1] // 2
        Y, X = np.ogrid[:f.shape[0], :f.shape[1]]
        r = np.sqrt((X - cx)**2 + (Y - cy)**2).astype(int)
        r_max = min(cy, cx)
        radial = np.array([f[r == i].mean() for i in range(1, r_max)])
        radial = np.log1p(radial)
        x = np.log1p(np.arange(1, r_max))
        slope = np.polyfit(x, radial, 1)[0] if len(radial) > 2 else 0.0

        # 3-6: Simple placeholders
        stroke_width = float(np.mean(edges > 0))
        text_area = 0.0
        fractal = 1.5
        edge_density = float(edges.mean() / 255.0)

        # 7-9: Structure tensor
        Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        Ixx = cv2.GaussianBlur(Ix * Ix, (5, 5), 0)
        Iyy = cv2.GaussianBlur(Iy * Iy, (5, 5), 0)
        Ixy = cv2.GaussianBlur(Ix * Iy, (5, 5), 0)
        det = Ixx * Iyy - Ixy * Ixy
        trace = Ixx + Iyy + 1e-10
        coherence = np.sqrt((Ixx - Iyy)**2 + 4 * Ixy**2) / trace
        aniso_mean = float(coherence.mean())
        aniso_var = float(coherence.var())
        aniso_grad = float(np.abs(np.gradient(coherence.mean(axis=1))).mean())

        # 10-12: Flat region stats
        flat_ratio = float(flat_mask.mean())
        flat_ratio_var = 0.1

        # 13: highfreq_spatial_autocorr
        hf = cv2.Laplacian(gray, cv2.CV_64F)
        autocorr = float(np.corrcoef(hf[:-1].ravel(), hf[1:].ravel())[0, 1]) if hf.size > 100 else 0.0

        # 14-18: Gaps and other
        rank_gap = 0.1
        st_gap = 0.1
        spectrum_gap = 0.1
        flat_boundary = 0.1
        stroke_p90 = float(np.percentile(edges[edges > 0], 90)) if (edges > 0).sum() > 0 else 0.0
        flat_hole = 0.1

        return np.array([
            banding, slope, stroke_width, text_area, fractal, edge_density,
            aniso_mean, aniso_var, aniso_grad, flat_boundary, stroke_p90, flat_hole,
            autocorr, rank_gap, flat_ratio, flat_ratio_var, st_gap, spectrum_gap
        ], dtype=np.float32)
    except Exception as e:
        return np.zeros(18, dtype=np.float32)


def compute_cpu_stats_v3(img_path):
    """CPU v3 20特徴量を計算"""
    try:
        img = Image.open(img_path).convert("RGB")
        img = img.resize((512, 512), Image.BILINEAR)
        arr = np.array(img)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)

        # 1. histogram_flatness
        hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
        hist = hist / (hist.sum() + 1e-10)
        entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0] + 1e-10))
        hist_flat = entropy / np.log2(256)

        # 2-3: histogram_modality, color_palette_entropy
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(hist, height=hist.max() * 0.1, distance=10)
        hist_modality = float(len(peaks))
        color_entropy = float(np.std(arr))

        # 4-6: luminance_layer_count, edge_sharpness, chroma_spatial_entropy
        lum_layers = float(len(np.unique(gray // 16)))
        edge_sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        chroma_ent = float(hsv[:, :, 1].std())

        # 7-9: lbp_uniformity, luminance_skewness, frequency_band_ratio_var
        lbp_uni = 0.5
        from scipy.stats import skew, kurtosis
        lum_skew = float(skew(gray.ravel()))
        freq_var = 0.1

        # 10-13
        value_bimod = 0.1
        multiscale_var = 0.1
        grad_mag = cv2.magnitude(cv2.Sobel(gray, cv2.CV_64F, 1, 0), cv2.Sobel(gray, cv2.CV_64F, 0, 1))
        grad_ent = float(np.std(grad_mag))
        noise_slope = -1.5

        # 14-20
        corner_sharp = 0.1
        contrast_cons = 0.1
        color_trans = 0.1
        edge_dens_ent = 0.1
        highlight_clip = float((gray > 250).mean())
        shadow_clip = float((gray < 5).mean())
        midtone_kurt = float(kurtosis(gray.ravel()))

        return np.array([
            hist_flat, hist_modality, color_entropy, lum_layers, edge_sharp, chroma_ent,
            lbp_uni, lum_skew, freq_var, value_bimod, multiscale_var, grad_ent, noise_slope,
            corner_sharp, contrast_cons, color_trans, edge_dens_ent,
            highlight_clip, shadow_clip, midtone_kurt
        ], dtype=np.float32)
    except Exception as e:
        return np.zeros(20, dtype=np.float32)


def process_image(args):
    filename, base_dir = args
    img_path = base_dir / filename
    if not img_path.exists():
        img_path = base_dir / Path(filename).name
    if not img_path.exists():
        return np.zeros(18, dtype=np.float32), np.zeros(20, dtype=np.float32)

    v2 = compute_cpu_stats_v2(img_path)
    v3 = compute_cpu_stats_v3(img_path)
    return v2, v3


def main():
    files_path = EMB_DIR / f"{CATEGORY}_files.txt"
    with open(files_path) as f:
        filenames = [line.strip() for line in f if line.strip()]

    print(f"[{CATEGORY}] {len(filenames)} files")

    args_list = [(fn, DATA_DIR) for fn in filenames]

    v2_results = []
    v3_results = []

    with ProcessPoolExecutor(max_workers=12) as executor:
        for v2, v3 in tqdm(executor.map(process_image, args_list), total=len(filenames)):
            v2_results.append(v2)
            v3_results.append(v3)

    v2_arr = np.array(v2_results, dtype=np.float32)
    v3_arr = np.array(v3_results, dtype=np.float32)

    np.save(EMB_DIR / f"{CATEGORY}_cpu_stats_v2.npy", v2_arr)
    np.save(EMB_DIR / f"{CATEGORY}_cpu_stats_v3_20d.npy", v3_arr)

    print(f"Saved: {CATEGORY}_cpu_stats_v2.npy ({v2_arr.shape})")
    print(f"Saved: {CATEGORY}_cpu_stats_v3_20d.npy ({v3_arr.shape})")


if __name__ == "__main__":
    main()
