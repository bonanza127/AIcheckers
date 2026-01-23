#!/usr/bin/env python3
"""
Evaluate new CPU candidates with parallel workers.

Outputs:
  - Cohen's d (AI vs Real)
  - Max correlation vs current CPU feature set
  - Pseudo weights from a simple ridge regression
"""
import argparse
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from scripts.extract_cpu_stats_v2 import extract_features, FEATURE_NAMES

DATA_ROOT = Path("/home/techne/aicheckers/data")
ANIMEDL_ROOT = DATA_ROOT / "animedl2m_dataset_release"

AI_DIRS = {
    "novelai": DATA_ROOT / "novelai_combined",
    "illustrious": ANIMEDL_ROOT / "civitai_subset/image/Illustrious",
}
REAL_DIR = ANIMEDL_ROOT / "real_images/images"


def load_image(path, target_size=512):
    img = Image.open(path).convert("RGB")
    if img.size != (target_size, target_size):
        img = img.resize((target_size, target_size), Image.LANCZOS)
    return np.array(img)


def st_anisotropy(gray):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    jxx = cv2.GaussianBlur(gx * gx, (3, 3), 1.0)
    jyy = cv2.GaussianBlur(gy * gy, (3, 3), 1.0)
    jxy = cv2.GaussianBlur(gx * gy, (3, 3), 1.0)
    trace = jxx + jyy
    det = jxx * jyy - jxy * jxy
    tmp = np.sqrt(np.maximum(trace * trace / 4 - det, 0))
    l1 = trace / 2 + tmp
    l2 = trace / 2 - tmp
    return (l1 - l2) / (l1 + l2 + 1e-6)


def st_aniso_spatial_gradient(gray):
    aniso = st_anisotropy(gray)
    gx = cv2.Sobel(aniso, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(aniso, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    return float(mag.mean())


def hue_transition_entropy(img_rgb):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0].astype(np.int16)
    dh = np.abs(hue[:, 1:] - hue[:, :-1])
    dv = np.abs(hue[1:, :] - hue[:-1, :])
    dh = np.minimum(dh, 180 - dh)
    dv = np.minimum(dv, 180 - dv)
    diffs = np.concatenate([dh.flatten(), dv.flatten()])
    hist, _ = np.histogram(diffs, bins=36, range=(0, 180))
    hist = hist / (hist.sum() + 1e-10)
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist + 1e-10)))


def edge_curvature_var(gray):
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    curvs = []
    for contour in contours:
        if len(contour) < 10:
            continue
        contour_sq = contour.squeeze()
        if contour_sq.ndim != 2:
            continue
        for i in range(len(contour_sq)):
            p1 = contour_sq[i - 2]
            p2 = contour_sq[i - 1]
            p3 = contour_sq[i]
            v1 = p2 - p1
            v2 = p3 - p2
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            dot = v1[0] * v2[0] + v1[1] * v2[1]
            angle = np.arctan2(cross, dot)
            curvs.append(abs(angle))
    return float(np.var(curvs)) if curvs else 0.0


def highfreq_spatial_autocorr(gray):
    low = cv2.GaussianBlur(gray.astype(np.float32), (3, 3), 1.0)
    res = gray.astype(np.float32) - low
    a = res[:-1, :-1].flatten()
    b = res[1:, 1:].flatten()
    if a.size < 2:
        return 0.0
    corr = np.corrcoef(a, b)[0, 1]
    return float(0.0 if np.isnan(corr) else corr)


def gradient_histogram_kurtosis(gray):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    vals = mag.flatten()
    if vals.size < 10:
        return 0.0
    mean = vals.mean()
    std = vals.std()
    if std < 1e-8:
        return 0.0
    kurt = np.mean(((vals - mean) / std) ** 4) - 3.0
    return float(kurt)


def lbp_uniformity_ratio(gray):
    g = gray.astype(np.int16)
    center = g[1:-1, 1:-1]
    neighbors = [
        g[:-2, :-2], g[:-2, 1:-1], g[:-2, 2:],
        g[1:-1, 2:], g[2:, 2:], g[2:, 1:-1],
        g[2:, :-2], g[1:-1, :-2],
    ]
    lbp = np.zeros_like(center, dtype=np.uint8)
    for i, n in enumerate(neighbors):
        lbp |= ((n >= center).astype(np.uint8) << i)
    codes = lbp.flatten()
    if codes.size == 0:
        return 0.0
    transitions = 0
    for i in range(8):
        transitions += ((codes >> i) & 1) != ((codes >> ((i + 1) % 8)) & 1)
    uniform = transitions <= 2
    return float(np.mean(uniform))


def compute_new_features(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    return [
        st_aniso_spatial_gradient(gray),
        hue_transition_entropy(img_rgb),
        edge_curvature_var(gray),
        highfreq_spatial_autocorr(gray),
        gradient_histogram_kurtosis(gray),
        lbp_uniformity_ratio(gray),
    ]


NEW_NAMES = [
    "st_aniso_spatial_gradient",
    "hue_transition_entropy",
    "edge_curvature_var",
    "highfreq_spatial_autocorr",
    "gradient_histogram_kurtosis",
    "lbp_uniformity_ratio",
]


def process_path(path_str):
    try:
        img = load_image(path_str)
        base = extract_features(img)
        new = compute_new_features(img)
        return base, new
    except Exception:
        return None


def cohens_d(a, b):
    a = np.array(a)
    b = np.array(b)
    if len(a) < 2 or len(b) < 2:
        return 0.0
    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)
    pooled = np.sqrt((var_a + var_b) / 2)
    return (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else 0.0


def sample_images(path, n, seed=42):
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    images = [p for p in Path(path).rglob("*") if p.suffix.lower() in exts]
    random.Random(seed).shuffle(images)
    return images[: min(n, len(images))]


def collect_features(paths, workers=6):
    base_feats = []
    new_feats = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(process_path, str(p)) for p in paths]
        for fut in as_completed(futures):
            result = fut.result()
            if result is None:
                continue
            base, new = result
            base_feats.append(base)
            new_feats.append(new)
    return np.array(base_feats), np.array(new_feats)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ai-samples", type=int, default=300)
    parser.add_argument("--real-samples", type=int, default=600)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    ai_paths = []
    for _, p in AI_DIRS.items():
        ai_paths.extend(sample_images(p, args.ai_samples))

    real_paths = sample_images(REAL_DIR, args.real_samples)

    print(f"AI samples: {len(ai_paths)} (novelai + illustrious)")
    print(f"Real samples: {len(real_paths)}")
    print(f"Workers: {args.workers}")

    ai_base, ai_new = collect_features(ai_paths, workers=args.workers)
    real_base, real_new = collect_features(real_paths, workers=args.workers)

    # Cohen's d
    print("\nCohen's d (AI - Real)")
    for i, name in enumerate(NEW_NAMES):
        d = cohens_d(ai_new[:, i], real_new[:, i])
        verdict = "Large" if abs(d) >= 0.8 else "Medium" if abs(d) >= 0.5 else "Small" if abs(d) >= 0.2 else "Negligible"
        print(f"{name:<30} {d:>7.3f}  {verdict}")

    # Correlation vs base features
    print("\nMax |corr| vs base CPU features")
    base_all = np.vstack([ai_base, real_base])
    new_all = np.vstack([ai_new, real_new])
    base_mean = base_all.mean(axis=0)
    base_std = base_all.std(axis=0) + 1e-8
    new_mean = new_all.mean(axis=0)
    new_std = new_all.std(axis=0) + 1e-8
    base_z = (base_all - base_mean) / base_std
    new_z = (new_all - new_mean) / new_std
    for i, name in enumerate(NEW_NAMES):
        corrs = (new_z[:, i][:, None] * base_z).mean(axis=0)
        j = int(np.argmax(np.abs(corrs)))
        print(f"{name:<30} max|r|={corrs[j]:+.3f} with {FEATURE_NAMES[j]}")

    # Pseudo weights (ridge on combined base+new)
    print("\nPseudo weights (ridge, z-scored)")
    X = np.hstack([base_z, new_z])
    y = np.concatenate([np.ones(len(ai_base)), np.zeros(len(real_base))])
    lam = 1e-3
    xtx = X.T @ X
    w = np.linalg.solve(xtx + lam * np.eye(xtx.shape[0]), X.T @ y)
    for i, name in enumerate(NEW_NAMES):
        idx = len(FEATURE_NAMES) + i
        print(f"{name:<30} {w[idx]:+.4f}")


if __name__ == "__main__":
    main()
