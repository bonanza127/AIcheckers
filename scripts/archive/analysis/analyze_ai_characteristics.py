#!/usr/bin/env python3
"""
AI画像特徴分析スクリプト（並列化版）
- NovelAI vs Illustrious vs Real の比較
- AI共通特徴の発見
- NovelAI固有特徴の発見
"""

import numpy as np
import cv2
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import random
from scipy import stats, ndimage
from scipy.signal import find_peaks
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# ===== 設定 =====
N_NOVELAI = 2000
N_ILLUSTRIOUS = 2000
N_REAL = 4000
N_WORKERS = 8

EMBEDDINGS_DIR = Path("/home/techne/aicheckers/embeddings")


def load_image_paths(category: str, n_samples: int) -> list:
    """カテゴリから画像パスをロード"""
    files_path = EMBEDDINGS_DIR / f"{category}_files.txt"
    if not files_path.exists():
        files_path = EMBEDDINGS_DIR / f"{category}_image_list.txt"

    with open(files_path) as f:
        paths = [line.strip() for line in f if line.strip()]

    if len(paths) > n_samples:
        random.seed(42)  # 再現性のため
        paths = random.sample(paths, n_samples)
    return paths


def load_image(path, target_size=512):
    """画像をロード・リサイズ"""
    img = cv2.imread(str(path))
    if img is None:
        return None
    img = cv2.resize(img, (target_size, target_size))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ===== 特徴量関数 =====

def luminance_mean(img_rgb):
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    return float(lab[:, :, 0].mean())

def luminance_std(img_rgb):
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    return float(lab[:, :, 0].std())

def luminance_skewness(img_rgb):
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l = lab[:, :, 0].astype(np.float64).ravel()
    return float(stats.skew(l))

def luminance_kurtosis(img_rgb):
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l = lab[:, :, 0].astype(np.float64).ravel()
    return float(stats.kurtosis(l))

def luminance_peak_count(img_rgb):
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l = lab[:, :, 0].ravel()
    hist, _ = np.histogram(l, bins=32, range=(0, 256))
    hist_smooth = ndimage.gaussian_filter1d(hist.astype(np.float64), sigma=1)
    peaks, _ = find_peaks(hist_smooth, height=hist_smooth.max() * 0.02)
    return float(len(peaks))

def color_hue_entropy(img_rgb):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h = hsv[:, :, 0].ravel()
    hist, _ = np.histogram(h, bins=36, range=(0, 180))
    hist = hist.astype(np.float64) / (hist.sum() + 1e-10)
    return float(-np.sum(hist * np.log2(hist + 1e-10)))

def saturation_mean(img_rgb):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    return float(hsv[:, :, 1].mean())

def saturation_std(img_rgb):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    return float(hsv[:, :, 1].std())

def saturation_skewness(img_rgb):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    s = hsv[:, :, 1].astype(np.float64).ravel()
    return float(stats.skew(s))

def color_a_mean(img_rgb):
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    return float(lab[:, :, 1].mean())

def color_a_std(img_rgb):
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    return float(lab[:, :, 1].std())

def color_b_mean(img_rgb):
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    return float(lab[:, :, 2].mean())

def color_b_std(img_rgb):
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    return float(lab[:, :, 2].std())

def color_ab_spread(img_rgb):
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float64)
    a, b = lab[:,:,1].ravel(), lab[:,:,2].ravel()
    ab_cov = np.cov(a, b)
    return float(np.sqrt(np.trace(ab_cov)))

def color_palette_entropy(img_rgb):
    quantized = (img_rgb // 8).astype(np.uint8)
    colors = quantized.reshape(-1, 3)
    color_ids = colors[:, 0].astype(np.int32) * 1024 + colors[:, 1] * 32 + colors[:, 2]
    _, counts = np.unique(color_ids, return_counts=True)
    probs = counts.astype(np.float64) / counts.sum()
    return float(-np.sum(probs * np.log2(probs + 1e-10)))

def edge_density(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return float(edges.sum() / edges.size / 255)

def edge_mean_strength(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(sobelx**2 + sobely**2)
    return float(gradient.mean())

def edge_std_strength(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(sobelx**2 + sobely**2)
    return float(gradient.std())

def edge_hv_ratio(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    h_edge = np.abs(sobelx).sum()
    v_edge = np.abs(sobely).sum()
    return float(h_edge / (v_edge + 1e-10))

def noise_std(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    smooth = cv2.GaussianBlur(gray, (5, 5), 0)
    noise = gray.astype(np.float64) - smooth.astype(np.float64)
    return float(noise.std())

def noise_kurtosis(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    smooth = cv2.GaussianBlur(gray, (5, 5), 0)
    noise = gray.astype(np.float64) - smooth.astype(np.float64)
    return float(stats.kurtosis(noise.ravel()))

def noise_autocorr(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    smooth = cv2.GaussianBlur(gray, (5, 5), 0)
    noise = gray.astype(np.float64) - smooth.astype(np.float64)
    noise_h = noise[:, 1:] * noise[:, :-1]
    noise_v = noise[1:, :] * noise[:-1, :]
    noise_var = noise.var()
    if noise_var < 1e-10:
        return 0.0
    return float((noise_h.mean() + noise_v.mean()) / 2 / noise_var)

def freq_low_ratio(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    f = np.fft.fft2(gray.astype(np.float64))
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    max_r = min(cx, cy)
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    low_mask = r < max_r // 4
    return float(magnitude[low_mask].sum() / (magnitude.sum() + 1e-10))

def freq_high_ratio(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    f = np.fft.fft2(gray.astype(np.float64))
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    max_r = min(cx, cy)
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    high_mask = r > max_r // 2
    return float(magnitude[high_mask].sum() / (magnitude.sum() + 1e-10))

def freq_slope(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    f = np.fft.fft2(gray.astype(np.float64))
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift) + 1e-10
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx)**2 + (y - cy)**2).astype(np.int32)
    max_r = r.max()
    radial_mean = np.bincount(r.ravel(), magnitude.ravel(), minlength=max_r+1)
    radial_cnt = np.bincount(r.ravel(), minlength=max_r+1)
    radial_mean = radial_mean / (radial_cnt + 1e-6)
    start_r = max_r // 4
    xs = np.log(np.arange(start_r, max_r) + 1)
    ys = np.log(radial_mean[start_r:max_r] + 1e-10)
    if len(xs) < 2:
        return 0.0
    slope, _ = np.polyfit(xs, ys, 1)
    return float(slope)

def texture_flat_ratio(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    patch_size = 32
    n_h, n_w = h // patch_size, w // patch_size
    flat_count = 0
    total = 0
    for i in range(n_h):
        for j in range(n_w):
            patch = gray[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
            if patch.std() < 10:
                flat_count += 1
            total += 1
    return float(flat_count / total) if total > 0 else 0.0

def texture_patch_std_var(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    patch_size = 32
    n_h, n_w = h // patch_size, w // patch_size
    stds = []
    for i in range(n_h):
        for j in range(n_w):
            patch = gray[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
            stds.append(patch.std())
    return float(np.var(stds)) if stds else 0.0

def compress_block_boundary(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float64)
    h, w = gray.shape
    block_size = 8
    h_boundaries = []
    for i in range(block_size, h - block_size, block_size):
        diff = np.abs(gray[i, :] - gray[i-1, :])
        h_boundaries.append(diff.mean())
    v_boundaries = []
    for j in range(block_size, w - block_size, block_size):
        diff = np.abs(gray[:, j] - gray[:, j-1])
        v_boundaries.append(diff.mean())
    h_mean = np.mean(h_boundaries) if h_boundaries else 0
    v_mean = np.mean(v_boundaries) if v_boundaries else 0
    overall = np.mean(np.abs(np.diff(gray, axis=1))) + np.mean(np.abs(np.diff(gray, axis=0)))
    return float((h_mean + v_mean) / (overall + 1e-6))

def jpeg_quant_pattern(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
    hist_fft = np.abs(np.fft.fft(hist))
    return float(hist_fft[32] / (hist_fft[1:64].mean() + 1e-10))

def high_saturation_ratio(img_rgb):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    s = hsv[:, :, 1]
    return float((s > 200).mean())

def skin_tone_ratio(img_rgb):
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = lab[:,:,0], lab[:,:,1], lab[:,:,2]
    skin_mask = (l > 150) & (a > 128) & (a < 145) & (b > 128) & (b < 145)
    return float(skin_mask.mean())


# 全特徴量リスト
FEATURE_FUNCS = {
    # 輝度系
    'luminance_mean': luminance_mean,
    'luminance_std': luminance_std,
    'luminance_skewness': luminance_skewness,
    'luminance_kurtosis': luminance_kurtosis,
    'luminance_peak_count': luminance_peak_count,
    # 色相・彩度系
    'color_hue_entropy': color_hue_entropy,
    'saturation_mean': saturation_mean,
    'saturation_std': saturation_std,
    'saturation_skewness': saturation_skewness,
    'high_saturation_ratio': high_saturation_ratio,
    # Lab色空間系
    'color_a_mean': color_a_mean,
    'color_a_std': color_a_std,
    'color_b_mean': color_b_mean,
    'color_b_std': color_b_std,
    'color_ab_spread': color_ab_spread,
    'color_palette_entropy': color_palette_entropy,
    'skin_tone_ratio': skin_tone_ratio,
    # エッジ系
    'edge_density': edge_density,
    'edge_mean_strength': edge_mean_strength,
    'edge_std_strength': edge_std_strength,
    'edge_hv_ratio': edge_hv_ratio,
    # ノイズ系
    'noise_std': noise_std,
    'noise_kurtosis': noise_kurtosis,
    'noise_autocorr': noise_autocorr,
    # 周波数系
    'freq_low_ratio': freq_low_ratio,
    'freq_high_ratio': freq_high_ratio,
    'freq_slope': freq_slope,
    # テクスチャ系
    'texture_flat_ratio': texture_flat_ratio,
    'texture_patch_std_var': texture_patch_std_var,
    # 圧縮系
    'compress_block_boundary': compress_block_boundary,
    'jpeg_quant_pattern': jpeg_quant_pattern,
}


def extract_features_single(path):
    """単一画像の特徴抽出"""
    try:
        img = load_image(path)
        if img is None:
            return None
        return {name: func(img) for name, func in FEATURE_FUNCS.items()}
    except Exception:
        return None


def extract_features_parallel(paths, desc="Extracting"):
    """並列特徴抽出"""
    results = []
    from tqdm import tqdm

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(extract_features_single, p): p for p in paths}
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
            result = future.result()
            if result is not None:
                results.append(result)

    return results


def compute_cohens_d(group1, group2):
    """Cohen's d計算"""
    pooled_std = np.sqrt((group1.std()**2 + group2.std()**2) / 2)
    if pooled_std < 1e-10:
        return 0.0
    return (group1.mean() - group2.mean()) / pooled_std


def main():
    print("=" * 80)
    print("AI画像特徴分析（並列化版）")
    print(f"  NovelAI: {N_NOVELAI} / Illustrious: {N_ILLUSTRIOUS} / Real: {N_REAL}")
    print(f"  Workers: {N_WORKERS}")
    print("=" * 80)

    # 画像パス取得
    print("\n[1/5] 画像パス取得...")
    novelai_paths = load_image_paths('novelai_combined_ai', N_NOVELAI)
    illustrious_paths = load_image_paths('illustrious_ai', N_ILLUSTRIOUS)
    real_paths = load_image_paths('danbooru_real', N_REAL)

    print(f"  NovelAI: {len(novelai_paths)}")
    print(f"  Illustrious: {len(illustrious_paths)}")
    print(f"  Real: {len(real_paths)}")

    # 特徴抽出（並列）
    print("\n[2/5] 特徴抽出（並列処理）...")
    novelai_features = extract_features_parallel(novelai_paths, "NovelAI")
    illustrious_features = extract_features_parallel(illustrious_paths, "Illustrious")
    real_features = extract_features_parallel(real_paths, "Real")

    print(f"  NovelAI: {len(novelai_features)} extracted")
    print(f"  Illustrious: {len(illustrious_features)} extracted")
    print(f"  Real: {len(real_features)} extracted")

    # numpy配列に変換
    feature_names = list(FEATURE_FUNCS.keys())
    novelai_data = np.array([[f[fn] for fn in feature_names] for f in novelai_features])
    illustrious_data = np.array([[f[fn] for fn in feature_names] for f in illustrious_features])
    real_data = np.array([[f[fn] for fn in feature_names] for f in real_features])

    # AI全体 = NovelAI + Illustrious
    ai_data = np.vstack([novelai_data, illustrious_data])

    # ===== 分析1: AI共通特徴（AI vs Real）=====
    print("\n[3/5] AI共通特徴分析（AI vs Real）...")
    print("\n" + "=" * 80)
    print("AI共通特徴（NovelAI + Illustrious vs Real）")
    print("=" * 80)

    ai_vs_real = []
    for i, fn in enumerate(feature_names):
        d = compute_cohens_d(ai_data[:, i], real_data[:, i])
        ai_vs_real.append((fn, d, ai_data[:, i].mean(), real_data[:, i].mean()))

    ai_vs_real.sort(key=lambda x: abs(x[1]), reverse=True)

    print(f"\n{'Feature':<28} {'Cohen d':>10} {'AI mean':>12} {'Real mean':>12}")
    print("-" * 64)

    for fn, d, ai_m, real_m in ai_vs_real[:20]:
        sign = '+' if d > 0 else ' '
        effect = '◎' if abs(d) > 0.8 else '○' if abs(d) > 0.5 else '△' if abs(d) > 0.3 else '✗'
        print(f"{fn:<28} {sign}{d:>8.3f} {effect}  {ai_m:>10.4f}  {real_m:>10.4f}")

    # ===== 分析2: NovelAI固有特徴（NovelAI vs Illustrious）=====
    print("\n[4/5] NovelAI固有特徴分析（NovelAI vs Illustrious）...")
    print("\n" + "=" * 80)
    print("NovelAI固有特徴（NovelAI vs Illustrious）")
    print("=" * 80)

    novelai_vs_illust = []
    for i, fn in enumerate(feature_names):
        d = compute_cohens_d(novelai_data[:, i], illustrious_data[:, i])
        novelai_vs_illust.append((fn, d, novelai_data[:, i].mean(), illustrious_data[:, i].mean()))

    novelai_vs_illust.sort(key=lambda x: abs(x[1]), reverse=True)

    print(f"\n{'Feature':<28} {'Cohen d':>10} {'NovelAI':>12} {'Illustrious':>12}")
    print("-" * 64)

    for fn, d, nov_m, ill_m in novelai_vs_illust[:20]:
        sign = '+' if d > 0 else ' '
        effect = '◎' if abs(d) > 0.8 else '○' if abs(d) > 0.5 else '△' if abs(d) > 0.3 else '✗'
        print(f"{fn:<28} {sign}{d:>8.3f} {effect}  {nov_m:>10.4f}  {ill_m:>10.4f}")

    # ===== 分析3: 特徴の分類 =====
    print("\n[5/5] 特徴分類...")
    print("\n" + "=" * 80)
    print("特徴分類マトリクス")
    print("=" * 80)

    # AI共通度 = |AI vs Real|, モデル固有度 = |NovelAI vs Illustrious|
    ai_common_d = {fn: d for fn, d, _, _ in ai_vs_real}
    model_specific_d = {fn: d for fn, d, _, _ in novelai_vs_illust}

    print(f"\n{'Feature':<28} {'AI共通':>10} {'モデル差':>10} {'分類':>15}")
    print("-" * 65)

    categories = {'AI共通（汎用）': [], 'NovelAI固有': [], 'Illustrious固有': [], '弱い': []}

    for fn in feature_names:
        ai_d = ai_common_d[fn]
        model_d = model_specific_d[fn]

        if abs(ai_d) < 0.3:
            cat = '弱い'
        elif abs(model_d) < 0.3:
            cat = 'AI共通（汎用）'
        elif model_d > 0.3:
            cat = 'NovelAI固有'
        else:
            cat = 'Illustrious固有'

        categories[cat].append((fn, ai_d, model_d))

        if abs(ai_d) >= 0.3 or abs(model_d) >= 0.3:
            print(f"{fn:<28} {ai_d:>+10.3f} {model_d:>+10.3f} {cat:>15}")

    # ===== サマリー =====
    print("\n" + "=" * 80)
    print("サマリー")
    print("=" * 80)

    print("\n【AI共通（汎用）特徴】（|AI vs Real| > 0.3 かつ |NovelAI vs Illustrious| < 0.3）")
    for fn, ai_d, _ in sorted(categories['AI共通（汎用）'], key=lambda x: abs(x[1]), reverse=True)[:10]:
        effect = '◎' if abs(ai_d) > 0.8 else '○' if abs(ai_d) > 0.5 else '△'
        print(f"  {effect} {fn:<28} d={ai_d:+.3f}")

    print("\n【NovelAI固有特徴】（モデル差 > 0.3）")
    for fn, ai_d, model_d in sorted(categories['NovelAI固有'], key=lambda x: abs(x[2]), reverse=True)[:10]:
        print(f"  → {fn:<28} AI共通={ai_d:+.3f}, NovelAI差={model_d:+.3f}")

    print("\n【Illustrious固有特徴】（モデル差 < -0.3）")
    for fn, ai_d, model_d in sorted(categories['Illustrious固有'], key=lambda x: abs(x[2]), reverse=True)[:10]:
        print(f"  → {fn:<28} AI共通={ai_d:+.3f}, Illustrious差={model_d:+.3f}")

    # ===== ロジスティック回帰 =====
    print("\n" + "=" * 80)
    print("ロジスティック回帰分析")
    print("=" * 80)

    # AI vs Real
    X_ai_real = np.vstack([ai_data, real_data])
    y_ai_real = np.array([1] * len(ai_data) + [0] * len(real_data))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_ai_real)
    valid_mask = ~(np.isnan(X_scaled).any(axis=1) | np.isinf(X_scaled).any(axis=1))
    X_scaled = X_scaled[valid_mask]
    y_ai_real = y_ai_real[valid_mask]

    model = LogisticRegression(max_iter=1000, C=0.1)
    scores = cross_val_score(model, X_scaled, y_ai_real, cv=5, scoring='roc_auc')
    print(f"\nAI vs Real 5-fold CV ROC-AUC: {scores.mean():.4f} (+/- {scores.std():.4f})")

    model.fit(X_scaled, y_ai_real)
    print("\nTop 10 特徴量（AI vs Real重み）:")
    weights = list(zip(feature_names, model.coef_[0]))
    weights.sort(key=lambda x: abs(x[1]), reverse=True)
    for fn, w in weights[:10]:
        ai_d = ai_common_d[fn]
        print(f"  {fn:<28} w={w:+.3f}  d={ai_d:+.3f}")


if __name__ == "__main__":
    main()
