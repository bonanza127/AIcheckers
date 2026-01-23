#!/usr/bin/env python3
"""
NovelAI分析から発見した有望特徴量のテスト
既存18d CPUとの相関・Cohen's d・ロジスティック重みを確認
"""

import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import random
from scipy import stats, fftpack
from scipy.signal import find_peaks
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


def load_image_paths(category: str, n_samples: int = 500) -> list:
    files_path = Path(f"/home/techne/aicheckers/embeddings/{category}_files.txt")
    if not files_path.exists():
        files_path = Path(f"/home/techne/aicheckers/embeddings/{category}_image_list.txt")
    with open(files_path) as f:
        paths = [line.strip() for line in f if line.strip()]
    if len(paths) > n_samples:
        paths = random.sample(paths, n_samples)
    return paths


# ==== 新規特徴量（AI共通・高効果） ====

def color_hue_entropy(img_rgb):
    """色相エントロピー - AIは色相の多様性が高い (d=+0.925)"""
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h = hsv[:, :, 0].ravel()
    hist, _ = np.histogram(h, bins=36, range=(0, 180))
    hist = hist / (hist.sum() + 1e-10)
    entropy = -np.sum(hist * np.log2(hist + 1e-10))
    return float(entropy)


def saturation_mean(img_rgb):
    """彩度平均 - AIは彩度が高い (d=+0.785)"""
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    return float(hsv[:, :, 1].mean())


def luminance_peak_count(img_rgb):
    """輝度分布のピーク数 - AIは多峰性 (d=+0.658)"""
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l = lab[:, :, 0].ravel()
    hist, _ = np.histogram(l, bins=32, range=(0, 256))
    hist = hist / hist.sum()
    peaks, _ = find_peaks(hist, height=0.02)
    return float(len(peaks))


def jpeg_quant_pattern(img_gray):
    """JPEG量子化パターン - Realのほうが強い (d=-0.515)"""
    hist, _ = np.histogram(img_gray.ravel(), bins=256, range=(0, 256))
    hist_fft = np.abs(np.fft.fft(hist))
    # 8周期のエネルギー（JPEG量子化）
    quant_energy = hist_fft[32] / (hist_fft[1:64].mean() + 1e-10)
    return float(quant_energy)


def luminance_mean(img_rgb):
    """輝度平均 - AIは暗め (d=-0.702)"""
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    return float(lab[:, :, 0].mean())


def color_b_std(img_rgb):
    """Lab b*チャンネルの標準偏差 (d=+0.635)"""
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    return float(lab[:, :, 2].std())


def noise_block16_pattern(img_gray):
    """16x16ブロックノイズパターン - Realが強い (d=-0.419)"""
    smooth = cv2.GaussianBlur(img_gray, (5, 5), 0)
    noise = img_gray.astype(np.float64) - smooth.astype(np.float64)
    h, w = noise.shape
    block_size = 16
    block_h, block_w = h // block_size, w // block_size
    if block_h < 2 or block_w < 2:
        return 0.0
    blocks = noise[:block_h*block_size, :block_w*block_size].reshape(
        block_h, block_size, block_w, block_size)
    block_means = blocks.mean(axis=(1, 3))
    return float(block_means.std())


def texture_patch_std_variance(img_gray):
    """パッチ標準偏差の分散 - Realが大きい (d=-0.455)"""
    h, w = img_gray.shape
    patch_size = 32
    n_h, n_w = h // patch_size, w // patch_size
    patch_stds = []
    for i in range(n_h):
        for j in range(n_w):
            patch = img_gray[i*patch_size:(i+1)*patch_size,
                           j*patch_size:(j+1)*patch_size]
            patch_stds.append(patch.std())
    return float(np.var(patch_stds))


def compress_block_boundary(img_gray):
    """8x8ブロック境界強度 - Realが強い (d=-0.422)"""
    h, w = img_gray.shape
    block_size = 8

    h_boundaries = []
    for i in range(block_size, h - block_size, block_size):
        diff = np.abs(img_gray[i, :].astype(float) - img_gray[i-1, :].astype(float))
        h_boundaries.append(diff.mean())

    v_boundaries = []
    for j in range(block_size, w - block_size, block_size):
        diff = np.abs(img_gray[:, j].astype(float) - img_gray[:, j-1].astype(float))
        v_boundaries.append(diff.mean())

    h_mean = np.mean(h_boundaries) if h_boundaries else 0
    v_mean = np.mean(v_boundaries) if v_boundaries else 0

    # 非境界との比較
    h_non = []
    for i in range(4, h - 4, 8):
        for offset in [2, 3, 5, 6]:
            if i + offset < h - 1:
                diff = np.abs(img_gray[i+offset, :].astype(float) -
                            img_gray[i+offset-1, :].astype(float))
                h_non.append(diff.mean())

    h_non_mean = np.mean(h_non) if h_non else 1
    return float((h_mean + v_mean) / (2 * h_non_mean + 1e-10))


def color_ab_spread(img_rgb):
    """Lab色空間のab平面での広がり (d=+0.565)"""
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float64)
    a, b = lab[:,:,1].ravel(), lab[:,:,2].ravel()
    ab_cov = np.cov(a, b)
    return float(np.sqrt(np.trace(ab_cov)))


# ==== 特徴抽出 ====

FEATURE_FUNCS = {
    'color_hue_entropy': lambda rgb, gray: color_hue_entropy(rgb),
    'saturation_mean': lambda rgb, gray: saturation_mean(rgb),
    'luminance_peak_count': lambda rgb, gray: luminance_peak_count(rgb),
    'jpeg_quant_pattern': lambda rgb, gray: jpeg_quant_pattern(gray),
    'luminance_mean': lambda rgb, gray: luminance_mean(rgb),
    'color_b_std': lambda rgb, gray: color_b_std(rgb),
    'noise_block16_pattern': lambda rgb, gray: noise_block16_pattern(gray),
    'texture_patch_std_variance': lambda rgb, gray: texture_patch_std_variance(gray),
    'compress_block_boundary': lambda rgb, gray: compress_block_boundary(gray),
    'color_ab_spread': lambda rgb, gray: color_ab_spread(rgb),
}


def extract_features(img_path):
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            return None
        img = cv2.resize(img, (512, 512))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        return {name: func(img_rgb, img_gray) for name, func in FEATURE_FUNCS.items()}
    except Exception as e:
        return None


def main():
    print("=" * 70)
    print("NovelAI分析発見特徴量のテスト")
    print("=" * 70)

    # 画像パス取得
    print("\n[1/4] 画像パス取得...")
    ai_categories = ['novelai_combined_ai', 'illustrious_ai', 'pony_ai']
    ai_paths = []
    for cat in ai_categories:
        try:
            paths = load_image_paths(cat, 200)
            ai_paths.extend(paths)
        except:
            pass
    random.shuffle(ai_paths)
    ai_paths = ai_paths[:500]

    real_paths = load_image_paths('danbooru_real', 500)
    print(f"  AI: {len(ai_paths)} samples")
    print(f"  Real: {len(real_paths)} samples")

    # 特徴抽出
    print("\n[2/4] 特徴量抽出...")
    ai_features = []
    for p in tqdm(ai_paths, desc="AI"):
        f = extract_features(p)
        if f:
            ai_features.append(f)

    real_features = []
    for p in tqdm(real_paths, desc="Real"):
        f = extract_features(p)
        if f:
            real_features.append(f)

    feature_names = list(FEATURE_FUNCS.keys())

    ai_data = np.array([[f[fn] for fn in feature_names] for f in ai_features])
    real_data = np.array([[f[fn] for fn in feature_names] for f in real_features])

    # Cohen's d計算
    print("\n[3/4] Cohen's d 計算...")
    print("\n" + "=" * 70)
    print(f"{'Feature':<30} {'Cohen d':>10} {'AI mean':>12} {'Real mean':>12}")
    print("=" * 70)

    results = []
    for i, fn in enumerate(feature_names):
        ai_vals = ai_data[:, i]
        real_vals = real_data[:, i]

        pooled_std = np.sqrt((ai_vals.std()**2 + real_vals.std()**2) / 2)
        d = (ai_vals.mean() - real_vals.mean()) / pooled_std if pooled_std > 1e-10 else 0

        results.append((fn, d, ai_vals.mean(), real_vals.mean()))

        sign = '+' if d > 0 else ' '
        effect = '◎' if abs(d) > 0.8 else '○' if abs(d) > 0.5 else '△' if abs(d) > 0.3 else '✗'
        print(f"{fn:<30} {sign}{d:>8.3f} {effect} {ai_vals.mean():>12.4f} {real_vals.mean():>12.4f}")

    # ランキング
    results.sort(key=lambda x: abs(x[1]), reverse=True)
    print("\n" + "-" * 70)
    print("Cohen's d ランキング (|d| 降順):")
    print("-" * 70)
    for rank, (fn, d, _, _) in enumerate(results, 1):
        sign = '+' if d > 0 else ' '
        effect = '◎' if abs(d) > 0.8 else '○' if abs(d) > 0.5 else '△' if abs(d) > 0.3 else '✗'
        print(f"  {rank:2d}. {fn:<30} {sign} {d:>6.3f} {effect}")

    # 既存CPU v2との相関確認
    print("\n[4/4] 既存CPU v2との相関確認...")

    # CPU v2データをロード
    try:
        cpu_ai = np.load('/home/techne/aicheckers/embeddings/novelai_combined_ai_cpu_stats_v2.npy')
        cpu_real = np.load('/home/techne/aicheckers/embeddings/danbooru_real_cpu_stats_v2.npy')

        # サンプル数を合わせる
        n_ai = min(len(ai_data), len(cpu_ai))
        n_real = min(len(real_data), len(cpu_real))

        combined_new = np.vstack([ai_data[:n_ai], real_data[:n_real]])
        combined_cpu = np.vstack([cpu_ai[:n_ai], cpu_real[:n_real]])

        cpu_feature_names = [
            'banding_score', 'radial_spectrum_slope', 'text_area_ratio',
            'fractal_dim_edge_512', 'patchwise_edge_density', 'st_aniso_mean',
            'st_aniso_var', 'st_aniso_spatial_gradient', 'flat_boundary_peri_area',
            'flat_hole_ratio', 'highfreq_spatial_autocorr', 'flat_ratio',
            'patch_vs_global_st_aniso_gap', 'cbcr_autocorr', 'edge_length_mean',
            'rank_entropy', 'flat_ratio_variance_across_tiles',
            'patch_vs_global_rank_entropy_gap'
        ]

        print("\n新規特徴量と既存CPU v2の最大相関:")
        print("-" * 60)

        for i, new_fn in enumerate(feature_names):
            max_corr = 0
            max_cpu_fn = ""
            for j, cpu_fn in enumerate(cpu_feature_names):
                if j < combined_cpu.shape[1]:
                    corr = np.corrcoef(combined_new[:, i], combined_cpu[:, j])[0, 1]
                    if abs(corr) > abs(max_corr):
                        max_corr = corr
                        max_cpu_fn = cpu_fn
            print(f"  {new_fn:<30} ↔ {max_cpu_fn:<35} r={max_corr:+.3f}")

    except Exception as e:
        print(f"  CPU v2ロードエラー: {e}")

    # ロジスティック回帰
    print("\n[5/5] ロジスティック回帰...")
    X = np.vstack([ai_data, real_data])
    y = np.array([1] * len(ai_data) + [0] * len(real_data))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # フィルタ（NaN/Inf除去）
    valid_mask = ~(np.isnan(X_scaled).any(axis=1) | np.isinf(X_scaled).any(axis=1))
    X_scaled = X_scaled[valid_mask]
    y = y[valid_mask]

    model = LogisticRegression(max_iter=1000, C=0.1)
    model.fit(X_scaled, y)

    scores = cross_val_score(model, X_scaled, y, cv=5, scoring='roc_auc')
    print(f"\n5-fold CV ROC-AUC: {scores.mean():.4f} (+/- {scores.std():.4f})")

    print("\n特徴量重み:")
    print("-" * 60)
    weights = list(zip(feature_names, model.coef_[0]))
    weights.sort(key=lambda x: abs(x[1]), reverse=True)
    for fn, w in weights:
        d_val = next((d for n, d, _, _ in results if n == fn), 0)
        print(f"  {fn:<30} w={w:+.3f}  |d|={abs(d_val):.3f}")

    # 推奨
    print("\n" + "=" * 70)
    print("推奨特徴量 (|d| > 0.5 かつ 既存との相関 < 0.7)")
    print("=" * 70)
    for fn, d, ai_m, real_m in results:
        if abs(d) > 0.5:
            sign = '+' if d > 0 else '-'
            print(f"  ○ {fn:<30} d={sign}{abs(d):.3f}")


if __name__ == "__main__":
    main()
