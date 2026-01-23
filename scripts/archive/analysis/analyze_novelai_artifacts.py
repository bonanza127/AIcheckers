#!/usr/bin/env python3
"""
NovelAI画像の特徴分析スクリプト
Real画像・他AIモデルとの比較から、NovelAI固有のアーティファクトを特定する
"""

import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import random
from scipy import stats, fftpack
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


def load_image_paths(category: str, n_samples: int = 300) -> list:
    """カテゴリから画像パスをロード"""
    files_path = Path(f"/home/techne/aicheckers/embeddings/{category}_files.txt")
    if not files_path.exists():
        files_path = Path(f"/home/techne/aicheckers/embeddings/{category}_image_list.txt")

    with open(files_path) as f:
        paths = [line.strip() for line in f if line.strip()]

    if len(paths) > n_samples:
        paths = random.sample(paths, n_samples)
    return paths


def analyze_frequency_spectrum(img_gray):
    """周波数スペクトルの詳細分析"""
    f = np.fft.fft2(img_gray.astype(np.float64))
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    h, w = img_gray.shape
    cy, cx = h // 2, w // 2

    # ラジアル周波数分布
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    r = r.astype(int)

    # 周波数帯域ごとのエネルギー
    max_r = min(cx, cy)
    radial_profile = np.zeros(max_r)
    for i in range(max_r):
        mask = (r == i)
        if mask.sum() > 0:
            radial_profile[i] = magnitude[mask].mean()

    # 低・中・高周波のエネルギー比
    low = radial_profile[:max_r//4].sum()
    mid = radial_profile[max_r//4:max_r//2].sum()
    high = radial_profile[max_r//2:].sum()
    total = low + mid + high + 1e-10

    # 方向性分析（水平・垂直・対角線）
    angles = np.arctan2(y - cy, x - cx)
    horizontal_mask = np.abs(angles) < np.pi/8
    vertical_mask = np.abs(np.abs(angles) - np.pi/2) < np.pi/8
    diagonal_mask = (np.abs(np.abs(angles) - np.pi/4) < np.pi/8) | (np.abs(np.abs(angles) - 3*np.pi/4) < np.pi/8)

    h_energy = magnitude[horizontal_mask].sum()
    v_energy = magnitude[vertical_mask].sum()
    d_energy = magnitude[diagonal_mask].sum()
    dir_total = h_energy + v_energy + d_energy + 1e-10

    return {
        'freq_low_ratio': low / total,
        'freq_mid_ratio': mid / total,
        'freq_high_ratio': high / total,
        'freq_h_v_ratio': h_energy / (v_energy + 1e-10),
        'freq_diagonal_ratio': d_energy / dir_total,
        'freq_slope': np.polyfit(np.arange(len(radial_profile)), np.log(radial_profile + 1e-10), 1)[0],
    }


def analyze_color_distribution(img_rgb):
    """色分布の詳細分析"""
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float64)
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.float64)

    l, a, b = lab[:,:,0], lab[:,:,1], lab[:,:,2]
    h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]

    # Lab空間の統計
    l_mean, l_std = l.mean(), l.std()
    a_mean, a_std = a.mean(), a.std()
    b_mean, b_std = b.mean(), b.std()

    # 色相の分布
    h_hist, _ = np.histogram(h.ravel(), bins=36, range=(0, 180))
    h_hist = h_hist / (h_hist.sum() + 1e-10)
    h_entropy = -np.sum(h_hist * np.log2(h_hist + 1e-10))

    # 彩度の分布
    s_mean, s_std = s.mean(), s.std()
    s_skew = stats.skew(s.ravel())

    # 色のクラスタリング度（ab平面での分散）
    ab_cov = np.cov(a.ravel(), b.ravel())
    ab_spread = np.sqrt(np.trace(ab_cov))

    # 特定の色への偏り
    # NovelAIは特定の肌色・髪色に偏る傾向？
    skin_mask = (l > 150) & (a > 128) & (a < 145) & (b > 128) & (b < 145)
    skin_ratio = skin_mask.sum() / skin_mask.size

    return {
        'color_l_mean': l_mean,
        'color_l_std': l_std,
        'color_a_mean': a_mean,
        'color_a_std': a_std,
        'color_b_mean': b_mean,
        'color_b_std': b_std,
        'color_h_entropy': h_entropy,
        'color_s_mean': s_mean,
        'color_s_std': s_std,
        'color_s_skew': s_skew,
        'color_ab_spread': ab_spread,
        'color_skin_ratio': skin_ratio,
    }


def analyze_noise_pattern(img_gray):
    """ノイズパターンの分析"""
    # ガウシアンブラーで滑らかな部分を取得
    smooth = cv2.GaussianBlur(img_gray, (5, 5), 0)
    noise = img_gray.astype(np.float64) - smooth.astype(np.float64)

    # ノイズの統計
    noise_std = noise.std()
    noise_kurtosis = stats.kurtosis(noise.ravel())
    noise_skew = stats.skew(noise.ravel())

    # ノイズの空間相関（隣接ピクセル間）
    noise_h = noise[:, 1:] * noise[:, :-1]
    noise_v = noise[1:, :] * noise[:-1, :]
    noise_autocorr = (noise_h.mean() + noise_v.mean()) / 2 / (noise_std**2 + 1e-10)

    # ノイズの周波数特性
    f = np.fft.fft2(noise)
    fshift = np.fft.fftshift(f)
    noise_mag = np.abs(fshift)
    h, w = noise.shape
    cy, cx = h // 2, w // 2

    # 高周波ノイズの比率
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    high_freq_mask = r > min(cx, cy) * 0.5
    noise_high_freq_ratio = noise_mag[high_freq_mask].sum() / (noise_mag.sum() + 1e-10)

    # ブロックノイズ（8x8, 16x16）の検出
    def detect_block_pattern(img, block_size):
        h, w = img.shape
        block_h, block_w = h // block_size, w // block_size
        blocks = img[:block_h*block_size, :block_w*block_size].reshape(block_h, block_size, block_w, block_size)
        block_means = blocks.mean(axis=(1, 3))
        return block_means.std()

    block8_pattern = detect_block_pattern(noise, 8)
    block16_pattern = detect_block_pattern(noise, 16)

    return {
        'noise_std': noise_std,
        'noise_kurtosis': noise_kurtosis,
        'noise_skew': noise_skew,
        'noise_autocorr': noise_autocorr,
        'noise_high_freq_ratio': noise_high_freq_ratio,
        'noise_block8': block8_pattern,
        'noise_block16': block16_pattern,
    }


def analyze_edge_characteristics(img_gray):
    """エッジ特性の分析"""
    # Canny edge
    edges = cv2.Canny(img_gray, 50, 150)
    edge_density = edges.sum() / edges.size / 255

    # Sobel勾配
    sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(sobelx**2 + sobely**2)
    gradient_dir = np.arctan2(sobely, sobelx)

    # エッジ強度の分布
    edge_mean = gradient_mag.mean()
    edge_std = gradient_mag.std()

    # エッジ方向のヒストグラム
    dir_hist, _ = np.histogram(gradient_dir.ravel(), bins=8, range=(-np.pi, np.pi), weights=gradient_mag.ravel())
    dir_hist = dir_hist / (dir_hist.sum() + 1e-10)
    dir_entropy = -np.sum(dir_hist * np.log2(dir_hist + 1e-10))

    # 水平・垂直エッジの比率
    h_edge = np.abs(sobelx).sum()
    v_edge = np.abs(sobely).sum()
    hv_ratio = h_edge / (v_edge + 1e-10)

    # エッジの連続性（エッジピクセルの隣接度）
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    edge_continuity = (edges & dilated).sum() / (edges.sum() + 1e-10)

    # アンチエイリアシングの検出
    # エッジ周辺の勾配の滑らかさ
    edge_mask = edges > 0
    if edge_mask.sum() > 100:
        edge_gradients = gradient_mag[edge_mask]
        aa_smoothness = np.std(edge_gradients) / (np.mean(edge_gradients) + 1e-10)
    else:
        aa_smoothness = 0

    return {
        'edge_density': edge_density,
        'edge_mean': edge_mean,
        'edge_std': edge_std,
        'edge_dir_entropy': dir_entropy,
        'edge_hv_ratio': hv_ratio,
        'edge_continuity': edge_continuity,
        'edge_aa_smoothness': aa_smoothness,
    }


def analyze_texture_pattern(img_gray):
    """テクスチャパターンの分析"""
    # 局所的なパターンの分散
    h, w = img_gray.shape
    patch_size = 32
    n_patches_h = h // patch_size
    n_patches_w = w // patch_size

    patch_stds = []
    patch_means = []
    for i in range(n_patches_h):
        for j in range(n_patches_w):
            patch = img_gray[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
            patch_stds.append(patch.std())
            patch_means.append(patch.mean())

    patch_std_var = np.var(patch_stds)
    patch_mean_var = np.var(patch_means)

    # 平坦領域の検出
    flat_threshold = 10
    flat_ratio = (np.array(patch_stds) < flat_threshold).mean()

    # グラデーションの検出
    # 水平・垂直方向の単調増加/減少パターン
    row_diffs = np.diff(img_gray.astype(np.float64), axis=1)
    col_diffs = np.diff(img_gray.astype(np.float64), axis=0)

    # 一貫した方向を持つピクセルの比率
    consistent_h = ((row_diffs > 0).all(axis=1) | (row_diffs < 0).all(axis=1)).mean()
    consistent_v = ((col_diffs > 0).all(axis=0) | (col_diffs < 0).all(axis=0)).mean()
    gradient_consistency = (consistent_h + consistent_v) / 2

    # DCT係数の分析
    h8, w8 = (h // 8) * 8, (w // 8) * 8
    img_crop = img_gray[:h8, :w8].astype(np.float64)

    # 8x8ブロックごとのDCT
    dct_energy_high = 0
    dct_energy_total = 0
    n_blocks = 0

    for i in range(0, h8, 8):
        for j in range(0, w8, 8):
            block = img_crop[i:i+8, j:j+8]
            dct = fftpack.dct(fftpack.dct(block.T, norm='ortho').T, norm='ortho')
            dct_energy_total += np.abs(dct).sum()
            dct_energy_high += np.abs(dct[4:, 4:]).sum()  # 高周波成分
            n_blocks += 1

    dct_high_ratio = dct_energy_high / (dct_energy_total + 1e-10)

    return {
        'texture_patch_std_var': patch_std_var,
        'texture_patch_mean_var': patch_mean_var,
        'texture_flat_ratio': flat_ratio,
        'texture_gradient_consistency': gradient_consistency,
        'texture_dct_high_ratio': dct_high_ratio,
    }


def analyze_compression_artifacts(img_gray):
    """圧縮アーティファクトの分析"""
    h, w = img_gray.shape

    # 8x8ブロック境界の検出（JPEG特有）
    def detect_block_boundary(img, block_size=8):
        h, w = img.shape
        h_boundaries = []
        v_boundaries = []

        for i in range(block_size, h - block_size, block_size):
            diff = np.abs(img[i, :].astype(float) - img[i-1, :].astype(float))
            h_boundaries.append(diff.mean())

        for j in range(block_size, w - block_size, block_size):
            diff = np.abs(img[:, j].astype(float) - img[:, j-1].astype(float))
            v_boundaries.append(diff.mean())

        return np.mean(h_boundaries) if h_boundaries else 0, np.mean(v_boundaries) if v_boundaries else 0

    h_boundary, v_boundary = detect_block_boundary(img_gray)

    # 非境界との比較
    h_non = []
    v_non = []
    for i in range(4, h - 4, 8):
        for offset in [2, 3, 5, 6]:
            if i + offset < h - 1:
                diff = np.abs(img_gray[i+offset, :].astype(float) - img_gray[i+offset-1, :].astype(float))
                h_non.append(diff.mean())

    h_non_mean = np.mean(h_non) if h_non else 1
    block_boundary_ratio = (h_boundary + v_boundary) / (2 * h_non_mean + 1e-10)

    # 量子化ステップの検出
    hist, _ = np.histogram(img_gray.ravel(), bins=256, range=(0, 256))
    # 周期的なパターンの検出
    hist_fft = np.abs(np.fft.fft(hist))
    # 8周期のエネルギー（JPEG量子化）
    quant_energy = hist_fft[32] / (hist_fft[1:64].mean() + 1e-10)

    return {
        'compress_block_boundary': block_boundary_ratio,
        'compress_quant_pattern': quant_energy,
    }


def analyze_unique_patterns(img_rgb, img_gray):
    """NovelAI固有パターンの可能性がある特徴"""
    h, w = img_gray.shape

    # 1. 特定のノイズシード由来のパターン
    # VAE由来の周期的なパターン
    f = np.fft.fft2(img_gray.astype(np.float64))
    fshift = np.fft.fftshift(f)
    magnitude = np.log(np.abs(fshift) + 1)

    # 特定の周波数でのピーク検出
    cy, cx = h // 2, w // 2
    # 1/4, 1/8, 1/16周期のパターン
    peaks = []
    for freq in [4, 8, 16]:
        if cx > freq and cy > freq:
            peak_val = magnitude[cy, cx + freq] + magnitude[cy + freq, cx]
            baseline = magnitude[cy, cx + freq + 2:cx + freq + 5].mean() if cx + freq + 5 < w else magnitude[cy, cx + freq - 5:cx + freq - 2].mean()
            peaks.append(peak_val / (baseline + 1e-10))

    periodic_pattern = np.mean(peaks) if peaks else 0

    # 2. 特定の色遷移パターン
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float64)
    l = lab[:,:,0]

    # エッジでの色遷移の急峻さ
    edges = cv2.Canny(img_gray, 100, 200)
    if edges.sum() > 0:
        sobelx = cv2.Sobel(l, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(l, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(sobelx**2 + sobely**2)
        edge_transition = gradient[edges > 0].mean()
    else:
        edge_transition = 0

    # 3. 特定の彩度分布
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    s = hsv[:,:,1].astype(np.float64)

    # 高彩度ピクセルの割合
    high_saturation_ratio = (s > 200).mean()

    # 4. 特定の輝度クラスタリング
    l_hist, _ = np.histogram(l.ravel(), bins=32, range=(0, 256))
    l_hist = l_hist / l_hist.sum()

    # ピーク数（多峰性）
    from scipy.signal import find_peaks
    peaks_idx, _ = find_peaks(l_hist, height=0.02)
    n_luminance_peaks = len(peaks_idx)

    return {
        'unique_periodic_pattern': periodic_pattern,
        'unique_edge_transition': edge_transition,
        'unique_high_sat_ratio': high_saturation_ratio,
        'unique_luminance_peaks': n_luminance_peaks,
    }


def extract_all_features(img_path):
    """全特徴量の抽出"""
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            return None

        # リサイズ（統一）
        img = cv2.resize(img, (512, 512))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        features = {}
        features.update(analyze_frequency_spectrum(img_gray))
        features.update(analyze_color_distribution(img_rgb))
        features.update(analyze_noise_pattern(img_gray))
        features.update(analyze_edge_characteristics(img_gray))
        features.update(analyze_texture_pattern(img_gray))
        features.update(analyze_compression_artifacts(img_gray))
        features.update(analyze_unique_patterns(img_rgb, img_gray))

        return features
    except Exception as e:
        return None


def main():
    print("=" * 70)
    print("NovelAI アーティファクト分析")
    print("=" * 70)

    # カテゴリ設定
    categories = {
        'novelai': 'novelai_combined_ai',
        'illustrious': 'illustrious_ai',
        'pony': 'pony_ai',
        'sdxl': 'sdxl10_ai',
        'real': 'danbooru_real',
    }

    n_samples = 300

    # 各カテゴリの特徴抽出
    all_features = {}

    for name, category in categories.items():
        print(f"\n[{name}] 画像パス取得...")
        try:
            paths = load_image_paths(category, n_samples)
            print(f"  {len(paths)} samples")
        except Exception as e:
            print(f"  Error: {e}")
            continue

        print(f"[{name}] 特徴抽出中...")
        features_list = []
        for path in tqdm(paths, desc=name):
            feat = extract_all_features(path)
            if feat is not None:
                features_list.append(feat)

        if features_list:
            # 辞書のリストをまとめる
            feature_names = list(features_list[0].keys())
            all_features[name] = {
                fn: np.array([f[fn] for f in features_list])
                for fn in feature_names
            }
            print(f"  Extracted: {len(features_list)} images, {len(feature_names)} features")

    if 'novelai' not in all_features or 'real' not in all_features:
        print("Error: Required categories not found")
        return

    # 分析結果
    print("\n" + "=" * 70)
    print("NovelAI vs Real 比較 (Cohen's d)")
    print("=" * 70)

    feature_names = list(all_features['novelai'].keys())
    results = []

    for fn in feature_names:
        novelai_vals = all_features['novelai'][fn]
        real_vals = all_features['real'][fn]

        # Cohen's d
        pooled_std = np.sqrt((novelai_vals.std()**2 + real_vals.std()**2) / 2)
        if pooled_std > 1e-10:
            d = (novelai_vals.mean() - real_vals.mean()) / pooled_std
        else:
            d = 0

        results.append((fn, d, novelai_vals.mean(), real_vals.mean()))

    # |d|でソート
    results.sort(key=lambda x: abs(x[1]), reverse=True)

    print(f"\n{'Feature':<40} {'Cohen d':>10} {'NovelAI':>12} {'Real':>12}")
    print("-" * 76)

    for fn, d, nov_mean, real_mean in results[:30]:
        sign = '+' if d > 0 else ' '
        effect = '◎' if abs(d) > 0.8 else '○' if abs(d) > 0.5 else '△' if abs(d) > 0.3 else '✗'
        print(f"{fn:<40} {sign}{d:>8.3f} {effect}  {nov_mean:>10.4f}  {real_mean:>10.4f}")

    # NovelAI vs 他AIモデル比較
    print("\n" + "=" * 70)
    print("NovelAI vs 他AIモデル 比較 (Cohen's d)")
    print("=" * 70)

    for other_name in ['illustrious', 'pony', 'sdxl']:
        if other_name not in all_features:
            continue

        print(f"\n--- NovelAI vs {other_name.upper()} ---")

        other_results = []
        for fn in feature_names:
            novelai_vals = all_features['novelai'][fn]
            other_vals = all_features[other_name][fn]

            pooled_std = np.sqrt((novelai_vals.std()**2 + other_vals.std()**2) / 2)
            if pooled_std > 1e-10:
                d = (novelai_vals.mean() - other_vals.mean()) / pooled_std
            else:
                d = 0

            other_results.append((fn, d))

        other_results.sort(key=lambda x: abs(x[1]), reverse=True)

        print(f"{'Feature':<40} {'Cohen d':>10}")
        print("-" * 52)
        for fn, d in other_results[:10]:
            sign = '+' if d > 0 else ' '
            effect = '◎' if abs(d) > 0.8 else '○' if abs(d) > 0.5 else '△' if abs(d) > 0.3 else '✗'
            print(f"{fn:<40} {sign}{d:>8.3f} {effect}")

    # NovelAI固有の特徴（Real差が大きく、他AI差が小さい）
    print("\n" + "=" * 70)
    print("NovelAI固有の特徴候補")
    print("（Real差が大きく、他AIとの差が小さいもの）")
    print("=" * 70)

    # Real差のdictを作成
    real_d = {fn: d for fn, d, _, _ in results}

    # 各特徴について、AI間の平均|d|を計算
    ai_avg_d = {}
    for fn in feature_names:
        ai_diffs = []
        for other_name in ['illustrious', 'pony', 'sdxl']:
            if other_name not in all_features:
                continue
            novelai_vals = all_features['novelai'][fn]
            other_vals = all_features[other_name][fn]
            pooled_std = np.sqrt((novelai_vals.std()**2 + other_vals.std()**2) / 2)
            if pooled_std > 1e-10:
                d = (novelai_vals.mean() - other_vals.mean()) / pooled_std
                ai_diffs.append(abs(d))
        ai_avg_d[fn] = np.mean(ai_diffs) if ai_diffs else 0

    # Real差が大きく(|d|>0.3)、AI間差が小さい(|d|<0.3)もの = AI共通の特徴
    # Real差が大きく(|d|>0.3)、AI間差も大きい(|d|>0.3)もの = NovelAI固有の特徴

    print(f"\n{'Feature':<40} {'vs Real':>10} {'vs OtherAI':>12} {'Type':>15}")
    print("-" * 80)

    for fn in feature_names:
        rd = real_d[fn]
        ad = ai_avg_d[fn]

        if abs(rd) < 0.3:
            continue

        if ad < 0.2:
            ftype = "AI共通"
        elif ad < 0.4:
            ftype = "やや固有"
        else:
            ftype = "NovelAI固有"

        print(f"{fn:<40} {rd:>+10.3f} {ad:>12.3f} {ftype:>15}")


if __name__ == "__main__":
    main()
