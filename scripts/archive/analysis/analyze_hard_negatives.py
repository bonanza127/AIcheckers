#!/usr/bin/env python3
"""
Hard Negative分析スクリプト
- 誤検出（AI画像なのに低スコア）の原因を特定
- Tier0特徴量が効いているか確認
- 補強すべき特徴の発見
"""

import numpy as np
import cv2
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy import stats, ndimage
from scipy.signal import find_peaks
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

N_WORKERS = 8
DATA_ROOT = Path("/home/techne/aicheckers")


def load_image(path, target_size=512):
    img = cv2.imread(str(path))
    if img is None:
        return None
    img = cv2.resize(img, (target_size, target_size))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ===== Tier0特徴量 =====
def color_palette_entropy(img_rgb):
    quantized = (img_rgb // 8).astype(np.uint8)
    colors = quantized.reshape(-1, 3)
    color_ids = colors[:, 0].astype(np.int32) * 1024 + colors[:, 1] * 32 + colors[:, 2]
    _, counts = np.unique(color_ids, return_counts=True)
    probs = counts.astype(np.float64) / counts.sum()
    return float(-np.sum(probs * np.log2(probs + 1e-10)))

def luminance_mean(img_rgb):
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    return float(lab[:, :, 0].mean())

def color_hue_entropy(img_rgb):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h = hsv[:, :, 0].ravel()
    hist, _ = np.histogram(h, bins=36, range=(0, 180))
    hist = hist.astype(np.float64) / (hist.sum() + 1e-10)
    return float(-np.sum(hist * np.log2(hist + 1e-10)))

def saturation_mean(img_rgb):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    return float(hsv[:, :, 1].mean())

def luminance_peak_count(img_rgb):
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l = lab[:, :, 0].ravel()
    hist, _ = np.histogram(l, bins=32, range=(0, 256))
    hist_smooth = ndimage.gaussian_filter1d(hist.astype(np.float64), sigma=1)
    peaks, _ = find_peaks(hist_smooth, height=hist_smooth.max() * 0.02)
    return float(len(peaks))

# ===== 追加分析用特徴量 =====
def edge_density(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return float(edges.sum() / edges.size / 255)

def noise_std(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    smooth = cv2.GaussianBlur(gray, (5, 5), 0)
    noise = gray.astype(np.float64) - smooth.astype(np.float64)
    return float(noise.std())

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

def jpeg_quant_pattern(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
    hist_fft = np.abs(np.fft.fft(hist))
    return float(hist_fft[32] / (hist_fft[1:64].mean() + 1e-10))


FEATURE_FUNCS = {
    # Tier0
    'color_palette_entropy': color_palette_entropy,
    'luminance_mean': luminance_mean,
    'color_hue_entropy': color_hue_entropy,
    'saturation_mean': saturation_mean,
    'luminance_peak_count': luminance_peak_count,
    # 追加分析用
    'edge_density': edge_density,
    'noise_std': noise_std,
    'freq_high_ratio': freq_high_ratio,
    'texture_flat_ratio': texture_flat_ratio,
    'jpeg_quant_pattern': jpeg_quant_pattern,
}


def extract_features_single(path):
    try:
        img = load_image(path)
        if img is None:
            return None
        return {name: func(img) for name, func in FEATURE_FUNCS.items()}
    except Exception:
        return None


def main():
    print("=" * 80)
    print("Hard Negative分析")
    print("=" * 80)

    # Hard Negative画像のロード（直接ディレクトリから）
    print("\n[1/4] Hard Negativeデータロード...")
    hard_neg_dir = DATA_ROOT / "data/hard_negatives"
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    hard_neg_paths = [p for p in hard_neg_dir.glob("*") if p.suffix.lower() in exts]
    print(f"  Hard Negative（検出失敗AI画像）: {len(hard_neg_paths)}")

    # 正常に検出されたAI画像（比較用）
    print("\n[2/4] 正常検出AI画像ロード...")
    ai_files = DATA_ROOT / "embeddings/novelai_combined_ai_files.txt"
    with open(ai_files) as f:
        ai_paths = [line.strip() for line in f if line.strip()][:2000]
    print(f"  正常AI画像（NovelAI）: {len(ai_paths)}")

    # Real画像（比較用）
    real_files = DATA_ROOT / "embeddings/danbooru_real_files.txt"
    with open(real_files) as f:
        real_paths = [line.strip() for line in f if line.strip()][:2000]
    print(f"  Real画像: {len(real_paths)}")

    # 特徴抽出（並列）
    print("\n[3/4] 特徴抽出...")

    def extract_parallel(paths, desc):
        results = []
        with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
            futures = {executor.submit(extract_features_single, p): p for p in paths}
            for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
                result = future.result()
                if result is not None:
                    results.append(result)
        return results

    hard_neg_features = extract_parallel(hard_neg_paths, "HardNeg")
    ai_features = extract_parallel(ai_paths, "AI")
    real_features = extract_parallel(real_paths, "Real")

    print(f"\n  Extracted - HardNeg: {len(hard_neg_features)}, AI: {len(ai_features)}, Real: {len(real_features)}")

    # numpy配列に変換
    feature_names = list(FEATURE_FUNCS.keys())
    hard_neg_data = np.array([[f[fn] for fn in feature_names] for f in hard_neg_features])
    ai_data = np.array([[f[fn] for fn in feature_names] for f in ai_features])
    real_data = np.array([[f[fn] for fn in feature_names] for f in real_features])

    # 分析
    print("\n[4/4] 分析結果...")
    print("\n" + "=" * 80)
    print("特徴量比較")
    print("=" * 80)

    print(f"\n{'Feature':<25} {'HardNeg':>10} {'AI':>10} {'Real':>10} {'HN vs AI':>10} {'AI vs Real':>12}")
    print("-" * 80)

    results = []
    for i, fn in enumerate(feature_names):
        hn_mean = hard_neg_data[:, i].mean()
        ai_mean = ai_data[:, i].mean()
        real_mean = real_data[:, i].mean()

        # Cohen's d: HardNeg vs AI（正常検出）
        pooled_std = np.sqrt((hard_neg_data[:, i].std()**2 + ai_data[:, i].std()**2) / 2)
        hn_vs_ai = (hn_mean - ai_mean) / pooled_std if pooled_std > 1e-10 else 0

        # Cohen's d: AI vs Real
        pooled_std2 = np.sqrt((ai_data[:, i].std()**2 + real_data[:, i].std()**2) / 2)
        ai_vs_real = (ai_mean - real_mean) / pooled_std2 if pooled_std2 > 1e-10 else 0

        results.append((fn, hn_mean, ai_mean, real_mean, hn_vs_ai, ai_vs_real))

        # HardNegがRealに近いかどうかをチェック
        hn_closer_to_real = abs(hn_mean - real_mean) < abs(hn_mean - ai_mean)
        marker = "← Real寄り" if hn_closer_to_real else ""

        print(f"{fn:<25} {hn_mean:>10.4f} {ai_mean:>10.4f} {real_mean:>10.4f} {hn_vs_ai:>+10.3f} {ai_vs_real:>+10.3f}  {marker}")

    # サマリー
    print("\n" + "=" * 80)
    print("分析サマリー")
    print("=" * 80)

    print("\n【Hard NegativeがRealに近い特徴】（検出を困難にしている原因）")
    for fn, hn_m, ai_m, real_m, hn_vs_ai, ai_vs_real in results:
        if abs(hn_m - real_m) < abs(hn_m - ai_m) and abs(hn_vs_ai) > 0.3:
            print(f"  ⚠️  {fn:<25} HN={hn_m:.3f}, AI={ai_m:.3f}, Real={real_m:.3f}")
            print(f"      → Hard NegativeはこのAI特徴を持っていない")

    print("\n【Hard NegativeがAIに近い特徴】（検出に効いている）")
    for fn, hn_m, ai_m, real_m, hn_vs_ai, ai_vs_real in results:
        if abs(hn_m - ai_m) < abs(hn_m - real_m) and abs(ai_vs_real) > 0.3:
            print(f"  ✓  {fn:<25} HN={hn_m:.3f}, AI={ai_m:.3f}, Real={real_m:.3f}")

    # Tier0の効果
    print("\n【Tier0特徴のHard Negativeへの効果】")
    tier0_features = ['color_palette_entropy', 'luminance_mean', 'color_hue_entropy',
                      'saturation_mean', 'luminance_peak_count']
    for fn, hn_m, ai_m, real_m, hn_vs_ai, ai_vs_real in results:
        if fn in tier0_features:
            effectiveness = "効果あり" if abs(hn_m - ai_m) < abs(hn_m - real_m) else "効果薄"
            print(f"  {fn:<25} {effectiveness} (HN vs AI: {hn_vs_ai:+.3f})")


if __name__ == "__main__":
    main()
