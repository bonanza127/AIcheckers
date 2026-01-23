#!/usr/bin/env python3
"""
Tier3: 新アプローチの特徴量テスト
- ピクセル差分の高次統計
- ノイズ特性
- 局所パターンの規則性
- 周波数の方向別分析
"""
import sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

EMBEDDINGS_DIR = Path("/home/techne/aicheckers/embeddings")
N_SAMPLES = 500


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


# ========== アプローチ1: ピクセル差分の高次統計 ==========

def pixel_diff_kurtosis(gray):
    """ピクセル差分の尖度 - AI生成は差分分布が正規分布から外れる"""
    diff_h = np.diff(gray.astype(np.float64), axis=1).ravel()
    diff_v = np.diff(gray.astype(np.float64), axis=0).ravel()
    all_diff = np.concatenate([diff_h, diff_v])
    if len(all_diff) < 100:
        return 0.0
    mean = all_diff.mean()
    std = all_diff.std()
    if std < 1e-6:
        return 0.0
    kurtosis = ((all_diff - mean) ** 4).mean() / (std ** 4) - 3
    return float(kurtosis)


def pixel_diff_skewness(gray):
    """ピクセル差分の歪度"""
    diff_h = np.diff(gray.astype(np.float64), axis=1).ravel()
    diff_v = np.diff(gray.astype(np.float64), axis=0).ravel()
    all_diff = np.concatenate([diff_h, diff_v])
    if len(all_diff) < 100:
        return 0.0
    mean = all_diff.mean()
    std = all_diff.std()
    if std < 1e-6:
        return 0.0
    skewness = ((all_diff - mean) ** 3).mean() / (std ** 3)
    return float(skewness)


def small_diff_ratio(gray, threshold=3):
    """小さい差分の割合 - AI生成は滑らかすぎて小差分が多い"""
    diff_h = np.abs(np.diff(gray.astype(np.float64), axis=1)).ravel()
    diff_v = np.abs(np.diff(gray.astype(np.float64), axis=0)).ravel()
    all_diff = np.concatenate([diff_h, diff_v])
    return float((all_diff < threshold).mean())


def large_diff_ratio(gray, threshold=30):
    """大きい差分の割合 - エッジの量"""
    diff_h = np.abs(np.diff(gray.astype(np.float64), axis=1)).ravel()
    diff_v = np.abs(np.diff(gray.astype(np.float64), axis=0)).ravel()
    all_diff = np.concatenate([diff_h, diff_v])
    return float((all_diff > threshold).mean())


# ========== アプローチ2: ノイズの統計的特性 ==========

def noise_kurtosis(gray):
    """高周波ノイズの尖度"""
    low = cv2.GaussianBlur(gray.astype(np.float64), (5, 5), 1.5)
    noise = gray.astype(np.float64) - low
    vals = noise.ravel()
    mean = vals.mean()
    std = vals.std()
    if std < 1e-6:
        return 0.0
    kurtosis = ((vals - mean) ** 4).mean() / (std ** 4) - 3
    return float(kurtosis)


def noise_spatial_correlation(gray):
    """ノイズの空間的相関 - AI生成のノイズは相関が低い"""
    low = cv2.GaussianBlur(gray.astype(np.float64), (5, 5), 1.5)
    noise = gray.astype(np.float64) - low
    h, w = noise.shape
    if h < 4 or w < 4:
        return 0.0
    noise_shifted = np.roll(noise, 1, axis=1)
    corr = np.corrcoef(noise[:, 1:].ravel(), noise_shifted[:, 1:].ravel())[0, 1]
    if np.isnan(corr):
        return 0.0
    return float(corr)


def noise_uniformity(gray, tiles=4):
    """ノイズの空間的均一性 - AI生成は均一すぎる"""
    low = cv2.GaussianBlur(gray.astype(np.float64), (5, 5), 1.5)
    noise = gray.astype(np.float64) - low
    h, w = noise.shape
    tile_h, tile_w = h // tiles, w // tiles
    if tile_h < 16 or tile_w < 16:
        return 0.0
    stds = []
    for i in range(tiles):
        for j in range(tiles):
            tile = noise[i*tile_h:(i+1)*tile_h, j*tile_w:(j+1)*tile_w]
            stds.append(tile.std())
    stds = np.array(stds)
    if stds.mean() < 1e-6:
        return 0.0
    cv = np.std(stds) / (np.mean(stds) + 1e-6)
    return float(1.0 / (1.0 + cv))


# ========== アプローチ3: 局所パターンの規則性 ==========

def patch_similarity_mean(gray, patch_size=16, n_patches=25):
    """ランダムパッチ間の類似度平均 - AI生成は類似度が高い"""
    h, w = gray.shape
    if h < patch_size * 2 or w < patch_size * 2:
        return 0.0

    np.random.seed(42)
    patches = []
    for _ in range(n_patches):
        y = np.random.randint(0, h - patch_size)
        x = np.random.randint(0, w - patch_size)
        patch = gray[y:y+patch_size, x:x+patch_size].astype(np.float64).ravel()
        patch = (patch - patch.mean()) / (patch.std() + 1e-6)
        patches.append(patch)

    similarities = []
    for i in range(len(patches)):
        for j in range(i + 1, len(patches)):
            sim = np.dot(patches[i], patches[j]) / len(patches[i])
            similarities.append(sim)

    return float(np.mean(similarities)) if similarities else 0.0


def local_variance_entropy(gray, tiles=8):
    """局所分散のエントロピー - AI生成は分散分布が偏る"""
    h, w = gray.shape
    tile_h, tile_w = h // tiles, w // tiles
    if tile_h < 8 or tile_w < 8:
        return 0.0
    variances = []
    for i in range(tiles):
        for j in range(tiles):
            tile = gray[i*tile_h:(i+1)*tile_h, j*tile_w:(j+1)*tile_w]
            variances.append(tile.var())
    variances = np.array(variances)
    if variances.max() < 1e-6:
        return 0.0
    # 正規化してヒストグラム
    variances_norm = variances / (variances.max() + 1e-10)
    hist, _ = np.histogram(variances_norm, bins=16, range=(0, 1))
    hist = hist.astype(np.float64) / (hist.sum() + 1e-10)
    entropy = -np.sum(hist * np.log2(hist + 1e-10))
    return float(entropy / np.log2(16))


# ========== アプローチ4: 周波数の方向別分析 ==========

def horizontal_vertical_freq_ratio(gray):
    """水平/垂直周波数比 - AI生成は方向バイアスがある"""
    f = np.fft.fft2(gray.astype(np.float64))
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)
    h, w = mag.shape
    cy, cx = h // 2, w // 2

    # 水平方向（中央の横帯）
    band_h = 5
    horiz = mag[cy-band_h:cy+band_h, :].sum()
    # 垂直方向（中央の縦帯）
    vert = mag[:, cx-band_h:cx+band_h].sum()

    if horiz + vert < 1e-6:
        return 0.0
    return float(horiz / (vert + 1e-6))


def diagonal_freq_strength(gray):
    """対角線方向の周波数強度"""
    f = np.fft.fft2(gray.astype(np.float64))
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)
    h, w = mag.shape
    cy, cx = h // 2, w // 2

    # 対角線マスク
    y, x = np.ogrid[:h, :w]
    diag1 = np.abs((x - cx) - (y - cy)) < 5
    diag2 = np.abs((x - cx) + (y - cy) - h) < 5
    diag_mask = diag1 | diag2

    diag_power = mag[diag_mask].mean()
    total_power = mag.mean()

    if total_power < 1e-6:
        return 0.0
    return float(diag_power / total_power)


# ========== アプローチ5: 色空間の詳細分析 ==========

def ab_plane_spread(img_rgb):
    """Lab色空間のa*b*平面での広がり"""
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    a = lab[:, :, 1].astype(np.float64).ravel() - 128
    b = lab[:, :, 2].astype(np.float64).ravel() - 128
    # 標準偏差の幾何平均
    std_a = np.std(a)
    std_b = np.std(b)
    return float(np.sqrt(std_a * std_b))


def chroma_luminance_correlation(img_rgb):
    """彩度-明度の相関"""
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l = lab[:, :, 0].astype(np.float64).ravel()
    a = lab[:, :, 1].astype(np.float64) - 128
    b = lab[:, :, 2].astype(np.float64) - 128
    chroma = np.sqrt(a**2 + b**2).ravel()
    corr = np.corrcoef(l, chroma)[0, 1]
    if np.isnan(corr):
        return 0.0
    return float(corr)


def color_channel_independence(img_rgb):
    """RGB各チャンネルの独立性"""
    r = img_rgb[:, :, 0].astype(np.float64).ravel()
    g = img_rgb[:, :, 1].astype(np.float64).ravel()
    b = img_rgb[:, :, 2].astype(np.float64).ravel()

    corr_rg = np.abs(np.corrcoef(r, g)[0, 1])
    corr_rb = np.abs(np.corrcoef(r, b)[0, 1])
    corr_gb = np.abs(np.corrcoef(g, b)[0, 1])

    if np.isnan(corr_rg) or np.isnan(corr_rb) or np.isnan(corr_gb):
        return 0.0
    # 独立性 = 1 - 平均相関
    return float(1.0 - (corr_rg + corr_rb + corr_gb) / 3)


# ========== 特徴量リスト ==========

FEATURE_FUNCS = {
    # アプローチ1: ピクセル差分
    "pixel_diff_kurtosis": lambda rgb, gray: pixel_diff_kurtosis(gray),
    "pixel_diff_skewness": lambda rgb, gray: pixel_diff_skewness(gray),
    "small_diff_ratio": lambda rgb, gray: small_diff_ratio(gray),
    "large_diff_ratio": lambda rgb, gray: large_diff_ratio(gray),
    # アプローチ2: ノイズ特性
    "noise_kurtosis": lambda rgb, gray: noise_kurtosis(gray),
    "noise_spatial_correlation": lambda rgb, gray: noise_spatial_correlation(gray),
    "noise_uniformity": lambda rgb, gray: noise_uniformity(gray),
    # アプローチ3: 局所パターン
    "patch_similarity_mean": lambda rgb, gray: patch_similarity_mean(gray),
    "local_variance_entropy": lambda rgb, gray: local_variance_entropy(gray),
    # アプローチ4: 周波数方向
    "horizontal_vertical_freq_ratio": lambda rgb, gray: horizontal_vertical_freq_ratio(gray),
    "diagonal_freq_strength": lambda rgb, gray: diagonal_freq_strength(gray),
    # アプローチ5: 色空間
    "ab_plane_spread": lambda rgb, gray: ab_plane_spread(rgb),
    "chroma_luminance_correlation": lambda rgb, gray: chroma_luminance_correlation(rgb),
    "color_channel_independence": lambda rgb, gray: color_channel_independence(rgb),
}


def extract_all_features(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    feats = []
    for name, func in FEATURE_FUNCS.items():
        try:
            val = func(img_rgb, gray)
        except Exception:
            val = 0.0
        feats.append(val)
    return np.array(feats, dtype=np.float32)


def get_image_paths(category, n_samples):
    files_path = EMBEDDINGS_DIR / f"{category}_files.txt"
    if files_path.exists():
        paths = [Path(l.strip()) for l in files_path.read_text().splitlines() if l.strip()]
    else:
        return []
    np.random.seed(42)
    if len(paths) > n_samples:
        indices = np.random.choice(len(paths), n_samples, replace=False)
        paths = [paths[i] for i in indices]
    return paths


def compute_cohens_d(ai_vals, real_vals):
    n1, n2 = len(ai_vals), len(real_vals)
    var1, var2 = np.var(ai_vals, ddof=1), np.var(real_vals, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std < 1e-10:
        return 0.0
    return float((np.mean(ai_vals) - np.mean(real_vals)) / pooled_std)


def main():
    print("=" * 60)
    print("Tier3 特徴量テスト（新アプローチ）")
    print("=" * 60)

    # 画像パス取得
    print("\n[1/4] 画像パス取得...")
    ai_paths = get_image_paths("illustrious_ai", N_SAMPLES)
    real_paths = get_image_paths("danbooru_real", N_SAMPLES)
    print(f"  AI: {len(ai_paths)} samples")
    print(f"  Real: {len(real_paths)} samples")

    if len(ai_paths) < 100 or len(real_paths) < 100:
        print("サンプル不足")
        return

    # 特徴量抽出
    print("\n[2/4] 特徴量抽出...")
    ai_features = []
    for p in tqdm(ai_paths, desc="AI"):
        try:
            img = load_image(str(p))
            ai_features.append(extract_all_features(img))
        except Exception:
            ai_features.append(np.zeros(len(FEATURE_FUNCS), dtype=np.float32))
    ai_features = np.array(ai_features)

    real_features = []
    for p in tqdm(real_paths, desc="Real"):
        try:
            img = load_image(str(p))
            real_features.append(extract_all_features(img))
        except Exception:
            real_features.append(np.zeros(len(FEATURE_FUNCS), dtype=np.float32))
    real_features = np.array(real_features)

    feature_names = list(FEATURE_FUNCS.keys())

    # Cohen's d
    print("\n[3/4] Cohen's d 計算...")
    print("\n" + "=" * 70)
    print(f"{'Feature':<35} {'Cohen d':>10} {'AI mean':>12} {'Real mean':>12}")
    print("=" * 70)

    cohens_d = []
    for i, name in enumerate(feature_names):
        d = compute_cohens_d(ai_features[:, i], real_features[:, i])
        cohens_d.append(d)
        ai_mean = np.mean(ai_features[:, i])
        real_mean = np.mean(real_features[:, i])
        sign = "+" if d > 0 else ""
        effect = "◎" if abs(d) > 0.8 else ("○" if abs(d) > 0.5 else ("△" if abs(d) > 0.3 else "✗"))
        print(f"{name:<35} {sign}{d:>8.3f} {effect} {ai_mean:>10.4f} {real_mean:>10.4f}")

    # ランキング
    print("\n" + "-" * 70)
    print("Cohen's d ランキング (|d| 降順):")
    print("-" * 70)
    sorted_idx = np.argsort(-np.abs(cohens_d))
    for rank, idx in enumerate(sorted_idx, 1):
        d = cohens_d[idx]
        name = feature_names[idx]
        sign = "+" if d > 0 else ""
        effect = "◎" if abs(d) > 0.8 else ("○" if abs(d) > 0.5 else ("△" if abs(d) > 0.3 else "✗"))
        print(f"  {rank:2d}. {name:<35} {sign}{d:>7.3f} {effect}")

    # 特徴量間相関
    print("\n[4/4] 特徴量間相関...")
    all_features = np.vstack([ai_features, real_features])
    corr_matrix = np.corrcoef(all_features.T)

    print("\n高相関ペア (|r| > 0.6):")
    print("-" * 60)
    high_corr_pairs = []
    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            r = corr_matrix[i, j]
            if abs(r) > 0.6:
                high_corr_pairs.append((feature_names[i], feature_names[j], r))

    if high_corr_pairs:
        for name1, name2, r in sorted(high_corr_pairs, key=lambda x: -abs(x[2])):
            print(f"  {name1:<30} ↔ {name2:<25} r={r:.3f}")
    else:
        print("  なし（独立性が高い）")

    # 簡易ロジスティック回帰
    print("\n[5/5] 簡易ロジスティック回帰...")
    labels = np.array([1] * len(ai_features) + [0] * len(real_features))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(all_features)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_scaled, labels)

    print(f"\n{'Feature':<35} {'Weight':>10} {'|d|':>8}")
    print("-" * 60)
    weights = clf.coef_[0]
    sorted_weight_idx = np.argsort(-np.abs(weights))
    for idx in sorted_weight_idx:
        w = weights[idx]
        d = cohens_d[idx]
        name = feature_names[idx]
        sign = "+" if w > 0 else ""
        print(f"{name:<35} {sign}{w:>8.3f} {abs(d):>8.3f}")

    scores = cross_val_score(clf, X_scaled, labels, cv=5, scoring="roc_auc")
    print(f"\n5-fold CV ROC-AUC: {scores.mean():.4f} (+/- {scores.std():.4f})")

    # サマリー
    print("\n" + "=" * 70)
    print("推奨特徴量 (|d| > 0.3)")
    print("=" * 70)
    for i, name in enumerate(feature_names):
        d = cohens_d[i]
        if abs(d) > 0.3:
            effect = "◎" if abs(d) > 0.8 else ("○" if abs(d) > 0.5 else "△")
            print(f"  {effect} {name:<32} d={d:+.3f}  w={weights[i]:+.3f}")


if __name__ == "__main__":
    main()
