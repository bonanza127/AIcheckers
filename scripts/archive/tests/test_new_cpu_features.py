#!/usr/bin/env python3
"""
新CPU特徴量の一括テスト
- Cohen's d
- DINO CLSとの相関
- 特徴量間相関
- 簡易ロジスティック回帰重み
"""
import sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

DATA_ROOT = Path("/home/techne/aicheckers/data")
ANIMEDL_ROOT = DATA_ROOT / "animedl2m_dataset_release"
EMBEDDINGS_DIR = Path("/home/techne/aicheckers/embeddings")

# サンプル数
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


# ========== Tier A: 高効果期待 ==========

def line_width_variance(gray):
    """エッジの太さのばらつき"""
    edges = cv2.Canny(gray, 50, 150)
    if edges.sum() == 0:
        return 0.0
    dist = cv2.distanceTransform((edges == 0).astype(np.uint8), cv2.DIST_L2, 5)
    edge_points = dist[edges > 0]
    if len(edge_points) < 10:
        return 0.0
    return float(np.std(edge_points) / (np.mean(edge_points) + 1e-6))


def hue_clustering_coefficient(img_rgb):
    """色相の集中度"""
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0].ravel()
    sat = hsv[:, :, 1].ravel()
    valid = sat > 30
    if valid.sum() < 100:
        return 0.0
    hue_valid = hue[valid]
    hist, _ = np.histogram(hue_valid, bins=36, range=(0, 180))
    hist = hist.astype(np.float64) / (hist.sum() + 1e-10)
    top3 = np.sort(hist)[-3:].sum()
    return float(top3)


def gradient_smoothness_score(gray):
    """グラデーションの滑らかさ"""
    grad1 = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)) + \
            np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3))
    grad2 = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
    p25, p75 = np.percentile(grad1, [25, 75])
    gradient_mask = (grad1 > p25) & (grad1 < p75)
    if gradient_mask.sum() < 100:
        return 0.0
    smoothness = 1.0 / (grad2[gradient_mask].mean() + 1e-6)
    return float(np.clip(smoothness / 10, 0, 1))


def local_texture_variance_ratio(gray, tiles=4):
    """パッチ間テクスチャ分散比"""
    h, w = gray.shape
    tile_h, tile_w = h // tiles, w // tiles
    if tile_h < 16 or tile_w < 16:
        return 0.0
    local_vars = []
    for i in range(tiles):
        for j in range(tiles):
            tile = gray[i*tile_h:(i+1)*tile_h, j*tile_w:(j+1)*tile_w]
            high_freq = cv2.Laplacian(tile, cv2.CV_64F)
            local_vars.append(high_freq.var())
    local_vars = np.array(local_vars)
    if local_vars.mean() < 1e-6:
        return 0.0
    return float(np.std(local_vars) / (np.mean(local_vars) + 1e-6))


def stroke_direction_entropy(gray):
    """線の方向エントロピー"""
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    edge_mask = magnitude > np.percentile(magnitude, 75)
    if edge_mask.sum() < 100:
        return 0.0
    angles = np.arctan2(sobely, sobelx) * 180 / np.pi
    angles = angles[edge_mask]
    angles = np.mod(angles, 180)
    hist, _ = np.histogram(angles, bins=36, range=(0, 180))
    hist = hist.astype(np.float64) / (hist.sum() + 1e-10)
    entropy = -np.sum(hist * np.log2(hist + 1e-10))
    return float(entropy / np.log2(36))


# ========== Tier B: 中効果 ==========

def saturation_spatial_autocorr(img_rgb):
    """彩度の空間的自己相関"""
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1].astype(np.float64)
    h, w = sat.shape
    if h < 4 or w < 4:
        return 0.0
    sat_shifted_h = np.roll(sat, 1, axis=0)
    sat_shifted_w = np.roll(sat, 1, axis=1)
    corr_h = np.corrcoef(sat[1:, :].ravel(), sat_shifted_h[1:, :].ravel())[0, 1]
    corr_w = np.corrcoef(sat[:, 1:].ravel(), sat_shifted_w[:, 1:].ravel())[0, 1]
    if np.isnan(corr_h) or np.isnan(corr_w):
        return 0.0
    return float((corr_h + corr_w) / 2)


def detail_density_gini(gray, tiles=8):
    """細部密度のジニ係数"""
    h, w = gray.shape
    tile_h, tile_w = h // tiles, w // tiles
    if tile_h < 8 or tile_w < 8:
        return 0.0
    densities = []
    for i in range(tiles):
        for j in range(tiles):
            tile = gray[i*tile_h:(i+1)*tile_h, j*tile_w:(j+1)*tile_w]
            edges = cv2.Canny(tile, 50, 150)
            densities.append(edges.sum() / (edges.size + 1e-10))
    densities = np.array(densities)
    if densities.sum() < 1e-10:
        return 0.0
    sorted_d = np.sort(densities)
    n = len(sorted_d)
    cum = np.cumsum(sorted_d)
    gini = (n + 1 - 2 * np.sum(cum) / (cum[-1] + 1e-10)) / n
    return float(gini)


def anti_aliasing_sharpness(gray):
    """アンチエイリアシング境界の鮮明度"""
    edges = cv2.Canny(gray, 100, 200)
    if edges.sum() == 0:
        return 0.0
    # エッジ周辺の勾配変化率
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(sobelx**2 + sobely**2)
    # エッジ上の勾配
    edge_gradient = gradient[edges > 0]
    if len(edge_gradient) < 10:
        return 0.0
    # 勾配の標準偏差/平均 = シャープネスの変動
    return float(np.std(edge_gradient) / (np.mean(edge_gradient) + 1e-6))


def color_transition_histogram(img_rgb):
    """色遷移パターンの分布"""
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l_channel = lab[:, :, 0].astype(np.float64)
    # 水平・垂直の色遷移
    diff_h = np.abs(np.diff(l_channel, axis=1)).ravel()
    diff_v = np.abs(np.diff(l_channel, axis=0)).ravel()
    all_diff = np.concatenate([diff_h, diff_v])
    # 遷移量のヒストグラムエントロピー
    hist, _ = np.histogram(all_diff, bins=32, range=(0, 64))
    hist = hist.astype(np.float64) / (hist.sum() + 1e-10)
    entropy = -np.sum(hist * np.log2(hist + 1e-10))
    return float(entropy / np.log2(32))


def dct_block_variance(gray):
    """DCT係数のブロック間分散"""
    h, w = gray.shape
    block_size = 8
    n_blocks_h = h // block_size
    n_blocks_w = w // block_size
    if n_blocks_h < 2 or n_blocks_w < 2:
        return 0.0

    dc_coeffs = []
    ac_energy = []
    for i in range(n_blocks_h):
        for j in range(n_blocks_w):
            block = gray[i*block_size:(i+1)*block_size,
                        j*block_size:(j+1)*block_size].astype(np.float64)
            dct = cv2.dct(block)
            dc_coeffs.append(dct[0, 0])
            ac_energy.append(np.sum(np.abs(dct[1:, :])) + np.sum(np.abs(dct[0, 1:])))

    dc_var = np.var(dc_coeffs)
    ac_var = np.var(ac_energy)
    return float(ac_var / (dc_var + 1e-6))


# ========== Tier C: 実験的 ==========

def high_frequency_coherence(gray):
    """高周波成分の空間的一貫性"""
    # 高周波を抽出
    low = cv2.GaussianBlur(gray.astype(np.float64), (11, 11), 3)
    high = gray.astype(np.float64) - low
    h, w = high.shape
    tiles = 4
    tile_h, tile_w = h // tiles, w // tiles
    if tile_h < 16 or tile_w < 16:
        return 0.0

    tile_stds = []
    for i in range(tiles):
        for j in range(tiles):
            tile = high[i*tile_h:(i+1)*tile_h, j*tile_w:(j+1)*tile_w]
            tile_stds.append(tile.std())

    tile_stds = np.array(tile_stds)
    if tile_stds.mean() < 1e-6:
        return 0.0
    # 一貫性 = 1 - 変動係数
    cv = np.std(tile_stds) / (np.mean(tile_stds) + 1e-6)
    return float(1.0 / (1.0 + cv))


def shadow_highlight_ratio(gray):
    """陰影のコントラスト比"""
    vals = gray.ravel()
    shadow_thresh = np.percentile(vals, 10)
    highlight_thresh = np.percentile(vals, 90)
    shadow_mean = vals[vals <= shadow_thresh].mean() if (vals <= shadow_thresh).sum() > 0 else 0
    highlight_mean = vals[vals >= highlight_thresh].mean() if (vals >= highlight_thresh).sum() > 0 else 255
    return float((highlight_mean - shadow_mean) / 255.0)


def flat_region_boundary_sharpness(gray):
    """平坦領域境界の鮮明度"""
    # 平坦領域を検出
    grad = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
    flat_mask = grad < np.percentile(grad, 30)

    # 平坦領域の境界
    kernel = np.ones((3, 3), dtype=np.uint8)
    dilated = cv2.dilate(flat_mask.astype(np.uint8), kernel, iterations=1)
    boundary = dilated - flat_mask.astype(np.uint8)

    if boundary.sum() == 0:
        return 0.0

    # 境界での勾配
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(sobelx**2 + sobely**2)

    boundary_grad = gradient[boundary > 0]
    if len(boundary_grad) < 10:
        return 0.0
    return float(boundary_grad.mean() / (gradient.mean() + 1e-6))


# ========== 特徴量リスト ==========

FEATURE_FUNCS = {
    # Tier A
    "line_width_variance": lambda rgb, gray: line_width_variance(gray),
    "hue_clustering_coefficient": lambda rgb, gray: hue_clustering_coefficient(rgb),
    "gradient_smoothness_score": lambda rgb, gray: gradient_smoothness_score(gray),
    "local_texture_variance_ratio": lambda rgb, gray: local_texture_variance_ratio(gray),
    "stroke_direction_entropy": lambda rgb, gray: stroke_direction_entropy(gray),
    # Tier B
    "saturation_spatial_autocorr": lambda rgb, gray: saturation_spatial_autocorr(rgb),
    "detail_density_gini": lambda rgb, gray: detail_density_gini(gray),
    "anti_aliasing_sharpness": lambda rgb, gray: anti_aliasing_sharpness(gray),
    "color_transition_histogram": lambda rgb, gray: color_transition_histogram(rgb),
    "dct_block_variance": lambda rgb, gray: dct_block_variance(gray),
    # Tier C
    "high_frequency_coherence": lambda rgb, gray: high_frequency_coherence(gray),
    "shadow_highlight_ratio": lambda rgb, gray: shadow_highlight_ratio(gray),
    "flat_region_boundary_sharpness": lambda rgb, gray: flat_region_boundary_sharpness(gray),
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
    """カテゴリの画像パスを取得"""
    files_path = EMBEDDINGS_DIR / f"{category}_files.txt"
    if files_path.exists():
        paths = [Path(l.strip()) for l in files_path.read_text().splitlines() if l.strip()]
    else:
        if category == "danbooru_real":
            img_dir = ANIMEDL_ROOT / "real_images/images"
        elif category == "illustrious_ai":
            img_dir = ANIMEDL_ROOT / "civitai_subset/image/Illustrious"
        else:
            return []
        exts = {".jpg", ".jpeg", ".png", ".webp"}
        paths = sorted([p for p in img_dir.rglob("*") if p.suffix.lower() in exts])

    # ランダムサンプリング
    np.random.seed(42)
    if len(paths) > n_samples:
        indices = np.random.choice(len(paths), n_samples, replace=False)
        paths = [paths[i] for i in indices]
    return paths


def compute_cohens_d(ai_vals, real_vals):
    """Cohen's d を計算"""
    n1, n2 = len(ai_vals), len(real_vals)
    var1, var2 = np.var(ai_vals, ddof=1), np.var(real_vals, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std < 1e-10:
        return 0.0
    return float((np.mean(ai_vals) - np.mean(real_vals)) / pooled_std)


def main():
    print("=" * 60)
    print("新CPU特徴量テスト")
    print("=" * 60)

    # 画像パス取得
    print("\n[1/5] 画像パス取得...")
    ai_paths = get_image_paths("illustrious_ai", N_SAMPLES)
    real_paths = get_image_paths("danbooru_real", N_SAMPLES)
    print(f"  AI: {len(ai_paths)} samples")
    print(f"  Real: {len(real_paths)} samples")

    if len(ai_paths) < 100 or len(real_paths) < 100:
        print("サンプル不足")
        return

    # 特徴量抽出
    print("\n[2/5] 特徴量抽出...")
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
    print("\n[3/5] Cohen's d 計算...")
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
        tier = "A" if i < 5 else ("B" if i < 10 else "C")
        effect = "◎" if abs(d) > 0.8 else ("○" if abs(d) > 0.5 else ("△" if abs(d) > 0.3 else "✗"))
        print(f"[{tier}] {name:<32} {sign}{d:>8.3f} {effect} {ai_mean:>10.4f} {real_mean:>10.4f}")

    # |d| でソート
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

    # DINO CLS との相関
    print("\n[4/5] DINO CLS との相関...")
    ai_cls_path = EMBEDDINGS_DIR / "illustrious_ai.npy"
    real_cls_path = EMBEDDINGS_DIR / "danbooru_real.npy"

    if ai_cls_path.exists() and real_cls_path.exists():
        ai_cls = np.load(ai_cls_path)[:N_SAMPLES]
        real_cls = np.load(real_cls_path)[:N_SAMPLES]

        # CLS の主成分（平均）との相関
        all_cls = np.vstack([ai_cls, real_cls])
        cls_mean = all_cls.mean(axis=1)  # 768次元の平均

        all_features = np.vstack([ai_features, real_features])

        print(f"\n{'Feature':<35} {'CLS corr':>10}")
        print("-" * 50)
        cls_corrs = []
        for i, name in enumerate(feature_names):
            corr = np.corrcoef(all_features[:, i], cls_mean)[0, 1]
            if np.isnan(corr):
                corr = 0.0
            cls_corrs.append(corr)
            print(f"{name:<35} {corr:>10.3f}")
    else:
        print("  DINO CLS が見つかりません")
        cls_corrs = [0.0] * len(feature_names)

    # 特徴量間相関
    print("\n[5/5] 特徴量間相関...")
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
            print(f"  {name1:<30} ↔ {name2:<30} r={r:.3f}")
    else:
        print("  なし（独立性が高い）")

    # 簡易ロジスティック回帰
    print("\n[6/6] 簡易ロジスティック回帰...")
    labels = np.array([1] * len(ai_features) + [0] * len(real_features))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(all_features)

    # NaN/Inf を0に置換
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

    # 精度
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(clf, X_scaled, labels, cv=5, scoring="roc_auc")
    print(f"\n5-fold CV ROC-AUC: {scores.mean():.4f} (+/- {scores.std():.4f})")

    # サマリー
    print("\n" + "=" * 70)
    print("サマリー: 推奨特徴量 (|d| > 0.3 かつ CLS相関 < 0.5)")
    print("=" * 70)
    recommended = []
    for i, name in enumerate(feature_names):
        d = cohens_d[i]
        cls_corr = cls_corrs[i] if cls_corrs else 0.0
        if abs(d) > 0.3 and abs(cls_corr) < 0.5:
            recommended.append((name, d, cls_corr, weights[i]))

    recommended.sort(key=lambda x: -abs(x[1]))
    for name, d, cls_corr, w in recommended:
        effect = "◎" if abs(d) > 0.8 else ("○" if abs(d) > 0.5 else "△")
        print(f"  {effect} {name:<32} d={d:+.3f}  cls_r={cls_corr:.3f}  w={w:+.3f}")


if __name__ == "__main__":
    main()
