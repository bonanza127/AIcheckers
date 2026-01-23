#!/usr/bin/env python3
"""
新候補特徴テスト:
- local_autocorr_isotropy: 局所自己相関の等方性
- effective_bit_depth: バンディング検出用の有効bit深度
"""
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
import random
from scipy import ndimage

N_SAMPLES = 150


def compute_local_autocorr_isotropy(img_gray):
    """
    局所自己相関の等方性を計算
    - AI画像はCNN/Attentionの方向バイアスで異方性が出る可能性
    - 人間絵は筆致のランダム性で等方的
    """
    img = img_gray.astype(np.float32)

    # 4方向の自己相関（1ピクセルシフト）
    h, w = img.shape

    # 水平方向
    corr_h = np.corrcoef(img[:, :-1].flatten(), img[:, 1:].flatten())[0, 1]

    # 垂直方向
    corr_v = np.corrcoef(img[:-1, :].flatten(), img[1:, :].flatten())[0, 1]

    # 対角方向（右下）
    corr_d1 = np.corrcoef(img[:-1, :-1].flatten(), img[1:, 1:].flatten())[0, 1]

    # 対角方向（左下）
    corr_d2 = np.corrcoef(img[:-1, 1:].flatten(), img[1:, :-1].flatten())[0, 1]

    # 等方性 = 4方向の相関の分散（低いほど等方的）
    correlations = [corr_h, corr_v, corr_d1, corr_d2]
    correlations = [c for c in correlations if not np.isnan(c)]

    if len(correlations) < 4:
        return 0.0

    isotropy_var = np.var(correlations)

    # 水平-垂直の差も計算（CNN特有のバイアス検出）
    hv_diff = abs(corr_h - corr_v)

    return isotropy_var, hv_diff


def compute_effective_bit_depth(img_gray):
    """
    有効bit深度を計算（バンディング検出）
    - AI画像は量子化アーティファクトでバンディングが出やすい
    - 実質的に使われているbit数を推定
    """
    # ヒストグラムを取得
    hist, _ = np.histogram(img_gray.flatten(), bins=256, range=(0, 256))

    # 使用されている値の数
    used_values = np.sum(hist > 0)

    # 有効bit深度 = log2(使用値数)
    effective_bits = np.log2(max(used_values, 1))

    # 隣接値の差分分布（バンディング検出）
    sorted_values = np.sort(img_gray.flatten())
    diffs = np.diff(sorted_values)

    # ゼロでない差分の最小値（量子化ステップ）
    nonzero_diffs = diffs[diffs > 0]
    if len(nonzero_diffs) > 0:
        min_step = np.percentile(nonzero_diffs, 10)  # 10%tile
        banding_score = min_step  # 大きいほどバンディング
    else:
        banding_score = 0

    return effective_bits, banding_score


def compute_gradient_histogram_entropy(img_gray):
    """
    勾配ヒストグラムのエントロピー
    - AI画像は勾配分布が偏る可能性
    """
    grad_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)

    # 勾配の大きさと方向
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    angle = np.arctan2(grad_y, grad_x)

    # 方向ヒストグラム（重み付き）
    hist, _ = np.histogram(angle.flatten(), bins=36, range=(-np.pi, np.pi),
                           weights=magnitude.flatten())

    hist = hist / (hist.sum() + 1e-10)
    hist = hist[hist > 0]

    entropy = -np.sum(hist * np.log2(hist + 1e-10))

    return entropy


def process_image(img_path):
    try:
        img = Image.open(img_path).convert('RGB')
        img = img.resize((512, 512), Image.LANCZOS)
        img_np = np.array(img)
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        # Local autocorrelation isotropy
        isotropy_var, hv_diff = compute_local_autocorr_isotropy(img_gray)

        # Effective bit depth
        effective_bits, banding_score = compute_effective_bit_depth(img_gray)

        # Gradient histogram entropy
        grad_entropy = compute_gradient_histogram_entropy(img_gray)

        return {
            'isotropy_var': isotropy_var,
            'hv_diff': hv_diff,
            'effective_bits': effective_bits,
            'banding_score': banding_score,
            'grad_entropy': grad_entropy,
        }
    except Exception as e:
        return None


def cohens_d(ai_vals, real_vals):
    ai_mean, ai_std = np.mean(ai_vals), np.std(ai_vals)
    real_mean, real_std = np.mean(real_vals), np.std(real_vals)
    pooled_std = np.sqrt((ai_std**2 + real_std**2) / 2)
    return abs(ai_mean - real_mean) / pooled_std if pooled_std > 0 else 0


def main():
    ai_dirs = {
        'novelai': Path("/home/techne/aicheckers/data/novelai_combined"),
        'illustrious': Path("/home/techne/aicheckers/data/animedl2m_dataset_release/civitai_subset/image/Illustrious"),
    }
    real_dir = Path("/home/techne/aicheckers/data/animedl2m_dataset_release/real_images/images")

    # Hard negatives
    hard_neg_files = Path("/home/techne/aicheckers/embeddings/novelai_combined_hard_neg_fixed_hard_neg_files.txt")
    hard_neg_dir = Path("/home/techne/aicheckers/data/novelai_combined")

    random.seed(42)

    # サンプリング
    real_images = list(real_dir.glob("*.jpg")) + list(real_dir.glob("*.png"))
    real_samples = random.sample(real_images, min(N_SAMPLES * 2, len(real_images)))

    ai_samples = []
    for name, ai_dir in ai_dirs.items():
        imgs = list(ai_dir.glob("*.jpeg")) + list(ai_dir.glob("*.jpg")) + list(ai_dir.glob("*.png"))
        samples = random.sample(imgs, min(N_SAMPLES, len(imgs)))
        ai_samples.extend(samples)
        print(f"  {name}: {len(samples)} samples")

    # Hard negatives
    hard_neg_samples = []
    if hard_neg_files.exists():
        with open(hard_neg_files) as f:
            names = [line.strip() for line in f if line.strip()]
        hard_neg_samples = [hard_neg_dir / n for n in names if (hard_neg_dir / n).exists()]
        hard_neg_samples = random.sample(hard_neg_samples, min(N_SAMPLES, len(hard_neg_samples)))
        print(f"  hard_neg: {len(hard_neg_samples)} samples")

    print(f"AI: {len(ai_samples)}, Real: {len(real_samples)}, HardNeg: {len(hard_neg_samples)}")

    keys = ['isotropy_var', 'hv_diff', 'effective_bits', 'banding_score', 'grad_entropy']
    ai_features = {k: [] for k in keys}
    real_features = {k: [] for k in keys}
    hard_features = {k: [] for k in keys}

    print("\nProcessing AI...")
    for img_path in ai_samples:
        feat = process_image(img_path)
        if feat:
            for k, v in feat.items():
                if not np.isnan(v) and not np.isinf(v):
                    ai_features[k].append(v)

    print("Processing Real...")
    for img_path in real_samples:
        feat = process_image(img_path)
        if feat:
            for k, v in feat.items():
                if not np.isnan(v) and not np.isinf(v):
                    real_features[k].append(v)

    print("Processing Hard Negatives...")
    for img_path in hard_neg_samples:
        feat = process_image(img_path)
        if feat:
            for k, v in feat.items():
                if not np.isnan(v) and not np.isinf(v):
                    hard_features[k].append(v)

    # 結果表示
    print("\n" + "=" * 85)
    print("NEW CANDIDATE FEATURES TEST")
    print("=" * 85)
    print(f"{'Feature':<18} {'AI mean':<12} {'Real mean':<12} {'Dir':<12} {'d(AI-R)':<10} {'d(HN-R)':<10}")
    print("-" * 85)

    for key in keys:
        ai_vals = np.array(ai_features[key])
        real_vals = np.array(real_features[key])
        hard_vals = np.array(hard_features[key])

        if len(ai_vals) < 10 or len(real_vals) < 10:
            print(f"{key:<18} データ不足")
            continue

        ai_mean = np.mean(ai_vals)
        real_mean = np.mean(real_vals)
        hard_mean = np.mean(hard_vals) if len(hard_vals) > 0 else 0

        d_ai = cohens_d(ai_vals, real_vals)
        d_hard = cohens_d(hard_vals, real_vals) if len(hard_vals) > 10 else 0

        direction = "AI < Real" if ai_mean < real_mean else "AI > Real"

        star_ai = "★★★" if d_ai >= 0.8 else "★★" if d_ai >= 0.5 else "★" if d_ai >= 0.2 else ""
        star_hard = "★★★" if d_hard >= 0.8 else "★★" if d_hard >= 0.5 else "★" if d_hard >= 0.2 else ""

        print(f"{key:<18} {ai_mean:<12.4f} {real_mean:<12.4f} {direction:<12} {d_ai:.3f} {star_ai:<4} {d_hard:.3f} {star_hard}")

    print("-" * 85)
    print("\n判定基準: d >= 0.8 ★★★(強), d >= 0.5 ★★(中), d >= 0.2 ★(弱)")
    print("HardNeg列が重要: 通常AIより下がると使えない")


if __name__ == "__main__":
    main()
