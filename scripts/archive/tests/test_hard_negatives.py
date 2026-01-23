#!/usr/bin/env python3
"""
Hard Negatives での特徴テスト
- Hard negatives = 現モデルが苦手なAI画像
- これらとReal画像で特徴が分離できるか確認
"""
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
import random
from scipy.stats import entropy

FLAT_PERCENTILE = 15

def get_flat_mask(img_gray):
    grad_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    threshold = np.percentile(grad_mag, FLAT_PERCENTILE)
    flat_mask = (grad_mag <= threshold).astype(np.uint8)
    return flat_mask


def compute_all_features(img_path):
    try:
        img = Image.open(img_path).convert('RGB')
        img = img.resize((512, 512), Image.LANCZOS)
        img_np = np.array(img)
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        flat_mask = get_flat_mask(img_gray)

        # fractal_dim & curvature_var
        contours, _ = cv2.findContours(flat_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        total_perimeter = 0
        total_area = 0
        all_curvatures = []

        for contour in contours:
            if len(contour) < 10:
                continue
            perimeter = cv2.arcLength(contour, True)
            area = cv2.contourArea(contour)
            if area < 100:
                continue
            total_perimeter += perimeter
            total_area += area

            contour_sq = contour.squeeze()
            if len(contour_sq.shape) == 1:
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
                all_curvatures.append(abs(angle))

        fractal_dim = total_perimeter / max(np.sqrt(total_area), 1)
        curvature_var = np.var(all_curvatures) if all_curvatures else 0

        # rank_entropy
        flat_values = img_gray[flat_mask > 0]
        if len(flat_values) >= 100:
            hist, _ = np.histogram(flat_values, bins=256, range=(0, 256))
            hist = hist / hist.sum()
            hist = hist[hist > 0]
            rank_entropy = -np.sum(hist * np.log2(hist + 1e-10))
        else:
            rank_entropy = 0

        # corr_decay
        low_freq = cv2.GaussianBlur(img_gray.astype(np.float32), (3, 3), 1.0)
        residual = img_gray.astype(np.float32) - low_freq
        res_flat = residual.flatten()
        try:
            lag1 = np.corrcoef(res_flat[:-1], res_flat[1:])[0, 1]
            lag5 = np.corrcoef(res_flat[:-5], res_flat[5:])[0, 1]
            corr_decay = lag1 - lag5 if not np.isnan(lag5) else 0
        except:
            corr_decay = 0

        return {
            'fractal_dim': fractal_dim,
            'curvature_var': curvature_var,
            'rank_entropy': rank_entropy,
            'corr_decay': corr_decay,
        }
    except Exception as e:
        return None


def cohens_d(ai_vals, real_vals):
    ai_mean, ai_std = np.mean(ai_vals), np.std(ai_vals)
    real_mean, real_std = np.mean(real_vals), np.std(real_vals)
    pooled_std = np.sqrt((ai_std**2 + real_std**2) / 2)
    return abs(ai_mean - real_mean) / pooled_std if pooled_std > 0 else 0


def main():
    # Hard negatives (AI画像だが判別困難なもの)
    hard_neg_dir = Path("/home/techne/aicheckers/data/novelai_combined")
    hard_neg_files = Path("/home/techne/aicheckers/embeddings/novelai_combined_hard_neg_fixed_hard_neg_files.txt")

    # Real
    real_dir = Path("/home/techne/aicheckers/data/animedl2m_dataset_release/real_images/images")

    random.seed(42)

    # Hard negatives読み込み
    with open(hard_neg_files) as f:
        hard_neg_names = [line.strip() for line in f if line.strip()]

    hard_neg_paths = [hard_neg_dir / name for name in hard_neg_names if (hard_neg_dir / name).exists()]
    print(f"Hard negatives found: {len(hard_neg_paths)}")

    # サンプリング
    n_samples = min(250, len(hard_neg_paths))
    hard_neg_samples = random.sample(hard_neg_paths, n_samples)

    real_images = list(real_dir.glob("*.jpg")) + list(real_dir.glob("*.png"))
    real_samples = random.sample(real_images, min(n_samples, len(real_images)))

    print(f"Hard Neg samples: {len(hard_neg_samples)}, Real samples: {len(real_samples)}")

    keys = ['fractal_dim', 'curvature_var', 'rank_entropy', 'corr_decay']
    ai_features = {k: [] for k in keys}
    real_features = {k: [] for k in keys}

    print("\nProcessing Hard Negatives...")
    for img_path in hard_neg_samples:
        feat = compute_all_features(img_path)
        if feat:
            for k, v in feat.items():
                if not np.isnan(v) and not np.isinf(v):
                    ai_features[k].append(v)

    print("Processing Real images...")
    for img_path in real_samples:
        feat = compute_all_features(img_path)
        if feat:
            for k, v in feat.items():
                if not np.isnan(v) and not np.isinf(v):
                    real_features[k].append(v)

    # 結果表示
    print("\n" + "=" * 70)
    print("HARD NEGATIVES TEST")
    print("=" * 70)
    print(f"{'Feature':<20} {'HardNeg mean':<14} {'Real mean':<12} {'Direction':<14} {'Cohen d':<10}")
    print("-" * 70)

    for key in keys:
        ai_vals = np.array(ai_features[key])
        real_vals = np.array(real_features[key])

        ai_mean = np.mean(ai_vals)
        real_mean = np.mean(real_vals)
        d = cohens_d(ai_vals, real_vals)
        direction = "HardNeg < Real" if ai_mean < real_mean else "HardNeg > Real"

        star = ""
        if d >= 0.8:
            star = "★★★"
        elif d >= 0.5:
            star = "★★"
        elif d >= 0.2:
            star = "★"

        print(f"{key:<20} {ai_mean:<14.4f} {real_mean:<12.4f} {direction:<14} {d:.3f} {star}")

    print("-" * 70)


if __name__ == "__main__":
    main()
