#!/usr/bin/env python3
"""
新特徴と既存flat特徴の相関を確認
"""
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
import random

N_SAMPLES = 200
FLAT_PERCENTILE = 15

def get_flat_mask(img_gray):
    grad_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    threshold = np.percentile(grad_mag, FLAT_PERCENTILE)
    flat_mask = (grad_mag <= threshold).astype(np.uint8)
    return flat_mask


def compute_features(img_path):
    try:
        img = Image.open(img_path).convert('RGB')
        img = img.resize((512, 512), Image.LANCZOS)
        img_np = np.array(img)
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        flat_mask = get_flat_mask(img_gray)

        # === 既存flat特徴 ===
        flat_ratio = flat_mask.sum() / flat_mask.size

        # flat clusters
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(flat_mask, connectivity=8)
        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]  # skip background
            flat_cluster_max = np.max(areas)
            flat_cluster_median = np.median(areas)
            flat_cluster_mean = np.mean(areas)
            flat_cluster_count = len(areas)
        else:
            flat_cluster_max = 0
            flat_cluster_median = 0
            flat_cluster_mean = 0
            flat_cluster_count = 0

        # === 新特徴 ===
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

        return {
            # 既存
            'flat_ratio': flat_ratio,
            'flat_cluster_max': flat_cluster_max,
            'flat_cluster_median': flat_cluster_median,
            'flat_cluster_mean': flat_cluster_mean,
            'flat_cluster_count': flat_cluster_count,
            # 新規
            'fractal_dim': fractal_dim,
            'curvature_var': curvature_var,
            'rank_entropy': rank_entropy,
        }
    except:
        return None


def main():
    # 全画像を混ぜてサンプリング
    dirs = [
        Path("/home/techne/aicheckers/data/novelai_combined"),
        Path("/home/techne/aicheckers/data/animedl2m_dataset_release/civitai_subset/image/Illustrious"),
        Path("/home/techne/aicheckers/data/animedl2m_dataset_release/real_images/images"),
    ]

    random.seed(42)
    all_images = []
    for d in dirs:
        all_images.extend(list(d.glob("*.jpg"))[:500])
        all_images.extend(list(d.glob("*.jpeg"))[:500])
        all_images.extend(list(d.glob("*.png"))[:500])

    samples = random.sample(all_images, min(N_SAMPLES, len(all_images)))
    print(f"Samples: {len(samples)}")

    # 特徴抽出
    data = {k: [] for k in ['flat_ratio', 'flat_cluster_max', 'flat_cluster_median',
                             'flat_cluster_mean', 'flat_cluster_count',
                             'fractal_dim', 'curvature_var', 'rank_entropy']}

    print("Processing...")
    for img_path in samples:
        feat = compute_features(img_path)
        if feat:
            for k, v in feat.items():
                if not np.isnan(v) and not np.isinf(v):
                    data[k].append(v)

    # 相関行列
    print("\n" + "=" * 80)
    print("CORRELATION MATRIX: 新特徴 vs 既存flat特徴")
    print("=" * 80)

    existing = ['flat_ratio', 'flat_cluster_max', 'flat_cluster_median', 'flat_cluster_mean', 'flat_cluster_count']
    new_feats = ['fractal_dim', 'curvature_var', 'rank_entropy']

    # ヘッダー
    print(f"{'':>22}", end="")
    for nf in new_feats:
        print(f"{nf:>14}", end="")
    print()
    print("-" * 80)

    for ef in existing:
        print(f"{ef:>22}", end="")
        for nf in new_feats:
            arr1 = np.array(data[ef])
            arr2 = np.array(data[nf])
            min_len = min(len(arr1), len(arr2))
            r = np.corrcoef(arr1[:min_len], arr2[:min_len])[0, 1]
            print(f"{r:>14.3f}", end="")
        print()

    print("\n" + "=" * 80)
    print("新特徴同士の相関")
    print("=" * 80)
    for i, f1 in enumerate(new_feats):
        for f2 in new_feats[i+1:]:
            arr1 = np.array(data[f1])
            arr2 = np.array(data[f2])
            min_len = min(len(arr1), len(arr2))
            r = np.corrcoef(arr1[:min_len], arr2[:min_len])[0, 1]
            print(f"{f1} <-> {f2}: r = {r:.3f}")


if __name__ == "__main__":
    main()
