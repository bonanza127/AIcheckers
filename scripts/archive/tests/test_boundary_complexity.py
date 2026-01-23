#!/usr/bin/env python3
"""
Boundary Complexity の有効性検証（小サンプル）
"""
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
import random

# サンプル数
N_SAMPLES = 100

# Flat領域判定の相対閾値
FLAT_PERCENTILE = 15

def get_flat_mask(img_gray):
    """勾配が小さい領域のマスクを取得"""
    grad_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    threshold = np.percentile(grad_mag, FLAT_PERCENTILE)
    flat_mask = (grad_mag <= threshold).astype(np.uint8)
    return flat_mask

def compute_boundary_complexity(flat_mask):
    """Flat領域境界の複雑度を計算"""
    # 輪郭抽出
    contours, _ = cv2.findContours(flat_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        return {'perimeter_area_ratio': 0, 'curvature_var': 0, 'fractal_dim': 0}

    total_perimeter = 0
    total_area = 0
    all_curvatures = []

    for contour in contours:
        if len(contour) < 10:
            continue

        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)

        if area < 100:  # 小さすぎるクラスタは無視
            continue

        total_perimeter += perimeter
        total_area += area

        # 曲率計算（簡易版：角度変化）
        contour = contour.squeeze()
        if len(contour.shape) == 1:
            continue

        # 各点での方向変化
        for i in range(len(contour)):
            p1 = contour[i - 2]
            p2 = contour[i - 1]
            p3 = contour[i]

            v1 = p2 - p1
            v2 = p3 - p2

            # 角度変化
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            dot = v1[0] * v2[0] + v1[1] * v2[1]
            angle = np.arctan2(cross, dot)
            all_curvatures.append(abs(angle))

    # 指標計算
    perimeter_area_ratio = total_perimeter / max(total_area, 1) * 100
    curvature_var = np.var(all_curvatures) if all_curvatures else 0

    # 簡易フラクタル次元（境界のギザギザ度）
    # perimeter / sqrt(area) で近似
    fractal_approx = total_perimeter / max(np.sqrt(total_area), 1)

    return {
        'perimeter_area_ratio': perimeter_area_ratio,
        'curvature_var': curvature_var,
        'fractal_dim': fractal_approx
    }

def process_image(img_path):
    """画像からBoundary Complexity特徴を抽出"""
    try:
        img = Image.open(img_path).convert('RGB')
        img = img.resize((512, 512), Image.LANCZOS)
        img_np = np.array(img)
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        flat_mask = get_flat_mask(img_gray)
        features = compute_boundary_complexity(flat_mask)
        return features
    except Exception as e:
        return None

def main():
    # パス設定
    ai_dir = Path("/home/techne/aicheckers/data/animedl2m_dataset_release/civitai_subset/image/Pony")
    real_dir = Path("/home/techne/aicheckers/data/animedl2m_dataset_release/real_images/images")

    # サンプル取得
    ai_images = list(ai_dir.glob("*.jpeg")) + list(ai_dir.glob("*.jpg")) + list(ai_dir.glob("*.png"))
    real_images = list(real_dir.glob("*.jpg")) + list(real_dir.glob("*.png"))

    random.seed(42)
    ai_samples = random.sample(ai_images, min(N_SAMPLES, len(ai_images)))
    real_samples = random.sample(real_images, min(N_SAMPLES, len(real_images)))

    print(f"AI samples: {len(ai_samples)}, Real samples: {len(real_samples)}")

    # 特徴抽出
    ai_features = {'perimeter_area_ratio': [], 'curvature_var': [], 'fractal_dim': []}
    real_features = {'perimeter_area_ratio': [], 'curvature_var': [], 'fractal_dim': []}

    print("\nProcessing AI images...")
    for img_path in ai_samples:
        feat = process_image(img_path)
        if feat:
            for k, v in feat.items():
                ai_features[k].append(v)

    print("Processing Real images...")
    for img_path in real_samples:
        feat = process_image(img_path)
        if feat:
            for k, v in feat.items():
                real_features[k].append(v)

    # 結果表示
    print("\n" + "=" * 60)
    print("BOUNDARY COMPLEXITY VALIDATION RESULTS")
    print("=" * 60)

    for key in ai_features.keys():
        ai_vals = np.array(ai_features[key])
        real_vals = np.array(real_features[key])

        ai_mean, ai_std = np.mean(ai_vals), np.std(ai_vals)
        real_mean, real_std = np.mean(real_vals), np.std(real_vals)

        # 分離度（Cohen's d）
        pooled_std = np.sqrt((ai_std**2 + real_std**2) / 2)
        cohens_d = abs(ai_mean - real_mean) / pooled_std if pooled_std > 0 else 0

        # 方向
        direction = "AI < Real" if ai_mean < real_mean else "AI > Real"

        print(f"\n{key}:")
        print(f"  AI:   mean={ai_mean:.4f}, std={ai_std:.4f}")
        print(f"  Real: mean={real_mean:.4f}, std={real_std:.4f}")
        print(f"  Direction: {direction}")
        print(f"  Cohen's d: {cohens_d:.3f} ", end="")

        if cohens_d >= 0.8:
            print("★★★ STRONG")
        elif cohens_d >= 0.5:
            print("★★ MEDIUM")
        elif cohens_d >= 0.2:
            print("★ SMALL")
        else:
            print("(negligible)")

if __name__ == "__main__":
    main()
