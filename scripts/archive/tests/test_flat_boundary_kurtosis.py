#!/usr/bin/env python3
"""
flat境界近傍の色差勾配の尖度
"""
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
import random
from scipy.stats import kurtosis

N_SAMPLES = 125  # per model

def compute_flat_boundary_kurtosis(img_path):
    """flat領域境界の色差勾配の尖度を計算"""
    try:
        img = Image.open(img_path).convert('RGB')
        img = img.resize((512, 512), Image.LANCZOS)
        img_np = np.array(img)

        # YCbCrに変換
        img_ycrcb = cv2.cvtColor(img_np, cv2.COLOR_RGB2YCrCb)
        y_channel = img_ycrcb[:, :, 0].astype(np.float32)
        cb_channel = img_ycrcb[:, :, 2].astype(np.float32)
        cr_channel = img_ycrcb[:, :, 1].astype(np.float32)

        # flat領域マスク
        grad_x = cv2.Sobel(y_channel, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(y_channel, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        flat_threshold = np.percentile(grad_mag, 15)
        flat_mask = (grad_mag <= flat_threshold).astype(np.uint8)

        # flat領域の境界を抽出（膨張 - 元 = 境界）
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(flat_mask, kernel, iterations=1)
        boundary_mask = (dilated - flat_mask) > 0

        # 境界ピクセルでの色差勾配を計算
        cb_grad_x = cv2.Sobel(cb_channel, cv2.CV_32F, 1, 0, ksize=3)
        cb_grad_y = cv2.Sobel(cb_channel, cv2.CV_32F, 0, 1, ksize=3)
        cb_grad_mag = np.sqrt(cb_grad_x**2 + cb_grad_y**2)

        cr_grad_x = cv2.Sobel(cr_channel, cv2.CV_32F, 1, 0, ksize=3)
        cr_grad_y = cv2.Sobel(cr_channel, cv2.CV_32F, 0, 1, ksize=3)
        cr_grad_mag = np.sqrt(cr_grad_x**2 + cr_grad_y**2)

        # 境界での勾配値を取得
        cb_boundary_grads = cb_grad_mag[boundary_mask]
        cr_boundary_grads = cr_grad_mag[boundary_mask]

        if len(cb_boundary_grads) < 100 or len(cr_boundary_grads) < 100:
            return None

        # 尖度を計算
        cb_kurtosis = kurtosis(cb_boundary_grads, fisher=True)
        cr_kurtosis = kurtosis(cr_boundary_grads, fisher=True)

        # 平均
        avg_kurtosis = (cb_kurtosis + cr_kurtosis) / 2

        return {
            'cb_kurtosis': cb_kurtosis,
            'cr_kurtosis': cr_kurtosis,
            'avg_kurtosis': avg_kurtosis
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

    random.seed(42)

    real_images = list(real_dir.glob("*.jpg")) + list(real_dir.glob("*.png"))
    real_samples = random.sample(real_images, min(N_SAMPLES * 2, len(real_images)))

    ai_samples = []
    for name, ai_dir in ai_dirs.items():
        imgs = list(ai_dir.glob("*.jpeg")) + list(ai_dir.glob("*.jpg")) + list(ai_dir.glob("*.png"))
        samples = random.sample(imgs, min(N_SAMPLES, len(imgs)))
        ai_samples.extend(samples)
        print(f"  {name}: {len(samples)} samples")

    print(f"AI samples: {len(ai_samples)}, Real samples: {len(real_samples)}")

    keys = ['cb_kurtosis', 'cr_kurtosis', 'avg_kurtosis']
    ai_features = {k: [] for k in keys}
    real_features = {k: [] for k in keys}

    print("\nProcessing AI images...")
    for img_path in ai_samples:
        feat = compute_flat_boundary_kurtosis(img_path)
        if feat:
            for k, v in feat.items():
                if not np.isnan(v) and not np.isinf(v):
                    ai_features[k].append(v)

    print("Processing Real images...")
    for img_path in real_samples:
        feat = compute_flat_boundary_kurtosis(img_path)
        if feat:
            for k, v in feat.items():
                if not np.isnan(v) and not np.isinf(v):
                    real_features[k].append(v)

    # 結果
    print("\n" + "=" * 60)
    print("FLAT BOUNDARY COLOR GRADIENT KURTOSIS")
    print("=" * 60)

    for key in keys:
        ai_vals = np.array(ai_features[key])
        real_vals = np.array(real_features[key])

        if len(ai_vals) < 10 or len(real_vals) < 10:
            print(f"\n{key}: データ不足")
            continue

        ai_mean, ai_std = np.mean(ai_vals), np.std(ai_vals)
        real_mean, real_std = np.mean(real_vals), np.std(real_vals)

        d = cohens_d(ai_vals, real_vals)
        direction = "AI < Real" if ai_mean < real_mean else "AI > Real"

        print(f"\n{key}:")
        print(f"  AI:   mean={ai_mean:.4f}, std={ai_std:.4f} (n={len(ai_vals)})")
        print(f"  Real: mean={real_mean:.4f}, std={real_std:.4f} (n={len(real_vals)})")
        print(f"  Direction: {direction}")
        print(f"  Cohen's d: {d:.3f} ", end="")

        if d >= 0.8:
            print("★★★ STRONG")
        elif d >= 0.5:
            print("★★ MEDIUM")
        elif d >= 0.2:
            print("★ SMALL")
        else:
            print("(negligible)")

if __name__ == "__main__":
    main()
