#!/usr/bin/env python3
"""
バンディング指標の有効性検証
"""
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
import random

N_SAMPLES = 125  # per model

def compute_banding_features(img_path):
    """バンディング特徴を計算"""
    try:
        img = Image.open(img_path).convert('RGB')
        img = img.resize((512, 512), Image.LANCZOS)
        img_np = np.array(img)
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(np.float32)

        # === 1. 局所勾配の量子化度 ===
        # 小ブロック(8x8)ごとに勾配のユニーク値数を計算
        grad_x = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)

        # 勾配を整数化（量子化）
        grad_quantized = (grad_mag / 4).astype(np.int32)  # 4段階で丸め

        block_size = 8
        unique_counts = []
        for y in range(0, img_gray.shape[0] - block_size, block_size):
            for x in range(0, img_gray.shape[1] - block_size, block_size):
                block = grad_quantized[y:y+block_size, x:x+block_size]
                unique_counts.append(len(np.unique(block)))

        # ユニーク値が少ない = バンディング寄り
        grad_quant_score = np.mean(unique_counts)

        # === 2. 平坦領域内の階調段差率 ===
        # まずflat領域を検出
        grad_threshold = np.percentile(grad_mag, 15)
        flat_mask = grad_mag <= grad_threshold

        # 近傍差分を計算
        diff_h = np.abs(np.diff(img_gray, axis=1))  # 水平差分
        diff_v = np.abs(np.diff(img_gray, axis=0))  # 垂直差分

        # flat領域内の差分のみ抽出
        flat_mask_h = flat_mask[:, :-1]
        flat_mask_v = flat_mask[:-1, :]

        flat_diffs_h = diff_h[flat_mask_h]
        flat_diffs_v = diff_v[flat_mask_v]
        flat_diffs = np.concatenate([flat_diffs_h, flat_diffs_v])

        if len(flat_diffs) > 100:
            # 方法A: 小さな値（1-3）に集中している割合
            small_step_ratio = np.mean((flat_diffs >= 1) & (flat_diffs <= 3))

            # 方法B: 差分ヒストグラムのエントロピー（バンディング=低エントロピー）
            hist, _ = np.histogram(flat_diffs, bins=20, range=(0, 20))
            hist = hist / (hist.sum() + 1e-6)
            hist = hist[hist > 0]
            diff_entropy = -np.sum(hist * np.log(hist + 1e-10))

            # 方法C: 特定値への集中度（mode / mean）
            from scipy import stats
            mode_val = stats.mode(flat_diffs, keepdims=False).mode
            mode_ratio = np.mean(flat_diffs == mode_val)
        else:
            small_step_ratio = 0
            diff_entropy = 5  # 高エントロピー（バンディングなし）
            mode_ratio = 0

        return {
            'grad_quant': grad_quant_score,
            'small_step_ratio': small_step_ratio,
            'diff_entropy': diff_entropy,
            'mode_ratio': mode_ratio
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

    # 特徴抽出
    keys = ['grad_quant', 'small_step_ratio', 'diff_entropy', 'mode_ratio']
    ai_features = {k: [] for k in keys}
    real_features = {k: [] for k in keys}

    print("\nProcessing AI images...")
    for img_path in ai_samples:
        feat = compute_banding_features(img_path)
        if feat:
            for k, v in feat.items():
                if not np.isnan(v) and not np.isinf(v):
                    ai_features[k].append(v)

    print("Processing Real images...")
    for img_path in real_samples:
        feat = compute_banding_features(img_path)
        if feat:
            for k, v in feat.items():
                if not np.isnan(v) and not np.isinf(v):
                    real_features[k].append(v)

    # 結果表示
    print("\n" + "=" * 60)
    print("BANDING FEATURE VALIDATION")
    print("=" * 60)

    for key in ai_features.keys():
        ai_vals = np.array(ai_features[key])
        real_vals = np.array(real_features[key])

        if len(ai_vals) == 0 or len(real_vals) == 0:
            print(f"\n{key}: データ不足")
            continue

        ai_mean, ai_std = np.mean(ai_vals), np.std(ai_vals)
        real_mean, real_std = np.mean(real_vals), np.std(real_vals)

        d = cohens_d(ai_vals, real_vals)
        direction = "AI < Real" if ai_mean < real_mean else "AI > Real"

        print(f"\n{key}:")
        print(f"  AI:   mean={ai_mean:.4f}, std={ai_std:.4f}")
        print(f"  Real: mean={real_mean:.4f}, std={real_std:.4f}")
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
