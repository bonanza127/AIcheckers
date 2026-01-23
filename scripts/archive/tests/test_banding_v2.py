#!/usr/bin/env python3
"""
バンディング指標 v2 - 正しい設計
A) YCbCrのCb/Crチャンネルのみ
B) flat領域限定
C) band_step_ratio 1指標
"""
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
import random

N_SAMPLES = 125  # per model

def compute_band_step_ratio(img_path):
    """正しいバンディング指標を計算"""
    try:
        img = Image.open(img_path).convert('RGB')
        img = img.resize((512, 512), Image.LANCZOS)
        img_np = np.array(img)

        # === A) YCbCrに変換、Cb/Crのみ使用 ===
        img_ycrcb = cv2.cvtColor(img_np, cv2.COLOR_RGB2YCrCb)
        y_channel = img_ycrcb[:, :, 0].astype(np.float32)
        cb_channel = img_ycrcb[:, :, 2].astype(np.float32)  # Cb
        cr_channel = img_ycrcb[:, :, 1].astype(np.float32)  # Cr

        # === B) flat領域マスク（Y channelの勾配で判定） ===
        grad_x = cv2.Sobel(y_channel, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(y_channel, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        flat_threshold = np.percentile(grad_mag, 15)
        flat_mask = grad_mag <= flat_threshold

        # === C) band_step_ratio: flat領域内のCb/Crで小さな段差の割合 ===
        def calc_step_ratio(channel, mask):
            # 水平・垂直差分
            diff_h = np.abs(np.diff(channel, axis=1))
            diff_v = np.abs(np.diff(channel, axis=0))

            # flat領域内のみ
            mask_h = mask[:, :-1]
            mask_v = mask[:-1, :]

            flat_diffs_h = diff_h[mask_h]
            flat_diffs_v = diff_v[mask_v]
            flat_diffs = np.concatenate([flat_diffs_h, flat_diffs_v])

            if len(flat_diffs) < 100:
                return np.nan

            # バンディング = 小さな非ゼロ段差（1-4）への集中
            nonzero_diffs = flat_diffs[flat_diffs > 0]
            if len(nonzero_diffs) < 50:
                return 0.0

            small_steps = (nonzero_diffs >= 1) & (nonzero_diffs <= 4)
            return np.mean(small_steps)

        cb_ratio = calc_step_ratio(cb_channel, flat_mask)
        cr_ratio = calc_step_ratio(cr_channel, flat_mask)

        # Cb/Crの平均
        ratios = [r for r in [cb_ratio, cr_ratio] if not np.isnan(r)]
        if ratios:
            band_step_ratio = np.mean(ratios)
        else:
            band_step_ratio = np.nan

        return {'band_step_ratio': band_step_ratio}

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

    ai_vals = []
    real_vals = []

    print("\nProcessing AI images...")
    for img_path in ai_samples:
        feat = compute_band_step_ratio(img_path)
        if feat and not np.isnan(feat['band_step_ratio']):
            ai_vals.append(feat['band_step_ratio'])

    print("Processing Real images...")
    for img_path in real_samples:
        feat = compute_band_step_ratio(img_path)
        if feat and not np.isnan(feat['band_step_ratio']):
            real_vals.append(feat['band_step_ratio'])

    # 結果
    print("\n" + "=" * 60)
    print("BANDING v2 (YCbCr + flat only + band_step_ratio)")
    print("=" * 60)

    ai_vals = np.array(ai_vals)
    real_vals = np.array(real_vals)

    ai_mean, ai_std = np.mean(ai_vals), np.std(ai_vals)
    real_mean, real_std = np.mean(real_vals), np.std(real_vals)

    d = cohens_d(ai_vals, real_vals)
    direction = "AI < Real" if ai_mean < real_mean else "AI > Real"

    print(f"\nband_step_ratio:")
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
