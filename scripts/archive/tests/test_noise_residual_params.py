#!/usr/bin/env python3
"""
ノイズ残差特徴 - パラメータ感度分析
"""
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
import random
from scipy.stats import kurtosis, entropy

N_SAMPLES = 150  # per class

def get_noise_residual(img_gray, kernel_size, sigma):
    """画像からノイズ残差を抽出"""
    low_freq = cv2.GaussianBlur(img_gray.astype(np.float32), (kernel_size, kernel_size), sigma)
    residual = img_gray.astype(np.float32) - low_freq
    return residual

def compute_features(img_gray, kernel_size, sigma):
    """指定パラメータでノイズ残差特徴を計算"""
    residual = get_noise_residual(img_gray, kernel_size, sigma)

    # Energy (variance)
    res_energy = np.var(residual)

    # Entropy
    hist, _ = np.histogram(residual.flatten(), bins=256, range=(-128, 128))
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    res_entropy = entropy(hist)

    # Corr decay
    res_flat = residual.flatten()
    lag1 = np.corrcoef(res_flat[:-1], res_flat[1:])[0, 1]
    lag5 = np.corrcoef(res_flat[:-5], res_flat[5:])[0, 1]
    corr_decay = lag1 - lag5 if not np.isnan(lag5) else 0

    return {
        'energy': res_energy,
        'entropy': res_entropy,
        'corr_decay': corr_decay
    }

def cohens_d(ai_vals, real_vals):
    ai_mean, ai_std = np.mean(ai_vals), np.std(ai_vals)
    real_mean, real_std = np.mean(real_vals), np.std(real_vals)
    pooled_std = np.sqrt((ai_std**2 + real_std**2) / 2)
    return abs(ai_mean - real_mean) / pooled_std if pooled_std > 0 else 0

def main():
    # パス設定
    ai_dirs = {
        'novelai': Path("/home/techne/aicheckers/data/novelai_combined"),
        'illustrious': Path("/home/techne/aicheckers/data/animedl2m_dataset_release/civitai_subset/image/Illustrious"),
    }
    real_dir = Path("/home/techne/aicheckers/data/animedl2m_dataset_release/real_images/images")

    random.seed(42)

    # サンプル収集
    real_images = list(real_dir.glob("*.jpg")) + list(real_dir.glob("*.png"))
    real_samples = random.sample(real_images, min(N_SAMPLES, len(real_images)))

    ai_samples = []
    for name, ai_dir in ai_dirs.items():
        imgs = list(ai_dir.glob("*.jpeg")) + list(ai_dir.glob("*.jpg")) + list(ai_dir.glob("*.png"))
        ai_samples.extend(random.sample(imgs, min(N_SAMPLES // 2, len(imgs))))

    print(f"AI: {len(ai_samples)}, Real: {len(real_samples)}")

    # 画像読み込み（1回だけ）
    print("Loading images...")
    ai_grays = []
    for p in ai_samples:
        try:
            img = Image.open(p).convert('RGB').resize((512, 512), Image.LANCZOS)
            ai_grays.append(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY))
        except:
            pass

    real_grays = []
    for p in real_samples:
        try:
            img = Image.open(p).convert('RGB').resize((512, 512), Image.LANCZOS)
            real_grays.append(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY))
        except:
            pass

    print(f"Loaded: AI={len(ai_grays)}, Real={len(real_grays)}")

    # パラメータ組み合わせ
    params = [
        # (kernel_size, sigma, description)
        (3, 0.5, "tiny (3, 0.5)"),
        (3, 1.0, "small (3, 1.0)"),
        (5, 1.5, "current (5, 1.5)"),
        (7, 2.0, "medium (7, 2.0)"),
        (11, 3.0, "large (11, 3.0)"),
        (15, 4.0, "xlarge (15, 4.0)"),
        (21, 6.0, "xxlarge (21, 6.0)"),
    ]

    print("\n" + "=" * 70)
    print("PARAMETER SENSITIVITY ANALYSIS")
    print("=" * 70)
    print(f"{'Params':<20} {'energy d':<12} {'entropy d':<12} {'corr_decay d':<12}")
    print("-" * 70)

    best_d = 0
    best_params = None
    best_feature = None

    for kernel, sigma, desc in params:
        ai_feats = {'energy': [], 'entropy': [], 'corr_decay': []}
        real_feats = {'energy': [], 'entropy': [], 'corr_decay': []}

        for g in ai_grays:
            f = compute_features(g, kernel, sigma)
            for k, v in f.items():
                if not np.isnan(v) and not np.isinf(v):
                    ai_feats[k].append(v)

        for g in real_grays:
            f = compute_features(g, kernel, sigma)
            for k, v in f.items():
                if not np.isnan(v) and not np.isinf(v):
                    real_feats[k].append(v)

        d_energy = cohens_d(ai_feats['energy'], real_feats['energy'])
        d_entropy = cohens_d(ai_feats['entropy'], real_feats['entropy'])
        d_corr = cohens_d(ai_feats['corr_decay'], real_feats['corr_decay'])

        # ベスト更新
        for feat, d in [('energy', d_energy), ('entropy', d_entropy), ('corr_decay', d_corr)]:
            if d > best_d:
                best_d = d
                best_params = desc
                best_feature = feat

        # 星マーク
        def star(d):
            if d >= 0.8: return "★★★"
            elif d >= 0.5: return "★★"
            elif d >= 0.2: return "★"
            else: return ""

        print(f"{desc:<20} {d_energy:.3f} {star(d_energy):<4} {d_entropy:.3f} {star(d_entropy):<4} {d_corr:.3f} {star(d_corr):<4}")

    print("-" * 70)
    print(f"BEST: {best_feature} @ {best_params} with d={best_d:.3f}")

if __name__ == "__main__":
    main()
