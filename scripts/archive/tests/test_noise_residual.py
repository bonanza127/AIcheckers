#!/usr/bin/env python3
"""
ノイズ残差特徴の有効性検証
"""
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
import random
from scipy import ndimage
from scipy.stats import kurtosis, entropy

N_SAMPLES = 125  # per model (125 x 2 = 250 total AI)

def get_noise_residual(img_gray):
    """画像からノイズ残差を抽出（高周波成分）"""
    # ガウシアンブラーで低周波成分を抽出
    low_freq = cv2.GaussianBlur(img_gray.astype(np.float32), (5, 5), 1.5)
    # 残差 = 元画像 - 低周波
    residual = img_gray.astype(np.float32) - low_freq
    return residual, low_freq

def compute_residual_features(img_path):
    """ノイズ残差から特徴量を計算"""
    try:
        img = Image.open(img_path).convert('RGB')
        img = img.resize((512, 512), Image.LANCZOS)
        img_np = np.array(img)
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        residual, low_freq = get_noise_residual(img_gray)

        # === 既存の3特徴 ===
        # 1. Residual Kurtosis（尖度）
        res_kurtosis = kurtosis(residual.flatten())

        # 2. Residual Entropy（エントロピー）
        hist, _ = np.histogram(residual.flatten(), bins=256, range=(-128, 128))
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        res_entropy = entropy(hist)

        # 3. Corr Decay（相関減衰率）
        res_flat = residual.flatten()
        lag1_corr = np.corrcoef(res_flat[:-1], res_flat[1:])[0, 1]
        lag5_corr = np.corrcoef(res_flat[:-5], res_flat[5:])[0, 1]
        corr_decay = lag1_corr - lag5_corr if not np.isnan(lag5_corr) else 0

        # === 追加の3特徴 ===
        # 4. Residual Energy（残差エネルギー = 分散）
        res_energy = np.var(residual)

        # 5. Residual MAD（中央絶対偏差）- 外れ値に強い
        res_mad = np.median(np.abs(residual - np.median(residual)))

        # 6. Residual HF Ratio（高周波比率）
        # 差分フィルタで近似：Laplacianの絶対値 / 元画像の範囲
        laplacian = cv2.Laplacian(img_gray.astype(np.float32), cv2.CV_32F)
        hf_energy = np.mean(np.abs(laplacian))
        img_range = img_gray.max() - img_gray.min() + 1e-6
        hf_ratio = hf_energy / img_range

        return {
            'residual_kurtosis': res_kurtosis,
            'residual_entropy': res_entropy,
            'corr_decay': corr_decay,
            'residual_energy': res_energy,
            'residual_MAD': res_mad,
            'hf_ratio': hf_ratio
        }
    except Exception as e:
        return None

def main():
    # パス設定（Illustrious + NovelAI のみ）
    ai_dirs = {
        'novelai': Path("/home/techne/aicheckers/data/novelai_combined"),
        'illustrious': Path("/home/techne/aicheckers/data/animedl2m_dataset_release/civitai_subset/image/Illustrious"),
    }
    real_dir = Path("/home/techne/aicheckers/data/animedl2m_dataset_release/real_images/images")

    random.seed(42)

    # Real画像サンプル（250枚）
    real_images = list(real_dir.glob("*.jpg")) + list(real_dir.glob("*.png"))
    real_samples = random.sample(real_images, min(250, len(real_images)))

    # AI画像サンプル（各モデル50枚ずつ）
    ai_samples = []
    for name, ai_dir in ai_dirs.items():
        imgs = list(ai_dir.glob("*.jpeg")) + list(ai_dir.glob("*.jpg")) + list(ai_dir.glob("*.png"))
        samples = random.sample(imgs, min(N_SAMPLES, len(imgs)))
        ai_samples.extend(samples)
        print(f"  {name}: {len(samples)} samples")

    print(f"AI samples: {len(ai_samples)}, Real samples: {len(real_samples)}")

    # 特徴抽出
    feature_keys = ['residual_kurtosis', 'residual_entropy', 'corr_decay',
                    'residual_energy', 'residual_MAD', 'hf_ratio']
    ai_features = {k: [] for k in feature_keys}
    real_features = {k: [] for k in feature_keys}

    print("\nProcessing AI images...")
    for img_path in ai_samples:
        feat = compute_residual_features(img_path)
        if feat:
            for k, v in feat.items():
                if not np.isnan(v) and not np.isinf(v):
                    ai_features[k].append(v)

    print("Processing Real images...")
    for img_path in real_samples:
        feat = compute_residual_features(img_path)
        if feat:
            for k, v in feat.items():
                if not np.isnan(v) and not np.isinf(v):
                    real_features[k].append(v)

    # 結果表示
    print("\n" + "=" * 60)
    print("NOISE RESIDUAL FEATURE VALIDATION")
    print("=" * 60)

    for key in ai_features.keys():
        ai_vals = np.array(ai_features[key])
        real_vals = np.array(real_features[key])

        if len(ai_vals) == 0 or len(real_vals) == 0:
            print(f"\n{key}: データ不足")
            continue

        ai_mean, ai_std = np.mean(ai_vals), np.std(ai_vals)
        real_mean, real_std = np.mean(real_vals), np.std(real_vals)

        # Cohen's d
        pooled_std = np.sqrt((ai_std**2 + real_std**2) / 2)
        cohens_d = abs(ai_mean - real_mean) / pooled_std if pooled_std > 0 else 0

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
