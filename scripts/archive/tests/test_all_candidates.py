#!/usr/bin/env python3
"""
全候補特徴の比較テスト
- fractal_dim (採用)
- curvature_var (採用)
- rank_entropy (アブレーション)
- corr_decay (アブレーション)
- edge_alignment (新規テスト)
"""
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
import random
from scipy.stats import entropy

N_SAMPLES = 125  # per model
FLAT_PERCENTILE = 15

def get_flat_mask(img_gray):
    """勾配が小さい領域のマスクを取得"""
    grad_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    threshold = np.percentile(grad_mag, FLAT_PERCENTILE)
    flat_mask = (grad_mag <= threshold).astype(np.uint8)
    return flat_mask, grad_mag


def compute_all_features(img_path):
    """全候補特徴を計算"""
    try:
        img = Image.open(img_path).convert('RGB')
        img = img.resize((512, 512), Image.LANCZOS)
        img_np = np.array(img)
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        flat_mask, grad_mag = get_flat_mask(img_gray)

        # === 1. fractal_dim & curvature_var ===
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

        # === 2. rank_entropy ===
        flat_values = img_gray[flat_mask > 0]
        if len(flat_values) >= 100:
            hist, _ = np.histogram(flat_values, bins=256, range=(0, 256))
            hist = hist / hist.sum()
            hist = hist[hist > 0]
            rank_entropy = -np.sum(hist * np.log2(hist + 1e-10))
        else:
            rank_entropy = 0

        # === 3. corr_decay (kernel=3, sigma=1.0) ===
        low_freq = cv2.GaussianBlur(img_gray.astype(np.float32), (3, 3), 1.0)
        residual = img_gray.astype(np.float32) - low_freq
        res_flat = residual.flatten()
        try:
            lag1 = np.corrcoef(res_flat[:-1], res_flat[1:])[0, 1]
            lag5 = np.corrcoef(res_flat[:-5], res_flat[5:])[0, 1]
            corr_decay = lag1 - lag5 if not np.isnan(lag5) else 0
        except:
            corr_decay = 0

        # === 4. edge_alignment (新規) ===
        # Cannyエッジ検出
        edges = cv2.Canny(img_gray, 50, 150)

        # flat境界を取得（膨張 - 元 = 境界）
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(flat_mask, kernel, iterations=1)
        flat_boundary = (dilated - flat_mask) > 0

        # flat境界とエッジの一致度
        if flat_boundary.sum() > 0:
            # flat境界上でエッジが検出されている割合
            edge_on_boundary = edges[flat_boundary].mean() / 255.0

            # エッジ上でflat境界である割合
            if edges.sum() > 0:
                boundary_on_edge = flat_boundary[edges > 0].mean()
            else:
                boundary_on_edge = 0

            # F1的な調和平均
            if edge_on_boundary + boundary_on_edge > 0:
                edge_alignment = 2 * edge_on_boundary * boundary_on_edge / (edge_on_boundary + boundary_on_edge)
            else:
                edge_alignment = 0
        else:
            edge_alignment = 0

        return {
            'fractal_dim': fractal_dim,
            'curvature_var': curvature_var,
            'rank_entropy': rank_entropy,
            'corr_decay': corr_decay,
            'edge_alignment': edge_alignment
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

    keys = ['fractal_dim', 'curvature_var', 'rank_entropy', 'corr_decay', 'edge_alignment']
    ai_features = {k: [] for k in keys}
    real_features = {k: [] for k in keys}

    print("\nProcessing AI images...")
    for img_path in ai_samples:
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
    print("ALL CANDIDATE FEATURES COMPARISON")
    print("=" * 70)
    print(f"{'Feature':<20} {'AI mean':<12} {'Real mean':<12} {'Direction':<12} {'Cohen d':<10}")
    print("-" * 70)

    results = []
    for key in keys:
        ai_vals = np.array(ai_features[key])
        real_vals = np.array(real_features[key])

        if len(ai_vals) < 10 or len(real_vals) < 10:
            print(f"{key:<20} データ不足")
            continue

        ai_mean = np.mean(ai_vals)
        real_mean = np.mean(real_vals)
        d = cohens_d(ai_vals, real_vals)
        direction = "AI < Real" if ai_mean < real_mean else "AI > Real"

        star = ""
        if d >= 0.8:
            star = "★★★"
        elif d >= 0.5:
            star = "★★"
        elif d >= 0.2:
            star = "★"

        print(f"{key:<20} {ai_mean:<12.4f} {real_mean:<12.4f} {direction:<12} {d:.3f} {star}")
        results.append((key, d, star))

    print("-" * 70)
    print("\n採用判断:")
    for key, d, star in sorted(results, key=lambda x: -x[1]):
        status = "✅ 採用" if d >= 0.5 else "⚠️ 要検討" if d >= 0.3 else "❌ 見送り"
        print(f"  {key}: d={d:.3f} {star} → {status}")


if __name__ == "__main__":
    main()
