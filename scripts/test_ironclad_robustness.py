#!/usr/bin/env python3
"""
Ironclad V3.1 耐性テスト - 攻撃者視点からの検証

攻撃ベクトル:
1. JPEG再圧縮（各種品質）
2. リサイズ（縮小・拡大）
3. ガウシアンノイズ追加
4. ガウシアンブラー
5. メディアンフィルタ
6. ヒストグラム正規化
7. シャープニング
8. 複合攻撃（圧縮→リサイズ→ノイズ）
"""

import os
import sys
import io
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageEnhance
import cv2

# Ironclad をインポート
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.ironclad_v2 import IroncladPoisoner, image_to_tensor, tensor_to_image


def apply_jpeg_compression(image: Image.Image, quality: int) -> Image.Image:
    """JPEG再圧縮"""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def apply_resize(image: Image.Image, scale: float) -> Image.Image:
    """リサイズ（縮小→元サイズに戻す）"""
    orig_size = image.size
    new_size = (int(orig_size[0] * scale), int(orig_size[1] * scale))
    resized = image.resize(new_size, Image.Resampling.LANCZOS)
    return resized.resize(orig_size, Image.Resampling.LANCZOS)


def apply_gaussian_noise(image: Image.Image, std: float) -> Image.Image:
    """ガウシアンノイズ追加"""
    arr = np.array(image).astype(np.float32)
    noise = np.random.normal(0, std, arr.shape)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


def apply_gaussian_blur(image: Image.Image, radius: float) -> Image.Image:
    """ガウシアンブラー"""
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def apply_median_filter(image: Image.Image, size: int) -> Image.Image:
    """メディアンフィルタ"""
    return image.filter(ImageFilter.MedianFilter(size=size))


def apply_histogram_equalization(image: Image.Image) -> Image.Image:
    """ヒストグラム平坦化（YCbCrのYのみ）"""
    arr = np.array(image)
    ycbcr = cv2.cvtColor(arr, cv2.COLOR_RGB2YCrCb)
    ycbcr[:, :, 0] = cv2.equalizeHist(ycbcr[:, :, 0])
    rgb = cv2.cvtColor(ycbcr, cv2.COLOR_YCrCb2RGB)
    return Image.fromarray(rgb)


def apply_clahe(image: Image.Image) -> Image.Image:
    """適応的ヒストグラム平坦化（CLAHE）"""
    arr = np.array(image)
    ycbcr = cv2.cvtColor(arr, cv2.COLOR_RGB2YCrCb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    ycbcr[:, :, 0] = clahe.apply(ycbcr[:, :, 0])
    rgb = cv2.cvtColor(ycbcr, cv2.COLOR_YCrCb2RGB)
    return Image.fromarray(rgb)


def apply_sharpening(image: Image.Image, factor: float) -> Image.Image:
    """シャープニング"""
    enhancer = ImageEnhance.Sharpness(image)
    return enhancer.enhance(factor)


def apply_nlm_denoising(image: Image.Image, h: float = 10) -> Image.Image:
    """Non-local Means デノイジング"""
    arr = np.array(image)
    denoised = cv2.fastNlMeansDenoisingColored(arr, None, h, h, 7, 21)
    return Image.fromarray(denoised)


def apply_bilateral_filter(image: Image.Image) -> Image.Image:
    """Bilateral フィルタ（エッジ保持デノイズ）"""
    arr = np.array(image)
    filtered = cv2.bilateralFilter(arr, 9, 75, 75)
    return Image.fromarray(filtered)


def apply_combined_attack(image: Image.Image) -> Image.Image:
    """複合攻撃: JPEG→リサイズ→ノイズ"""
    img = apply_jpeg_compression(image, 70)
    img = apply_resize(img, 0.75)
    img = apply_gaussian_noise(img, 5)
    return img


def apply_aggressive_attack(image: Image.Image) -> Image.Image:
    """積極的攻撃: 強めの複合処理"""
    img = apply_jpeg_compression(image, 50)
    img = apply_resize(img, 0.5)
    img = apply_gaussian_blur(img, 1.0)
    img = apply_nlm_denoising(img, 15)
    return img


def main():
    # テスト画像を用意（既存の画像があれば使用）
    test_images_dir = Path("/home/techne/aicheckers/data/novelai")
    test_image_path = None

    for ext in ["*.png", "*.jpg", "*.jpeg"]:
        files = list(test_images_dir.glob(ext))
        if files:
            test_image_path = files[0]
            break

    if not test_image_path:
        print("テスト画像が見つかりません")
        sys.exit(1)

    print(f"テスト画像: {test_image_path}")
    original = Image.open(test_image_path).convert("RGB")

    # Ironclad初期化（CPUで実行）
    poisoner = IroncladPoisoner(
        secret_key="AICHECKERS_DEFAULT_KEY",
        version="202501",
        device="cpu"
    )

    # 1. オリジナル画像をポイズニング
    print("\n=== ポイズニング実行 ===")
    orig_tensor = image_to_tensor(original)
    # CPUでは遅いので反復回数を減らす
    poisoned_tensor = poisoner.poison(orig_tensor, iterations=20)
    poisoned_image = tensor_to_image(poisoned_tensor)

    # ポイズン後の署名検出（ベースライン）
    baseline_result = poisoner.detect_signature(poisoned_tensor)
    print(f"ベースライン相関: {baseline_result['correlation']:.4f}")
    print(f"署名検出: {baseline_result['detected']}")

    # 攻撃テスト
    attacks = [
        ("JPEG Q=95", lambda img: apply_jpeg_compression(img, 95)),
        ("JPEG Q=80", lambda img: apply_jpeg_compression(img, 80)),
        ("JPEG Q=60", lambda img: apply_jpeg_compression(img, 60)),
        ("JPEG Q=40", lambda img: apply_jpeg_compression(img, 40)),
        ("JPEG Q=20", lambda img: apply_jpeg_compression(img, 20)),
        ("Resize 80%", lambda img: apply_resize(img, 0.8)),
        ("Resize 60%", lambda img: apply_resize(img, 0.6)),
        ("Resize 40%", lambda img: apply_resize(img, 0.4)),
        ("Gaussian Noise σ=5", lambda img: apply_gaussian_noise(img, 5)),
        ("Gaussian Noise σ=15", lambda img: apply_gaussian_noise(img, 15)),
        ("Gaussian Noise σ=30", lambda img: apply_gaussian_noise(img, 30)),
        ("Gaussian Blur r=1", lambda img: apply_gaussian_blur(img, 1)),
        ("Gaussian Blur r=2", lambda img: apply_gaussian_blur(img, 2)),
        ("Gaussian Blur r=3", lambda img: apply_gaussian_blur(img, 3)),
        ("Median Filter 3x3", lambda img: apply_median_filter(img, 3)),
        ("Median Filter 5x5", lambda img: apply_median_filter(img, 5)),
        ("Histogram Eq", apply_histogram_equalization),
        ("CLAHE", apply_clahe),
        ("Sharpen 2.0", lambda img: apply_sharpening(img, 2.0)),
        ("Sharpen 3.0", lambda img: apply_sharpening(img, 3.0)),
        ("NLM Denoise h=10", lambda img: apply_nlm_denoising(img, 10)),
        ("NLM Denoise h=20", lambda img: apply_nlm_denoising(img, 20)),
        ("Bilateral Filter", apply_bilateral_filter),
        ("Combined Attack", apply_combined_attack),
        ("Aggressive Attack", apply_aggressive_attack),
    ]

    print("\n=== 攻撃耐性テスト ===")
    print(f"{'攻撃手法':<25} {'相関値':>10} {'検出':>8} {'状態':<10}")
    print("-" * 60)

    results = []
    for name, attack_fn in attacks:
        try:
            attacked_image = attack_fn(poisoned_image)
            attacked_tensor = image_to_tensor(attacked_image)
            result = poisoner.detect_signature(attacked_tensor)

            status = "✓ 耐性あり" if result['detected'] else "✗ 破壊"
            print(f"{name:<25} {result['correlation']:>10.4f} {str(result['detected']):>8} {status}")

            results.append({
                "attack": name,
                "correlation": result['correlation'],
                "detected": result['detected']
            })
        except Exception as e:
            print(f"{name:<25} ERROR: {e}")

    # サマリー
    survived = sum(1 for r in results if r['detected'])
    total = len(results)
    print("\n" + "=" * 60)
    print(f"結果サマリー: {survived}/{total} 攻撃に耐性あり ({100*survived/total:.1f}%)")

    failed_attacks = [r['attack'] for r in results if not r['detected']]
    if failed_attacks:
        print(f"\n⚠ 署名が破壊された攻撃:")
        for attack in failed_attacks:
            print(f"  - {attack}")
    else:
        print("\n✓ 全ての攻撃に耐性あり！")


if __name__ == "__main__":
    main()
