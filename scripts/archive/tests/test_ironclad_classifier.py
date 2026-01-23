#!/usr/bin/env python3
"""
Ironclad効果検証 - AIcheckers分類器での検出変化

ポイズン前後で「AI生成」と判定されるスコアがどう変わるか確認
"""

import sys
from pathlib import Path
import io
import requests

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.ironclad_v2 import IroncladPoisoner, image_to_tensor, tensor_to_image


def apply_jpeg_compression(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def classify_image(image: Image.Image) -> dict:
    """ローカルAPIで分類"""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    response = requests.post(
        "http://localhost:8000/detect",
        files={"file": ("test.png", buffer, "image/png")},
        timeout=30
    )
    return response.json()


def main():
    # AI生成画像をテスト（NovelAI）
    test_images_dir = Path("/home/techne/aicheckers/data/novelai")
    test_files = list(test_images_dir.glob("*.png"))[:5]

    print("=" * 70)
    print("Ironclad V3.1 AI検出回避テスト")
    print("=" * 70)

    poisoner = IroncladPoisoner(
        secret_key="AICHECKERS_DEFAULT_KEY",
        version="202501",
        device="cpu"
    )

    results = []

    for i, test_path in enumerate(test_files):
        print(f"\n[{i+1}/{len(test_files)}] {test_path.name}")
        original = Image.open(test_path).convert("RGB")

        # オリジナルの分類
        orig_result = classify_image(original)
        orig_score = orig_result.get("ai_probability", 0)
        print(f"  原画 AI確率: {orig_score*100:.1f}%")

        # ポイズニング
        print("  ポイズニング中...")
        orig_tensor = image_to_tensor(original)
        poisoned_tensor = poisoner.poison(orig_tensor, iterations=30)
        poisoned_image = tensor_to_image(poisoned_tensor)

        # ポイズン後の分類
        poison_result = classify_image(poisoned_image)
        poison_score = poison_result.get("ai_probability", 0)
        print(f"  ポイズン後 AI確率: {poison_score*100:.1f}%")

        # JPEG圧縮後（SNS投稿シミュレーション）
        jpeg_image = apply_jpeg_compression(poisoned_image, 80)
        jpeg_result = classify_image(jpeg_image)
        jpeg_score = jpeg_result.get("ai_probability", 0)
        print(f"  JPEG(Q=80)後 AI確率: {jpeg_score*100:.1f}%")

        results.append({
            "file": test_path.name,
            "original": orig_score,
            "poisoned": poison_score,
            "jpeg": jpeg_score,
        })

        change = orig_score - poison_score
        if poison_score < 0.5:
            print(f"  ✓ AI検出を回避！（-{change*100:.1f}%）")
        elif change > 0.1:
            print(f"  △ スコア低下（-{change*100:.1f}%）")
        else:
            print(f"  ✗ 効果なし")

    # サマリー
    print("\n" + "=" * 70)
    print("サマリー")
    print("=" * 70)

    evaded = sum(1 for r in results if r["poisoned"] < 0.5)
    reduced = sum(1 for r in results if r["original"] - r["poisoned"] > 0.1)

    print(f"テスト画像数: {len(results)}")
    print(f"AI検出回避成功: {evaded}/{len(results)}")
    print(f"スコア10%以上低下: {reduced}/{len(results)}")

    avg_orig = np.mean([r["original"] for r in results])
    avg_poison = np.mean([r["poisoned"] for r in results])
    avg_jpeg = np.mean([r["jpeg"] for r in results])

    print(f"\n平均スコア:")
    print(f"  原画: {avg_orig*100:.1f}%")
    print(f"  ポイズン後: {avg_poison*100:.1f}%")
    print(f"  JPEG後: {avg_jpeg*100:.1f}%")


if __name__ == "__main__":
    main()
