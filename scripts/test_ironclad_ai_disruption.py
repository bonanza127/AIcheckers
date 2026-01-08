#!/usr/bin/env python3
"""
Ironclad V3.1 AI学習破壊効果テスト

本当に重要なのは「署名が残るか」ではなく「AIが学習できなくなるか」
- DINOv2 embedding の変化を測定
- 攻撃前後でembeddingがどれだけ変わるか
- 変化が大きい = AIにとって「別の画像」になっている
"""

import sys
from pathlib import Path
import io

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.ironclad_v2 import IroncladPoisoner, image_to_tensor, tensor_to_image


def apply_jpeg_compression(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def apply_resize(image: Image.Image, scale: float) -> Image.Image:
    orig_size = image.size
    new_size = (int(orig_size[0] * scale), int(orig_size[1] * scale))
    resized = image.resize(new_size, Image.Resampling.LANCZOS)
    return resized.resize(orig_size, Image.Resampling.LANCZOS)


class DINOEmbedder:
    """DINOv2でembedding抽出"""
    def __init__(self, device="cpu"):
        self.device = device
        print("Loading DINOv2...")
        self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        self.model = AutoModel.from_pretrained("facebook/dinov2-base").eval().to(device)

    def embed(self, image: Image.Image) -> torch.Tensor:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0]  # CLSトークン


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a, b, dim=-1).item()


def main():
    # テスト画像
    test_images_dir = Path("/home/techne/aicheckers/data/novelai")
    test_image_path = list(test_images_dir.glob("*.png"))[0]
    print(f"テスト画像: {test_image_path}")
    original = Image.open(test_image_path).convert("RGB")

    # モデル初期化
    embedder = DINOEmbedder(device="cpu")
    poisoner = IroncladPoisoner(
        secret_key="AICHECKERS_DEFAULT_KEY",
        version="202501",
        device="cpu"
    )

    # オリジナルのembedding
    orig_embedding = embedder.embed(original)
    print(f"\nオリジナルembedding取得完了")

    # ポイズニング
    print("\n=== ポイズニング実行 (iterations=30) ===")
    orig_tensor = image_to_tensor(original)
    poisoned_tensor = poisoner.poison(orig_tensor, iterations=30)
    poisoned_image = tensor_to_image(poisoned_tensor)

    # ポイズン後のembedding
    poisoned_embedding = embedder.embed(poisoned_image)

    # 類似度計算
    orig_to_poisoned = cosine_similarity(orig_embedding, poisoned_embedding)
    print(f"\n原画 ↔ ポイズン済み 類似度: {orig_to_poisoned:.4f}")

    if orig_to_poisoned < 0.85:
        print("✓ AIにとって「別の画像」になっている可能性大")
    else:
        print("⚠ AIにとってほぼ同じ画像として認識される")

    # 攻撃後の類似度
    print("\n=== 攻撃後のembedding類似度 ===")
    print(f"{'攻撃手法':<20} {'原画↔攻撃後':>12} {'ポイズン↔攻撃後':>15} {'効果':<10}")
    print("-" * 65)

    attacks = [
        ("JPEG Q=80", lambda img: apply_jpeg_compression(img, 80)),
        ("JPEG Q=50", lambda img: apply_jpeg_compression(img, 50)),
        ("Resize 70%", lambda img: apply_resize(img, 0.7)),
        ("Resize 50%", lambda img: apply_resize(img, 0.5)),
    ]

    for name, attack_fn in attacks:
        # オリジナル → 攻撃
        attacked_orig = attack_fn(original)
        attacked_orig_emb = embedder.embed(attacked_orig)
        orig_to_attacked_orig = cosine_similarity(orig_embedding, attacked_orig_emb)

        # ポイズン → 攻撃
        attacked_poisoned = attack_fn(poisoned_image)
        attacked_poisoned_emb = embedder.embed(attacked_poisoned)
        poisoned_to_attacked_poisoned = cosine_similarity(poisoned_embedding, attacked_poisoned_emb)

        # 原画 ↔ ポイズン攻撃後
        orig_to_attacked_poisoned = cosine_similarity(orig_embedding, attacked_poisoned_emb)

        # 効果判定: 攻撃後もオリジナルと違えば効果あり
        effect = "✓ 有効" if orig_to_attacked_poisoned < 0.9 else "△ 弱"

        print(f"{name:<20} {orig_to_attacked_orig:>12.4f} {poisoned_to_attacked_poisoned:>15.4f} {effect}")

    print("\n" + "=" * 65)
    print("解釈:")
    print("- 原画↔攻撃後: 通常の画像処理でembeddingがどれだけ変わるか")
    print("- ポイズン↔攻撃後: ポイズン効果が攻撃で消えるか")
    print("- 効果: 攻撃後も原画と異なるembeddingなら、AI学習を妨害できている")


if __name__ == "__main__":
    main()
