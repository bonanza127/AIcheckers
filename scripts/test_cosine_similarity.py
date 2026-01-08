#!/usr/bin/env python3
"""
FastProtect Cosine Similarity Test
オリジナル vs 保護済み画像のVAE latent cos simを計算
"""
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
from diffusers import AutoencoderKL
import argparse


def compute_cosine_similarity(image1_path, image2_path, device="cuda", resize_to=None):
    """
    2つの画像のVAE latent空間でのコサイン類似度を計算

    Args:
        image1_path: 画像1のパス（オリジナル）
        image2_path: 画像2のパス（保護済み）
        device: デバイス
        resize_to: リサイズサイズ（None=元サイズ維持）

    Returns:
        cos_sim: コサイン類似度
    """
    # VAEロード
    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sdxl-vae",
        torch_dtype=torch.bfloat16,
    ).to(device)
    vae.eval()

    # 画像ロード
    img1 = Image.open(image1_path).convert("RGB")
    img2 = Image.open(image2_path).convert("RGB")

    original_size = img1.size
    print(f"Original size: {original_size}")

    # リサイズ処理
    if resize_to:
        transform = T.Compose([
            T.Resize((resize_to, resize_to)),
            T.ToTensor(),
        ])
        print(f"Resizing to: {resize_to}x{resize_to}")
    else:
        transform = T.ToTensor()
        print("No resizing (original size)")

    img1_tensor = transform(img1).unsqueeze(0).to(device)
    img2_tensor = transform(img2).unsqueeze(0).to(device)

    # VAE encode
    with torch.no_grad():
        img1_normalized = (img1_tensor * 2 - 1).to(torch.bfloat16)
        img2_normalized = (img2_tensor * 2 - 1).to(torch.bfloat16)

        z1 = vae.encode(img1_normalized).latent_dist.mean.float()
        z2 = vae.encode(img2_normalized).latent_dist.mean.float()

    # コサイン類似度計算
    z1_flat = z1.view(-1)
    z2_flat = z2.view(-1)

    cos_sim = F.cosine_similarity(z1_flat.unsqueeze(0), z2_flat.unsqueeze(0)).item()

    return cos_sim, original_size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", required=True, help="Original image path")
    parser.add_argument("--protected", required=True, help="Protected image path")
    parser.add_argument("--device", default="cuda", help="Device")
    args = parser.parse_args()

    print("=" * 60)
    print("FastProtect VAE Latent Cosine Similarity Test")
    print("=" * 60)

    # Test 1: 512x512（学習時のサイズ）
    print("\n[Test 1] 512x512 (training size)")
    cos_sim_512, original_size = compute_cosine_similarity(
        args.original,
        args.protected,
        device=args.device,
        resize_to=512,
    )
    print(f"Cosine Similarity (512x512): {cos_sim_512:.4f}")

    # Test 2: 元サイズ
    print(f"\n[Test 2] Original size ({original_size[0]}x{original_size[1]})")
    cos_sim_orig, _ = compute_cosine_similarity(
        args.original,
        args.protected,
        device=args.device,
        resize_to=None,
    )
    print(f"Cosine Similarity (original): {cos_sim_orig:.4f}")

    # 結果サマリー
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Original size:        {original_size[0]}x{original_size[1]}")
    print(f"Cos Sim (512x512):    {cos_sim_512:.4f}")
    print(f"Cos Sim (original):   {cos_sim_orig:.4f}")
    print(f"Difference:           {abs(cos_sim_512 - cos_sim_orig):.4f}")

    # 解釈
    print("\n" + "=" * 60)
    print("Interpretation")
    print("=" * 60)
    if cos_sim_512 > 0.9:
        print("⚠️  Cos Sim > 0.9: 攻撃効果が弱い可能性（latentがほぼ同一）")
    elif cos_sim_512 > 0.85:
        print("⚠️  Cos Sim 0.85-0.9: やや弱め（要確認）")
    elif cos_sim_512 > 0.8:
        print("✅ Cos Sim 0.8-0.85: 良好なバランス")
    else:
        print("✅ Cos Sim < 0.8: 強力な攻撃効果")

    if abs(cos_sim_512 - cos_sim_orig) < 0.01:
        print("✅ リサイズの影響: ほぼなし（差 < 0.01）")
    elif abs(cos_sim_512 - cos_sim_orig) < 0.05:
        print("⚠️  リサイズの影響: 小（差 0.01-0.05）")
    else:
        print("❌ リサイズの影響: 大（差 > 0.05）")


if __name__ == "__main__":
    main()
