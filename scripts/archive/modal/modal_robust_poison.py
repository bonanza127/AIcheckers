#!/usr/bin/env python3
"""
Modal wrapper for robust_poison.py
GPU上でAIポイズニングを実行し、コストを計測
"""

import modal
import time

# Modal app定義
app = modal.App("robust-poison-test")

# GPU付きイメージ
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch",
    "torchvision",
    "diffusers",
    "transformers",
    "accelerate",
    "Pillow",
    "numpy",
    "tqdm",
)

@app.function(
    image=image,
    gpu="T4",  # 安価なGPU (16GB VRAM)
    timeout=600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def poison_image(image_bytes: bytes, strength: float = 0.08, iterations: int = 100):
    """画像にポイズンを適用"""
    import os
    import io
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    from PIL import Image
    from tqdm import tqdm

    start_time = time.time()

    # HF Token設定
    hf_token = os.environ.get("HF_TOKEN", "")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # モデルロード
    print("Loading VAE...")
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
        torch_dtype=torch.float32
    ).to(device)
    vae.eval()

    print("Loading DINOv3...")
    from transformers import AutoImageProcessor, AutoModel
    dino_processor = AutoImageProcessor.from_pretrained(
        "facebook/dinov2-base",
        token=hf_token
    )
    dino = AutoModel.from_pretrained(
        "facebook/dinov2-base",
        token=hf_token
    ).to(device)
    dino.eval()

    model_load_time = time.time() - start_time
    print(f"Models loaded in {model_load_time:.1f}s")

    # 画像読み込み
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    orig_size = image.size
    print(f"Input image: {orig_size[0]}x{orig_size[1]}")

    # リサイズ（8の倍数）
    max_size = 512
    w, h = image.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        w, h = int(w * scale), int(h * scale)
        image = image.resize((w, h), Image.LANCZOS)
    new_w = (w // 8) * 8
    new_h = (h // 8) * 8
    if new_w != w or new_h != h:
        image = image.resize((new_w, new_h), Image.LANCZOS)
    print(f"Resized to: {new_w}x{new_h}")

    # Tensor変換
    arr = np.array(image).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    img_tensor = img_tensor * 2.0 - 1.0  # [0,1] -> [-1,1]
    img_tensor = img_tensor.to(device)
    original_tensor = img_tensor.clone()

    def get_dino_features(tensor):
        """DINO特徴量を取得"""
        img_01 = (tensor + 1.0) / 2.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        img_norm = (img_01 - mean) / std
        img_resized = F.interpolate(img_norm, size=(224, 224), mode="bilinear", align_corners=False)
        with torch.no_grad():
            outputs = dino(pixel_values=img_resized)
            features = outputs.last_hidden_state[:, 0, :]
        return features

    def low_freq_mask(shape, cutoff=0.3):
        """低周波マスク"""
        h, w = shape[-2:]
        cy, cx = h // 2, w // 2
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        dist = torch.sqrt((x - cx).float()**2 + (y - cy).float()**2)
        max_dist = np.sqrt(cx**2 + cy**2)
        mask = (dist / max_dist < cutoff).float()
        return mask.to(device)

    def apply_low_freq_constraint(perturbation, cutoff=0.4):
        """低周波制約"""
        fft = torch.fft.fft2(perturbation)
        fft_shifted = torch.fft.fftshift(fft)
        mask = low_freq_mask(perturbation.shape, cutoff)
        fft_filtered = fft_shifted * mask
        fft_unshifted = torch.fft.ifftshift(fft_filtered)
        filtered = torch.fft.ifft2(fft_unshifted).real
        return filtered

    # 元画像の特徴量
    with torch.no_grad():
        original_dino = get_dino_features(original_tensor)
        original_latent = vae.encode(original_tensor).latent_dist.mean

    # 最適化
    perturbation = torch.zeros_like(img_tensor, requires_grad=True)
    optimizer = torch.optim.Adam([perturbation], lr=0.01)

    poison_start = time.time()
    print(f"Starting optimization ({iterations} iterations)...")

    for i in tqdm(range(iterations), desc="Poisoning"):
        optimizer.zero_grad()

        # 低周波制約
        constrained_pert = apply_low_freq_constraint(perturbation, 0.4)
        constrained_pert = constrained_pert.clamp(-strength, strength)

        # ポイズン画像
        poisoned = (original_tensor + constrained_pert).clamp(-1, 1)

        # VAE耐性ロス
        poisoned_latent = vae.encode(poisoned).latent_dist.mean
        reconstructed = vae.decode(poisoned_latent).sample
        residual = (reconstructed - original_tensor).abs().mean()
        vae_loss = -residual

        # DINO特徴変化ロス
        poisoned_dino = get_dino_features(poisoned)
        dino_loss = -F.cosine_similarity(poisoned_dino, original_dino).mean()

        # 知覚的制約
        perceptual_loss = constrained_pert.abs().mean()

        # 合計ロス
        total_loss = 1.0 * vae_loss + 0.5 * dino_loss + 0.1 * perceptual_loss

        total_loss.backward()
        optimizer.step()

        if (i + 1) % 25 == 0:
            print(f"  Iter {i+1}: VAE={-vae_loss.item():.4f}, DINO={-dino_loss.item():.4f}")

    poison_time = time.time() - poison_start
    print(f"Optimization completed in {poison_time:.1f}s")

    # 最終画像生成
    with torch.no_grad():
        final_pert = apply_low_freq_constraint(perturbation, 0.4)
        final_pert = final_pert.clamp(-strength, strength)
        final_poisoned = (original_tensor + final_pert).clamp(-1, 1)

        # 検証
        pois_latent = vae.encode(final_poisoned).latent_dist.mean
        reconstructed = vae.decode(pois_latent).sample

        original_pert = (final_poisoned - original_tensor).abs().mean().item()
        after_vae_pert = (reconstructed - original_tensor).abs().mean().item()
        survival_rate = after_vae_pert / (original_pert + 1e-8)

        pois_dino = get_dino_features(final_poisoned)
        recon_dino = get_dino_features(reconstructed)
        dino_shift_before = 1 - F.cosine_similarity(pois_dino, original_dino).item()
        dino_shift_after = 1 - F.cosine_similarity(recon_dino, original_dino).item()

    # PIL変換
    final_poisoned = final_poisoned.clamp(-1, 1)
    final_poisoned = (final_poisoned + 1.0) / 2.0
    arr = final_poisoned.squeeze(0).permute(1, 2, 0).cpu().numpy()
    arr = (arr * 255).clip(0, 255).astype(np.uint8)
    result_image = Image.fromarray(arr)

    # 元サイズに戻す
    if result_image.size != orig_size:
        result_image = result_image.resize(orig_size, Image.LANCZOS)

    # bytes変換
    output_buffer = io.BytesIO()
    result_image.save(output_buffer, format="PNG")
    result_bytes = output_buffer.getvalue()

    total_time = time.time() - start_time

    return {
        "image_bytes": result_bytes,
        "stats": {
            "model_load_time": model_load_time,
            "poison_time": poison_time,
            "total_time": total_time,
            "perturbation_magnitude": original_pert,
            "vae_survival_rate": survival_rate,
            "dino_shift_before_vae": dino_shift_before,
            "dino_shift_after_vae": dino_shift_after,
            "iterations": iterations,
            "strength": strength,
        }
    }


@app.local_entrypoint()
def main():
    """ローカルから実行"""
    import sys
    from pathlib import Path

    # テスト画像
    test_images = [
        Path("/home/techne/aicheckers/data/novelai/10004.jpg"),
        Path("/home/techne/aicheckers/data/animedl2m_dataset_release/real_images/images/000001.png"),
    ]

    test_image = None
    for p in test_images:
        if p.exists():
            test_image = p
            break

    if test_image is None:
        print("Error: No test image found")
        sys.exit(1)

    print(f"Test image: {test_image}")

    with open(test_image, "rb") as f:
        image_bytes = f.read()

    print("Running on Modal (T4 GPU)...")
    print("-" * 50)

    result = poison_image.remote(
        image_bytes=image_bytes,
        strength=0.08,
        iterations=100,  # テスト用に少なめ
    )

    print("-" * 50)
    print("\n=== Results ===")
    stats = result["stats"]
    print(f"Model load time: {stats['model_load_time']:.1f}s")
    print(f"Poison time: {stats['poison_time']:.1f}s")
    print(f"Total time: {stats['total_time']:.1f}s")
    print(f"Perturbation magnitude: {stats['perturbation_magnitude']:.4f}")
    print(f"VAE survival rate: {stats['vae_survival_rate']*100:.1f}%")
    print(f"DINO shift (before VAE): {stats['dino_shift_before_vae']:.4f}")
    print(f"DINO shift (after VAE): {stats['dino_shift_after_vae']:.4f}")

    # 保存
    output_path = Path("/home/techne/aicheckers/data/test_poisoned.png")
    with open(output_path, "wb") as f:
        f.write(result["image_bytes"])
    print(f"\nOutput saved: {output_path}")

    # コスト概算
    # T4: $0.000164/sec = ~$0.59/hour
    cost_per_sec = 0.000164
    estimated_cost = stats["total_time"] * cost_per_sec
    print(f"\n=== Cost Estimate ===")
    print(f"T4 GPU time: {stats['total_time']:.1f}s")
    print(f"Estimated cost: ${estimated_cost:.4f}")
    print(f"Per image (100 iter): ~${estimated_cost:.4f}")
    print(f"Per image (200 iter): ~${estimated_cost * 2:.4f}")
