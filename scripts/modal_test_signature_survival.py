#!/usr/bin/env python3
"""
署名がポイズニング後も残るかテスト
1. 署名を埋め込む
2. ポイズニングを適用
3. 署名が読み取れるか確認
4. VAE通過後も署名が残るか確認
"""

import modal
import time

app = modal.App("signature-survival-test")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch",
        "torchvision",
        "diffusers",
        "transformers",
        "accelerate",
        "Pillow",
        "numpy",
        "tqdm",
        "invisible-watermark",
        "opencv-python-headless",
    )
)

SIGNATURE = b"AICHECKERS_HUMAN_VERIFIED"


@app.function(
    image=image,
    gpu="T4",
    timeout=600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def test_signature_survival(image_bytes: bytes, strength: float = 0.08, iterations: int = 100):
    """署名→ポイズニング→検証"""
    import os
    import io
    import cv2
    import torch
    import torch.nn.functional as F
    import numpy as np
    from PIL import Image
    from tqdm import tqdm
    from imwatermark import WatermarkEncoder, WatermarkDecoder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 画像読み込み
    img_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    print(f"Original image: {img_cv.shape[1]}x{img_cv.shape[0]}")

    # ===== STEP 1: 署名を埋め込む =====
    print("\n[STEP 1] Embedding signature...")
    encoder = WatermarkEncoder()
    encoder.set_watermark('bytes', SIGNATURE)
    img_signed = encoder.encode(img_cv, 'dwtDct')

    # 署名確認
    decoder = WatermarkDecoder('bytes', len(SIGNATURE) * 8)
    sig_after_embed = decoder.decode(img_signed, 'dwtDct')
    print(f"  Signature after embedding: {sig_after_embed}")
    print(f"  Match: {sig_after_embed == SIGNATURE}")

    # ===== STEP 2: ポイズニング適用 =====
    print("\n[STEP 2] Applying poisoning...")

    # モデルロード
    hf_token = os.environ.get("HF_TOKEN", "")

    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
        torch_dtype=torch.float32
    ).to(device)
    vae.eval()

    from transformers import AutoModel
    dino = AutoModel.from_pretrained(
        "facebook/dinov2-base",
        token=hf_token
    ).to(device)
    dino.eval()

    # 署名済み画像をテンソルに変換
    img_rgb = cv2.cvtColor(img_signed, cv2.COLOR_BGR2RGB)

    # リサイズ（8の倍数、最大512）
    h, w = img_rgb.shape[:2]
    max_size = 512
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        w, h = int(w * scale), int(h * scale)
        img_rgb = cv2.resize(img_rgb, (w, h), interpolation=cv2.INTER_LANCZOS4)
    new_w = (w // 8) * 8
    new_h = (h // 8) * 8
    if new_w != w or new_h != h:
        img_rgb = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    img_tensor = torch.from_numpy(img_rgb.astype(np.float32) / 255.0)
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
    img_tensor = img_tensor * 2.0 - 1.0
    img_tensor = img_tensor.to(device)
    original_tensor = img_tensor.clone()

    def get_dino_features(tensor):
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
        h, w = shape[-2:]
        cy, cx = h // 2, w // 2
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        dist = torch.sqrt((x - cx).float()**2 + (y - cy).float()**2)
        max_dist = np.sqrt(cx**2 + cy**2)
        mask = (dist / max_dist < cutoff).float()
        return mask.to(device)

    def apply_low_freq_constraint(perturbation, cutoff=0.4):
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

    for i in tqdm(range(iterations), desc="Poisoning"):
        optimizer.zero_grad()
        constrained_pert = apply_low_freq_constraint(perturbation, 0.4)
        constrained_pert = constrained_pert.clamp(-strength, strength)
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

        total_loss = 1.0 * vae_loss + 0.5 * dino_loss + 0.1 * perceptual_loss
        total_loss.backward()
        optimizer.step()

    # 最終画像生成
    with torch.no_grad():
        final_pert = apply_low_freq_constraint(perturbation, 0.4)
        final_pert = final_pert.clamp(-strength, strength)
        final_poisoned = (original_tensor + final_pert).clamp(-1, 1)

    # テンソル→numpy
    poisoned_np = final_poisoned.squeeze(0).permute(1, 2, 0).cpu().numpy()
    poisoned_np = ((poisoned_np + 1.0) / 2.0 * 255).clip(0, 255).astype(np.uint8)
    poisoned_cv = cv2.cvtColor(poisoned_np, cv2.COLOR_RGB2BGR)

    # ===== STEP 3: ポイズニング後の署名確認 =====
    print("\n[STEP 3] Checking signature after poisoning...")
    sig_after_poison = decoder.decode(poisoned_cv, 'dwtDct')
    print(f"  Signature after poisoning: {sig_after_poison}")
    print(f"  Match: {sig_after_poison == SIGNATURE}")

    # バイト単位の一致率
    match_bytes = sum(a == b for a, b in zip(sig_after_poison or b'', SIGNATURE))
    match_rate_poison = match_bytes / len(SIGNATURE) * 100
    print(f"  Byte match rate: {match_rate_poison:.1f}%")

    # ===== STEP 4: VAE通過後の署名確認 =====
    print("\n[STEP 4] Checking signature after VAE loop...")
    with torch.no_grad():
        poisoned_tensor = torch.from_numpy(poisoned_np.astype(np.float32) / 255.0)
        poisoned_tensor = poisoned_tensor.permute(2, 0, 1).unsqueeze(0)
        poisoned_tensor = poisoned_tensor * 2.0 - 1.0
        poisoned_tensor = poisoned_tensor.to(device)

        latent = vae.encode(poisoned_tensor).latent_dist.mean
        reconstructed = vae.decode(latent).sample

        recon_np = reconstructed.squeeze(0).permute(1, 2, 0).cpu().numpy()
        recon_np = ((recon_np + 1.0) / 2.0 * 255).clip(0, 255).astype(np.uint8)
        recon_cv = cv2.cvtColor(recon_np, cv2.COLOR_RGB2BGR)

    sig_after_vae = decoder.decode(recon_cv, 'dwtDct')
    print(f"  Signature after VAE: {sig_after_vae}")
    print(f"  Match: {sig_after_vae == SIGNATURE}")

    match_bytes_vae = sum(a == b for a, b in zip(sig_after_vae or b'', SIGNATURE))
    match_rate_vae = match_bytes_vae / len(SIGNATURE) * 100
    print(f"  Byte match rate: {match_rate_vae:.1f}%")

    # ===== 結果まとめ =====
    return {
        "signature": SIGNATURE.decode('utf-8'),
        "after_embed": {
            "decoded": sig_after_embed.decode('utf-8') if sig_after_embed else None,
            "match": sig_after_embed == SIGNATURE,
        },
        "after_poison": {
            "decoded": sig_after_poison.decode('utf-8', errors='replace') if sig_after_poison else None,
            "match": sig_after_poison == SIGNATURE,
            "match_rate": match_rate_poison,
        },
        "after_vae": {
            "decoded": sig_after_vae.decode('utf-8', errors='replace') if sig_after_vae else None,
            "match": sig_after_vae == SIGNATURE,
            "match_rate": match_rate_vae,
        },
    }


@app.local_entrypoint()
def main():
    from pathlib import Path

    test_image = Path("/home/techne/aicheckers/data/novelai/10004.jpg")
    if not test_image.exists():
        print(f"Error: {test_image} not found")
        return

    print(f"Test image: {test_image}")
    print("=" * 60)

    with open(test_image, "rb") as f:
        image_bytes = f.read()

    result = test_signature_survival.remote(
        image_bytes=image_bytes,
        strength=0.08,
        iterations=100,
    )

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Original signature: {result['signature']}")
    print()
    print(f"After embedding:  {result['after_embed']['match']} ✓" if result['after_embed']['match'] else f"After embedding:  FAIL")
    print(f"After poisoning:  {result['after_poison']['match_rate']:.1f}% match")
    print(f"After VAE loop:   {result['after_vae']['match_rate']:.1f}% match")
    print()

    if result['after_poison']['match']:
        print("✓ Signature SURVIVES poisoning!")
    else:
        print("✗ Signature CORRUPTED by poisoning")

    if result['after_vae']['match']:
        print("✓ Signature SURVIVES VAE loop!")
    else:
        print("✗ Signature CORRUPTED by VAE loop")
