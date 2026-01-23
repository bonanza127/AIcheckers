#!/usr/bin/env python3
"""
Modal wrapper for Ironclad v3.1
GPU上でAIポイズニング + 署名埋め込みを実行

使用方法:
    modal run scripts/modal_ironclad_v2.py --input image.png --output poisoned.png
    modal run scripts/modal_ironclad_v2.py --input image.png --detect  # 署名検出のみ
"""

import modal
import time

app = modal.App("ironclad-poison")

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
        "ptwt",
        "kornia",
        "imagehash",
    )
)


@app.function(
    image=image,
    gpu="T4",
    timeout=600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def poison_image(
    image_bytes: bytes,
    secret_key: str = "AICHECKERS_DEFAULT_KEY",
    strength_mid: float = 0.08,
    strength_low: float = 0.06,
    iterations: int = 50,
    normalize_resolution: bool = True,
    canonical_size: int = 512,
    version: str = None,
):
    """画像にポイズニング + 署名を適用"""
    import io
    import hashlib
    import hmac
    from datetime import datetime

    import numpy as np
    import torch
    import torch.nn.functional as F
    import ptwt
    import kornia
    import imagehash
    from PIL import Image
    from diffusers import AutoencoderKL

    start_time = time.time()
    device = torch.device("cuda")

    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # バージョン設定
    ver = version or datetime.now().strftime("%Y%m")
    full_key = f"{secret_key}_{ver}".encode()

    # VAEロード
    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
        torch_dtype=torch.float32
    ).eval().to(device)

    model_load_time = time.time() - start_time
    print(f"VAE loaded in {model_load_time:.1f}s")

    # 画像読み込み
    img_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    orig_size = img_pil.size
    print(f"Input: {orig_size[0]}x{orig_size[1]}")

    # テンソル変換
    arr = np.array(img_pil).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)

    # 解像度処理
    _, _, h, w = img_tensor.shape
    if normalize_resolution:
        scale = canonical_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        new_h = (new_h // 8) * 8
        new_w = (new_w // 8) * 8
        img_processed = F.interpolate(img_tensor, (new_h, new_w), mode='bilinear', align_corners=False)
    else:
        new_h = (h // 8) * 8
        new_w = (w // 8) * 8
        if new_h != h or new_w != w:
            img_processed = F.interpolate(img_tensor, (new_h, new_w), mode='bilinear', align_corners=False)
        else:
            img_processed = img_tensor

    print(f"Processing size: {img_processed.shape[-1]}x{img_processed.shape[-2]}")

    # YCbCr変換
    img_ycbcr = kornia.color.rgb_to_ycbcr(img_processed)
    Y, Cb, Cr = torch.chunk(img_ycbcr, 3, dim=1)

    # 知覚マスク
    edges = kornia.filters.sobel(Y)
    edge_magnitude = edges.abs().mean(dim=1, keepdim=True)
    mask = 0.2 + 0.8 * torch.sigmoid((edge_magnitude - 0.1) * 20)

    # DWT
    WAVELET = 'bior1.3'
    coeffs = ptwt.wavedec2(Y, WAVELET, level=1)
    LL = coeffs[0]
    LH, HL, HH = coeffs[1]

    # マスクダウンサンプリング
    mask_mid = F.interpolate(mask, size=LH.shape[-2:], mode='bilinear', align_corners=False)

    # pHashベースソルト
    Y_np = Y.squeeze(0).squeeze(0).cpu().numpy()
    Y_pil = Image.fromarray((Y_np * 255).clip(0, 255).astype(np.uint8), mode='L')
    phash = imagehash.phash(Y_pil, hash_size=8)
    image_salt = str(phash)
    print(f"Image salt (pHash): {image_salt}")

    # 署名パターン生成
    key = hmac.new(full_key, image_salt.encode(), hashlib.sha256).digest()
    seed = int.from_bytes(key[:8], 'big')
    rng = torch.Generator().manual_seed(seed)
    sig_pattern = (torch.rand(LH.shape, generator=rng) * 2 - 1).to(device)

    # ===== Untargeted Semantic Attack (LL) =====
    print(f"Starting semantic attack ({iterations} iterations)...")
    poison_start = time.time()

    with torch.no_grad():
        upscaled_orig = F.interpolate(LL, size=(canonical_size, canonical_size), mode='bilinear', align_corners=False)
        rgb_orig = upscaled_orig.repeat(1, 3, 1, 1)
        rgb_orig_scaled = rgb_orig * 2.0 - 1.0
        target_latent = vae.encode(rgb_orig_scaled).latent_dist.mean

    optimized_LL = LL.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([optimized_LL], lr=0.01)

    for i in range(iterations):
        upscaled_opt = F.interpolate(optimized_LL, size=(canonical_size, canonical_size), mode='bilinear', align_corners=False)
        rgb_opt = upscaled_opt.repeat(1, 3, 1, 1)
        rgb_opt_scaled = rgb_opt * 2.0 - 1.0
        current_latent = vae.encode(rgb_opt_scaled).latent_dist.mean

        loss = F.cosine_similarity(current_latent.flatten(), target_latent.flatten(), dim=0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            perturbation = optimized_LL - LL
            perturbation = torch.clamp(perturbation, -strength_low, strength_low)
            optimized_LL.data = LL + perturbation

        if (i + 1) % 25 == 0:
            print(f"  Iter {i+1}: loss={loss.item():.4f}")

    LL_poisoned = optimized_LL.detach()
    poison_time = time.time() - poison_start
    print(f"Semantic attack done in {poison_time:.1f}s")

    # ===== 署名埋め込み (LH/HL) =====
    perturbation_mid = sig_pattern * strength_mid * mask_mid
    LH_poisoned = LH + perturbation_mid
    HL_poisoned = HL + perturbation_mid

    # HH (微小ノイズ)
    HH_poisoned = HH + (torch.randn_like(HH) * 0.02)

    # DWT逆変換
    Y_poisoned = ptwt.waverec2([LL_poisoned, (LH_poisoned, HL_poisoned, HH_poisoned)], WAVELET)
    if Y_poisoned.shape != Y.shape:
        Y_poisoned = Y_poisoned[:, :, :Y.shape[2], :Y.shape[3]]

    # YCbCr→RGB
    poisoned_ycbcr = torch.cat([Y_poisoned, Cb, Cr], dim=1)
    poisoned_rgb = kornia.color.ycbcr_to_rgb(poisoned_ycbcr)

    # 元サイズに戻す
    if poisoned_rgb.shape[-2:] != (h, w):
        poisoned_rgb = F.interpolate(poisoned_rgb, (h, w), mode='bilinear', align_corners=False)

    poisoned_rgb = torch.clamp(poisoned_rgb, 0, 1)

    # PIL変換
    result_arr = poisoned_rgb.squeeze(0).permute(1, 2, 0).cpu().numpy()
    result_arr = (result_arr * 255).clip(0, 255).astype(np.uint8)
    result_pil = Image.fromarray(result_arr)

    # bytes変換
    output_buffer = io.BytesIO()
    result_pil.save(output_buffer, format="PNG")
    result_bytes = output_buffer.getvalue()

    # ===== 署名検証 =====
    print("\nVerifying signature...")
    # 再度DWT
    if normalize_resolution:
        verify_tensor = F.interpolate(poisoned_rgb, (new_h, new_w), mode='bilinear', align_corners=False)
    else:
        verify_tensor = poisoned_rgb
    verify_ycbcr = kornia.color.rgb_to_ycbcr(verify_tensor)
    Y_verify = verify_ycbcr[:, 0:1, :, :]

    # pHash再計算
    Y_verify_np = Y_verify.squeeze(0).squeeze(0).cpu().numpy()
    Y_verify_pil = Image.fromarray((Y_verify_np * 255).clip(0, 255).astype(np.uint8), mode='L')
    verify_phash = imagehash.phash(Y_verify_pil, hash_size=8)
    verify_salt = str(verify_phash)

    coeffs_verify = ptwt.wavedec2(Y_verify, WAVELET, level=1)
    LH_verify = coeffs_verify[1][0]

    # 期待パターン
    key_verify = hmac.new(full_key, verify_salt.encode(), hashlib.sha256).digest()
    seed_verify = int.from_bytes(key_verify[:8], 'big')
    rng_verify = torch.Generator().manual_seed(seed_verify)
    expected = (torch.rand(LH_verify.shape, generator=rng_verify) * 2 - 1).to(device)

    # 相関
    a = LH_verify.flatten().float()
    b = expected.flatten().float()
    a_norm = (a - a.mean()) / (a.std() + 1e-8)
    b_norm = (b - b.mean()) / (b.std() + 1e-8)
    correlation = (a_norm * b_norm).mean().item()

    detection_threshold = 0.20
    detected = correlation > detection_threshold

    total_time = time.time() - start_time

    print(f"\n=== Results ===")
    print(f"Signature detected: {detected}")
    print(f"Correlation: {correlation:.4f} (threshold: {detection_threshold})")
    print(f"Salt match: {image_salt == verify_salt}")
    print(f"Total time: {total_time:.1f}s")

    return {
        "image_bytes": result_bytes,
        "stats": {
            "model_load_time": model_load_time,
            "poison_time": poison_time,
            "total_time": total_time,
            "signature_detected": detected,
            "correlation": correlation,
            "image_salt": image_salt,
            "version": ver,
            "iterations": iterations,
            "strength_mid": strength_mid,
            "strength_low": strength_low,
            "canonical_size": canonical_size,
        }
    }


@app.function(
    image=image,
    gpu="T4",
    timeout=300,
)
def detect_signature(
    image_bytes: bytes,
    secret_key: str = "AICHECKERS_DEFAULT_KEY",
    version: str = None,
    normalize_resolution: bool = True,
    canonical_size: int = 512,
):
    """署名検出のみ（VAE不要で高速）"""
    import io
    import hashlib
    import hmac
    from datetime import datetime

    import numpy as np
    import torch
    import torch.nn.functional as F
    import ptwt
    import kornia
    import imagehash
    from PIL import Image

    start_time = time.time()
    device = torch.device("cuda")

    ver = version or datetime.now().strftime("%Y%m")
    full_key = f"{secret_key}_{ver}".encode()

    # 画像読み込み
    img_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    print(f"Input: {img_pil.size[0]}x{img_pil.size[1]}")

    # テンソル変換
    arr = np.array(img_pil).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)

    # 解像度処理
    _, _, h, w = img_tensor.shape
    if normalize_resolution:
        scale = canonical_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        new_h = (new_h // 8) * 8
        new_w = (new_w // 8) * 8
        img_processed = F.interpolate(img_tensor, (new_h, new_w), mode='bilinear', align_corners=False)
    else:
        new_h = (h // 8) * 8
        new_w = (w // 8) * 8
        if new_h != h or new_w != w:
            img_processed = F.interpolate(img_tensor, (new_h, new_w), mode='bilinear', align_corners=False)
        else:
            img_processed = img_tensor

    # YCbCr
    img_ycbcr = kornia.color.rgb_to_ycbcr(img_processed)
    Y = img_ycbcr[:, 0:1, :, :]

    # pHash
    Y_np = Y.squeeze(0).squeeze(0).cpu().numpy()
    Y_pil = Image.fromarray((Y_np * 255).clip(0, 255).astype(np.uint8), mode='L')
    phash = imagehash.phash(Y_pil, hash_size=8)
    image_salt = str(phash)

    # DWT
    WAVELET = 'bior1.3'
    coeffs = ptwt.wavedec2(Y, WAVELET, level=1)
    LH, HL = coeffs[1][0], coeffs[1][1]

    # 期待パターン
    key = hmac.new(full_key, image_salt.encode(), hashlib.sha256).digest()
    seed = int.from_bytes(key[:8], 'big')
    rng = torch.Generator().manual_seed(seed)
    expected = (torch.rand(LH.shape, generator=rng) * 2 - 1).to(device)

    # 相関
    def corr(a, b):
        a = a.flatten().float()
        b = b.flatten().float()
        a_norm = (a - a.mean()) / (a.std() + 1e-8)
        b_norm = (b - b.mean()) / (b.std() + 1e-8)
        return (a_norm * b_norm).mean().item()

    correlation_lh = corr(LH, expected)
    correlation_hl = corr(HL, expected)
    avg_correlation = (correlation_lh + correlation_hl) / 2

    detection_threshold = 0.20
    detected = avg_correlation > detection_threshold

    total_time = time.time() - start_time

    return {
        "detected": detected,
        "correlation": avg_correlation,
        "correlation_lh": correlation_lh,
        "correlation_hl": correlation_hl,
        "threshold": detection_threshold,
        "version": ver,
        "image_salt": image_salt,
        "processing_time": total_time,
    }


@app.local_entrypoint()
def main():
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
    print("=" * 60)

    with open(test_image, "rb") as f:
        image_bytes = f.read()

    print("Running poison_image on Modal (T4 GPU)...")
    result = poison_image.remote(
        image_bytes=image_bytes,
        strength_mid=0.08,
        strength_low=0.06,
        iterations=50,
        normalize_resolution=True,
    )

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    stats = result["stats"]
    print(f"Total time: {stats['total_time']:.1f}s")
    print(f"Signature detected: {stats['signature_detected']}")
    print(f"Correlation: {stats['correlation']:.4f}")
    print(f"Image salt: {stats['image_salt']}")
    print(f"Version: {stats['version']}")

    # 保存
    output_path = Path("/home/techne/aicheckers/data/test_ironclad_poisoned.png")
    with open(output_path, "wb") as f:
        f.write(result["image_bytes"])
    print(f"\nOutput saved: {output_path}")

    # コスト概算
    cost_per_sec = 0.000164  # T4
    estimated_cost = stats["total_time"] * cost_per_sec
    print(f"\n=== Cost Estimate ===")
    print(f"T4 GPU time: {stats['total_time']:.1f}s")
    print(f"Estimated cost: ${estimated_cost:.4f}")
