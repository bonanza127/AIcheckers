#!/usr/bin/env python3
"""
Single Image Attack Test - 1枚テスト用

magnitude 0.01でVAE攻撃 + High-Frequency攻撃を検証

Usage:
    modal run scripts/test_single_attack.py --magnitude 0.01
"""

import modal
from pathlib import Path

app = modal.App("test-single-attack")

volume = modal.Volume.from_name("ironclad-test-vol", create_if_missing=True)
VOLUME_PATH = "/vol"

attack_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch==2.1.2",
        "torchvision==0.16.2",
        "diffusers==0.25.1",
        "transformers==4.38.2",
        "huggingface_hub==0.21.4",
        "accelerate",
        "safetensors",
        "lpips",
        "pillow",
        "numpy<2.0",
        "tqdm",
        "opencv-python-headless",
    )
)


@app.function(
    image=attack_image,
    volumes={VOLUME_PATH: volume},
    gpu="A10G",
    timeout=1800,
)
def test_attack(
    magnitude: float = 0.01,
    iterations: int = 200,
    lpips_weight: float = 3.0,
):
    """
    1枚のテスト画像に対してVAE攻撃を実行

    Args:
        magnitude: 摂動の上限（0.01 = 1%）
        iterations: 最適化イテレーション数
        lpips_weight: LPIPS制約の重み
    """
    import torch
    import torch.nn.functional as F
    import lpips
    import cv2
    from PIL import Image
    from pathlib import Path
    from diffusers import AutoencoderKL
    from torchvision import transforms
    from tqdm import tqdm
    import numpy as np

    device = torch.device("cuda")

    # Load SDXL VAE
    print("Loading SDXL VAE...")
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sdxl-vae",
        torch_dtype=torch.float32,
    ).to(device)
    vae.eval()

    # Load LPIPS
    print("Loading LPIPS...")
    lpips_fn = lpips.LPIPS(net='vgg').to(device)
    lpips_fn.eval()

    print(f"\n=== Attack Parameters ===")
    print(f"magnitude (epsilon): {magnitude}")
    print(f"lpips_weight: {lpips_weight}")
    print(f"iterations: {iterations}")

    # Image transforms
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
    ])

    # テスト画像を探す（train_normalから1枚 = オリジナル未攻撃画像）
    input_path = Path(VOLUME_PATH) / "train_normal"
    image_files = list(input_path.glob("*.png"))
    if not image_files:
        print("No PNG files found in train_normal")
        return {"error": "No original image found in train_normal"}

    img_file = image_files[0]
    print(f"\nTest image: {img_file.name}")

    # Load image
    img = Image.open(img_file).convert("RGB")
    img_np = np.array(img.resize((1024, 1024)))
    x_orig = transform(img).unsqueeze(0).to(device)

    # === Create edge mask (Canny) ===
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_mask_np = (edges > 0).astype(np.float32)

    # Dilate edges
    kernel = np.ones((3, 3), np.uint8)
    edge_mask_np = cv2.dilate(edge_mask_np, kernel, iterations=2)
    edge_ratio = edge_mask_np.mean()
    print(f"Edge ratio: {edge_ratio*100:.1f}%")

    edge_mask = torch.from_numpy(edge_mask_np).to(device)
    edge_mask = edge_mask.unsqueeze(0).unsqueeze(0).expand(1, 3, 1024, 1024)

    # Normalize to [-1, 1] for VAE
    x_orig_norm = x_orig * 2.0 - 1.0

    # Get original latent
    with torch.no_grad():
        z_orig = vae.encode(x_orig_norm).latent_dist.sample()
        z_orig = z_orig * vae.config.scaling_factor

    # Epsilon設定: エッジ部分はmagnitude、平坦部分はmagnitude/2
    epsilon_edge = magnitude
    epsilon_flat = magnitude / 2

    print(f"epsilon_edge: {epsilon_edge}, epsilon_flat: {epsilon_flat}")

    # Initialize perturbation
    delta = torch.zeros_like(x_orig, requires_grad=True, device=device)
    optimizer = torch.optim.Adam([delta], lr=0.005)

    # Optimization loop
    pbar = tqdm(range(iterations), desc="Optimizing")
    for i in pbar:
        optimizer.zero_grad()

        # Apply perturbation with adaptive clipping
        epsilon_map = edge_mask * epsilon_edge + (1 - edge_mask) * epsilon_flat
        delta_clipped = torch.clamp(delta, -epsilon_map, epsilon_map)

        x_adv = torch.clamp(x_orig + delta_clipped, 0, 1)
        x_adv_norm = x_adv * 2.0 - 1.0

        # Get adversarial latent
        z_adv = vae.encode(x_adv_norm).latent_dist.sample()
        z_adv = z_adv * vae.config.scaling_factor

        # Cosine similarity loss
        z_orig_flat = z_orig.view(1, -1)
        z_adv_flat = z_adv.view(1, -1)
        cos_sim = F.cosine_similarity(z_orig_flat, z_adv_flat)

        # LPIPS loss
        lpips_loss = lpips_fn(x_orig_norm, x_adv_norm)

        # Total loss
        loss = -cos_sim + lpips_weight * lpips_loss.mean()

        loss.backward()
        optimizer.step()

        # Clip
        with torch.no_grad():
            delta.data = torch.clamp(delta.data, -epsilon_edge, epsilon_edge)

        if i % 40 == 0:
            pbar.set_postfix({
                'cos_sim': f'{cos_sim.item():.4f}',
                'lpips': f'{lpips_loss.item():.4f}',
            })

    # Final result
    with torch.no_grad():
        epsilon_map = edge_mask * epsilon_edge + (1 - edge_mask) * epsilon_flat
        delta_final = torch.clamp(delta, -epsilon_map, epsilon_map)
        x_adv_final = torch.clamp(x_orig + delta_final, 0, 1)

        x_adv_final_norm = x_adv_final * 2.0 - 1.0
        z_adv_final = vae.encode(x_adv_final_norm).latent_dist.sample()
        z_adv_final = z_adv_final * vae.config.scaling_factor

        final_cos_sim = F.cosine_similarity(
            z_orig.view(1, -1),
            z_adv_final.view(1, -1)
        ).item()

        final_lpips = lpips_fn(x_orig_norm, x_adv_final_norm).item()

        # Perturbation stats
        delta_abs = delta_final.abs()
        max_delta = delta_abs.max().item()
        mean_delta = delta_abs.mean().item()

    print(f"\n{'='*60}")
    print(f"=== Results (magnitude={magnitude}) ===")
    print(f"{'='*60}")
    print(f"Final Cosine Similarity: {final_cos_sim:.4f}")
    print(f"Final LPIPS: {final_lpips:.4f}")
    print(f"Max perturbation: {max_delta:.4f} ({max_delta*100:.2f}%)")
    print(f"Mean perturbation: {mean_delta:.4f} ({mean_delta*100:.2f}%)")

    # Save result
    output_path = Path(VOLUME_PATH) / f"test_magnitude_{magnitude}.png"
    x_adv_np = (x_adv_final.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    out_img = Image.fromarray(x_adv_np)
    out_img.save(output_path, quality=95)
    print(f"\nSaved to: {output_path}")

    # Save original for comparison
    orig_path = Path(VOLUME_PATH) / f"test_original.png"
    x_orig_np = (x_orig.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    orig_img = Image.fromarray(x_orig_np)
    orig_img.save(orig_path, quality=95)

    # Save diff visualization (amplified)
    diff_path = Path(VOLUME_PATH) / f"test_diff_{magnitude}.png"
    diff = (delta_final.squeeze(0).permute(1, 2, 0).cpu().numpy() + magnitude) / (2 * magnitude)  # normalize to 0-1
    diff = (diff * 255).clip(0, 255).astype(np.uint8)
    diff_img = Image.fromarray(diff)
    diff_img.save(diff_path)

    volume.commit()

    return {
        'magnitude': magnitude,
        'cos_sim': final_cos_sim,
        'lpips': final_lpips,
        'max_delta': max_delta,
        'mean_delta': mean_delta,
        'output': str(output_path),
    }


@app.local_entrypoint()
def main(magnitude: float = 0.01, iterations: int = 200, lpips_weight: float = 3.0):
    """
    1枚テスト実行

    Usage:
        modal run scripts/test_single_attack.py --magnitude 0.01
        modal run scripts/test_single_attack.py --magnitude 0.015
    """
    print(f"=== Single Image Attack Test ===")
    print(f"magnitude: {magnitude}")

    result = test_attack.remote(
        magnitude=magnitude,
        iterations=iterations,
        lpips_weight=lpips_weight,
    )
    print(f"\nResult: {result}")
