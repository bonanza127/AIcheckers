#!/usr/bin/env python3
"""
High-Frequency Attack - エッジ領域への強摂動

High-Frequency Anti-DreamBooth (2024) の手法を参考に、
画像のエッジ部分に強い摂動を集中させる。

- エッジ部分: 強い摂動 (epsilon_edge)
- 平坦部分: 弱い摂動 (epsilon_flat)

これにより:
1. 視覚的には目立ちにくい
2. DiffPure等のpurificationに耐性がある
3. LoRA学習で重要な線画・輪郭情報を攻撃

Usage:
    modal run scripts/highfreq_attack.py --setup
    modal run scripts/highfreq_attack.py --attack
"""

import modal
from pathlib import Path

app = modal.App("highfreq-attack")

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
    timeout=3600,
)
def apply_highfreq_attack(
    input_dir: str,
    output_dir: str,
    iterations: int = 200,
    lr: float = 0.008,
    lpips_weight: float = 2.0,     # LPIPS≤0.07を許容（Nightshade同等）
    epsilon_edge: float = 0.05,    # エッジ部分の摂動上限（5%）
    epsilon_flat: float = 0.01,    # 平坦部分の摂動上限（1%）
    edge_threshold: float = 0.03,  # エッジ検出閾値
    use_canny: bool = True,        # Cannyエッジ検出
    target_lpips: float = 0.07,    # 目標LPIPS（Nightshade基準）
):
    """
    High-Frequency攻撃 + VAE Latent攻撃の統合（改良版）

    改良点:
    - epsilon削減: エッジ15%→5%、平坦3%→1%
    - LPIPS重み増加: 0.5→3.0
    - Cannyエッジ検出オプション追加

    1. Laplacian/Cannyフィルタでエッジマスクを生成
    2. エッジ部分には適度な摂動、平坦部分には最小限の摂動
    3. VAE latent空間での類似度も同時に最小化
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

    # Image transforms
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
    ])

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_files = list(input_path.glob("*.png")) + list(input_path.glob("*.jpg"))
    print(f"Found {len(image_files)} images")

    results = []

    for img_file in image_files:
        print(f"\nProcessing: {img_file.name}")

        # Load image
        img = Image.open(img_file).convert("RGB")
        img_np = np.array(img.resize((1024, 1024)))
        x_orig = transform(img).unsqueeze(0).to(device)

        # === Create edge mask ===
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        if use_canny:
            # Canny edge detection (より精密なエッジ検出)
            edges = cv2.Canny(gray, 50, 150)
            edge_mask_np = (edges > 0).astype(np.float32)
        else:
            # Laplacian filter
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            laplacian_abs = np.abs(laplacian)
            laplacian_norm = laplacian_abs / (laplacian_abs.max() + 1e-8)
            edge_mask_np = (laplacian_norm > edge_threshold).astype(np.float32)

        # Dilate to widen edge regions (エッジを少し太くする)
        kernel = np.ones((3, 3), np.uint8)
        edge_mask_np = cv2.dilate(edge_mask_np, kernel, iterations=2)

        edge_ratio = edge_mask_np.mean()
        print(f"Edge ratio: {edge_ratio*100:.1f}%")

        # Convert to tensor and expand to 3 channels
        edge_mask = torch.from_numpy(edge_mask_np).to(device)
        edge_mask = edge_mask.unsqueeze(0).unsqueeze(0).expand(1, 3, 1024, 1024)

        # Normalize to [-1, 1] for VAE
        x_orig_norm = x_orig * 2.0 - 1.0

        # Get original latent
        with torch.no_grad():
            z_orig = vae.encode(x_orig_norm).latent_dist.sample()
            z_orig = z_orig * vae.config.scaling_factor

        # Initialize perturbation
        delta = torch.zeros_like(x_orig, requires_grad=True, device=device)
        optimizer = torch.optim.Adam([delta], lr=lr)

        # Optimization loop
        pbar = tqdm(range(iterations), desc="Optimizing")
        for i in pbar:
            optimizer.zero_grad()

            # Apply perturbation with adaptive clipping based on edge mask
            # Edge regions: larger epsilon, Flat regions: smaller epsilon
            epsilon_map = edge_mask * epsilon_edge + (1 - edge_mask) * epsilon_flat
            delta_clipped = torch.clamp(delta, -epsilon_map, epsilon_map)

            x_adv = torch.clamp(x_orig + delta_clipped, 0, 1)
            x_adv_norm = x_adv * 2.0 - 1.0

            # Get adversarial latent
            z_adv = vae.encode(x_adv_norm).latent_dist.sample()
            z_adv = z_adv * vae.config.scaling_factor

            # Cosine similarity loss (minimize = make different)
            z_orig_flat = z_orig.view(1, -1)
            z_adv_flat = z_adv.view(1, -1)
            cos_sim = F.cosine_similarity(z_orig_flat, z_adv_flat)

            # LPIPS loss (minimize = keep visually similar)
            lpips_loss = lpips_fn(x_orig_norm, x_adv_norm)

            # Edge-weighted perturbation magnitude penalty
            # Encourage stronger perturbation on edges
            edge_perturbation = (delta.abs() * edge_mask).mean()
            flat_perturbation = (delta.abs() * (1 - edge_mask)).mean()

            # Total loss
            loss = -cos_sim + lpips_weight * lpips_loss.mean()

            loss.backward()
            optimizer.step()

            # Apply adaptive clipping after update
            with torch.no_grad():
                delta.data = torch.clamp(delta.data, -epsilon_edge, epsilon_edge)

            if i % 20 == 0:
                pbar.set_postfix({
                    'cos_sim': f'{cos_sim.item():.4f}',
                    'lpips': f'{lpips_loss.item():.4f}',
                    'loss': f'{loss.item():.4f}'
                })

        # Final adversarial image with adaptive clipping
        with torch.no_grad():
            epsilon_map = edge_mask * epsilon_edge + (1 - edge_mask) * epsilon_flat
            delta_final = torch.clamp(delta, -epsilon_map, epsilon_map)
            x_adv_final = torch.clamp(x_orig + delta_final, 0, 1)

            # Verify latent dissimilarity
            x_adv_final_norm = x_adv_final * 2.0 - 1.0
            z_adv_final = vae.encode(x_adv_final_norm).latent_dist.sample()
            z_adv_final = z_adv_final * vae.config.scaling_factor

            final_cos_sim = F.cosine_similarity(
                z_orig.view(1, -1),
                z_adv_final.view(1, -1)
            ).item()

            final_lpips = lpips_fn(x_orig_norm, x_adv_final_norm).item()

        print(f"Final - Cosine Sim: {final_cos_sim:.4f}, LPIPS: {final_lpips:.4f}")

        # Save
        x_adv_np = (x_adv_final.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        out_img = Image.fromarray(x_adv_np)
        out_path = output_path / img_file.name
        out_img.save(out_path, quality=95)

        # Copy caption if exists
        caption_file = img_file.with_suffix(".txt")
        if caption_file.exists():
            (output_path / caption_file.name).write_text(caption_file.read_text())

        results.append({
            'file': img_file.name,
            'cos_sim': final_cos_sim,
            'lpips': final_lpips,
            'edge_ratio': edge_ratio,
        })

    volume.commit()

    # Summary
    avg_cos_sim = np.mean([r['cos_sim'] for r in results])
    avg_lpips = np.mean([r['lpips'] for r in results])
    avg_edge = np.mean([r['edge_ratio'] for r in results])

    print(f"\n{'='*60}")
    print(f"High-Frequency Attack Complete")
    print(f"{'='*60}")
    print(f"Images processed: {len(results)}")
    print(f"Average Cosine Similarity: {avg_cos_sim:.4f}")
    print(f"Average LPIPS: {avg_lpips:.4f}")
    print(f"Average Edge Ratio: {avg_edge*100:.1f}%")

    return {
        'status': 'success',
        'count': len(results),
        'avg_cos_sim': avg_cos_sim,
        'avg_lpips': avg_lpips,
        'avg_edge_ratio': avg_edge,
        'results': results,
    }


@app.function(
    image=attack_image,
    volumes={VOLUME_PATH: volume},
    gpu="A10G",
    timeout=600,
)
def setup_check():
    """環境確認"""
    import torch
    import cv2
    from diffusers import AutoencoderKL
    import lpips

    print(f"PyTorch: {torch.__version__}")
    print(f"OpenCV: {cv2.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")

    # Check VAE
    print("\nLoading SDXL VAE...")
    vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")
    print(f"VAE loaded: {type(vae)}")

    # Check LPIPS
    print("\nLoading LPIPS...")
    lpips_fn = lpips.LPIPS(net='vgg')
    print(f"LPIPS loaded: {type(lpips_fn)}")

    # Check training data
    from pathlib import Path
    for folder in ["train_normal", "train_vae_attack"]:
        data_path = Path(VOLUME_PATH) / folder
        if data_path.exists():
            images = list(data_path.glob("*.png"))
            print(f"\n{folder}: {len(images)} images")

    volume.commit()
    return {"status": "ready"}


@app.local_entrypoint()
def main(
    setup: bool = False,
    attack: bool = False,
    input_folder: str = "train_normal",
    output_folder: str = "train_hf_attack",
    iterations: int = 200,
    lpips_weight: float = 2.0,
    epsilon_edge: float = 0.05,
    epsilon_flat: float = 0.01,
    use_canny: bool = True,
):
    """
    Main entrypoint

    Nightshade-level settings (LPIPS ≤ 0.07):
    - epsilon_edge: 0.05 (5%)
    - epsilon_flat: 0.01 (1%)
    - lpips_weight: 2.0 (allows LPIPS up to ~0.07)
    - iterations: 200
    """

    if setup:
        print("=== Setup Check ===")
        result = setup_check.remote()
        print(f"Result: {result}")

    if attack:
        print("\n=== High-Frequency Attack (Stealth Mode) ===")
        result = apply_highfreq_attack.remote(
            input_dir=f"{VOLUME_PATH}/{input_folder}",
            output_dir=f"{VOLUME_PATH}/{output_folder}",
            iterations=iterations,
            lpips_weight=lpips_weight,
            epsilon_edge=epsilon_edge,
            epsilon_flat=epsilon_flat,
            use_canny=use_canny,
        )
        print(f"Result: {result}")


if __name__ == "__main__":
    print("Use: modal run scripts/highfreq_attack.py --help")
