#!/usr/bin/env python3
"""
VAE Latent Attack - VAEボトルネック攻撃

DWT空間ではなく、VAE latent空間を直接攻撃することで、
LoRA学習を効果的に妨害する。

Loss = -CosineSim(E(x + δx), E(x)) + λ·LPIPS(x, x + δx)

Usage:
    modal run scripts/vae_latent_attack.py --setup
    modal run scripts/vae_latent_attack.py --attack
    modal run scripts/vae_latent_attack.py --train
"""

import modal
from pathlib import Path

app = modal.App("vae-latent-attack")

volume = modal.Volume.from_name("ironclad-test-vol", create_if_missing=True)
VOLUME_PATH = "/vol"

# Image with VAE and LPIPS support
attack_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch==2.1.2",
        "torchvision==0.16.2",
        "diffusers==0.25.1",
        "transformers==4.38.2",
        "huggingface_hub==0.21.4",  # バージョン固定
        "accelerate",
        "safetensors",
        "lpips",
        "pillow",
        "numpy<2.0",
        "tqdm",
    )
)


def compute_lowfreq_energy(delta: "torch.Tensor", lowpass_filter: "torch.Tensor"):
    """
    摂動の低周波エネルギーを計算（ペナライズ用）
    低周波にノイズがあると目立つので、これを最小化したい。

    Returns:
        低周波成分のエネルギー（スカラー）
    """
    import torch

    device = delta.device
    batch, channels, h, w = delta.shape

    lp = lowpass_filter.to(device)
    total_energy = 0.0

    for c in range(channels):
        # FFT
        freq = torch.fft.fft2(delta[0, c])
        freq_shifted = torch.fft.fftshift(freq)

        # 低周波成分のマグニチュード
        lowfreq_mag = torch.abs(freq_shifted) * lp

        # エネルギー（L2ノルム）
        total_energy = total_energy + torch.sum(lowfreq_mag ** 2)

    return total_energy / (channels * h * w)


def create_lowpass_filter(size: int, cutoff: float = 0.1):
    """
    低周波通過フィルタを作成（ペナライズ用）

    Args:
        size: 画像サイズ
        cutoff: カットオフ周波数（0-1、低いほど中心部のみ）
    """
    import torch
    import numpy as np

    center = size // 2
    y, x = np.ogrid[:size, :size]
    distance = np.sqrt((x - center) ** 2 + (y - center) ** 2)
    max_dist = np.sqrt(2) * center
    normalized_dist = distance / max_dist

    # ガウシアン低周波フィルタ（中心部を1、周辺を0）
    lowpass = np.exp(-0.5 * (normalized_dist / cutoff) ** 2)

    return torch.from_numpy(lowpass).float()


@app.function(
    image=attack_image,
    volumes={VOLUME_PATH: volume},
    gpu="A10G",
    timeout=3600,
)
def apply_vae_latent_attack(
    input_dir: str,
    output_dir: str,
    iterations: int = 150,
    lr: float = 0.005,
    lpips_weight: float = 3.0,
    epsilon: float = 0.02,  # 2%摂動（視認困難レベル）
    frequency_mask: bool = True,  # 高周波にノイズを集中
    highpass_cutoff: float = 0.15,  # 高周波カットオフ
):
    """
    VAE latent空間を攻撃する摂動を画像に適用

    改良版: ノイズの視認性を最小化
    - epsilon削減: 0.1 → 0.02
    - LPIPS重み増加: 0.5 → 3.0
    - 高周波フィルタ: ノイズをエッジに集中（オプション）

    Loss = -CosineSim(E(x + δx), E(x)) + λ·LPIPS(x, x + δx)
    """
    import torch
    import torch.nn.functional as F
    import lpips
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
        torch_dtype=torch.float32,  # 勾配計算のためfp32
    ).to(device)
    vae.eval()

    # Load LPIPS
    print("Loading LPIPS...")
    lpips_fn = lpips.LPIPS(net='vgg').to(device)
    lpips_fn.eval()

    # 低周波ペナルティ用フィルタ（オプション）
    lowpass_filter = None
    if frequency_mask:
        print(f"Creating lowpass filter for penalty (cutoff={highpass_cutoff})...")
        lowpass_filter = create_lowpass_filter(1024, highpass_cutoff)

    print(f"\n=== Attack Parameters ===")
    print(f"epsilon: {epsilon} (max perturbation)")
    print(f"lpips_weight: {lpips_weight}")
    print(f"frequency_mask: {frequency_mask}")
    print(f"iterations: {iterations}")

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
        x_orig = transform(img).unsqueeze(0).to(device)  # [1, 3, 1024, 1024]

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
        lowfreq_weight = 0.1  # 低周波ペナルティの重み
        pbar = tqdm(range(iterations), desc="Optimizing")
        for i in pbar:
            optimizer.zero_grad()

            # Apply perturbation with clipping
            x_adv = torch.clamp(x_orig + delta, 0, 1)
            x_adv_norm = x_adv * 2.0 - 1.0

            # Get adversarial latent
            z_adv = vae.encode(x_adv_norm).latent_dist.sample()
            z_adv = z_adv * vae.config.scaling_factor

            # Cosine similarity loss (minimize = make different)
            z_orig_flat = z_orig.view(1, -1)
            z_adv_flat = z_adv.view(1, -1)
            cos_sim = F.cosine_similarity(z_orig_flat, z_adv_flat)

            # LPIPS loss (minimize = keep visually similar)
            # LPIPS expects [-1, 1] range
            lpips_loss = lpips_fn(x_orig_norm, x_adv_norm)

            # Low-frequency penalty (push noise to high frequencies)
            lowfreq_loss = torch.tensor(0.0, device=device)
            if frequency_mask and lowpass_filter is not None:
                lowfreq_loss = compute_lowfreq_energy(delta, lowpass_filter)

            # Total loss:
            # - minimize -cos_sim (maximize latent dissimilarity)
            # + LPIPS (keep visually similar)
            # + lowfreq_penalty (push noise to high frequencies)
            loss = -cos_sim + lpips_weight * lpips_loss.mean() + lowfreq_weight * lowfreq_loss

            loss.backward()
            optimizer.step()

            # Clip perturbation to epsilon ball
            with torch.no_grad():
                delta.data = torch.clamp(delta.data, -epsilon, epsilon)

            if i % 30 == 0:
                pbar.set_postfix({
                    'cos': f'{cos_sim.item():.3f}',
                    'lpips': f'{lpips_loss.item():.4f}',
                    'lowf': f'{lowfreq_loss.item():.4f}' if frequency_mask else 'N/A',
                })

        # Final adversarial image
        with torch.no_grad():
            x_adv_final = torch.clamp(x_orig + delta, 0, 1)

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
        })

    volume.commit()

    # Summary
    avg_cos_sim = np.mean([r['cos_sim'] for r in results])
    avg_lpips = np.mean([r['lpips'] for r in results])

    print(f"\n{'='*60}")
    print(f"VAE Latent Attack Complete")
    print(f"{'='*60}")
    print(f"Images processed: {len(results)}")
    print(f"Average Cosine Similarity: {avg_cos_sim:.4f}")
    print(f"Average LPIPS: {avg_lpips:.4f}")

    return {
        'status': 'success',
        'count': len(results),
        'avg_cos_sim': avg_cos_sim,
        'avg_lpips': avg_lpips,
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
    from diffusers import AutoencoderKL
    import lpips

    print(f"PyTorch: {torch.__version__}")
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
    train_normal = Path(VOLUME_PATH) / "train_normal"
    if train_normal.exists():
        images = list(train_normal.glob("*.png"))
        print(f"\ntrain_normal: {len(images)} images")

    volume.commit()
    return {"status": "ready"}


@app.local_entrypoint()
def main(
    setup: bool = False,
    attack: bool = False,
    iterations: int = 150,
    lpips_weight: float = 3.0,
    epsilon: float = 0.02,
    frequency_mask: bool = True,
    highpass_cutoff: float = 0.15,
):
    """
    Main entrypoint

    Improved defaults for invisible perturbation:
    - epsilon: 0.02 (2% max, down from 10%)
    - lpips_weight: 3.0 (up from 0.5)
    - frequency_mask: True (concentrate noise in high frequencies)
    """

    if setup:
        print("=== Setup Check ===")
        result = setup_check.remote()
        print(f"Result: {result}")

    if attack:
        print("\n=== VAE Latent Attack (Stealth Mode) ===")
        result = apply_vae_latent_attack.remote(
            input_dir=f"{VOLUME_PATH}/train_normal",
            output_dir=f"{VOLUME_PATH}/train_vae_attack",
            iterations=iterations,
            lpips_weight=lpips_weight,
            epsilon=epsilon,
            frequency_mask=frequency_mask,
            highpass_cutoff=highpass_cutoff,
        )
        print(f"Result: {result}")


if __name__ == "__main__":
    print("Use: modal run scripts/vae_latent_attack.py --help")
