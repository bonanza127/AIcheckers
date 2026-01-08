#!/usr/bin/env python3
"""
FastProtect Local Inference V3 - Low VRAM Optimized
Analysis @ 512px -> Protection @ Original Resolution

Optimizations:
1. Low VRAM usage (Analysis on 512x512 downscaled version)
2. Alpha Channel Support (Preserves transparency)
3. Async Saving (Parallel I/O for speed)
4. High Quality Upscaling (Bicubic interpolation for sharper protection)
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm
import argparse
import sys
import pickle
import numpy as np
import json
import concurrent.futures

class FastProtectPerturbations:
    def __init__(self, K=4, image_size=512, eta=8/255, device="cuda", num_targets=3):
        self.K = K
        self.eta = eta
        self.eta_half = eta / 2
        self.device = device
        self.image_size = image_size
        self.num_targets = num_targets

        self.delta_g = [self._init_perturbation() for _ in range(num_targets)]
        self.Delta = [[self._init_perturbation() for _ in range(K)] for _ in range(num_targets)]

    def _init_perturbation(self):
        import torch.nn as nn
        delta = nn.Parameter(
            torch.randn(3, self.image_size, self.image_size, device=self.device) * 0.001
        )
        return delta

    @classmethod
    def load(cls, path, device="cuda"):
        """摂動をロード"""
        checkpoint = torch.load(path, map_location=device)
        instance = cls(
            K=checkpoint["K"],
            image_size=checkpoint["image_size"],
            eta=checkpoint["eta"],
            device=device,
            num_targets=checkpoint.get("num_targets", 3),
        )

        for t, d in enumerate(checkpoint["delta_g"]):
            instance.delta_g[t].data.copy_(d.to(device))
        for t, deltas in enumerate(checkpoint["Delta"]):
            for k, d in enumerate(deltas):
                instance.Delta[t][k].data.copy_(d.to(device))
        
        return instance, checkpoint


def predict_cluster(latent, kmeans):
    """K-meansでクラスタ予測"""
    if latent.dim() == 4:
        latent = latent.squeeze(0)

    latent_flat = latent.cpu().numpy().reshape(1, -1).astype(np.float32)
    cluster_idx = kmeans.predict(latent_flat)[0]
    return cluster_idx


def compute_latent_entropy(z):
    """Latentコードのエントロピーを計算"""
    if z.dim() == 4:
        B = z.shape[0]
        z_flat = z.view(B, -1)
    else:
        z_flat = z.view(1, -1)

    entropy = z_flat.var(dim=1)
    return entropy


def select_target_by_entropy(z, target_entropies):
    """エントロピーに基づいてターゲットを選択"""
    input_entropy = compute_latent_entropy(z).item()
    distances = [abs(input_entropy - te) for te in target_entropies]
    target_idx = distances.index(min(distances))
    return target_idx


def compute_lpips_map(lpips_model, original, protected):
    """LPIPS空間マップを計算"""
    # LPIPS expects inputs in [-1, 1]
    original_normalized = original * 2 - 1
    protected_normalized = protected * 2 - 1

    with torch.no_grad():
        # normalize=False because we already scaled to [-1, 1]
        distance_map = lpips_model(original_normalized, protected_normalized, normalize=False)

    return distance_map


def scale_perturbation_strength(
    perceptual_map,
    base_scale=0.5,
    sensitivity=3.0,
    min_scale=0.02,
):
    """知覚マップに基づいて摂動強度をスケーリング"""
    # Normalize map to [0, 1]
    M_normalized = (perceptual_map - perceptual_map.min()) / (perceptual_map.max() - perceptual_map.min() + 1e-8)

    # Invert and scale: High distance areas gets LOWER perturbation
    scaling = base_scale * (1 - M_normalized * sensitivity)
    scaling = torch.clamp(scaling, min_scale, 1.0)

    return scaling


def save_image_task(image_np, output_path, alpha_channel=None):
    """
    非同期保存用のタスク関数
    image_np: (H, W, 3) uint8 numpy array
    output_path: Path object
    alpha_channel: Optional (H, W) PIL Image or numpy array
    """
    try:
        img_out = Image.fromarray(image_np)
        
        # Alphaチャンネルの復元
        if alpha_channel is not None:
             # サイズが合わない場合はリサイズ（念のため）
            if alpha_channel.size != img_out.size:
                alpha_channel = alpha_channel.resize(img_out.size, Image.LANCZOS)
            img_out.putalpha(alpha_channel)

        # 保存形式の決定と保存
        suffix = output_path.suffix.lower()
        if suffix in ['.jpg', '.jpeg']:
            # JPEGはAlphaをサポートしないので背景を白にするなどの処理が必要だが、
            # ここではそのまま保存（Alphaは捨てられる）か、変換する。
            # 今回はユーザーの入力を尊重し、拡張子そのまま保存。
            # もしAlphaがあるのにJPEGなら、RGB変換して保存
            if alpha_channel is not None:
                bg = Image.new("RGB", img_out.size, (255, 255, 255))
                bg.paste(img_out, mask=img_out.split()[3])
                bg.save(output_path, quality=95, subsampling=0)
            else:
                img_out.save(output_path, quality=95, subsampling=0)
        else:
            # PNG, WEBP等はそのまま
            img_out.save(output_path)
            
    except Exception as e:
        print(f"Error saving {output_path.name}: {e}")


def protect_images_optimized(
    input_dir,
    output_dir,
    model_path="models/fastprotect/checkpoint_step25000.pt",
    kmeans_path="models/fastprotect/kmeans_model.pkl",
    entropies_path="models/fastprotect/target_entropies.json",
    device="cuda",
    use_adaptive=True,
    strength_multiplier=0.6,
    base_scale=0.5,
    sensitivity=3.0,
    max_workers=4, # Async save workers
):
    """
    画像保護処理（最適化版）
    - Analysis @ 512px
    - Output @ Original Resolution
    - Alpha Channel Support
    - Async Saving
    - Bicubic Upscaling
    """
    print(f"Device: {device}")
    print(f"Mode: Low VRAM Optimized (High Quality & Alpha capable)")
    print(f"Adaptive Protection: {'Enabled' if use_adaptive else 'Disabled'}")
    print(f"Strength Multiplier: {strength_multiplier}")

    # Load FastProtect Model
    print("Loading FastProtect perturbation model...")
    perturbations, _ = FastProtectPerturbations.load(model_path, device=device)

    # Load K-means
    print("Loading K-means model...")
    with open(kmeans_path, "rb") as f:
        kmeans_model = pickle.load(f)

    # Load Entropies
    print("Loading target entropies...")
    with open(entropies_path, "r") as f:
        entropy_data = json.load(f)
        target_entropies = entropy_data["entropies"]

    # Load LPIPS (if needed)
    lpips_model = None
    if use_adaptive:
        print("Loading LPIPS model...")
        import lpips
        lpips_model = lpips.LPIPS(net='alex').to(device)
        lpips_model.eval()

    # Load VAE
    print("Loading VAE (using float32 for stability against NaNs)...")
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sdxl-vae",
        torch_dtype=torch.float32, 
    ).to(device)
    vae.eval()

    # Input/Output Setup
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_files = []
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.webp"]:
        image_files.extend(input_path.glob(ext))
    
    image_files.sort()
    print(f"Found {len(image_files)} images")

    # Transform for Analysis (Fixed 512x512)
    transform_analysis = T.Compose([
        T.Resize((512, 512), interpolation=T.InterpolationMode.LANCZOS),
        T.ToTensor(),
    ])

    # Transform for Full Res (Just ToTensor)
    transform_full = T.ToTensor()

    # ThreadPool for saving images
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    futures = []

    with torch.no_grad():
        for img_file in tqdm(image_files, desc="Protecting"):
            try:
                # 1. Load Image
                img_raw = Image.open(img_file)
                w_orig, h_orig = img_raw.size
                
                # Alpha Channel Handling
                alpha_channel = None
                if img_raw.mode in ('RGBA', 'LA') or (img_raw.mode == 'P' and 'transparency' in img_raw.info):
                    img_raw = img_raw.convert("RGBA")
                    alpha_channel = img_raw.split()[3]
                    img_pil = img_raw.convert("RGB") # Analysis works on RGB
                else:
                    img_pil = img_raw.convert("RGB")
                
                # 2. Analysis Step (on 512x512 resized copy)
                img_analysis = transform_analysis(img_pil).unsqueeze(0).to(device)
                
                # VAE Encode (in float32 for stability)
                img_norm_analysis = (img_analysis * 2 - 1).to(dtype=torch.float32)
                z = vae.encode(img_norm_analysis).latent_dist.mean.float()
                
                # Decision Making
                target_idx = select_target_by_entropy(z, target_entropies)
                cluster_idx = predict_cluster(z, kmeans_model)
                
                # Get Base Perturbation (512x512)
                raw_delta = perturbations.delta_g[target_idx] + perturbations.Delta[target_idx][cluster_idx]
                
                # 3. Calculate Scaling Map (Adaptive) if enabled
                scaling_map_512 = None
                
                if use_adaptive and lpips_model is not None:
                    delta_scaled_512 = raw_delta * strength_multiplier
                    
                    if delta_scaled_512.shape[1:] != img_analysis.shape[2:]:
                         delta_scaled_512 = F.interpolate(
                            delta_scaled_512.unsqueeze(0),
                            size=img_analysis.shape[2:],
                            mode="bilinear", align_corners=False
                        ).squeeze(0)

                    surrogate_512 = torch.clamp(img_analysis + delta_scaled_512.unsqueeze(0), 0, 1)
                    
                    perceptual_map = compute_lpips_map(lpips_model, img_analysis, surrogate_512)
                    
                    scaling_map_512 = scale_perturbation_strength(
                        perceptual_map,
                        base_scale=base_scale,
                        sensitivity=sensitivity
                    )
                
                # 4. Protection Application (On Original Size)
                img_full = transform_full(img_pil).unsqueeze(0).to(device)
                
                # Upscale Delta: Using BICUBIC for better quality on large upsizing
                delta_full = F.interpolate(
                    raw_delta.unsqueeze(0),
                    size=(h_orig, w_orig),
                    mode="bicubic", # Changed from bilinear to bicubic
                    align_corners=False,
                    antialias=True # Enable antialiasing for smoothness
                ).squeeze(0)
                
                final_perturbation = delta_full * strength_multiplier
                
                # Upscale Scaling Map if Adaptive
                if scaling_map_512 is not None:
                    scaling_full = F.interpolate(
                        scaling_map_512,
                        size=(h_orig, w_orig),
                        mode="bicubic", # Changed to bicubic
                        align_corners=False,
                        antialias=True
                    ).squeeze(0)
                    
                    final_perturbation = final_perturbation * scaling_full
                
                # Apply and Clamp
                protected_full = torch.clamp(img_full + final_perturbation.unsqueeze(0), 0, 1)
                
                # 5. Async Save
                protected_np = protected_full.squeeze(0).cpu().numpy()
                protected_np = (protected_np * 255).astype("uint8").transpose(1, 2, 0)
                
                output_file = output_path / img_file.name
                
                # Submit to thread pool
                future = executor.submit(save_image_task, protected_np, output_file, alpha_channel)
                futures.append(future)
                
                tqdm.write(f"  {img_file.name}: {w_orig}x{h_orig} -> target={target_idx} (Queued)")
                
                # Clean up periodic futures to keep memory clean if list gets too long
                if len(futures) > 100:
                    futures = [f for f in futures if not f.done()]

            except Exception as e:
                print(f"Error processing {img_file.name}: {e}")
                continue

    # Wait for all saves to complete
    print("Waiting for file saving to complete...")
    concurrent.futures.wait(futures)
    print(f"\nCompleted! Protected images saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="FastProtect Local V3 - Optimized")
    parser.add_argument("--input", required=True, help="Input directory")
    parser.add_argument("--output", required=True, help="Output directory")
    
    parser.add_argument("--model", default="models/fastprotect/checkpoint_step25000.pt")
    parser.add_argument("--kmeans", default="models/fastprotect/kmeans_model.pkl")
    parser.add_argument("--entropies", default="models/fastprotect/target_entropies.json")
    
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-adaptive", action="store_true", help="Disable Adaptive Protection")
    
    # Parameters
    parser.add_argument("--strength", type=float, default=0.6, help="Global strength (0.4-0.8)")
    parser.add_argument("--base-scale", type=float, default=0.5, help="Adaptive base scale")
    parser.add_argument("--sensitivity", type=float, default=3.0, help="Adaptive sensitivity")
    
    args = parser.parse_args()

    protect_images_optimized(
        input_dir=args.input,
        output_dir=args.output,
        model_path=args.model,
        kmeans_path=args.kmeans,
        entropies_path=args.entropies,
        device=args.device,
        use_adaptive=not args.no_adaptive,
        strength_multiplier=args.strength,
        base_scale=args.base_scale,
        sensitivity=args.sensitivity,
    )

if __name__ == "__main__":
    main()
