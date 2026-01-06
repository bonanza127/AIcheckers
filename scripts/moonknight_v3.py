#!/usr/bin/env python3
"""
MoonKnight V3 (formerly FastProtect V3 Local)
Analysis @ 512px -> Protection @ Original Resolution with Bicubic Upscaling.
Optimized for local execution and web API integration.
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

# ==================== Core Classes ====================

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

class MoonKnightV3:
    """
    MoonKnight V3 Protection Engine
    Wraps FastProtect logic for easy API use.
    """
    def __init__(
        self,
        model_dir="models/fastprotect",
        device="cuda",
        use_adaptive=True
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.use_adaptive = use_adaptive
        self.model_dir = Path(model_dir)
        
        # Paths
        self.checkpoint_path = self.model_dir / "checkpoint_step25000.pt"
        self.kmeans_path = self.model_dir / "kmeans_model.pkl"
        self.entropies_path = self.model_dir / "target_entropies.json"
        
        # Load Components
        self._load_models()
        self._setup_transforms()
        
    def _load_models(self):
        print(f"[MoonKnight] Loading models on {self.device}...")
        
        # 1. FastProtect Perturbations
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Model not found: {self.checkpoint_path}")
        self.perturbations, _ = FastProtectPerturbations.load(self.checkpoint_path, device=self.device)
        
        # 2. K-means
        with open(self.kmeans_path, "rb") as f:
            self.kmeans_model = pickle.load(f)
            
        # 3. Entropies
        with open(self.entropies_path, "r") as f:
            entropy_data = json.load(f)
            self.target_entropies = entropy_data["entropies"]
            
        # 4. LPIPS (Optional)
        self.lpips_model = None
        if self.use_adaptive:
            try:
                import lpips
                self.lpips_model = lpips.LPIPS(net='alex').to(self.device)
                self.lpips_model.eval()
            except ImportError:
                print("[MoonKnight] Warning: lpips not found. Adaptive mode disabled.")
                self.use_adaptive = False

        # 5. VAE (float32 for stability)
        from diffusers import AutoencoderKL
        self.vae = AutoencoderKL.from_pretrained(
            "stabilityai/sdxl-vae",
            torch_dtype=torch.float32, 
        ).to(self.device).eval()

    def _setup_transforms(self):
        # Transform for Analysis (Fixed 512x512)
        self.transform_analysis = T.Compose([
            T.Resize((512, 512), interpolation=T.InterpolationMode.LANCZOS),
            T.ToTensor(),
        ])
        # Transform for Full Res (Just ToTensor)
        self.transform_full = T.ToTensor()

    def _compute_latent_entropy(self, z):
        """Latentコードのエントロピーを計算"""
        if z.dim() == 4:
            B = z.shape[0]
            z_flat = z.view(B, -1)
        else:
            z_flat = z.view(1, -1)
        return z_flat.var(dim=1)

    def _select_target_by_entropy(self, z):
        """エントロピーに基づいてターゲットを選択"""
        input_entropy = self._compute_latent_entropy(z).item()
        distances = [abs(input_entropy - te) for te in self.target_entropies]
        return distances.index(min(distances))

    def _predict_cluster(self, latent):
        """K-meansでクラスタ予測"""
        if latent.dim() == 4:
            latent = latent.squeeze(0)
        latent_flat = latent.cpu().numpy().reshape(1, -1).astype(np.float32)
        return self.kmeans_model.predict(latent_flat)[0]

    def _compute_lpips_map(self, original, protected):
        """LPIPS空間マップを計算"""
        original_normalized = original * 2 - 1
        protected_normalized = protected * 2 - 1
        with torch.no_grad():
            distance_map = self.lpips_model(original_normalized, protected_normalized, normalize=False)
        return distance_map

    def _scale_perturbation_strength(self, perceptual_map, base_scale=0.5, sensitivity=3.0, min_scale=0.15):
        """知覚マップに基づいて摂動強度をスケーリング"""
        M_normalized = (perceptual_map - perceptual_map.min()) / (perceptual_map.max() - perceptual_map.min() + 1e-8)
        scaling = base_scale * (1 - M_normalized * sensitivity)
        return torch.clamp(scaling, min_scale, 1.0)

    def poison(self, image: Image.Image, strength: float = 0.6, base_scale=0.5, sensitivity=3.0, progress_callback=None) -> Image.Image:
        """
        Apply MoonKnight protection to a single image.
        Args:
            image: PIL Image (RGB)
            strength: Protection strength (0.0 - 1.0)
            progress_callback: function(progress: int, total: int)
        Returns:
            Protected PIL Image
        """
        def report_progress(p):
            if progress_callback:
                progress_callback(p, 100)

        report_progress(5) # Start

        w_orig, h_orig = image.size
        
        # Handle Alpha
        alpha_channel = None
        if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
            image = image.convert("RGBA")
            alpha_channel = image.split()[3]
            img_pil = image.convert("RGB")
        else:
            img_pil = image.convert("RGB")

        with torch.no_grad():
            # 1. Analysis Step (on 512x512 resized copy)
            report_progress(10) # Analysis Start
            img_analysis = self.transform_analysis(img_pil).unsqueeze(0).to(self.device)
            
            # VAE Encode (float32)
            report_progress(20) # VAE Encoding
            img_norm_analysis = (img_analysis * 2 - 1).to(dtype=torch.float32)
            z = self.vae.encode(img_norm_analysis).latent_dist.mean.float()
            
            # Strategy Selection
            report_progress(30) # Strategy Selection
            target_idx = self._select_target_by_entropy(z)
            cluster_idx = self._predict_cluster(z)
            
            # Base Perturbation
            report_progress(40) # Calculating Perturbation
            raw_delta = self.perturbations.delta_g[target_idx] + self.perturbations.Delta[target_idx][cluster_idx]
            
            # 2. Compute Scaling Map (Adaptive)
            scaling_map_512 = None
            if self.use_adaptive and self.lpips_model is not None:
                report_progress(50) # Adaptive Scaling
                
                # Use fixed reference strength for scaling map to ensure linearity
                # If we use args.strength, higher strength triggers stronger suppression, canceling out the increase.
                reference_strength = 0.5
                delta_scaled_512 = raw_delta * reference_strength
                
                # Check dims and resize if needed (rare case if raw_delta != 512)
                if delta_scaled_512.shape[1:] != img_analysis.shape[2:]:
                        delta_scaled_512 = F.interpolate(
                        delta_scaled_512.unsqueeze(0),
                        size=img_analysis.shape[2:],
                        mode="bilinear", align_corners=False
                    ).squeeze(0)

                surrogate_512 = torch.clamp(img_analysis + delta_scaled_512.unsqueeze(0), 0, 1)
                perceptual_map = self._compute_lpips_map(img_analysis, surrogate_512)
                
                scaling_map_512 = self._scale_perturbation_strength(
                    perceptual_map,
                    base_scale=base_scale,
                    sensitivity=sensitivity
                )
            
            # 3. Application (On Original Size)
            report_progress(70) # Upscaling
            img_full = self.transform_full(img_pil).unsqueeze(0).to(self.device)
            
            # Upscale Delta (Bicubic)
            delta_full = F.interpolate(
                raw_delta.unsqueeze(0),
                size=(h_orig, w_orig),
                mode="bicubic",
                align_corners=False,
                antialias=True
            ).squeeze(0)
            
            final_perturbation = delta_full * strength
            
            # Upscale Scanning Map
            if scaling_map_512 is not None:
                scaling_full = F.interpolate(
                    scaling_map_512,
                    size=(h_orig, w_orig),
                    mode="bicubic",
                    align_corners=False,
                    antialias=True
                ).squeeze(0)
                final_perturbation = final_perturbation * scaling_full
            
            # Apply
            report_progress(90) # Applying
            protected_full = torch.clamp(img_full + final_perturbation.unsqueeze(0), 0, 1)
            
            # Convert back to PIL
            protected_np = protected_full.squeeze(0).cpu().numpy()
            protected_np = (protected_np * 255).astype("uint8").transpose(1, 2, 0)
            img_out = Image.fromarray(protected_np)
            
            # Restore Alpha
            if alpha_channel is not None:
                if alpha_channel.size != img_out.size:
                    alpha_channel = alpha_channel.resize(img_out.size, Image.LANCZOS)
                img_out.putalpha(alpha_channel)
                
            report_progress(100) # Done
            return img_out

# ==================== CLI Support ====================

def save_image_task(image, output_path):
    try:
        if output_path.suffix.lower() in ['.jpg', '.jpeg'] and image.mode == 'RGBA':
            bg = Image.new("RGB", image.size, (255, 255, 255))
            bg.paste(image, mask=image.split()[3])
            bg.save(output_path, quality=95, subsampling=0)
        else:
            image.save(output_path)
    except Exception as e:
        print(f"Error saving {output_path.name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="MoonKnight V3 (FastProtect Local)")
    parser.add_argument("--input", required=True, help="Input directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--strength", type=float, default=0.6, help="Global strength")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize Engine
    engine = MoonKnightV3()
    
    # Process
    image_files = []
    if input_path.is_file():
        image_files = [input_path]
    elif input_path.is_dir():
        supported_exts = ["*.png", "*.jpg", "*.jpeg", "*.webp"]
        for ext in supported_exts:
            image_files.extend(input_path.glob(ext))
        
    print(f"Found {len(image_files)} images")
    
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1) # Reduce workers for better logging
    futures = []
    
    import numpy as np
    
    for img_file in tqdm(image_files, desc="Protecting"):
        try:
            img = Image.open(img_file).convert("RGB")
            # Calculate stats
            original_np = np.array(img).astype(np.float32) / 255.0
            
            protected_img = engine.poison(img, strength=args.strength)
            
            protected_np = np.array(protected_img).astype(np.float32) / 255.0
            diff = np.abs(original_np - protected_np)
            l_inf = np.max(diff)
            l2 = np.mean(diff ** 2)
            
            print(f"[{img_file.name}] Protection Stats: L_inf={l_inf:.4f}, L2={l2:.6f}")
            
            # Add _protected suffix and ensure png for alpha support consistency
            base_name = img_file.stem
            out_name = f"{base_name}_protected.png"
            # If output is dir, use it. If output is file (and input is single file), use output filename.
            if output_path.suffix:
                 # Output is a file path
                 out_file = output_path
            else:
                 out_file = output_path / out_name
            
            futures.append(executor.submit(save_image_task, protected_img, out_file))
            
        except Exception as e:
            print(f"Failed to process {img_file.name}: {e}")
            import traceback
            traceback.print_exc()
            
    concurrent.futures.wait(futures)
    print("All tasks completed.")

if __name__ == "__main__":
    main()
