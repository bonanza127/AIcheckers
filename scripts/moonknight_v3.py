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
import hashlib

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
        use_adaptive=True,
        use_warping=False,
        warp_magnitude=0.006,
        use_gamma=True,
        gamma_strength=0.03,
        use_edge_aware_warp=True,
        edge_avoid_strength=0.7,
        chrominance_only_warp=True,
        use_coupled_tps=False,
        tps_steps=2,
        tps_grid=4,
        tps_magnitude=0.004,
        tps_margin=0.08,
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.use_adaptive = use_adaptive
        self.use_warping = use_warping
        self.warp_magnitude = warp_magnitude
        self.use_gamma = use_gamma
        self.gamma_strength = gamma_strength
        self.use_edge_aware_warp = use_edge_aware_warp
        self.edge_avoid_strength = edge_avoid_strength
        self.chrominance_only_warp = chrominance_only_warp
        self.use_coupled_tps = use_coupled_tps
        self.tps_steps = tps_steps
        self.tps_grid = tps_grid
        self.tps_magnitude = tps_magnitude
        self.tps_margin = tps_margin
        self.use_chromatic_aberration = True
        self.chromatic_magnitude = 0.003
        self.use_hue_rotation = True
        self.hue_rotation_max_degrees = 2.0
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

        # 4. Integrity checks (silent mismatch causes severe degradation)
        if self.perturbations.K != self.kmeans_model.n_clusters:
            raise ValueError(
                f"K mismatch: model={self.perturbations.K}, "
                f"kmeans={self.kmeans_model.n_clusters}"
            )
        if self.perturbations.num_targets != len(self.target_entropies):
            raise ValueError(
                f"num_targets mismatch: model={self.perturbations.num_targets}, "
                f"entropies={len(self.target_entropies)}"
            )
        print(
            f"[MoonKnight] Integrity check passed: "
            f"K={self.perturbations.K}, targets={self.perturbations.num_targets}"
        )
            
        # 5. LPIPS (Optional)
        self.lpips_model = None
        if self.use_adaptive:
            try:
                import lpips
                self.lpips_model = lpips.LPIPS(net='alex').to(self.device)
                self.lpips_model.eval()
            except ImportError:
                print("[MoonKnight] Warning: lpips not found. Adaptive mode disabled.")
                self.use_adaptive = False

        # 6. VAE (float32 for stability)
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
        """Latentコードのエントロピーを計算 (trainと一致する分散ベース)"""
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

    def _compute_lpips_map(self, original, protected, patch_size=64):
        """パッチ単位でLPIPS空間マップを計算"""
        B, C, H, W = original.shape
        orig_norm = original * 2 - 1
        prot_norm = protected * 2 - 1

        # 512x512前提だが、念のため整数割りにならない場合は切り捨てる
        grid_h = H // patch_size
        grid_w = W // patch_size
        distance_map = torch.zeros(B, 1, grid_h, grid_w, device=original.device)

        with torch.no_grad():
            for i in range(0, grid_h * patch_size, patch_size):
                for j in range(0, grid_w * patch_size, patch_size):
                    orig_patch = orig_norm[:, :, i : i + patch_size, j : j + patch_size]
                    prot_patch = prot_norm[:, :, i : i + patch_size, j : j + patch_size]
                    dist = self.lpips_model(orig_patch, prot_patch)
                    distance_map[:, :, i // patch_size, j // patch_size] = dist

        distance_map = F.interpolate(
            distance_map, size=(H, W), mode="bilinear", align_corners=False
        )
        return distance_map

    def _scale_perturbation_strength(self, perceptual_map, base_scale=0.5, sensitivity=3.0, min_scale=0.15):
        """知覚マップに基づいて摂動強度をスケーリング"""
        M_normalized = (perceptual_map - perceptual_map.min()) / (perceptual_map.max() - perceptual_map.min() + 1e-8)
        scaling = base_scale * (1 - M_normalized * sensitivity)
        return torch.clamp(scaling, min_scale, 1.0)

    def _apply_micro_warping(
        self,
        image_tensor,
        magnitude=0.006,
        seed=None,
        kernel_size=63,
        sigma=12.0,
        anisotropy=(1.0, 1.0),
        edge_source=None,
        edge_avoid_strength=0.7,
        chrominance_only=False,
    ):
        """
        Micro-Warping: 微細な幾何学的変形

        chrominance_only=True: LAB空間のa/bチャンネル（色度）のみに変形を適用。
        人間の目は輝度(L)に敏感だが色度(a,b)には鈍感なため、視認性への影響を最小化。
        """
        try:
            import kornia
        except ImportError:
            print("[MoonKnight] Warning: kornia not found. Micro-warping skipped.")
            return image_tensor

        if seed is not None:
            torch.manual_seed(seed)

        B, C, H, W = image_tensor.shape
        noise = torch.randn(B, 2, H, W, device=image_tensor.device) * magnitude
        noise[:, 0] *= anisotropy[0]
        noise[:, 1] *= anisotropy[1]

        # Edge-aware attenuation (reduce warp near strong edges)
        if edge_source is not None and edge_avoid_strength > 0:
            rgb = edge_source
            Y = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
            edge_mag = kornia.filters.sobel(Y)
            edge_mag = edge_mag / (edge_mag.max() + 1e-8)
            atten = 1.0 - edge_avoid_strength * edge_mag
            noise = noise * atten

        if chrominance_only:
            # 色度チャンネル限定ワープ: 輝度(L)を保持、色度(a,b)のみ変形
            lab = kornia.color.rgb_to_lab(image_tensor)
            L_original = lab[:, 0:1].clone()  # 輝度を保存

            # a,bチャンネルにのみelastic transform適用
            ab = lab[:, 1:3]  # (B, 2, H, W)
            ab_warped = kornia.geometry.transform.elastic_transform2d(
                ab,
                noise,
                kernel_size=(kernel_size, kernel_size),
                sigma=(sigma, sigma),
                align_corners=True,
            )

            # 元のLとwarpedのa,bを合成
            lab_warped = torch.cat([L_original, ab_warped], dim=1)
            warped = kornia.color.lab_to_rgb(lab_warped)
        else:
            # 従来のRGB全体ワープ
            warped = kornia.geometry.transform.elastic_transform2d(
                image_tensor,
                noise,
                kernel_size=(kernel_size, kernel_size),
                sigma=(sigma, sigma),
                align_corners=True,
            )

        return torch.clamp(warped, 0, 1)

    def _apply_gamma_fluctuation(self, image_tensor, lpips_map=None, strength=0.02, seed=None):
        """低周波ガンマゆらぎ: 輝度チャンネルに微細な明暗変化を追加"""
        B, C, H, W = image_tensor.shape

        if seed is not None:
            torch.manual_seed(seed)

        # 低解像度ノイズ生成（16x16）→ アップスケール
        low_res = 16
        noise_low = torch.randn(B, 1, low_res, low_res, device=image_tensor.device)
        gamma_map = F.interpolate(noise_low, size=(H, W), mode='bilinear', align_corners=False)
        gamma_map = gamma_map * strength  # ±strength の範囲

        # LPIPSマップで強度調整（目立つ部分は弱める）
        if lpips_map is not None:
            M_norm = (lpips_map - lpips_map.min()) / (lpips_map.max() - lpips_map.min() + 1e-8)
            gamma_map = gamma_map * (1 - M_norm * 0.8)  # 目立つ部分は20%まで減衰

        # RGB → Y (輝度) 変換係数
        # Y = 0.299*R + 0.587*G + 0.114*B
        rgb = image_tensor
        Y = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]

        # ガンマ適用: Y' = Y ^ (1 + gamma_map)
        eps = 1e-6
        Y_new = torch.clamp(Y, eps, 1.0) ** (1.0 + gamma_map)

        # 輝度比でRGBをスケール
        scale = Y_new / (Y + eps)
        rgb_new = rgb * scale

        return torch.clamp(rgb_new, 0, 1)

    def _apply_chromatic_aberration(
        self,
        image_tensor,
        magnitude=0.003,
        seed=None,
    ):
        """
        Chromatic Aberration: RGBチャンネルを放射状に微小シフト

        レンズの色収差を模倣。中心から離れるほどシフト量が増加。
        - Rチャンネル: 外側へシフト
        - Gチャンネル: 固定（基準）
        - Bチャンネル: 内側へシフト

        幾何学的変形のため、LightShed等の摂動除去手法に耐性あり。
        """
        if seed is not None:
            torch.manual_seed(seed)

        B, C, H, W = image_tensor.shape
        device = image_tensor.device

        # 正規化座標グリッド [-1, 1]
        y_coords = torch.linspace(-1, 1, H, device=device)
        x_coords = torch.linspace(-1, 1, W, device=device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')

        # 中心からの距離（放射状）
        r = torch.sqrt(xx**2 + yy**2)
        r_normalized = r / (r.max() + 1e-8)  # [0, 1]に正規化

        # 放射方向の単位ベクトル
        r_safe = r + 1e-8
        dir_x = xx / r_safe
        dir_y = yy / r_safe

        # ランダム要素: 各画像で微妙に異なる収差パターン
        r_noise = torch.randn(1, device=device).item() * 0.3 + 1.0  # 0.7-1.3
        b_noise = torch.randn(1, device=device).item() * 0.3 + 1.0

        # シフト量（距離に比例、magnitudeでスケール）
        r_shift = magnitude * r_normalized * r_noise  # Rは外側へ
        b_shift = -magnitude * r_normalized * b_noise  # Bは内側へ

        # サンプリング座標を計算
        # grid_sample用: [-1, 1]の範囲
        base_grid = torch.stack([xx, yy], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

        # Rチャンネル用グリッド（外側へシフト = 元の位置から内側をサンプル）
        r_offset = torch.stack([dir_x * r_shift, dir_y * r_shift], dim=-1).unsqueeze(0)
        r_grid = base_grid - r_offset  # 逆方向にサンプル

        # Bチャンネル用グリッド（内側へシフト = 元の位置から外側をサンプル）
        b_offset = torch.stack([dir_x * b_shift, dir_y * b_shift], dim=-1).unsqueeze(0)
        b_grid = base_grid - b_offset

        # 各チャンネルをサンプリング
        r_channel = F.grid_sample(
            image_tensor[:, 0:1], r_grid,
            mode='bilinear', padding_mode='border', align_corners=True
        )
        g_channel = image_tensor[:, 1:2]  # Gは固定
        b_channel = F.grid_sample(
            image_tensor[:, 2:3], b_grid,
            mode='bilinear', padding_mode='border', align_corners=True
        )

        result = torch.cat([r_channel, g_channel, b_channel], dim=1)
        return torch.clamp(result, 0, 1)

    def _apply_hue_micro_rotation(
        self,
        image_tensor,
        max_degrees=3.0,
        seed=None,
        low_freq_size=8,
    ):
        """
        Hue Micro-Rotation: 色相を局所的に微小回転

        HSV空間でHue値を微小回転。低周波ノイズマップで
        滑らかに変化させ、不自然さを回避。

        Args:
            image_tensor: (B, 3, H, W) RGB画像 [0, 1]
            max_degrees: 最大回転角度（±degrees）
            seed: 再現性のためのシード
            low_freq_size: 低周波ノイズの解像度
        """
        if seed is not None:
            torch.manual_seed(seed)

        B, C, H, W = image_tensor.shape
        device = image_tensor.device

        # RGB → HSV
        rgb = image_tensor
        max_val, _ = rgb.max(dim=1, keepdim=True)
        min_val, _ = rgb.min(dim=1, keepdim=True)
        diff = max_val - min_val + 1e-8

        # Value
        v = max_val

        # Saturation
        s = diff / (max_val + 1e-8)

        # Hue calculation
        r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]

        # Determine which channel is max
        r_is_max = (rgb[:, 0:1] == max_val).float()
        g_is_max = (rgb[:, 1:2] == max_val).float() * (1 - r_is_max)
        b_is_max = 1 - r_is_max - g_is_max

        h_r = (g - b) / diff
        h_g = 2.0 + (b - r) / diff
        h_b = 4.0 + (r - g) / diff

        h = h_r * r_is_max + h_g * g_is_max + h_b * b_is_max
        h = (h / 6.0) % 1.0  # Normalize to [0, 1]

        # 低周波ノイズマップ生成（滑らかな局所回転）
        noise_low = torch.randn(B, 1, low_freq_size, low_freq_size, device=device)
        rotation_map = F.interpolate(noise_low, size=(H, W), mode='bilinear', align_corners=False)
        rotation_map = rotation_map * (max_degrees / 360.0)  # degrees → [0,1]範囲の割合

        # Hue回転
        h_rotated = (h + rotation_map) % 1.0

        # HSV → RGB
        h6 = h_rotated * 6.0
        sector = h6.floor()
        f = h6 - sector

        p = v * (1 - s)
        q = v * (1 - s * f)
        t = v * (1 - s * (1 - f))

        sector = sector % 6

        # 各セクターごとのRGB値
        rgb_out = torch.zeros_like(rgb)

        mask0 = (sector == 0).float()
        mask1 = (sector == 1).float()
        mask2 = (sector == 2).float()
        mask3 = (sector == 3).float()
        mask4 = (sector == 4).float()
        mask5 = (sector == 5).float()

        rgb_out[:, 0:1] = v * mask0 + q * mask1 + p * mask2 + p * mask3 + t * mask4 + v * mask5
        rgb_out[:, 1:2] = t * mask0 + v * mask1 + v * mask2 + q * mask3 + p * mask4 + p * mask5
        rgb_out[:, 2:3] = p * mask0 + p * mask1 + t * mask2 + v * mask3 + v * mask4 + q * mask5

        # 彩度が低い部分は元の値を維持（グレー領域の保護）
        gray_mask = (s < 0.05).float()
        rgb_out = rgb_out * (1 - gray_mask) + rgb * gray_mask

        return torch.clamp(rgb_out, 0, 1)

    def _make_tps_control_points(self, grid_size=4, margin=0.08, device="cpu"):
        """TPS制御点を正規化座標 [0,1] で作成"""
        coords = torch.linspace(margin, 1.0 - margin, grid_size, device=device)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        points = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
        return points.unsqueeze(0)

    def _apply_coupled_tps(
        self,
        image_tensor,
        steps=2,
        grid_size=4,
        magnitude=0.004,
        margin=0.08,
        seed=None,
    ):
        """Coupled TPS: 少数制御点TPSを複数回カップリング"""
        try:
            from kornia.geometry.transform.thin_plate_spline import get_tps_transform, warp_image_tps
        except ImportError:
            print("[MoonKnight] Warning: kornia TPS not available. CoupledTPS skipped.")
            return image_tensor

        if seed is not None:
            torch.manual_seed(seed)

        warped = image_tensor
        points_src = self._make_tps_control_points(
            grid_size=grid_size, margin=margin, device=image_tensor.device
        )

        for _ in range(max(1, steps)):
            offsets = torch.randn_like(points_src) * magnitude
            points_dst = torch.clamp(points_src + offsets, 0.0, 1.0)
            # Reverse transform: dst -> src
            kernel_weights, affine_weights = get_tps_transform(points_dst, points_src)
            warped = warp_image_tps(
                warped,
                points_src,
                kernel_weights,
                affine_weights,
                align_corners=False,
                padding_mode="reflection",
            )

        return torch.clamp(warped, 0, 1)

    def poison(
        self,
        image: Image.Image,
        strength: float = 0.6,
        base_scale=0.5,
        sensitivity=3.0,
        progress_callback=None,
        use_warping=None,
        warp_magnitude=None,
        use_gamma=None,
        gamma_strength=None,
        use_edge_aware_warp=None,
        edge_avoid_strength=None,
        chrominance_only_warp=None,
        use_coupled_tps=None,
        tps_steps=None,
        tps_grid=None,
        tps_magnitude=None,
        tps_margin=None,
        use_chromatic_aberration=None,
        chromatic_magnitude=None,
        use_hue_rotation=None,
        hue_rotation_max_degrees=None,
    ) -> Image.Image:
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
            perceptual_map = None
            if self.use_adaptive and self.lpips_model is not None:
                report_progress(50) # Adaptive Scaling
                
                # Use actual strength for scaling map calculation
                # This ensures different strength values produce visibly different results
                delta_scaled_512 = raw_delta * strength
                
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

            # Optional micro-warping (浄化耐性向上)
            apply_warping = self.use_warping if use_warping is None else use_warping
            if apply_warping:
                # 3パターンのwarp設定（画像ごとに1つを決定的に選択）
                warp_configs = [
                    {"kernel_size": 31, "sigma": 6.0, "magnitude": 0.004, "anisotropy": (1.0, 0.85)},   # 細かい・弱い
                    {"kernel_size": 63, "sigma": 12.0, "magnitude": 0.0055, "anisotropy": (0.9, 1.0)},  # 中程度
                    {"kernel_size": 95, "sigma": 18.0, "magnitude": 0.0065, "anisotropy": (1.0, 0.7)},  # 粗い・強い
                ]
                # 画像ハッシュから決定的にconfig選択（再現可能 + 画像ごとに異なる）
                img_bytes = img_full.detach().cpu().numpy().tobytes()
                seed = int(hashlib.sha256(img_bytes).hexdigest()[:8], 16)
                config = warp_configs[seed % len(warp_configs)]
                # magnitudeを明示指定された場合のみ上書き（configのmagnitudeは維持）
                if warp_magnitude is not None:
                    config = {**config, "magnitude": warp_magnitude}
                apply_edge_aware = self.use_edge_aware_warp if use_edge_aware_warp is None else use_edge_aware_warp
                edge_strength = self.edge_avoid_strength if edge_avoid_strength is None else edge_avoid_strength
                chroma_only = self.chrominance_only_warp if chrominance_only_warp is None else chrominance_only_warp
                protected_full = self._apply_micro_warping(
                    protected_full,
                    seed=seed,
                    edge_source=img_full if apply_edge_aware else None,
                    edge_avoid_strength=edge_strength if apply_edge_aware else 0.0,
                    chrominance_only=chroma_only,
                    **config
                )

            # CoupledTPS (optional, low-magnitude)
            apply_tps = self.use_coupled_tps if use_coupled_tps is None else use_coupled_tps
            if apply_tps:
                tps_seed = int(hashlib.sha256(img_full.detach().cpu().numpy().tobytes()).hexdigest()[:8], 16) ^ 0xA5A5
                protected_full = self._apply_coupled_tps(
                    protected_full,
                    steps=self.tps_steps if tps_steps is None else tps_steps,
                    grid_size=self.tps_grid if tps_grid is None else tps_grid,
                    magnitude=self.tps_magnitude if tps_magnitude is None else tps_magnitude,
                    margin=self.tps_margin if tps_margin is None else tps_margin,
                    seed=tps_seed,
                )

            # 低周波ガンマゆらぎ（denoise耐性向上）
            apply_gamma = self.use_gamma if use_gamma is None else use_gamma
            gamma_strength_value = self.gamma_strength if gamma_strength is None else gamma_strength
            if apply_gamma:
                # strengthと連動（視認性を壊しにくくする）
                gamma_strength_value = gamma_strength_value * max(0.5, min(strength / 0.6, 1.5))
                lpips_map_full = None
                if self.use_adaptive and perceptual_map is not None:
                    lpips_map_full = F.interpolate(
                        perceptual_map,
                        size=(h_orig, w_orig),
                        mode="bilinear",
                        align_corners=False
                    )
                gamma_seed = int(hashlib.sha256(img_full.detach().cpu().numpy().tobytes()).hexdigest()[:8], 16)
                protected_full = self._apply_gamma_fluctuation(
                    protected_full,
                    lpips_map=lpips_map_full,
                    strength=gamma_strength_value,
                    seed=gamma_seed
                )

            # Chromatic Aberration (LightShed耐性・幾何学的変形)
            apply_chromatic = self.use_chromatic_aberration if use_chromatic_aberration is None else use_chromatic_aberration
            if apply_chromatic:
                chroma_mag = self.chromatic_magnitude if chromatic_magnitude is None else chromatic_magnitude
                chroma_seed = int(hashlib.sha256(img_full.detach().cpu().numpy().tobytes()).hexdigest()[:8], 16) ^ 0xCAFE
                protected_full = self._apply_chromatic_aberration(
                    protected_full,
                    magnitude=chroma_mag,
                    seed=chroma_seed,
                )

            # Hue Micro-Rotation (色空間変換・色相の局所回転)
            apply_hue = self.use_hue_rotation if use_hue_rotation is None else use_hue_rotation
            if apply_hue:
                hue_degrees = self.hue_rotation_max_degrees if hue_rotation_max_degrees is None else hue_rotation_max_degrees
                hue_seed = int(hashlib.sha256(img_full.detach().cpu().numpy().tobytes()).hexdigest()[:8], 16) ^ 0xBEEF
                protected_full = self._apply_hue_micro_rotation(
                    protected_full,
                    max_degrees=hue_degrees,
                    seed=hue_seed,
                )

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
