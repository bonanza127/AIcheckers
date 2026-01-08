#!/usr/bin/env python3
"""
FastProtect Inference Module - サイト統合用

GTX 1660対応のバッチ処理機能付き
"""

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
import pickle
import json
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np


class FastProtectInference:
    """
    FastProtect推論クラス（論文準拠 + Adaptive Protection）

    Features:
    - Mixture-of-Perturbations (K=4)
    - Adaptive Protection Strength
    - バッチ処理対応（GTX 1660最適化）
    """

    def __init__(
        self,
        model_path: str,
        kmeans_path: str,
        entropies_path: str,
        device: str = "cuda",
        use_adaptive: bool = True,
        max_batch_size: int = 4,
    ):
        """
        Args:
            model_path: FastProtect摂動モデルパス
            kmeans_path: K-meansモデルパス
            entropies_path: Target entropiesパス
            device: 'cuda' or 'cpu'
            use_adaptive: Adaptive Protection Strengthを使用
            max_batch_size: 最大バッチサイズ（VRAM制限対応）
        """
        self.device = device
        self.use_adaptive = use_adaptive
        self.max_batch_size = max_batch_size

        print(f"[FastProtect] Device: {device}")
        print(f"[FastProtect] Adaptive Protection: {'Enabled' if use_adaptive else 'Disabled'}")
        print(f"[FastProtect] Max Batch Size: {max_batch_size}")

        # 摂動モデルロード
        print("[FastProtect] Loading perturbations...")
        self.perturbations, self.checkpoint = self._load_perturbations(model_path)

        # K-meansロード
        print("[FastProtect] Loading K-means...")
        with open(kmeans_path, "rb") as f:
            self.kmeans = pickle.load(f)

        # エントロピーロード
        print("[FastProtect] Loading target entropies...")
        with open(entropies_path, "r") as f:
            entropy_data = json.load(f)
            self.target_entropies = entropy_data["entropies"]

        # VAEロード
        print("[FastProtect] Loading VAE...")
        from diffusers import AutoencoderKL
        self.vae = AutoencoderKL.from_pretrained(
            "stabilityai/sdxl-vae",
            torch_dtype=torch.bfloat16,
        ).to(device)
        self.vae.eval()

        # LPIPSロード（Adaptive用）
        self.lpips_model = None
        if use_adaptive:
            print("[FastProtect] Loading LPIPS...")
            import lpips
            self.lpips_model = lpips.LPIPS(net='alex').to(device)
            self.lpips_model.eval()

        print("[FastProtect] Initialization complete!")

    def _load_perturbations(self, path: str):
        """摂動をロード"""
        checkpoint = torch.load(path, map_location=self.device)

        # FastProtectPerturbationsクラスを再構築
        K = checkpoint["K"]
        image_size = checkpoint["image_size"]
        eta = checkpoint["eta"]
        num_targets = checkpoint.get("num_targets", 3)

        # パラメータ初期化
        delta_g = []
        Delta = []

        for t in range(num_targets):
            dg = torch.nn.Parameter(checkpoint["delta_g"][t].to(self.device))
            delta_g.append(dg)

            deltas_t = []
            for k in range(K):
                dk = torch.nn.Parameter(checkpoint["Delta"][t][k].to(self.device))
                deltas_t.append(dk)
            Delta.append(deltas_t)

        perturbations = {
            "delta_g": delta_g,
            "Delta": Delta,
            "K": K,
            "image_size": image_size,
            "eta": eta,
            "num_targets": num_targets,
        }

        return perturbations, checkpoint

    def _select_target_by_entropy(self, z: torch.Tensor) -> int:
        """エントロピーに基づいてターゲットを選択"""
        if z.dim() == 4:
            z_flat = z.view(z.shape[0], -1)
        else:
            z_flat = z.view(1, -1)

        entropy = z_flat.var(dim=1).item()

        distances = [abs(entropy - te) for te in self.target_entropies]
        target_idx = distances.index(min(distances))

        return target_idx

    def _predict_cluster(self, z: torch.Tensor) -> int:
        """K-meansでクラスタ予測"""
        if z.dim() == 4:
            z = z.squeeze(0)

        z_flat = z.cpu().numpy().reshape(1, -1)
        cluster_idx = self.kmeans.predict(z_flat)[0]

        return cluster_idx

    def _apply_perturbation(
        self,
        image: torch.Tensor,
        target_idx: int,
        cluster_idx: int,
        use_adaptive: bool = True,
    ) -> torch.Tensor:
        """
        画像に摂動を適用

        Args:
            image: (1, 3, H, W) テンソル [0, 1]
            target_idx: ターゲットインデックス
            cluster_idx: クラスタインデックス
            use_adaptive: Adaptive Protection使用

        Returns:
            protected: (1, 3, H, W) 保護済み画像
        """
        # 摂動取得
        delta = self.perturbations["delta_g"][target_idx] + self.perturbations["Delta"][target_idx][cluster_idx]

        # 画像サイズに合わせてリサイズ
        if delta.shape[1:] != image.shape[2:]:
            delta_resized = F.interpolate(
                delta.unsqueeze(0),
                size=image.shape[2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        else:
            delta_resized = delta

        # Adaptive Protection
        if use_adaptive and self.lpips_model is not None:
            # サロゲート（スケーリングなし）
            surrogate = torch.clamp(image + delta_resized, 0, 1)

            # LPIPS距離マップ
            orig_norm = image * 2 - 1
            surr_norm = surrogate * 2 - 1

            with torch.no_grad():
                distance_map = self.lpips_model(orig_norm, surr_norm, normalize=False)

            # スケーリングマップ生成
            M_normalized = (distance_map - distance_map.min()) / (distance_map.max() - distance_map.min() + 1e-8)
            scaling = torch.clamp(1.0 - M_normalized * 2.0, 0.1, 1.0)

            # リサイズ
            if scaling.shape[2:] != image.shape[2:]:
                scaling = F.interpolate(scaling, size=image.shape[2:], mode="bilinear", align_corners=False)

            # スケーリング適用
            scaled_delta = scaling.squeeze(0) * delta_resized
            protected = torch.clamp(image + scaled_delta, 0, 1)
        else:
            # 通常の摂動適用
            protected = torch.clamp(image + delta_resized, 0, 1)

        return protected

    @torch.no_grad()
    def protect_single(
        self,
        image: Image.Image,
        return_pil: bool = True,
    ) -> Tuple[Image.Image, dict]:
        """
        単一画像を保護

        Args:
            image: PIL Image (RGB)
            return_pil: PIL形式で返すか（Falseでnumpy配列）

        Returns:
            protected_image: 保護済み画像
            metadata: {target_idx, cluster_idx, original_size}
        """
        original_size = image.size

        # テンソル変換（512x512にリサイズ）
        transform = T.Compose([
            T.Resize((512, 512)),
            T.ToTensor(),
        ])
        img_tensor = transform(image).unsqueeze(0).to(self.device)

        # VAE encode
        img_normalized = (img_tensor * 2 - 1).to(torch.bfloat16)
        z = self.vae.encode(img_normalized).latent_dist.mean.float()

        # ターゲット＆クラスタ選択
        target_idx = self._select_target_by_entropy(z)
        cluster_idx = self._predict_cluster(z)

        # 摂動適用
        protected = self._apply_perturbation(
            img_tensor,
            target_idx,
            cluster_idx,
            use_adaptive=self.use_adaptive,
        )

        # 元のサイズに戻す
        if original_size != (512, 512):
            protected = F.interpolate(
                protected,
                size=(original_size[1], original_size[0]),
                mode="bilinear",
                align_corners=False,
            )

        # PIL or Numpy変換
        protected_np = protected.squeeze(0).cpu().numpy()
        protected_np = (protected_np * 255).astype("uint8").transpose(1, 2, 0)

        if return_pil:
            protected_image = Image.fromarray(protected_np)
        else:
            protected_image = protected_np

        metadata = {
            "target_idx": target_idx,
            "cluster_idx": cluster_idx,
            "original_size": original_size,
        }

        return protected_image, metadata

    @torch.no_grad()
    def protect_batch(
        self,
        images: List[Image.Image],
        return_pil: bool = True,
    ) -> List[Tuple[Image.Image, dict]]:
        """
        バッチ処理で複数画像を保護（VRAM効率化）

        Args:
            images: PIL Images のリスト
            return_pil: PIL形式で返すか

        Returns:
            results: [(protected_image, metadata), ...]
        """
        results = []

        # バッチサイズに分割
        for i in range(0, len(images), self.max_batch_size):
            batch = images[i:i + self.max_batch_size]

            # 各画像を個別処理（バッチは将来の最適化用）
            for img in batch:
                protected, metadata = self.protect_single(img, return_pil=return_pil)
                results.append((protected, metadata))

        return results

    def estimate_vram_usage(self, batch_size: int = 1) -> dict:
        """
        VRAM使用量を推定

        Returns:
            dict: {vae, lpips, perturbations, batch_buffer, total}
        """
        vram_vae = 1.5  # GB
        vram_lpips = 0.5 if self.use_adaptive else 0.0
        vram_perturbations = 0.1

        # バッチバッファ（512x512, fp32）
        # 入力 + 出力 + 中間バッファ
        vram_batch = (512 * 512 * 3 * 4 * 3 * batch_size) / (1024**3)

        total = vram_vae + vram_lpips + vram_perturbations + vram_batch

        return {
            "vae_gb": vram_vae,
            "lpips_gb": vram_lpips,
            "perturbations_gb": vram_perturbations,
            "batch_buffer_gb": round(vram_batch, 2),
            "total_gb": round(total, 2),
            "recommended_max_batch": 4 if total < 6 else 2,
        }
