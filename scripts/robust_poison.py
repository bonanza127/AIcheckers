#!/usr/bin/env python3
"""
Robust AI Poisoning Tool (Codename: Ironclad)

LightShed等のカウンターに耐性のあるAIポイズニング
- VAEループ耐性: 潜在空間での最適化
- デノイジング耐性: 低周波・構造的摂動
- AI検出回避: DINOv3特徴空間での最適化

使用方法:
    python robust_poison.py --input image.png --output poisoned.png
    python robust_poison.py --input image.png --output poisoned.png --strength 0.1
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# 設定
HF_TOKEN = os.getenv("HF_TOKEN", "")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RobustPoisoner:
    """LightShed耐性ポイズニング"""

    def __init__(self, device=DEVICE):
        self.device = device
        self.vae = None
        self.dino = None
        self.dino_processor = None
        self._load_models()

    def _load_models(self):
        """モデルをロード"""
        print("Loading models...")

        # Stable Diffusion VAE (潜在空間アクセス用)
        from diffusers import AutoencoderKL
        self.vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse",
            torch_dtype=torch.float32
        ).to(self.device)
        self.vae.eval()

        # DINOv3 (AI検出回避用)
        from transformers import AutoImageProcessor, AutoModel
        self.dino_processor = AutoImageProcessor.from_pretrained(
            "facebook/dinov3-vitb16-pretrain-lvd1689m",
            token=HF_TOKEN
        )
        self.dino = AutoModel.from_pretrained(
            "facebook/dinov3-vitb16-pretrain-lvd1689m",
            token=HF_TOKEN
        ).to(self.device)
        self.dino.eval()

        print(f"Models loaded on {self.device}")

    def _image_to_tensor(self, image: Image.Image, max_size: int = 512) -> torch.Tensor:
        """PIL Image -> Tensor [-1, 1]"""
        img = image.convert("RGB")
        # メモリ節約のため最大サイズを制限
        w, h = img.size
        if max(w, h) > max_size:
            scale = max_size / max(w, h)
            w, h = int(w * scale), int(h * scale)
            img = img.resize((w, h), Image.LANCZOS)
        # VAE用にリサイズ（8の倍数）
        new_w = (w // 8) * 8
        new_h = (h // 8) * 8
        if new_w != w or new_h != h:
            img = img.resize((new_w, new_h), Image.LANCZOS)

        arr = np.array(img).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor * 2.0 - 1.0  # [0,1] -> [-1,1]
        return tensor.to(self.device)

    def _tensor_to_image(self, tensor: torch.Tensor) -> Image.Image:
        """Tensor [-1, 1] -> PIL Image"""
        tensor = tensor.clamp(-1, 1)
        tensor = (tensor + 1.0) / 2.0  # [-1,1] -> [0,1]
        arr = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    def _get_dino_features(self, tensor: torch.Tensor) -> torch.Tensor:
        """DINOv3特徴量を取得"""
        # [-1,1] -> [0,1] -> DINOv3前処理
        img_01 = (tensor + 1.0) / 2.0
        # DINOv3の正規化
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        img_norm = (img_01 - mean) / std
        # リサイズ
        img_resized = F.interpolate(img_norm, size=(224, 224), mode="bilinear", align_corners=False)

        with torch.no_grad():
            outputs = self.dino(pixel_values=img_resized)
            features = outputs.last_hidden_state[:, 0, :]  # CLSトークン
        return features

    def _low_freq_mask(self, shape, cutoff=0.3):
        """低周波マスクを生成（DCTの低周波成分のみ通過）"""
        h, w = shape[-2:]
        cy, cx = h // 2, w // 2
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        dist = torch.sqrt((x - cx).float()**2 + (y - cy).float()**2)
        max_dist = np.sqrt(cx**2 + cy**2)
        mask = (dist / max_dist < cutoff).float()
        return mask.to(self.device)

    def _apply_low_freq_constraint(self, perturbation: torch.Tensor, cutoff=0.3) -> torch.Tensor:
        """摂動を低周波成分に制限"""
        # FFT
        fft = torch.fft.fft2(perturbation)
        fft_shifted = torch.fft.fftshift(fft)

        # 低周波マスク適用
        mask = self._low_freq_mask(perturbation.shape, cutoff)
        fft_filtered = fft_shifted * mask

        # IFFT
        fft_unshifted = torch.fft.ifftshift(fft_filtered)
        filtered = torch.fft.ifft2(fft_unshifted).real

        return filtered

    def poison(
        self,
        image: Image.Image,
        strength: float = 0.08,
        iterations: int = 200,
        lr: float = 0.01,
        vae_weight: float = 1.0,
        dino_weight: float = 0.5,
        low_freq_cutoff: float = 0.4,
        verbose: bool = True
    ) -> Image.Image:
        """
        画像にロバストポイズンを適用

        Args:
            image: 入力画像
            strength: 摂動の強さ (0.0-1.0)
            iterations: 最適化イテレーション数
            lr: 学習率
            vae_weight: VAE耐性ロスの重み
            dino_weight: DINO特徴変化ロスの重み
            low_freq_cutoff: 低周波フィルタのカットオフ
            verbose: 進捗表示

        Returns:
            ポイズン適用済み画像
        """
        # 画像をテンソルに変換
        img_tensor = self._image_to_tensor(image)
        original_tensor = img_tensor.clone()

        # 摂動を初期化（小さなランダムノイズ）
        perturbation = torch.zeros_like(img_tensor, requires_grad=True)
        optimizer = torch.optim.Adam([perturbation], lr=lr)

        # 元画像のDINO特徴
        with torch.no_grad():
            original_dino = self._get_dino_features(original_tensor)

        # 元画像の潜在表現
        with torch.no_grad():
            original_latent = self.vae.encode(original_tensor).latent_dist.mean

        iterator = tqdm(range(iterations), desc="Poisoning") if verbose else range(iterations)

        for i in iterator:
            optimizer.zero_grad()

            # 低周波制約を適用した摂動
            constrained_pert = self._apply_low_freq_constraint(perturbation, low_freq_cutoff)

            # 強度制限
            constrained_pert = constrained_pert.clamp(-strength, strength)

            # ポイズン画像
            poisoned = (original_tensor + constrained_pert).clamp(-1, 1)

            # === ロス計算 ===

            # 1. VAE耐性ロス: VAEループ後も摂動が残るように
            poisoned_latent = self.vae.encode(poisoned).latent_dist.mean
            reconstructed = self.vae.decode(poisoned_latent).sample

            # 再構成画像と元画像の差 = 残存摂動
            # これを最大化したい（摂動が消えないように）
            residual = (reconstructed - original_tensor).abs().mean()
            vae_loss = -residual  # 負にして最大化→最小化

            # 2. DINO特徴変化ロス: AI検出を回避
            poisoned_dino = self._get_dino_features(poisoned)
            dino_loss = -F.cosine_similarity(poisoned_dino, original_dino).mean()
            # 負にして類似度を下げる

            # 3. 知覚的制約: 人間に見えないように
            perceptual_loss = constrained_pert.abs().mean()

            # 合計ロス
            total_loss = vae_weight * vae_loss + dino_weight * dino_loss + 0.1 * perceptual_loss

            total_loss.backward()
            optimizer.step()

            if verbose and (i + 1) % 50 == 0:
                tqdm.write(f"  Iter {i+1}: VAE={-vae_loss.item():.4f}, DINO={-dino_loss.item():.4f}")

        # 最終的なポイズン画像を生成
        with torch.no_grad():
            final_pert = self._apply_low_freq_constraint(perturbation, low_freq_cutoff)
            final_pert = final_pert.clamp(-strength, strength)
            final_poisoned = (original_tensor + final_pert).clamp(-1, 1)

        return self._tensor_to_image(final_poisoned)

    def verify_robustness(self, original: Image.Image, poisoned: Image.Image) -> dict:
        """ポイズンのロバスト性を検証"""
        orig_tensor = self._image_to_tensor(original)
        pois_tensor = self._image_to_tensor(poisoned)

        with torch.no_grad():
            # VAEループテスト
            pois_latent = self.vae.encode(pois_tensor).latent_dist.mean
            reconstructed = self.vae.decode(pois_latent).sample

            # 摂動の残存率
            original_pert = (pois_tensor - orig_tensor).abs().mean().item()
            after_vae_pert = (reconstructed - orig_tensor).abs().mean().item()
            survival_rate = after_vae_pert / (original_pert + 1e-8)

            # DINO特徴変化
            orig_dino = self._get_dino_features(orig_tensor)
            pois_dino = self._get_dino_features(pois_tensor)
            recon_dino = self._get_dino_features(reconstructed)

            dino_shift_before = 1 - F.cosine_similarity(pois_dino, orig_dino).item()
            dino_shift_after = 1 - F.cosine_similarity(recon_dino, orig_dino).item()

        return {
            "perturbation_magnitude": original_pert,
            "vae_survival_rate": survival_rate,
            "dino_shift_before_vae": dino_shift_before,
            "dino_shift_after_vae": dino_shift_after,
        }


def main():
    parser = argparse.ArgumentParser(description="Robust AI Poisoning Tool")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input image path")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output image path")
    parser.add_argument("--strength", "-s", type=float, default=0.08,
                        help="Perturbation strength (0.0-1.0, default: 0.08)")
    parser.add_argument("--iterations", type=int, default=200,
                        help="Optimization iterations (default: 200)")
    parser.add_argument("--verify", action="store_true",
                        help="Verify robustness after poisoning")
    args = parser.parse_args()

    # 入力画像を読み込み
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found")
        sys.exit(1)

    image = Image.open(input_path).convert("RGB")
    print(f"Input: {input_path} ({image.size[0]}x{image.size[1]})")

    # ポイズナーを初期化
    poisoner = RobustPoisoner()

    # ポイズン適用
    poisoned = poisoner.poison(
        image,
        strength=args.strength,
        iterations=args.iterations,
        verbose=True
    )

    # 保存
    output_path = Path(args.output)
    poisoned.save(output_path, quality=95)
    print(f"Output: {output_path}")

    # ロバスト性検証
    if args.verify:
        print("\nVerifying robustness...")
        results = poisoner.verify_robustness(image, poisoned)
        print(f"  Perturbation magnitude: {results['perturbation_magnitude']:.4f}")
        print(f"  VAE survival rate: {results['vae_survival_rate']*100:.1f}%")
        print(f"  DINO shift (before VAE): {results['dino_shift_before_vae']:.4f}")
        print(f"  DINO shift (after VAE): {results['dino_shift_after_vae']:.4f}")


if __name__ == "__main__":
    main()
