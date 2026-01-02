#!/usr/bin/env python3
"""
Ironclad v3.1 - LightShed耐性ポイズニング + Human Signature

特徴:
- YCbCr色空間: 署名はYチャンネルのみ（色収差回避）
- DWT (bior1.3): JPEG親和性、アーティファクト抑制
- 知覚マスキング: テクスチャ領域に強く、平坦領域に弱く
- Untargeted Attack: VAE潜在空間で類似度最小化
- 秘密鍵ベース署名: 画像固有のソルト + 動的バージョン

使用方法:
    python ironclad_v2.py --input image.png --output poisoned.png
    python ironclad_v2.py --input image.png --detect  # 署名検出のみ
"""

import os
import sys
import argparse
import hashlib
import hmac
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# 遅延インポート（GPUがない環境でもimportだけは通るように）
ptwt = None
kornia = None
AutoencoderKL = None


def lazy_imports():
    """必要なライブラリを遅延インポート"""
    global ptwt, kornia, AutoencoderKL
    if ptwt is None:
        import ptwt as _ptwt
        import kornia as _kornia
        from diffusers import AutoencoderKL as _VAE
        ptwt = _ptwt
        kornia = _kornia
        AutoencoderKL = _VAE


class IroncladPoisoner:
    """LightShed耐性ポイズニング + Human Signature (v3.1)"""

    WAVELET = 'bior1.3'  # Biorthogonal (JPEG親和性高、アーティファクト少)
    CANONICAL_SIZE = 512  # 署名検出時の基準解像度

    def __init__(self, secret_key: str, version: str = None, device: str = None):
        lazy_imports()

        self.version = version or datetime.now().strftime("%Y%m")
        self.secret_key = f"{secret_key}_{self.version}".encode()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.vae = None  # 遅延ロード

        # 強度パラメータ（YCbCrベースなので少し強めでもOK）
        self.strength_low = 0.08   # LL: セマンティック攻撃
        self.strength_mid = 0.06   # LH/HL: 署名+毒（重点）
        self.strength_high = 0.02  # HH: 微小ノイズのみ
        self.detection_threshold = 0.20

    def _load_vae(self):
        """VAEを遅延ロード"""
        if self.vae is None:
            print(f"Loading VAE on {self.device}...")
            self.vae = AutoencoderKL.from_pretrained(
                "stabilityai/sd-vae-ft-mse",
                torch_dtype=torch.float32
            ).eval().to(self.device)
        return self.vae

    def _compute_perceptual_mask(self, Y: torch.Tensor) -> torch.Tensor:
        """
        知覚マスキング: エッジ密度が高い領域ほど強くノイズを入れられる
        平坦領域（空、肌）は弱く、テクスチャ領域（髪、服）は強く
        """
        # Sobelフィルタでエッジ検出
        edges = kornia.filters.sobel(Y)
        edge_magnitude = edges.abs().mean(dim=1, keepdim=True)

        # シグモイドで0-1に正規化（エッジあり=1.0, 平坦=0.2程度）
        mask = 0.2 + 0.8 * torch.sigmoid((edge_magnitude - 0.1) * 20)
        return mask

    def _normalize_resolution(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """署名埋め込み/検出時に固定解像度に正規化（幾何学的耐性）"""
        _, _, h, w = image_tensor.shape
        if max(h, w) != self.CANONICAL_SIZE:
            scale = self.CANONICAL_SIZE / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            # 8の倍数に丸める（DWT用）
            new_h = (new_h // 8) * 8
            new_w = (new_w // 8) * 8
            image_tensor = F.interpolate(
                image_tensor, (new_h, new_w), mode='bilinear', align_corners=False
            )
        return image_tensor

    def poison(self, image_tensor: torch.Tensor, iterations: int = 50,
               normalize_resolution: bool = False) -> torch.Tensor:
        """
        メイン処理: ポイズニング + 署名埋め込み

        Args:
            image_tensor: (B, C, H, W) in range [0, 1], RGB
            iterations: セマンティック攻撃のイテレーション数
            normalize_resolution: 解像度正規化を行うか（Falseで元サイズ維持）

        Returns:
            ポイズン適用済み画像 (B, C, H, W) in range [0, 1], RGB
        """
        self._load_vae()
        device = self.device
        original_shape = image_tensor.shape

        # GPU転送
        image_tensor = image_tensor.to(device)

        # 1. 8の倍数にリサイズ（DWT用）- 正規化はオプション
        if normalize_resolution:
            img_processed = self._normalize_resolution(image_tensor)
        else:
            # 8の倍数に調整するだけ
            _, _, h, w = image_tensor.shape
            new_h = (h // 8) * 8
            new_w = (w // 8) * 8
            if new_h != h or new_w != w:
                img_processed = F.interpolate(
                    image_tensor, (new_h, new_w), mode='bilinear', align_corners=False
                )
            else:
                img_processed = image_tensor

        img_normalized = img_processed

        # 2. RGB -> YCbCr変換 (Korniaを使用)
        img_ycbcr = kornia.color.rgb_to_ycbcr(img_normalized)
        Y, Cb, Cr = torch.chunk(img_ycbcr, 3, dim=1)

        # 3. 知覚マスク計算 (Yチャンネルのみで計算)
        mask = self._compute_perceptual_mask(Y)

        # 4. DWT分解 (Yチャンネルのみ)
        coeffs = ptwt.wavedec2(Y, self.WAVELET, level=1)
        LL = coeffs[0]
        LH, HL, HH = coeffs[1]

        # マスクをLHサイズにダウンサンプリング（DWTではなくbilinear）
        mask_mid = F.interpolate(
            mask, size=LH.shape[-2:], mode='bilinear', align_corners=False
        )

        # 5. 画像固有ソルト & 署名パターン
        image_salt = self._get_image_salt(Y)
        sig_pattern = self._generate_signature_pattern(LH.shape, image_salt).to(device)

        # 6. Untargeted Semantic Attack (LLに対して実施)
        LL_poisoned = self._untargeted_semantic_attack(LL, iterations)

        # 7. 署名埋め込み (LH/HL) - 知覚マスク適用
        perturbation = sig_pattern * self.strength_mid * mask_mid
        LH_poisoned = LH + perturbation
        HL_poisoned = HL + perturbation

        # 8. HH (微小ノイズ) - SNSで消える前提
        HH_poisoned = HH + (torch.randn_like(HH) * self.strength_high)

        # 9. 逆変換
        Y_poisoned = ptwt.waverec2([LL_poisoned, (LH_poisoned, HL_poisoned, HH_poisoned)], self.WAVELET)

        # サイズ整合性チェック（パディングの影響で数ピクセルずれることがある）
        if Y_poisoned.shape != Y.shape:
            Y_poisoned = Y_poisoned[:, :, :Y.shape[2], :Y.shape[3]]

        # 10. YCbCr結合 & RGB変換（Cb, Crはそのまま = 色味を変えない）
        poisoned_ycbcr = torch.cat([Y_poisoned, Cb, Cr], dim=1)
        poisoned_rgb = kornia.color.ycbcr_to_rgb(poisoned_ycbcr)

        # 11. 元のサイズに戻す
        if poisoned_rgb.shape != original_shape:
            poisoned_rgb = F.interpolate(
                poisoned_rgb, original_shape[-2:], mode='bilinear', align_corners=False
            )

        return torch.clamp(poisoned_rgb, 0, 1)

    def _untargeted_semantic_attack(self, LL: torch.Tensor, steps: int = 50) -> torch.Tensor:
        """
        Untargeted Attack: VAE潜在空間で元画像との類似度を最小化
        - ターゲット概念の指定不要
        - 「AIにとって理解不能な何か」にする
        """
        device = LL.device

        # LLを3チャンネルRGBとしてVAEに通す（グレースケール→RGB疑似変換）
        # 512pxにアップスケールしてからVAEに通す
        original_size = self.CANONICAL_SIZE

        with torch.no_grad():
            # LLをアップスケール
            upscaled_orig = F.interpolate(
                LL, size=(original_size, original_size), mode='bilinear', align_corners=False
            )
            # 3チャンネル化
            rgb_orig = upscaled_orig.repeat(1, 3, 1, 1)
            # VAEの入力範囲 [-1, 1] に変換
            rgb_orig_scaled = rgb_orig * 2.0 - 1.0
            target_latent = self.vae.encode(rgb_orig_scaled).latent_dist.mean

        optimized_LL = LL.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([optimized_LL], lr=0.01)

        for _ in range(steps):
            upscaled_opt = F.interpolate(
                optimized_LL, size=(original_size, original_size), mode='bilinear', align_corners=False
            )
            rgb_opt = upscaled_opt.repeat(1, 3, 1, 1)
            rgb_opt_scaled = rgb_opt * 2.0 - 1.0
            current_latent = self.vae.encode(rgb_opt_scaled).latent_dist.mean

            # コサイン類似度を最小化（Untargeted）
            loss = F.cosine_similarity(
                current_latent.flatten(),
                target_latent.flatten(),
                dim=0
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 摂動の大きさを制限
            with torch.no_grad():
                perturbation = optimized_LL - LL
                perturbation = torch.clamp(perturbation, -self.strength_low, self.strength_low)
                optimized_LL.data = LL + perturbation

        return optimized_LL.detach()

    def _get_image_salt(self, image_tensor: torch.Tensor) -> str:
        """
        画像固有のソルト（pHashベース）
        pHashは知覚的に類似した画像で同じ値になるため、
        ポイズニング前後でも同じソルトが得られる
        """
        import imagehash
        from PIL import Image as PILImage

        # テンソル→PIL変換（グレースケール）
        arr = image_tensor.detach().cpu().squeeze(0).squeeze(0).numpy()
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
        pil_img = PILImage.fromarray(arr, mode='L')

        # pHash計算（64ビット）
        phash = imagehash.phash(pil_img, hash_size=8)
        return str(phash)

    def _generate_signature_pattern(self, shape, image_salt: str) -> torch.Tensor:
        """秘密鍵 + 画像ソルト から固有パターン生成"""
        key = hmac.new(self.secret_key, image_salt.encode(), hashlib.sha256).digest()
        seed = int.from_bytes(key[:8], 'big')
        rng = torch.Generator().manual_seed(seed)
        pattern = torch.rand(shape, generator=rng) * 2 - 1
        return pattern

    def detect_signature(self, image_tensor: torch.Tensor, version: str = None,
                          normalize_resolution: bool = False) -> dict:
        """
        署名検出（AIcheckers統合用）
        - VAE不要（高速）
        """
        lazy_imports()

        ver = version or self.version
        # バージョン付き秘密鍵を再構築
        base_key = self.secret_key.decode().rsplit('_', 1)[0]
        temp_key = f"{base_key}_{ver}".encode()

        device = image_tensor.device if image_tensor.is_cuda else self.device
        image_tensor = image_tensor.to(device)

        # 8の倍数に調整（正規化はオプション）
        if normalize_resolution:
            img_processed = self._normalize_resolution(image_tensor)
        else:
            _, _, h, w = image_tensor.shape
            new_h = (h // 8) * 8
            new_w = (w // 8) * 8
            if new_h != h or new_w != w:
                img_processed = F.interpolate(
                    image_tensor, (new_h, new_w), mode='bilinear', align_corners=False
                )
            else:
                img_processed = image_tensor

        img_normalized = img_processed

        # RGB -> YCbCr -> Y
        img_ycbcr = kornia.color.rgb_to_ycbcr(img_normalized)
        Y = img_ycbcr[:, 0:1, :, :]  # Yチャンネルのみ

        # 画像ソルト（正規化後のYから計算）
        image_salt = self._get_image_salt(Y)

        # DWT分解
        coeffs = ptwt.wavedec2(Y, self.WAVELET, level=1)
        LH, HL = coeffs[1][0], coeffs[1][1]

        # 期待されるパターン
        key = hmac.new(temp_key, image_salt.encode(), hashlib.sha256).digest()
        seed = int.from_bytes(key[:8], 'big')
        rng = torch.Generator().manual_seed(seed)
        expected = torch.rand(LH.shape, generator=rng) * 2 - 1
        expected = expected.to(device)

        # 相関係数計算
        correlation_lh = self._correlation(LH, expected)
        correlation_hl = self._correlation(HL, expected)
        avg_correlation = (correlation_lh + correlation_hl) / 2

        return {
            "detected": avg_correlation > self.detection_threshold,
            "correlation": avg_correlation,
            "correlation_lh": correlation_lh,
            "correlation_hl": correlation_hl,
            "version": ver,
            "threshold": self.detection_threshold
        }

    def _correlation(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """正規化相関係数"""
        a_flat = a.flatten().float()
        b_flat = b.flatten().float()
        a_norm = (a_flat - a_flat.mean()) / (a_flat.std() + 1e-8)
        b_norm = (b_flat - b_flat.mean()) / (b_flat.std() + 1e-8)
        return (a_norm * b_norm).mean().item()


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    """PIL Image -> Tensor [0, 1]"""
    arr = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """Tensor [0, 1] -> PIL Image"""
    tensor = tensor.clamp(0, 1)
    arr = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    arr = (arr * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def main():
    parser = argparse.ArgumentParser(description="Ironclad v3.1 - Robust AI Poisoning + Signature")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input image path")
    parser.add_argument("--output", "-o", type=str, help="Output image path")
    parser.add_argument("--detect", action="store_true", help="Detect signature only (no poisoning)")
    parser.add_argument("--strength", type=float, default=0.06, help="Signature strength (default: 0.06)")
    parser.add_argument("--iterations", type=int, default=50, help="Semantic attack iterations (default: 50)")
    parser.add_argument("--secret-key", type=str, default="AICHECKERS_DEFAULT_KEY", help="Secret key for signature")
    parser.add_argument("--version", type=str, help="Signature version (default: current month YYYYMM)")
    parser.add_argument("--verify", action="store_true", help="Verify signature after poisoning")
    args = parser.parse_args()

    # 入力画像を読み込み
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found")
        sys.exit(1)

    image = Image.open(input_path).convert("RGB")
    print(f"Input: {input_path} ({image.size[0]}x{image.size[1]})")

    # ポイズナー初期化
    poisoner = IroncladPoisoner(
        secret_key=args.secret_key,
        version=args.version
    )
    poisoner.strength_mid = args.strength

    # テンソル変換
    img_tensor = image_to_tensor(image)

    if args.detect:
        # 署名検出のみ
        print("\n=== Signature Detection ===")
        result = poisoner.detect_signature(img_tensor)
        print(f"Detected: {result['detected']}")
        print(f"Correlation: {result['correlation']:.4f}")
        print(f"  LH: {result['correlation_lh']:.4f}")
        print(f"  HL: {result['correlation_hl']:.4f}")
        print(f"Threshold: {result['threshold']}")
        print(f"Version: {result['version']}")
    else:
        # ポイズニング実行
        if not args.output:
            print("Error: --output is required for poisoning")
            sys.exit(1)

        print(f"\n=== Poisoning (strength={args.strength}, iterations={args.iterations}) ===")
        poisoned_tensor = poisoner.poison(img_tensor, iterations=args.iterations)

        # 保存
        poisoned_image = tensor_to_image(poisoned_tensor)
        output_path = Path(args.output)
        poisoned_image.save(output_path, quality=95)
        print(f"Output: {output_path}")

        # 検証
        if args.verify:
            print("\n=== Verification ===")
            result = poisoner.detect_signature(poisoned_tensor)
            print(f"Signature detected: {result['detected']}")
            print(f"Correlation: {result['correlation']:.4f}")


if __name__ == "__main__":
    main()
