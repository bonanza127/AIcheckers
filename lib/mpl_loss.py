#!/usr/bin/env python3
"""
Multi-Layer Protection (MLP) Loss

FastProtect論文の式(5)に基づく実装。

LT(x) = -||z - zy||²₂ - (λ/L) Σ ||Fl - Fly||²₂

- z: 保護画像のlatent
- zy: ターゲット画像のlatent
- Fl: 保護画像の中間層特徴
- Fly: ターゲット画像の中間層特徴
- λ: 中間層重み係数 (3.5×10⁻⁵)
- L: 中間層の数 (4)
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Optional


def mpl_loss(
    z: torch.Tensor,
    z_target: torch.Tensor,
    features: List[torch.Tensor],
    features_target: List[torch.Tensor],
    lambda_: float = 3.5e-5,
) -> torch.Tensor:
    """
    Multi-Layer Protection Loss

    目的: 保護画像のlatentと中間層特徴をターゲット画像から最大限離す

    Args:
        z: 保護画像のlatent (B, C, H, W)
        z_target: ターゲット画像のlatent (B, C, H, W)
        features: 保護画像の中間層特徴リスト
        features_target: ターゲット画像の中間層特徴リスト
        lambda_: 中間層重み係数

    Returns:
        Loss値（最小化するため負値を返す）
    """
    # Latent距離（L2ノルム²を最大化 → 負にして最小化）
    latent_loss = -torch.sum((z - z_target) ** 2)

    # 中間層距離
    L = len(features)
    feature_loss = 0.0

    for f, f_t in zip(features, features_target):
        feature_loss -= (lambda_ / L) * torch.sum((f - f_t) ** 2)

    total_loss = latent_loss + feature_loss
    return total_loss


def compute_latent_and_features(
    image: torch.Tensor,
    vae,
    extractor,
) -> tuple:
    """
    画像からlatentと中間層特徴を取得

    Args:
        image: 入力画像 (B, C, H, W) [0, 1]
        vae: AutoencoderKLインスタンス
        extractor: VAEFeatureExtractorインスタンス

    Returns:
        (latent, features_list)
    """
    # VAEは[-1, 1]を期待
    image_normalized = image * 2 - 1

    # エンコード
    extractor.clear()
    latent = vae.encode(image_normalized).latent_dist.mean

    # 特徴量取得
    features = extractor.get_feature_list()

    return latent, features


class FastProtectLoss:
    """
    FastProtect用の統合Loss計算クラス

    Usage:
        loss_fn = FastProtectLoss(vae, device)
        target_latent, target_features = loss_fn.precompute_target(target_image)

        for step in range(num_steps):
            loss = loss_fn.compute(protected_image, target_latent, target_features)
            loss.backward()
    """

    def __init__(self, vae, device: str = "cuda", lambda_: float = 3.5e-5):
        from .vae_hooks import VAEFeatureExtractor

        self.vae = vae
        self.device = device
        self.lambda_ = lambda_
        self.extractor = VAEFeatureExtractor(vae)

    def precompute_target(self, target_image: torch.Tensor) -> tuple:
        """
        ターゲット画像のlatentと特徴量を事前計算

        Args:
            target_image: ターゲット画像 (B, C, H, W) [0, 1]

        Returns:
            (target_latent, target_features)
        """
        with torch.no_grad():
            target_latent, target_features = compute_latent_and_features(
                target_image, self.vae, self.extractor
            )
            # 勾配計算から切り離す
            target_latent = target_latent.detach()
            target_features = [f.detach() for f in target_features]

        return target_latent, target_features

    def compute(
        self,
        protected_image: torch.Tensor,
        target_latent: torch.Tensor,
        target_features: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        MLP Lossを計算

        Args:
            protected_image: 保護画像 (B, C, H, W) [0, 1]
            target_latent: ターゲットlatent
            target_features: ターゲット特徴量

        Returns:
            Loss値
        """
        latent, features = compute_latent_and_features(
            protected_image, self.vae, self.extractor
        )

        return mpl_loss(latent, target_latent, features, target_features, self.lambda_)

    def cleanup(self):
        """リソースを解放"""
        self.extractor.remove_hooks()


def compute_entropy(z: torch.Tensor) -> torch.Tensor:
    """
    Latentコードのエントロピーを計算

    Adaptive Targeted Protectionでターゲット画像を選択する際に使用。

    Args:
        z: latentコード (B, C, H, W)

    Returns:
        エントロピー値
    """
    # フラット化してヒストグラム計算
    z_flat = z.flatten()

    # ヒストグラム（256 bins）
    hist = torch.histc(z_flat, bins=256, min=z_flat.min(), max=z_flat.max())

    # 確率分布
    p = hist / hist.sum()

    # ゼロを除外してエントロピー計算
    p = p[p > 0]
    entropy = -torch.sum(p * torch.log(p))

    return entropy


if __name__ == "__main__":
    # テスト
    print("Testing MLP Loss...")

    # ダミーテンソル
    B, C, H, W = 2, 4, 64, 64
    z = torch.randn(B, C, H, W)
    z_target = torch.randn(B, C, H, W)

    # ダミー特徴量
    features = [torch.randn(B, 128, 128, 128) for _ in range(4)]
    features_target = [torch.randn(B, 128, 128, 128) for _ in range(4)]

    loss = mpl_loss(z, z_target, features, features_target)
    print(f"MLP Loss: {loss.item():.4f}")

    # エントロピーテスト
    entropy = compute_entropy(z)
    print(f"Entropy: {entropy.item():.4f}")
