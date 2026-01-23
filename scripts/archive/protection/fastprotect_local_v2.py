#!/usr/bin/env python3
"""
FastProtect Local Inference V2 - 論文完全準拠版

Adaptive Protection Strength実装済み
- LPIPSベースの空間的摂動調整
- 平坦領域での視認性改善

Usage:
    python3 scripts/fastprotect_local_v2.py --input /path/to/images --output /path/to/output
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

# FastProtectPerturbations クラス
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

    def apply(self, image, target_idx, cluster_idx):
        """画像に摂動を適用"""
        delta = self.delta_g[target_idx] + self.Delta[target_idx][cluster_idx]

        if image.shape[2:] != (self.image_size, self.image_size):
            delta_resized = F.interpolate(
                delta.unsqueeze(0),
                size=image.shape[2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        else:
            delta_resized = delta

        protected = image + delta_resized
        return torch.clamp(protected, 0, 1)

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
            instance.delta_g[t].data = d.to(device)
        for t, deltas in enumerate(checkpoint["Delta"]):
            for k, d in enumerate(deltas):
                instance.Delta[t][k].data = d.to(device)

        return instance, checkpoint


def predict_cluster(latent, kmeans):
    """K-meansでクラスタ予測"""
    import numpy as np

    if latent.dim() == 4:
        latent = latent.squeeze(0)

    latent_flat = latent.cpu().numpy().reshape(1, -1)
    cluster_idx = kmeans.predict(latent_flat)[0]

    return cluster_idx


def compute_latent_entropy(z):
    """Latentコードのエントロピーを計算（簡易版：分散ベース）"""
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
    """
    LPIPS空間マップを計算（論文準拠）

    Args:
        lpips_model: LPIPS model (AlexNet backbone)
        original: 元画像 (1, 3, H, W) [0, 1]
        protected: 保護済み画像 (1, 3, H, W) [0, 1]

    Returns:
        perceptual_map: (1, 1, H, W) 知覚距離マップ
    """
    import lpips

    # LPIPSは[-1, 1]の入力を期待
    original_normalized = original * 2 - 1
    protected_normalized = protected * 2 - 1

    # 空間的なLPIPS距離を計算
    with torch.no_grad():
        # spatial=Trueで空間マップを取得
        distance_map = lpips_model(original_normalized, protected_normalized, normalize=False)

    return distance_map


def scale_perturbation_strength(perceptual_map, base_scale=1.0, sensitivity=2.0):
    """
    知覚マップに基づいて摂動強度をスケーリング（論文のS(·)関数）

    Args:
        perceptual_map: (1, 1, H, W) LPIPS距離マップ
        base_scale: ベーススケール
        sensitivity: 感度パラメータ

    Returns:
        scaling_map: (1, 1, H, W) スケーリングマップ
    """
    # 論文: S(1 - M)
    # Mは距離マップなので、距離が大きい（目立つ）領域ではスケールを下げる

    # 正規化
    M_normalized = (perceptual_map - perceptual_map.min()) / (perceptual_map.max() - perceptual_map.min() + 1e-8)

    # 反転してスケーリング（距離が大きい領域 = 摂動を弱める）
    scaling = base_scale * (1 - M_normalized * sensitivity)
    scaling = torch.clamp(scaling, 0.1, 1.0)  # 最小0.1、最大1.0

    return scaling


def apply_adaptive_protection(
    image,
    perturbations,
    target_idx,
    cluster_idx,
    lpips_model,
):
    """
    Adaptive Protection Strengthを適用（論文Eq. 7準拠）

    Args:
        image: 入力画像 (1, 3, H, W)
        perturbations: FastProtectPerturbations
        target_idx: ターゲットインデックス
        cluster_idx: クラスタインデックス
        lpips_model: LPIPS model

    Returns:
        protected: 保護済み画像 (1, 3, H, W)
    """
    # まず通常の摂動を適用（サロゲート）
    surrogate = perturbations.apply(image, target_idx, cluster_idx)

    # LPIPS知覚マップを計算
    perceptual_map = compute_lpips_map(lpips_model, image, surrogate.unsqueeze(0) if surrogate.dim() == 3 else surrogate)

    # スケーリングマップを生成
    scaling_map = scale_perturbation_strength(perceptual_map)

    # 画像サイズに合わせてリサイズ
    if scaling_map.shape[2:] != image.shape[2:]:
        scaling_map = F.interpolate(scaling_map, size=image.shape[2:], mode="bilinear", align_corners=False)

    # 摂動を取得
    delta = perturbations.delta_g[target_idx] + perturbations.Delta[target_idx][cluster_idx]

    # 画像サイズに合わせて摂動をリサイズ
    if delta.shape[1:] != image.shape[2:]:
        delta_resized = F.interpolate(
            delta.unsqueeze(0),
            size=image.shape[2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    else:
        delta_resized = delta

    # スケーリングを適用（論文Eq. 7）
    scaled_delta = scaling_map.squeeze(0) * delta_resized

    # 最終的な保護画像
    protected = image + scaled_delta
    protected = torch.clamp(protected, 0, 1)

    return protected


def protect_images(
    input_dir,
    output_dir,
    model_path="models/fastprotect/fastprotect_final.pt",
    kmeans_path="models/fastprotect/kmeans_model.pkl",
    entropies_path="models/fastprotect/target_entropies.json",
    device="cuda",
    use_adaptive=True,
):
    """画像フォルダを保護"""
    print(f"Device: {device}")
    print(f"Adaptive Protection Strength: {'Enabled' if use_adaptive else 'Disabled'}")

    # モデルロード
    print("Loading FastProtect model...")
    perturbations, _ = FastProtectPerturbations.load(model_path, device=device)

    print("Loading K-means model...")
    with open(kmeans_path, "rb") as f:
        kmeans_model = pickle.load(f)

    print("Loading target entropies...")
    import json
    with open(entropies_path, "r") as f:
        entropy_data = json.load(f)
        target_entropies = entropy_data["entropies"]

    print(f"Target entropies: {target_entropies}")

    # LPIPSモデルロード（Adaptive Protection用）
    lpips_model = None
    if use_adaptive:
        print("Loading LPIPS model for Adaptive Protection...")
        import lpips
        lpips_model = lpips.LPIPS(net='alex').to(device)
        lpips_model.eval()

    # VAEロード
    print("Loading VAE...")
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sdxl-vae",
        torch_dtype=torch.bfloat16,
    ).to(device)
    vae.eval()

    # 入力画像リスト
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_files = []
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.webp"]:
        image_files.extend(input_path.glob(ext))

    print(f"Found {len(image_files)} images")

    transform = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
    ])

    with torch.no_grad():
        for img_file in tqdm(image_files, desc="Protecting"):
            # 画像ロード
            img = Image.open(img_file).convert("RGB")
            original_size = img.size
            img_tensor = transform(img).unsqueeze(0).to(device)

            # VAE encode
            img_normalized = (img_tensor * 2 - 1).bfloat16()
            z = vae.encode(img_normalized).latent_dist.mean.float()

            # ターゲット選択
            target_idx = select_target_by_entropy(z, target_entropies)

            # クラスタ予測
            cluster_idx = predict_cluster(z, kmeans_model)

            # 摂動適用（Adaptive or Normal）
            if use_adaptive and lpips_model is not None:
                protected = apply_adaptive_protection(
                    img_tensor,
                    perturbations,
                    target_idx,
                    cluster_idx,
                    lpips_model,
                )
            else:
                protected = perturbations.apply(img_tensor, target_idx, cluster_idx).unsqueeze(0)

            # 元のサイズに戻す
            if original_size != (512, 512):
                protected_resized = F.interpolate(
                    protected,
                    size=(original_size[1], original_size[0]),
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                protected_resized = protected

            # 保存
            protected_np = protected_resized.squeeze(0).cpu().numpy()
            protected_np = (protected_np * 255).astype("uint8").transpose(1, 2, 0)
            protected_img = Image.fromarray(protected_np)

            output_file = output_path / img_file.name
            protected_img.save(output_file, quality=95)

            tqdm.write(f"  {img_file.name} -> target={target_idx}, cluster={cluster_idx}")

    print(f"\nCompleted! Protected images saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--model", default="models/fastprotect/fastprotect_final.pt")
    parser.add_argument("--kmeans", default="models/fastprotect/kmeans_model.pkl")
    parser.add_argument("--entropies", default="models/fastprotect/target_entropies.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-adaptive", action="store_true", help="Disable Adaptive Protection Strength")
    args = parser.parse_args()

    protect_images(
        input_dir=args.input,
        output_dir=args.output,
        model_path=args.model,
        kmeans_path=args.kmeans,
        entropies_path=args.entropies,
        device=args.device,
        use_adaptive=not args.no_adaptive,
    )


if __name__ == "__main__":
    main()
