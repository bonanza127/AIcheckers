#!/usr/bin/env python3
"""
FastProtect Local Inference - GTX 1660対応軽量版

Usage:
    python3 scripts/fastprotect_local.py --input /path/to/images --output /path/to/output
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

# FastProtectPerturbations クラス（train.pyから抽出）
class FastProtectPerturbations:
    def __init__(self, K=4, image_size=512, eta=8/255, device="cuda", num_targets=3):
        self.K = K
        self.eta = eta
        self.eta_half = eta / 2
        self.device = device
        self.image_size = image_size
        self.num_targets = num_targets

        # 3セットのMoP（ロード時に上書きされる）
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

        # 画像サイズに合わせてリサイズ
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

        # 3セット分をロード
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

    # 簡易エントロピー: 分散を使用
    entropy = z_flat.var(dim=1)
    return entropy


def select_target_by_entropy(z, target_entropies):
    """エントロピーに基づいてターゲットを選択"""
    input_entropy = compute_latent_entropy(z).item()

    # 最も近いターゲットを選択
    distances = [abs(input_entropy - te) for te in target_entropies]
    target_idx = distances.index(min(distances))

    return target_idx


def protect_images(
    input_dir,
    output_dir,
    model_path="models/fastprotect/fastprotect_final.pt",
    kmeans_path="models/fastprotect/kmeans_model.pkl",
    entropies_path="models/fastprotect/target_entropies.json",
    device="cuda",
):
    """画像フォルダを保護"""
    print(f"Device: {device}")

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

    # VAEロード（軽量：bf16）
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

    # 処理
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

            # ターゲット選択（エントロピーベース）
            target_idx = select_target_by_entropy(z, target_entropies)

            # クラスタ予測
            cluster_idx = predict_cluster(z, kmeans_model)

            # 摂動適用
            protected = perturbations.apply(img_tensor, target_idx, cluster_idx)

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

            print(f"  {img_file.name} -> target={target_idx}, cluster={cluster_idx}")

    print(f"\nCompleted! Protected images saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--model", default="models/fastprotect/fastprotect_final.pt")
    parser.add_argument("--kmeans", default="models/fastprotect/kmeans_model.pkl")
    parser.add_argument("--entropies", default="models/fastprotect/target_entropies.json")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    protect_images(
        input_dir=args.input,
        output_dir=args.output,
        model_path=args.model,
        kmeans_path=args.kmeans,
        entropies_path=args.entropies,
        device=args.device,
    )


if __name__ == "__main__":
    main()
