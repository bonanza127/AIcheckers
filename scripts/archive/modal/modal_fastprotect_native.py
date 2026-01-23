#!/usr/bin/env python3
"""
FastProtect Native Resolution - Modal版

元解像度のまま保護を適用し、VAE latent cos simを測定

Usage:
    modal run scripts/modal_fastprotect_native.py --test-single --strength 0.4
    modal run scripts/modal_fastprotect_native.py --test-strengths  # 0.4, 0.5, 0.6比較
"""
import modal
import io
from pathlib import Path

# Modal App
app = modal.App("fastprotect-native")

# GPU設定
GPU_CONFIG = "A100-40GB"

# Volume
fastprotect_vol = modal.Volume.from_name("fastprotect-vol", create_if_missing=True)

# Docker image
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch==2.1.2",
        "torchvision==0.16.2",
        "diffusers==0.27.0",
        "transformers==4.38.0",
        "accelerate==0.27.0",
        "huggingface-hub<1.0,>=0.20.0",  # diffusers 0.27.0互換
        "Pillow",
        "numpy<2",  # torch 2.1.2互換
        "scikit-learn",
        "tqdm",
        "lpips",
    )
)


@app.function(
    image=image,
    gpu=GPU_CONFIG,
    timeout=3600,
    volumes={"/vol": fastprotect_vol},
)
def protect_and_measure(
    image_path: str,
    output_path: str,
    checkpoint_path: str = "/vol/fastprotect_model/checkpoint_step25000.pt",
    strength: float = 0.4,
    base_scale: float = 0.5,
    sensitivity: float = 3.0,
    measure_cosine: bool = True,
):
    """
    元解像度で保護を適用し、VAE latent cos simを測定

    Args:
        image_path: 入力画像パス（/vol内）
        output_path: 出力画像パス（/vol内）
        checkpoint_path: FastProtectチェックポイントパス
        strength: グローバル強度
        base_scale: Adaptive base scale
        sensitivity: Adaptive sensitivity
        measure_cosine: cos sim測定

    Returns:
        dict: {cos_sim, original_size, protected_path}
    """
    import torch
    import torch.nn.functional as F
    from PIL import Image
    import torchvision.transforms as T
    import pickle
    import json

    device = "cuda"
    print(f"[Modal] Device: {device}")
    print(f"[Modal] Strength: {strength}, Base Scale: {base_scale}, Sensitivity: {sensitivity}")

    # FastProtectPerturbationsクラス
    class FastProtectPerturbations:
        def __init__(self, K=4, image_size=512, eta=8/255, device="cuda", num_targets=3):
            self.K = K
            self.eta = eta
            self.device = device
            self.image_size = image_size
            self.num_targets = num_targets
            self.delta_g = []
            self.Delta = []

        @classmethod
        def load(cls, path, device="cuda"):
            checkpoint = torch.load(path, map_location=device)
            instance = cls(
                K=checkpoint["K"],
                image_size=checkpoint["image_size"],
                eta=checkpoint["eta"],
                device=device,
                num_targets=checkpoint.get("num_targets", 3),
            )

            # 摂動ロード
            for t, d in enumerate(checkpoint["delta_g"]):
                param = torch.nn.Parameter(d.to(device))
                param.requires_grad_(True)
                instance.delta_g.append(param)

            for t, deltas in enumerate(checkpoint["Delta"]):
                deltas_t = []
                for k, d in enumerate(deltas):
                    param = torch.nn.Parameter(d.to(device))
                    param.requires_grad_(True)
                    deltas_t.append(param)
                instance.Delta.append(deltas_t)

            return instance, checkpoint

    # モデルロード
    print("[Modal] Loading FastProtect checkpoint...")
    perturbations, ckpt = FastProtectPerturbations.load(checkpoint_path, device=device)

    print("[Modal] Loading K-means...")
    kmeans_path = "/vol/fastprotect_model/kmeans_model.pkl"
    with open(kmeans_path, "rb") as f:
        kmeans = pickle.load(f)

    print("[Modal] Loading target entropies...")
    entropies_path = "/vol/fastprotect_model/target_entropies.json"
    with open(entropies_path, "r") as f:
        entropy_data = json.load(f)
        target_entropies = entropy_data["entropies"]

    print("[Modal] Loading VAE...")
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sdxl-vae",
        torch_dtype=torch.bfloat16,
    ).to(device)
    vae.eval()

    # LPIPS
    print("[Modal] Loading LPIPS...")
    import lpips
    lpips_model = lpips.LPIPS(net='alex').to(device)
    lpips_model.eval()

    # 画像ロード
    print(f"[Modal] Loading image: {image_path}")
    img = Image.open(image_path).convert("RGB")
    original_size = img.size
    print(f"[Modal] Original size: {original_size}")

    # 8の倍数にパディング（VAE要件）
    w, h = original_size
    new_w = ((w + 7) // 8) * 8
    new_h = ((h + 7) // 8) * 8
    if (new_w, new_h) != (w, h):
        print(f"[Modal] Padding to: {new_w}x{new_h}")
        img_padded = img.resize((new_w, new_h), Image.LANCZOS)
    else:
        img_padded = img

    transform = T.ToTensor()
    img_tensor = transform(img_padded).unsqueeze(0).to(device)

    # VAE encode（元画像）
    print("[Modal] VAE encoding original image...")
    img_normalized = (img_tensor * 2 - 1).to(torch.bfloat16)
    with torch.no_grad():
        z_original = vae.encode(img_normalized).latent_dist.mean.float()

    # ターゲット＆クラスタ選択
    def select_target_by_entropy(z):
        z_flat = z.view(1, -1)
        entropy = z_flat.var(dim=1).item()
        distances = [abs(entropy - te) for te in target_entropies]
        return distances.index(min(distances))

    def predict_cluster(z):
        z_flat = z.cpu().numpy().reshape(1, -1)
        return kmeans.predict(z_flat)[0]

    target_idx = select_target_by_entropy(z_original)
    cluster_idx = predict_cluster(z_original)
    print(f"[Modal] Target: {target_idx}, Cluster: {cluster_idx}")

    # 摂動適用（Adaptive Protection）
    print("[Modal] Applying adaptive protection...")
    delta = perturbations.delta_g[target_idx] + perturbations.Delta[target_idx][cluster_idx]

    # 画像サイズに合わせてリサイズ
    if delta.shape[1:] != img_tensor.shape[2:]:
        delta_resized = F.interpolate(
            delta.unsqueeze(0),
            size=img_tensor.shape[2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    else:
        delta_resized = delta

    # グローバル強度適用
    delta_scaled = delta_resized * strength

    # サロゲート
    surrogate = torch.clamp(img_tensor + delta_scaled, 0, 1)

    # LPIPS距離マップ
    with torch.no_grad():
        orig_norm = img_tensor * 2 - 1
        surr_norm = surrogate * 2 - 1
        distance_map = lpips_model(orig_norm, surr_norm, normalize=False)

    # スケーリングマップ生成
    M_normalized = (distance_map - distance_map.min()) / (distance_map.max() - distance_map.min() + 1e-8)
    scaling = base_scale * (1 - M_normalized * sensitivity)
    scaling = torch.clamp(scaling, 0.02, 1.0)

    # リサイズ
    if scaling.shape[2:] != img_tensor.shape[2:]:
        scaling = F.interpolate(scaling, size=img_tensor.shape[2:], mode="bilinear", align_corners=False)

    # 最終摂動適用
    final_delta = scaling.squeeze(0) * delta_scaled
    protected = torch.clamp(img_tensor + final_delta, 0, 1)

    # パディングを元に戻す
    if (new_w, new_h) != (w, h):
        protected = F.interpolate(
            protected,
            size=(h, w),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )

    # 保存
    print(f"[Modal] Saving to: {output_path}")
    protected_np = protected.squeeze(0).cpu().numpy()
    protected_np = (protected_np * 255).astype("uint8").transpose(1, 2, 0)
    protected_img = Image.fromarray(protected_np)
    protected_img.save(output_path, quality=95)

    # cos sim測定
    cos_sim = None
    if measure_cosine:
        print("[Modal] Measuring VAE latent cosine similarity...")
        protected_tensor = transform(protected_img).unsqueeze(0).to(device)

        # パディング（VAE要件）
        if (new_w, new_h) != (w, h):
            protected_padded = F.interpolate(
                protected_tensor,
                size=(new_h, new_w),
                mode="bilinear",
                align_corners=False,
            )
        else:
            protected_padded = protected_tensor

        protected_normalized = (protected_padded * 2 - 1).to(torch.bfloat16)
        with torch.no_grad():
            z_protected = vae.encode(protected_normalized).latent_dist.mean.float()

        # cos sim計算
        z1_flat = z_original.view(-1)
        z2_flat = z_protected.view(-1)
        cos_sim = F.cosine_similarity(z1_flat.unsqueeze(0), z2_flat.unsqueeze(0)).item()
        print(f"[Modal] VAE Latent Cosine Similarity: {cos_sim:.4f}")

    return {
        "cos_sim": cos_sim,
        "original_size": original_size,
        "protected_path": output_path,
        "target_idx": target_idx,
        "cluster_idx": cluster_idx,
        "strength": strength,
    }


@app.local_entrypoint()
def main(
    test_single: bool = False,
    test_strengths: bool = False,
    input_image: str = None,
    strength: float = 0.4,
):
    """
    FastProtect Native Resolution テスト

    Examples:
        modal run scripts/modal_fastprotect_native.py --test-single --strength 0.4
        modal run scripts/modal_fastprotect_native.py --test-strengths
        modal run scripts/modal_fastprotect_native.py --input-image /vol/my_image.jpg --strength 0.5
    """
    if test_single:
        # 単一画像テスト
        print("=== Single Image Test ===")
        result = protect_and_measure.remote(
            image_path="/vol/test_images/anime_char.jpg",
            output_path=f"/vol/protected_anime_char_s{strength}.jpg",
            strength=strength,
        )
        print(f"\n[Result] Strength: {strength}")
        print(f"  Cosine Similarity: {result['cos_sim']:.4f}")
        print(f"  Original Size: {result['original_size']}")
        print(f"  Protected Path: {result['protected_path']}")

    elif test_strengths:
        # 複数strength比較
        print("=== Strength Comparison Test ===")
        strengths = [0.4, 0.5, 0.6, 0.7]
        results = []

        for s in strengths:
            print(f"\n[Testing] Strength: {s}")
            result = protect_and_measure.remote(
                image_path="/vol/test_images/anime_char.jpg",
                output_path=f"/vol/protected_anime_char_s{s}.jpg",
                strength=s,
            )
            results.append(result)

        # サマリー
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"{'Strength':<10} {'Cos Sim':<10} {'判定'}")
        print("-" * 60)
        for r in results:
            s = r['strength']
            cs = r['cos_sim']

            if cs > 0.9:
                judge = "⚠️  弱い（要強化）"
            elif cs > 0.85:
                judge = "⚠️  やや弱め"
            elif cs > 0.8:
                judge = "✅ 良好"
            else:
                judge = "✅ 強力"

            print(f"{s:<10.1f} {cs:<10.4f} {judge}")

    elif input_image:
        # カスタム画像
        print(f"=== Custom Image: {input_image} ===")
        output_name = Path(input_image).stem + f"_protected_s{strength}.jpg"
        output_path = f"/vol/{output_name}"

        result = protect_and_measure.remote(
            image_path=input_image,
            output_path=output_path,
            strength=strength,
        )
        print(f"\n[Result]")
        print(f"  Cosine Similarity: {result['cos_sim']:.4f}")
        print(f"  Original Size: {result['original_size']}")
        print(f"  Protected Path: {result['protected_path']}")

    else:
        print("Usage:")
        print("  modal run scripts/modal_fastprotect_native.py --test-single --strength 0.4")
        print("  modal run scripts/modal_fastprotect_native.py --test-strengths")
        print("  modal run scripts/modal_fastprotect_native.py --input-image /vol/my_image.jpg --strength 0.5")
