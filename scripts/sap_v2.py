#!/usr/bin/env python3
"""
SAP v2 - Semantic Adaptive Protection v2

Dynamic Targeted Attack + Micro-Warping による次世代AIイラストガード。

特徴:
1. Dynamic Targeted Attack: 画像ごとに異なるターゲット概念に誘導
2. VAE + CLIP 同時攻撃: LoRA学習の両方の経路を攻撃
3. Micro-Warping: 幾何学的変形でLightShed耐性を向上
4. 解像度の壁への対処: CLIPは224x224にリサイズして損失計算

Usage:
    # ローカルテスト
    python scripts/sap_v2.py --test --input path/to/image.png

    # Modal実行
    modal run scripts/sap_v2.py --attack
"""

import modal
from pathlib import Path

app = modal.App("sap-v2")

volume = modal.Volume.from_name("ironclad-test-vol", create_if_missing=True)
VOLUME_PATH = "/vol"

# Image with VAE, CLIP, LPIPS, Kornia support
sap_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch==2.1.2",
        "torchvision==0.16.2",
        "diffusers==0.25.1",
        "transformers==4.38.2",
        "huggingface_hub==0.21.4",
        "accelerate",
        "safetensors",
        "lpips",
        "pillow",
        "numpy<2.0",
        "tqdm",
        "kornia",  # Micro-Warping用
        "open_clip_torch",  # CLIP用
        "ftfy",  # CLIPのテキスト処理用
    )
)


# ターゲット概念プール（多様性を確保）
TARGET_CONCEPTS = [
    "a photo of a dog",
    "a photo of a cat",
    "abstract geometric painting",
    "a landscape photograph",
    "a kitchen appliance",
    "colorful geometric shapes",
    "watercolor painting of flowers",
    "pixel art character",
    "oil painting of fruits",
    "a blurry photograph of trees",
    "pencil sketch of a building",
    "neon lights in the dark",
]


def get_image_seed(image_tensor):
    """画像固有のシードを生成"""
    import torch
    import hashlib

    # 画像のハッシュからシードを生成
    img_bytes = image_tensor.cpu().numpy().tobytes()
    hash_hex = hashlib.sha256(img_bytes).hexdigest()[:8]
    return int(hash_hex, 16)


def apply_micro_warping(image_tensor, magnitude=0.015, seed=None):
    """
    Micro-Warping: 微細な幾何学的変形

    Args:
        image_tensor: (B, C, H, W) tensor in [0, 1]
        magnitude: 変形の強さ（0.01-0.03推奨）
        seed: 再現性のためのシード

    Returns:
        Warped image tensor
    """
    import torch
    import kornia

    if seed is not None:
        torch.manual_seed(seed)

    B, C, H, W = image_tensor.shape
    device = image_tensor.device

    # ノイズフィールドを生成（変形の方向）
    noise = torch.randn(B, 2, H, W, device=device) * magnitude

    # Elastic Transform（ぐにゃりと曲げる）
    # sigma: ぼかしの強さ（大きいほど滑らか）
    # kernel_size: カーネルサイズ（奇数）
    warped = kornia.geometry.transform.elastic_transform2d(
        image_tensor,
        noise,
        kernel_size=(63, 63),  # 大きめで滑らかに
        sigma=(12.0, 12.0),
        align_corners=True,
    )

    return torch.clamp(warped, 0, 1)


@app.function(
    image=sap_image,
    volumes={VOLUME_PATH: volume},
    gpu="A10G",
    timeout=3600,
)
def apply_sap_v2(
    input_dir: str,
    output_dir: str,
    iterations: int = 200,
    lr: float = 0.01,
    # LPIPS制約
    target_lpips: float = 0.10,  # 視認性重視（0.07より緩和、0.12以下）
    lpips_weight: float = 3.0,
    # 攻撃パラメータ
    vae_weight: float = 1.5,
    clip_weight: float = 5.0,
    # 摂動上限
    epsilon: float = 0.12,  # 12%（視認性とのバランス）
    # Micro-Warping
    use_warping: bool = True,
    warp_magnitude: float = 0.015,
):
    """
    SAP v2: Dynamic Targeted Attack + Micro-Warping

    Phase 1: VAE + CLIP攻撃（動的ターゲット）
    Phase 2: Micro-Warping（オプション）
    """
    import torch
    import torch.nn.functional as F
    import lpips
    import open_clip
    from PIL import Image
    from pathlib import Path
    from diffusers import AutoencoderKL
    from torchvision import transforms
    from tqdm import tqdm
    import numpy as np

    device = torch.device("cuda")

    # === Load Models ===
    print("Loading SDXL VAE...")
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sdxl-vae",
        torch_dtype=torch.float32,
    ).to(device)
    vae.eval()

    print("Loading CLIP ViT-L/14...")
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        'ViT-L-14', pretrained='openai'
    )
    clip_model = clip_model.to(device)
    clip_model.eval()
    tokenizer = open_clip.get_tokenizer('ViT-L-14')

    print("Loading LPIPS...")
    lpips_fn = lpips.LPIPS(net='alex').to(device)
    lpips_fn.eval()

    # === Precompute target embeddings ===
    print("Precomputing target concept embeddings...")
    target_embeddings = []
    for concept in TARGET_CONCEPTS:
        tokens = tokenizer([concept]).to(device)
        with torch.no_grad():
            emb = clip_model.encode_text(tokens)
            emb = F.normalize(emb, dim=-1)
        target_embeddings.append(emb)
    target_embeddings = torch.cat(target_embeddings, dim=0)  # (N_concepts, 768)

    # === CLIP preprocessing ===
    # CLIP用の正規化パラメータ
    clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(device)
    clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(device)

    def get_clip_embedding(img):
        """解像度の壁を考慮したCLIP embedding取得"""
        # 224x224にリサイズ（微分可能）
        img_resized = F.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False)
        # 正規化
        img_norm = (img_resized - clip_mean) / clip_std
        return clip_model.encode_image(img_norm)

    # === Image transforms ===
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
    ])

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_files = list(input_path.glob("*.png")) + list(input_path.glob("*.jpg"))
    print(f"Found {len(image_files)} images")

    print(f"\n=== SAP v2 Parameters ===")
    print(f"iterations: {iterations}")
    print(f"target_lpips: {target_lpips}")
    print(f"epsilon: {epsilon}")
    print(f"use_warping: {use_warping}")
    print(f"warp_magnitude: {warp_magnitude}")

    results = []

    for img_file in image_files:
        print(f"\n{'='*60}")
        print(f"Processing: {img_file.name}")

        # Load image
        img = Image.open(img_file).convert("RGB")
        x_orig = transform(img).unsqueeze(0).to(device)  # [1, 3, 1024, 1024]

        # Get image-specific seed
        seed = get_image_seed(x_orig)
        torch.manual_seed(seed)

        # Select random target concept
        target_idx = seed % len(TARGET_CONCEPTS)
        target_clip = target_embeddings[target_idx:target_idx+1]  # (1, 768)
        print(f"Target concept: '{TARGET_CONCEPTS[target_idx]}'")

        # === Phase 1: VAE + CLIP Attack ===
        print("Phase 1: VAE + CLIP Attack")

        # Normalize to [-1, 1] for VAE
        x_orig_norm = x_orig * 2.0 - 1.0

        # Get original embeddings
        with torch.no_grad():
            z_orig = vae.encode(x_orig_norm).latent_dist.mean
            clip_orig = get_clip_embedding(x_orig)
            clip_orig = F.normalize(clip_orig, dim=-1)

        # Initialize perturbation
        delta = torch.zeros_like(x_orig, requires_grad=True, device=device)
        optimizer = torch.optim.Adam([delta], lr=lr)

        # Optimization loop
        pbar = tqdm(range(iterations), desc="Optimizing")
        best_delta = None
        best_score = float('-inf')

        for i in pbar:
            optimizer.zero_grad()

            # Apply perturbation
            prot = torch.clamp(x_orig + delta, 0, 1)
            prot_norm = prot * 2.0 - 1.0

            # === Loss calculation ===

            # 1. LPIPS (視覚品質制約)
            loss_lpips = lpips_fn(x_orig_norm, prot_norm).mean()

            # 2. VAE latent乖離（最大化 → 負号）
            z_prot = vae.encode(prot_norm).latent_dist.mean
            loss_vae = F.mse_loss(z_prot, z_orig)  # 乖離を最大化したいので負号

            # 3. CLIP: ターゲット概念に近づける
            clip_prot = get_clip_embedding(prot)
            clip_prot = F.normalize(clip_prot, dim=-1)
            # ターゲットとの類似度を最大化（= 損失を最小化）
            loss_clip = -F.cosine_similarity(clip_prot, target_clip, dim=-1).mean()

            # === Constrained optimization ===
            if loss_lpips > target_lpips:
                # LPIPS制約違反 → 視覚品質を優先
                total_loss = loss_lpips * 20.0
            else:
                # 通常の最適化
                total_loss = (
                    lpips_weight * loss_lpips
                    - vae_weight * loss_vae  # 乖離最大化
                    + clip_weight * loss_clip  # ターゲットに近づける
                )

            total_loss.backward()
            optimizer.step()

            # 摂動クリップ
            with torch.no_grad():
                delta.data = torch.clamp(delta.data, -epsilon, epsilon)

            # Best state tracking
            with torch.no_grad():
                current_lpips = loss_lpips.item()
                if current_lpips <= target_lpips:
                    score = -loss_vae.item() - loss_clip.item()
                    if score > best_score:
                        best_score = score
                        best_delta = delta.data.clone()

            if i % 40 == 0:
                pbar.set_postfix({
                    'lpips': f'{loss_lpips.item():.4f}',
                    'vae': f'{loss_vae.item():.4f}',
                    'clip': f'{loss_clip.item():.4f}',
                })

        # Use best delta if available
        if best_delta is not None:
            delta.data = best_delta

        # === Phase 2: Micro-Warping (Optional) ===
        with torch.no_grad():
            x_attacked = torch.clamp(x_orig + delta, 0, 1)

            if use_warping:
                print("Phase 2: Micro-Warping")
                x_protected = apply_micro_warping(x_attacked, warp_magnitude, seed)
            else:
                x_protected = x_attacked

        # === Final evaluation ===
        with torch.no_grad():
            x_prot_norm = x_protected * 2.0 - 1.0

            # Final LPIPS
            final_lpips = lpips_fn(x_orig_norm, x_prot_norm).item()

            # Final VAE similarity
            z_prot_final = vae.encode(x_prot_norm).latent_dist.mean
            final_vae_sim = F.cosine_similarity(
                z_orig.view(1, -1),
                z_prot_final.view(1, -1)
            ).item()

            # Final CLIP similarity (to original)
            clip_prot_final = get_clip_embedding(x_protected)
            clip_prot_final = F.normalize(clip_prot_final, dim=-1)
            final_clip_sim = F.cosine_similarity(clip_prot_final, clip_orig, dim=-1).item()

            # CLIP similarity to target
            final_clip_target = F.cosine_similarity(clip_prot_final, target_clip, dim=-1).item()

        print(f"\nFinal Results:")
        print(f"  LPIPS: {final_lpips:.4f} (target: {target_lpips})")
        print(f"  VAE Cosine Sim: {final_vae_sim:.4f}")
        print(f"  CLIP Sim (to orig): {final_clip_sim:.4f}")
        print(f"  CLIP Sim (to target): {final_clip_target:.4f}")

        # Check visual quality
        if final_lpips > target_lpips * 1.2:
            print(f"  ⚠️ Warning: LPIPS exceeds target by >20%")

        # Save
        x_prot_np = (x_protected.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        out_img = Image.fromarray(x_prot_np)
        out_path = output_path / img_file.name
        out_img.save(out_path, quality=95)

        # Copy caption if exists
        caption_file = img_file.with_suffix(".txt")
        if caption_file.exists():
            (output_path / caption_file.name).write_text(caption_file.read_text())

        results.append({
            'file': img_file.name,
            'lpips': final_lpips,
            'vae_sim': final_vae_sim,
            'clip_sim_orig': final_clip_sim,
            'clip_sim_target': final_clip_target,
            'target_concept': TARGET_CONCEPTS[target_idx],
        })

    volume.commit()

    # Summary
    avg_lpips = np.mean([r['lpips'] for r in results])
    avg_vae_sim = np.mean([r['vae_sim'] for r in results])
    avg_clip_orig = np.mean([r['clip_sim_orig'] for r in results])
    avg_clip_target = np.mean([r['clip_sim_target'] for r in results])

    print(f"\n{'='*60}")
    print(f"SAP v2 Complete")
    print(f"{'='*60}")
    print(f"Images processed: {len(results)}")
    print(f"Average LPIPS: {avg_lpips:.4f}")
    print(f"Average VAE Similarity: {avg_vae_sim:.4f}")
    print(f"Average CLIP Sim (orig): {avg_clip_orig:.4f}")
    print(f"Average CLIP Sim (target): {avg_clip_target:.4f}")

    return {
        'status': 'success',
        'count': len(results),
        'avg_lpips': avg_lpips,
        'avg_vae_sim': avg_vae_sim,
        'avg_clip_sim_orig': avg_clip_orig,
        'avg_clip_sim_target': avg_clip_target,
        'results': results,
    }


@app.function(
    image=sap_image,
    volumes={VOLUME_PATH: volume},
    gpu="A10G",
    timeout=600,
)
def setup_check():
    """環境確認"""
    import torch
    import kornia
    import open_clip
    from diffusers import AutoencoderKL
    import lpips

    print(f"PyTorch: {torch.__version__}")
    print(f"Kornia: {kornia.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Check VAE
    print("\nLoading SDXL VAE...")
    vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")
    print(f"VAE loaded: {type(vae)}")

    # Check CLIP
    print("\nLoading CLIP ViT-L/14...")
    clip_model, _, _ = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
    print(f"CLIP loaded: {type(clip_model)}")

    # Check LPIPS
    print("\nLoading LPIPS...")
    lpips_fn = lpips.LPIPS(net='alex')
    print(f"LPIPS loaded: {type(lpips_fn)}")

    # Check Kornia
    print("\nTesting Kornia elastic transform...")
    test_img = torch.randn(1, 3, 256, 256)
    noise = torch.randn(1, 2, 256, 256) * 0.02
    warped = kornia.geometry.transform.elastic_transform2d(
        test_img, noise, kernel_size=(31, 31), sigma=(6.0, 6.0)
    )
    print(f"Kornia test: {warped.shape}")

    # Check training data
    from pathlib import Path
    for folder in ["train_normal", "train_hf_stealth", "train_sap_v2"]:
        data_path = Path(VOLUME_PATH) / folder
        if data_path.exists():
            images = list(data_path.glob("*.png")) + list(data_path.glob("*.jpg"))
            print(f"\n{folder}: {len(images)} images")

    volume.commit()
    return {"status": "ready"}


@app.function(
    image=sap_image,
    volumes={VOLUME_PATH: volume},
    gpu="A10G",
    timeout=600,
)
def test_single_image(
    input_file: str = "train_normal/00001.png",
    output_file: str = "test_sap_v2_sample.png",
    iterations: int = 100,
    target_lpips: float = 0.10,
    epsilon: float = 0.12,
    use_warping: bool = True,
    warp_magnitude: float = 0.015,
):
    """1枚の画像でSAP v2をテスト（視認性確認用）"""
    import torch
    import torch.nn.functional as F
    import lpips
    import open_clip
    from PIL import Image
    from pathlib import Path
    from diffusers import AutoencoderKL
    from torchvision import transforms
    import numpy as np

    device = torch.device("cuda")

    # Load models
    print("Loading models...")
    vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae", torch_dtype=torch.float32).to(device)
    vae.eval()

    lpips_fn = lpips.LPIPS(net='alex').to(device)
    lpips_fn.eval()

    # Load image - find first available file if not found
    img_path = Path(VOLUME_PATH) / input_file
    if not img_path.exists():
        # Auto-detect first file in directory
        folder = Path(VOLUME_PATH) / input_file.rsplit('/', 1)[0]
        files = list(folder.glob("*.png")) + list(folder.glob("*.jpg"))
        if files:
            img_path = sorted(files)[0]
            print(f"Auto-detected: {img_path.name}")
        else:
            raise FileNotFoundError(f"No images found in {folder}")
    img = Image.open(img_path).convert('RGB')
    orig_size = img.size
    print(f"元画像サイズ: {orig_size}")

    # 1. 1024にリサイズ
    transform_1024 = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
    ])
    x_1024 = transform_1024(img).unsqueeze(0).to(device)

    seed = get_image_seed(x_1024)
    torch.manual_seed(seed)

    # 2. Micro-Warping
    if use_warping:
        print("Applying Micro-Warping...")
        x_warped = apply_micro_warping(x_1024, warp_magnitude, seed)
    else:
        x_warped = x_1024

    # 3. VAE攻撃
    print(f"Applying VAE attack ({iterations} iterations)...")
    x_base_norm = x_warped * 2.0 - 1.0

    with torch.no_grad():
        z_orig = vae.encode(x_base_norm).latent_dist.mean

    delta = torch.zeros_like(x_warped, requires_grad=True, device=device)
    optimizer = torch.optim.Adam([delta], lr=0.01)

    for i in range(iterations):
        optimizer.zero_grad()

        prot = torch.clamp(x_warped + delta, 0, 1)
        prot_norm = prot * 2.0 - 1.0

        loss_lpips = lpips_fn(x_base_norm, prot_norm).mean()
        z_prot = vae.encode(prot_norm).latent_dist.mean
        loss_vae = F.mse_loss(z_prot, z_orig)

        if loss_lpips > target_lpips:
            total_loss = loss_lpips * 20.0
        else:
            total_loss = 3.0 * loss_lpips - 1.5 * loss_vae

        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            delta.data = torch.clamp(delta.data, -epsilon, epsilon)

        if i % 20 == 0:
            print(f"  iter {i}: lpips={loss_lpips.item():.4f}, vae_loss={loss_vae.item():.4f}")

    # 4. 元サイズに戻す
    with torch.no_grad():
        x_protected = torch.clamp(x_warped + delta, 0, 1)
        x_final = F.interpolate(x_protected, size=(orig_size[1], orig_size[0]), mode='bilinear', align_corners=False)

        # 最終LPIPS
        x_orig_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
        final_lpips = lpips_fn(x_orig_tensor * 2 - 1, x_final * 2 - 1).item()

    print(f"\n最終LPIPS: {final_lpips:.4f}")
    print(f"最終サイズ: ({orig_size[0]}, {orig_size[1]})")

    # 保存
    x_final_np = (x_final.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    out_img = Image.fromarray(x_final_np)
    out_path = Path(VOLUME_PATH) / output_file
    out_img.save(out_path, quality=95)
    print(f"Saved: {out_path}")

    volume.commit()
    return {"lpips": final_lpips, "size": orig_size}


@app.local_entrypoint()
def main(
    setup: bool = False,
    attack: bool = False,
    test: bool = False,
    input_folder: str = "train_normal",
    output_folder: str = "train_sap_v2",
    iterations: int = 200,
    target_lpips: float = 0.10,
    epsilon: float = 0.12,
    use_warping: bool = True,
    warp_magnitude: float = 0.015,
):
    """
    SAP v2 Main Entrypoint

    Default parameters optimized for:
    - Visual quality: LPIPS <= 0.10 (slightly more permissive than Nightshade)
    - Attack strength: epsilon=0.12 (12% max perturbation)
    - LightShed resistance: Micro-Warping enabled
    """

    if setup:
        print("=== Setup Check ===")
        result = setup_check.remote()
        print(f"Result: {result}")

    if test:
        print("\n=== SAP v2: Single Image Test ===")
        result = test_single_image.remote(
            input_file=f"{input_folder}/00001.png",
            output_file="test_sap_v2_sample.png",
            iterations=100,
            target_lpips=target_lpips,
            epsilon=epsilon,
            use_warping=use_warping,
            warp_magnitude=warp_magnitude,
        )
        print(f"Result: {result}")

    if attack:
        print("\n=== SAP v2: Dynamic Targeted Attack + Micro-Warping ===")
        result = apply_sap_v2.remote(
            input_dir=f"{VOLUME_PATH}/{input_folder}",
            output_dir=f"{VOLUME_PATH}/{output_folder}",
            iterations=iterations,
            target_lpips=target_lpips,
            epsilon=epsilon,
            use_warping=use_warping,
            warp_magnitude=warp_magnitude,
        )
        print(f"Result: {result}")


# === Local testing ===
def test_local(image_path: str, output_path: str = None):
    """
    ローカル環境でのテスト（GPU必要）

    Usage:
        python scripts/sap_v2.py --test --input path/to/image.png
    """
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from torchvision import transforms

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if device.type == "cpu":
        print("Warning: Running on CPU. This will be slow.")

    # Load image
    transform = transforms.Compose([
        transforms.Resize((512, 512)),  # Smaller for local testing
        transforms.ToTensor(),
    ])

    img = Image.open(image_path).convert("RGB")
    x_orig = transform(img).unsqueeze(0).to(device)

    print(f"Input shape: {x_orig.shape}")

    # Test Micro-Warping only (no GPU-heavy models)
    print("\nTesting Micro-Warping...")
    seed = get_image_seed(x_orig)
    x_warped = apply_micro_warping(x_orig, magnitude=0.02, seed=seed)

    # Compute difference
    diff = (x_warped - x_orig).abs().mean().item()
    print(f"Mean absolute difference: {diff:.6f}")

    # Save if output path provided
    if output_path:
        x_warped_np = (x_warped.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
        Image.fromarray(x_warped_np).save(output_path)
        print(f"Saved to: {output_path}")

    return x_warped


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SAP v2")
    parser.add_argument("--test", action="store_true", help="Run local test")
    parser.add_argument("--input", type=str, help="Input image path")
    parser.add_argument("--output", type=str, help="Output image path")

    args = parser.parse_args()

    if args.test:
        if args.input:
            test_local(args.input, args.output)
        else:
            print("Usage: python scripts/sap_v2.py --test --input path/to/image.png")
    else:
        print("Use: modal run scripts/sap_v2.py --help")
