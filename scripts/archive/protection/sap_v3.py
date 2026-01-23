#!/usr/bin/env python3
"""
SAP v3 - Semantic Adaptive Protection v3

改良点:
1. CLIP攻撃: ネガティブ概念誘導 + 概念混乱（元画像からの離脱）
2. VAE攻撃: ホワイトノイズではなく「間違った構造」への誘導
3. 適応型マスク: エッジ5%、平坦1%
4. Micro-Warping: 幾何学的変形（オプション）

Usage:
    modal run scripts/sap_v3.py --test --warp-magnitude 0.01
    modal run scripts/sap_v3.py --attack
"""

import modal
from pathlib import Path

app = modal.App("sap-v3")

volume = modal.Volume.from_name("ironclad-test-vol", create_if_missing=True)
VOLUME_PATH = "/vol"

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
        "kornia",
        "open_clip_torch",
        "ftfy",
    )
)

# ネガティブ概念（AIが「低品質」と認識するもの）
NEGATIVE_CONCEPTS = [
    "low quality, worst quality, blurry",
    "jpeg artifacts, noise, grainy",
    "text, watermark, signature",
    "error, glitch, corrupted",
    "out of focus, motion blur",
]

# 概念混乱用（アニメとは無関係なもの）
CONFUSION_CONCEPTS = [
    "a photograph of mountains and trees",
    "3d render of geometric shapes",
    "satellite image of earth",
    "medical x-ray scan",
    "infrared thermal image",
    "microscope image of cells",
    "architectural blueprint drawing",
    "stock chart financial graph",
]


def get_image_seed(image_tensor):
    """画像固有のシードを生成"""
    import torch
    import hashlib
    img_bytes = image_tensor.cpu().numpy().tobytes()
    hash_hex = hashlib.sha256(img_bytes).hexdigest()[:8]
    return int(hash_hex, 16)


def apply_micro_warping(image_tensor, magnitude=0.01, seed=None):
    """Micro-Warping: 微細な幾何学的変形"""
    import torch
    import kornia

    if seed is not None:
        torch.manual_seed(seed)

    B, C, H, W = image_tensor.shape
    device = image_tensor.device

    noise = torch.randn(B, 2, H, W, device=device) * magnitude

    warped = kornia.geometry.transform.elastic_transform2d(
        image_tensor,
        noise,
        kernel_size=(63, 63),
        sigma=(12.0, 12.0),
        align_corners=True,
    )

    return torch.clamp(warped, 0, 1)


def compute_adaptive_mask(image_tensor):
    """
    適応型マスク: エッジ部5%、平坦部1%
    Sobelフィルタでエッジ検出
    """
    import torch
    import kornia

    # Sobelフィルタでエッジ検出
    grads = kornia.filters.sobel(image_tensor)
    magnitude = torch.sqrt(grads[:, 0:1] ** 2 + grads[:, 1:2] ** 2 + 1e-8)

    # 正規化
    magnitude = magnitude / (magnitude.max() + 1e-8)

    # エッジ部: 5% (0.05), 平坦部: 1% (0.01)
    # sigmoidで滑らかに遷移
    mask = 0.01 + (0.04 * torch.sigmoid((magnitude - 0.3) * 20))

    # 3チャンネルに拡張
    mask = mask.expand(-1, 3, -1, -1)

    return mask


@app.function(
    image=sap_image,
    volumes={VOLUME_PATH: volume},
    gpu="A10G",
    timeout=1800,
)
def test_sap_v3(
    iterations: int = 50,
    target_lpips: float = 0.08,
    use_warping: bool = True,
    warp_magnitude: float = 0.01,
    vae_weight: float = 10.0,
    clip_neg_weight: float = 3.0,
    clip_conf_weight: float = 2.0,
):
    """
    SAP v3 単体テスト

    攻撃構成:
    1. VAE: 元latentから離脱（乖離最大化）
    2. CLIP: ネガティブ概念に近づける + 元概念から離脱
    3. 適応型マスク: エッジ5%、平坦1%
    4. Micro-Warping: オプション
    """
    import torch
    import torch.nn.functional as F
    import lpips
    import open_clip
    import kornia
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
    clip_model, _, _ = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
    clip_model = clip_model.to(device)
    clip_model.eval()
    tokenizer = open_clip.get_tokenizer('ViT-L-14')

    print("Loading LPIPS...")
    lpips_fn = lpips.LPIPS(net='alex').to(device)
    lpips_fn.eval()

    # === Precompute CLIP embeddings ===
    print("Precomputing CLIP embeddings...")

    # ネガティブ概念の平均embedding
    neg_embeddings = []
    for concept in NEGATIVE_CONCEPTS:
        tokens = tokenizer([concept]).to(device)
        with torch.no_grad():
            emb = clip_model.encode_text(tokens)
            emb = F.normalize(emb, dim=-1)
        neg_embeddings.append(emb)
    negative_embedding = torch.cat(neg_embeddings, dim=0).mean(dim=0, keepdim=True)
    negative_embedding = F.normalize(negative_embedding, dim=-1)

    # 混乱概念のembeddings
    conf_embeddings = []
    for concept in CONFUSION_CONCEPTS:
        tokens = tokenizer([concept]).to(device)
        with torch.no_grad():
            emb = clip_model.encode_text(tokens)
            emb = F.normalize(emb, dim=-1)
        conf_embeddings.append(emb)
    confusion_embeddings = torch.cat(conf_embeddings, dim=0)  # (N, 768)

    # CLIP preprocessing
    clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(device)
    clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(device)

    def get_clip_embedding(img):
        img_resized = F.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False)
        img_norm = (img_resized - clip_mean) / clip_std
        return clip_model.encode_image(img_norm)

    # === Load test image ===
    input_path = Path(VOLUME_PATH) / "train_normal"
    image_files = sorted(list(input_path.glob("*.png")) + list(input_path.glob("*.jpg")))
    if not image_files:
        return {"error": "No images found"}

    img_file = image_files[0]
    print(f"\nTest image: {img_file.name}")

    img = Image.open(img_file).convert("RGB")
    orig_size = img.size
    print(f"Original size: {orig_size}")

    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
    ])
    x_orig = transform(img).unsqueeze(0).to(device)

    seed = get_image_seed(x_orig)
    torch.manual_seed(seed)

    # 画像固有の混乱ターゲットを選択
    conf_idx = seed % len(CONFUSION_CONCEPTS)
    target_confusion = confusion_embeddings[conf_idx:conf_idx+1]
    print(f"Confusion target: '{CONFUSION_CONCEPTS[conf_idx]}'")

    # === Phase 1: Micro-Warping (Optional) ===
    if use_warping:
        print(f"Phase 1: Micro-Warping (magnitude={warp_magnitude})")
        x_base = apply_micro_warping(x_orig, warp_magnitude, seed)
    else:
        x_base = x_orig

    # === Phase 2: VAE + CLIP Attack ===
    print(f"Phase 2: VAE + CLIP Attack ({iterations} iterations)")

    x_base_norm = x_base * 2.0 - 1.0

    with torch.no_grad():
        z_orig = vae.encode(x_base_norm).latent_dist.mean
        clip_orig = get_clip_embedding(x_base)
        clip_orig = F.normalize(clip_orig, dim=-1)

    # 適応型マスク計算
    adaptive_mask = compute_adaptive_mask(x_base)
    edge_ratio = (adaptive_mask > 0.03).float().mean().item()
    print(f"Edge ratio: {edge_ratio*100:.1f}%")

    # 摂動の初期化
    delta = torch.zeros_like(x_base, requires_grad=True, device=device)
    optimizer = torch.optim.Adam([delta], lr=0.01)

    # Optimization
    pbar = tqdm(range(iterations), desc="Optimizing")
    for i in pbar:
        optimizer.zero_grad()

        # Apply perturbation with adaptive mask
        delta_masked = torch.clamp(delta, -adaptive_mask, adaptive_mask)
        x_adv = torch.clamp(x_base + delta_masked, 0, 1)
        x_adv_norm = x_adv * 2.0 - 1.0

        # === Loss 1: LPIPS (視覚品質制約) ===
        loss_lpips = lpips_fn(x_base_norm, x_adv_norm).mean()

        # === Loss 2: VAE latent乖離 ===
        z_adv = vae.encode(x_adv_norm).latent_dist.mean
        # cos simを最小化（= 乖離を最大化）
        vae_cos_sim = F.cosine_similarity(z_orig.view(1, -1), z_adv.view(1, -1))
        loss_vae = vae_cos_sim  # 最小化したい

        # === Loss 3: CLIP ネガティブ誘導 ===
        clip_adv = get_clip_embedding(x_adv)
        clip_adv = F.normalize(clip_adv, dim=-1)
        # ネガティブ概念への類似度を最大化
        loss_clip_neg = -F.cosine_similarity(clip_adv, negative_embedding).mean()

        # === Loss 4: CLIP 概念混乱（元画像から離脱 + 混乱ターゲットに近づく） ===
        # 元画像からの離脱
        loss_clip_orig = F.cosine_similarity(clip_adv, clip_orig).mean()
        # 混乱ターゲットへの誘導
        loss_clip_conf = -F.cosine_similarity(clip_adv, target_confusion).mean()

        # === 統合Loss ===
        if loss_lpips > target_lpips:
            # LPIPS制約違反時は視覚品質を優先
            total_loss = loss_lpips * 50.0
        else:
            total_loss = (
                3.0 * loss_lpips
                + vae_weight * loss_vae
                + clip_neg_weight * loss_clip_neg
                + clip_conf_weight * (loss_clip_orig + loss_clip_conf)
            )

        total_loss.backward()
        optimizer.step()

        # Clamp delta to adaptive mask
        with torch.no_grad():
            delta.data = torch.clamp(delta.data, -adaptive_mask, adaptive_mask)

        if i % 10 == 0:
            pbar.set_postfix({
                'lpips': f'{loss_lpips.item():.4f}',
                'vae_sim': f'{vae_cos_sim.item():.4f}',
                'clip_neg': f'{-loss_clip_neg.item():.4f}',
            })

    # === Final evaluation ===
    with torch.no_grad():
        delta_final = torch.clamp(delta, -adaptive_mask, adaptive_mask)
        x_protected = torch.clamp(x_base + delta_final, 0, 1)

        # 元サイズに戻す
        x_final = F.interpolate(x_protected, size=(orig_size[1], orig_size[0]), mode='bilinear', align_corners=False)

        # Final metrics
        x_prot_norm = x_protected * 2.0 - 1.0

        # LPIPS (1024x1024で計算)
        final_lpips = lpips_fn(x_base_norm, x_prot_norm).item()

        # VAE cos sim
        z_final = vae.encode(x_prot_norm).latent_dist.mean
        final_vae_sim = F.cosine_similarity(z_orig.view(1, -1), z_final.view(1, -1)).item()

        # CLIP metrics
        clip_final = get_clip_embedding(x_protected)
        clip_final = F.normalize(clip_final, dim=-1)

        final_clip_orig = F.cosine_similarity(clip_final, clip_orig).item()
        final_clip_neg = F.cosine_similarity(clip_final, negative_embedding).item()
        final_clip_conf = F.cosine_similarity(clip_final, target_confusion).item()

    print(f"\n{'='*60}")
    print(f"=== SAP v3 Results ===")
    print(f"{'='*60}")
    print(f"LPIPS: {final_lpips:.4f} (target: {target_lpips})")
    print(f"VAE Cosine Sim: {final_vae_sim:.4f} (lower = better attack)")
    print(f"CLIP to Original: {final_clip_orig:.4f} (lower = better)")
    print(f"CLIP to Negative: {final_clip_neg:.4f} (higher = better)")
    print(f"CLIP to Confusion: {final_clip_conf:.4f} (higher = better)")

    # Save
    x_final_np = (x_final.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    out_img = Image.fromarray(x_final_np)
    out_path = Path(VOLUME_PATH) / "test_sap_v3.png"
    out_img.save(out_path, quality=95)
    print(f"\nSaved: {out_path}")

    # Save original for comparison
    x_orig_resized = F.interpolate(x_orig, size=(orig_size[1], orig_size[0]), mode='bilinear', align_corners=False)
    x_orig_np = (x_orig_resized.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    orig_out = Image.fromarray(x_orig_np)
    orig_path = Path(VOLUME_PATH) / "test_sap_v3_original.png"
    orig_out.save(orig_path, quality=95)

    volume.commit()

    return {
        'lpips': final_lpips,
        'vae_cos_sim': final_vae_sim,
        'clip_to_orig': final_clip_orig,
        'clip_to_negative': final_clip_neg,
        'clip_to_confusion': final_clip_conf,
        'confusion_target': CONFUSION_CONCEPTS[conf_idx],
        'size': orig_size,
    }


@app.local_entrypoint()
def main(
    test: bool = False,
    attack: bool = False,
    iterations: int = 50,
    target_lpips: float = 0.08,
    use_warping: bool = True,
    warp_magnitude: float = 0.01,
):
    """
    SAP v3 Main Entrypoint

    攻撃構成:
    - VAE latent乖離（cos sim最小化）
    - CLIP ネガティブ誘導（low quality等）
    - CLIP 概念混乱（元画像から離脱 + 無関係概念へ誘導）
    - 適応型マスク（エッジ5%、平坦1%）
    - Micro-Warping（オプション）
    """

    if test:
        print("=== SAP v3: Single Image Test ===")
        result = test_sap_v3.remote(
            iterations=iterations,
            target_lpips=target_lpips,
            use_warping=use_warping,
            warp_magnitude=warp_magnitude,
        )
        print(f"\nResult: {result}")


if __name__ == "__main__":
    print("Use: modal run scripts/sap_v3.py --test")
