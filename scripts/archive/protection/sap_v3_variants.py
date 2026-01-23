#!/usr/bin/env python3
"""
SAP v3 Variants - Perlin Noise Experiment

v3の適応型マスクにPerlinノイズを導入し、
平坦部の摂動をバラけさせて視認性を改善する実験

Usage:
    # 1枚テスト
    modal run scripts/sap_v3_variants.py --test --warp-magnitude 0.01

    # 非同期バッチ処理（推奨 - タイムアウトしない）
    modal run scripts/sap_v3_variants.py --submit-batch

    # ステータス確認
    modal run scripts/sap_v3_variants.py --status
"""

import modal
from pathlib import Path
import json
from datetime import datetime

app = modal.App("sap-v3-variants")

volume = modal.Volume.from_name("ironclad-test-vol", create_if_missing=True)
VOLUME_PATH = "/vol"
BATCH_STATUS_FILE = "sap_v3_batch_status.json"


# ==================== Status Management ====================

def read_batch_status_local() -> dict:
    """Read batch status from Volume (called from local entrypoint)"""
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name

    try:
        result = subprocess.run(
            ["modal", "volume", "get", "ironclad-test-vol", BATCH_STATUS_FILE, temp_path],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode != 0:
            return {"status": "no_job", "message": "No batch job found"}

        with open(temp_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        import os
        if os.path.exists(temp_path):
            os.remove(temp_path)

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


def generate_perlin_noise(shape, scale=16, device='cuda', seed=None):
    """
    Perlin-like noise生成（簡易版）

    低解像度ノイズを補間して滑らかな空間的ばらつきを生成

    Args:
        shape: (B, C, H, W)
        scale: ノイズのスケール（大きいほど粗い）
        device: デバイス
        seed: 再現性のためのシード（Noneならランダム）
    """
    import torch
    import torch.nn.functional as F

    if seed is not None:
        torch.manual_seed(seed)

    B, C, H, W = shape
    small_h, small_w = max(1, H // scale), max(1, W // scale)

    # 低解像度ランダムノイズ
    noise = torch.randn(B, 1, small_h, small_w, device=device)

    # バイリニア補間で拡大
    noise = F.interpolate(noise, size=(H, W), mode='bilinear', align_corners=False)

    # 0-1に正規化
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)

    # 3チャンネルに拡張
    return noise.expand(-1, C, -1, -1)


def generate_fractal_perlin(shape, base_scale=64, octaves=3, persistence=0.5, device='cuda', seed=None):
    """
    Fractal Brownian Motion (fBm) - 複数スケールのPerlinノイズを重ね合わせ

    Args:
        shape: (B, C, H, W)
        base_scale: 最も粗いスケール
        octaves: 重ね合わせる層数
        persistence: 各層の振幅減衰率
        device: デバイス
        seed: 再現性のためのシード
    """
    import torch

    B, C, H, W = shape
    noise = torch.zeros(B, 1, H, W, device=device)
    amplitude = 1.0
    total_amplitude = 0.0

    for i in range(octaves):
        scale = base_scale // (2 ** i)
        if scale < 8:  # 最小スケール制限
            break
        # 各オクターブで異なるシード
        octave_seed = (seed + i * 12345) if seed is not None else None
        octave_noise = generate_perlin_noise(shape, scale=scale, device=device, seed=octave_seed)
        noise += amplitude * octave_noise[:, :1, :, :]  # 1チャンネルのみ
        total_amplitude += amplitude
        amplitude *= persistence

    # 正規化して0-1に
    noise = noise / total_amplitude
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)

    # 3チャンネルに拡張
    return noise.expand(-1, C, -1, -1)


def compute_adaptive_mask_perlin(image_tensor, use_perlin=True, perlin_scale=16, perlin_seed=None, use_fractal=False, fractal_octaves=3):
    """
    Perlin Noise付き適応型マスク

    エッジ部: 5% (固定)
    平坦部: 0.75〜1.5% (Perlinでバラける、平均1.125%)

    Args:
        image_tensor: (B, C, H, W) tensor
        use_perlin: Perlinノイズを使用するか
        perlin_scale: Perlinノイズのスケール（大きいほど粗い）
        perlin_seed: Perlinノイズのシード（再現性確保用）
        use_fractal: フラクタルPerlin (fBm) を使用するか
        fractal_octaves: フラクタルの層数
    """
    import torch
    import kornia

    B, C, H, W = image_tensor.shape
    device = image_tensor.device

    # Sobelフィルタでエッジ検出
    grads = kornia.filters.sobel(image_tensor)
    magnitude = torch.sqrt(grads[:, 0:1] ** 2 + grads[:, 1:2] ** 2 + 1e-8)

    # 正規化
    magnitude = magnitude / (magnitude.max() + 1e-8)

    # エッジ重み (0-1)
    edge_weight = torch.sigmoid((magnitude - 0.3) * 20)

    # 平坦部重み (1 - edge_weight)
    flat_weight = 1.0 - edge_weight

    if use_perlin:
        if use_fractal:
            # フラクタルPerlin (fBm): 複数スケールの重ね合わせ
            perlin = generate_fractal_perlin((B, C, H, W), base_scale=perlin_scale, octaves=fractal_octaves, device=device, seed=perlin_seed)
        else:
            # 単一スケールPerlin
            perlin = generate_perlin_noise((B, C, H, W), scale=perlin_scale, device=device, seed=perlin_seed)
        # 平坦部: 0.5〜1.0%の範囲でバラける (perlinは0-1なので、0.005 + 0.005*perlin = 0.005〜0.01)
        flat_mask = 0.005 + 0.005 * perlin
    else:
        # Perlinなし: 一様に1%
        flat_mask = torch.full((B, C, H, W), 0.01, device=device)

    # エッジ部: 5%固定
    edge_mask = 0.05 * edge_weight.expand(-1, C, -1, -1)

    # 統合: エッジ部 + 平坦部（重複部分は大きい方を採用）
    mask = torch.max(edge_mask, flat_mask * flat_weight.expand(-1, C, -1, -1))

    return mask


@app.function(
    image=sap_image,
    volumes={VOLUME_PATH: volume},
    gpu="A10G",
    timeout=1800,
)
def test_sap_v3_perlin(
    iterations: int = 50,
    target_lpips: float = 0.08,
    use_warping: bool = True,
    warp_magnitude: float = 0.01,
    vae_weight: float = 10.0,
    clip_neg_weight: float = 3.0,
    clip_conf_weight: float = 2.0,
    use_perlin: bool = True,
    perlin_scale: int = 64,  # デフォルトを64に
    perlin_seed: int = None,
    use_fractal: bool = False,
    fractal_octaves: int = 3,
):
    """
    SAP v3 + Perlin Noise テスト

    v3と同じ攻撃構成だが、適応型マスクにPerlinノイズを追加:
    - エッジ部: 5%（固定）
    - 平坦部: 0.75〜1.5%（Perlinでバラける）
    - フラクタルモード: 複数スケールの重ね合わせ
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

    # Perlinシード: 指定がなければ画像ハッシュベースを使用（再現可能かつ画像ごとに異なる）
    effective_perlin_seed = perlin_seed if perlin_seed is not None else seed
    fractal_str = f", fractal={fractal_octaves}oct" if use_fractal else ""
    print(f"Perlin noise: {'enabled' if use_perlin else 'disabled'}, scale={perlin_scale}{fractal_str}, seed={effective_perlin_seed} ({'user-specified' if perlin_seed is not None else 'image-hash'})")

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

    # Perlin付き適応型マスク計算
    adaptive_mask = compute_adaptive_mask_perlin(x_base, use_perlin=use_perlin, perlin_scale=perlin_scale, perlin_seed=effective_perlin_seed, use_fractal=use_fractal, fractal_octaves=fractal_octaves)
    edge_ratio = (adaptive_mask > 0.03).float().mean().item()
    flat_mean = adaptive_mask[adaptive_mask < 0.03].mean().item() if (adaptive_mask < 0.03).any() else 0
    print(f"Edge ratio: {edge_ratio*100:.1f}%, Flat mean: {flat_mean*100:.2f}%")

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
    print(f"=== SAP v3 + Perlin Results ===")
    print(f"{'='*60}")
    print(f"LPIPS: {final_lpips:.4f} (target: {target_lpips})")
    print(f"VAE Cosine Sim: {final_vae_sim:.4f} (lower = better attack)")
    print(f"CLIP to Original: {final_clip_orig:.4f} (lower = better)")
    print(f"CLIP to Negative: {final_clip_neg:.4f} (higher = better)")
    print(f"CLIP to Confusion: {final_clip_conf:.4f} (higher = better)")

    # Save (scale毎に異なるファイル名)
    x_final_np = (x_final.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    out_img = Image.fromarray(x_final_np)
    suffix = f"_scale{perlin_scale}" if use_perlin else "_noperlin"
    if use_fractal:
        suffix += f"_fractal{fractal_octaves}oct"
    if perlin_seed is not None:
        suffix += f"_seed{perlin_seed}"
    out_path = Path(VOLUME_PATH) / f"test_sap_v3_perlin{suffix}.png"
    out_img.save(out_path, quality=95)
    print(f"\nSaved: {out_path}")

    # Save original for comparison (1回だけ)
    orig_path = Path(VOLUME_PATH) / "test_sap_v3_perlin_original.png"
    if not orig_path.exists():
        x_orig_resized = F.interpolate(x_orig, size=(orig_size[1], orig_size[0]), mode='bilinear', align_corners=False)
        x_orig_np = (x_orig_resized.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        orig_out = Image.fromarray(x_orig_np)
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
        'perlin_enabled': use_perlin,
        'perlin_scale': perlin_scale,
        'perlin_seed': effective_perlin_seed,
        'perlin_seed_type': 'user-specified' if perlin_seed is not None else 'image-hash',
        'use_fractal': use_fractal,
        'fractal_octaves': fractal_octaves if use_fractal else None,
        'clip_neg_weight': clip_neg_weight,
    }


@app.local_entrypoint()
def main(
    test: bool = False,
    batch: bool = False,
    submit_batch: bool = False,  # 非同期バッチ（タイムアウトしない）
    status: bool = False,        # ステータス確認
    search_seeds: bool = False,
    iterations: int = 50,
    target_lpips: float = 0.08,
    use_warping: bool = True,
    warp_magnitude: float = 0.01,
    use_perlin: bool = True,
    perlin_scale: int = 64,
    perlin_seed: int = None,
    use_fractal: bool = False,
    fractal_octaves: int = 3,
):
    """
    SAP v3 Variants - Perlin Noise Experiment

    v3の適応型マスクにPerlinノイズを導入:
    - エッジ部: 5%（固定、v3と同じ）
    - 平坦部: 0.75〜1.5%（Perlinでバラける、平均1.125%）
    - フラクタルモード: 複数スケール(64,32,16)の重ね合わせ
    """

    # === ステータス確認（最優先） ===
    if status:
        batch_status = read_batch_status_local()
        print(f"\n=== SAP v3 Variants Batch Status ===")
        print(json.dumps(batch_status, indent=2, default=str))
        return

    # === 非同期バッチ投入（spawn使用、即座に戻る） ===
    if submit_batch:
        print("=== SAP v3 + Perlin: Async Batch Processing ===")
        print("Submitting 11 images to train_sap_v3_variants (async)")
        print("Use --status to check progress\n")

        image_names = [f"image_{i:03d}.png" for i in range(11)]

        # 全画像を spawn() で投入（即座に戻る）
        spawned = []
        for img_name in image_names:
            print(f"Spawning {img_name}...")
            call = process_single_image_with_status.spawn(
                image_name=img_name,
                total_images=len(image_names),
                iterations=iterations,
                target_lpips=target_lpips,
                use_warping=use_warping,
                warp_magnitude=warp_magnitude,
                perlin_scale=perlin_scale,
            )
            spawned.append((img_name, call))

        print(f"\n✓ {len(spawned)} jobs submitted!")
        print(f"  Check progress: modal run scripts/sap_v3_variants.py --status")
        print(f"  Or view at: https://modal.com/apps")
        return

    if search_seeds:
        # 複数シードで探索
        print("=== SAP v3 + Perlin: Seed Search ===")
        seeds = [42, 123, 256, 512, 1024, 2048, 4096, 8192]
        results = []
        for s in seeds:
            print(f"\n--- Testing seed={s} ---")
            result = test_sap_v3_perlin.remote(
                iterations=iterations,
                target_lpips=target_lpips,
                use_warping=use_warping,
                warp_magnitude=warp_magnitude,
                use_perlin=use_perlin,
                perlin_scale=perlin_scale,
                perlin_seed=s,
            )
            results.append((s, result))
            print(f"seed={s}: VAE={result['vae_cos_sim']:.4f}, CLIP_orig={result['clip_to_orig']:.4f}")

        # 最良のシードを表示（CLIP to Origが最も低いもの）
        best = min(results, key=lambda x: x[1]['clip_to_orig'])
        print(f"\n=== Best Seed: {best[0]} ===")
        print(f"VAE Cos Sim: {best[1]['vae_cos_sim']:.4f}")
        print(f"CLIP to Orig: {best[1]['clip_to_orig']:.4f}")

    elif test:
        print("=== SAP v3 + Perlin: Single Image Test ===")
        result = test_sap_v3_perlin.remote(
            iterations=iterations,
            target_lpips=target_lpips,
            use_warping=use_warping,
            warp_magnitude=warp_magnitude,
            use_perlin=use_perlin,
            perlin_scale=perlin_scale,
            perlin_seed=perlin_seed,
            use_fractal=use_fractal,
            fractal_octaves=fractal_octaves,
        )
        print(f"\nResult: {result}")

    elif batch:
        print("=== SAP v3 + Perlin: Batch Processing ===")
        print("Processing all images in train_normal -> train_sap_v3_variants")
        print("Each image processed in separate GPU instance to avoid OOM")

        # 画像リスト
        image_names = [f"image_{i:03d}.png" for i in range(11)]

        results = []
        for img_name in image_names:
            print(f"\nProcessing {img_name}...")
            result = process_single_image.remote(
                image_name=img_name,
                iterations=iterations,
                target_lpips=target_lpips,
                use_warping=use_warping,
                warp_magnitude=warp_magnitude,
                perlin_scale=perlin_scale,
            )
            results.append(result)
            print(f"  {img_name}: {result}")

        processed = sum(1 for r in results if r.get('status') == 'processed')
        print(f"\nTotal: {processed}/{len(image_names)} processed")

    else:
        print("Use --test for single image test, --batch for batch processing")


@app.function(
    image=sap_image,
    volumes={VOLUME_PATH: volume},
    gpu="A100",  # 40GB VRAM
    timeout=300,  # 5分/画像
)
def process_single_image(
    image_name: str,
    iterations: int = 50,
    target_lpips: float = 0.08,
    use_warping: bool = True,
    warp_magnitude: float = 0.01,
    perlin_scale: int = 64,
):
    """1枚の画像にSAP v3 Perlinを適用"""
    import torch
    import torch.nn.functional as F
    import lpips
    import open_clip
    from PIL import Image
    from pathlib import Path
    from torchvision import transforms
    import numpy as np

    device = "cuda"
    torch.cuda.empty_cache()

    # Setup paths
    input_path = Path(VOLUME_PATH) / "train_normal" / image_name
    output_dir = Path(VOLUME_PATH) / "train_sap_v3_variants"
    output_dir.mkdir(exist_ok=True)
    out_path = output_dir / image_name

    if out_path.exists():
        return {'status': 'skipped', 'image': image_name}

    # Load models - A100 has enough VRAM for fp32
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae", torch_dtype=torch.float32).to(device)
    vae.eval()

    clip_model, _, _ = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
    clip_model = clip_model.to(device).eval()
    tokenizer = open_clip.get_tokenizer('ViT-L-14')

    lpips_fn = lpips.LPIPS(net='alex').to(device)

    def get_clip_embedding(img_tensor):
        img_resized = F.interpolate(img_tensor, size=(224, 224), mode='bilinear', align_corners=False)
        img_normalized = (img_resized - torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)) / \
                         torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
        with torch.no_grad():
            return clip_model.encode_image(img_normalized)

    # Precompute embeddings
    with torch.no_grad():
        neg_texts = tokenizer(NEGATIVE_CONCEPTS).to(device)
        neg_embeddings = clip_model.encode_text(neg_texts)
        negative_embedding = F.normalize(neg_embeddings.mean(dim=0, keepdim=True), dim=-1)

        conf_texts = tokenizer(CONFUSION_CONCEPTS).to(device)
        confusion_embeddings = F.normalize(clip_model.encode_text(conf_texts), dim=-1)

    torch.cuda.empty_cache()

    # Process image
    transform = transforms.Compose([transforms.Resize((1024, 1024)), transforms.ToTensor()])
    img = Image.open(input_path).convert("RGB")
    orig_size = img.size
    x_orig = transform(img).unsqueeze(0).to(device)

    seed = get_image_seed(x_orig)
    torch.manual_seed(seed)
    conf_idx = seed % len(CONFUSION_CONCEPTS)
    target_confusion = confusion_embeddings[conf_idx:conf_idx+1]

    if use_warping:
        x_base = apply_micro_warping(x_orig, warp_magnitude, seed)
    else:
        x_base = x_orig

    x_base_norm = x_base * 2.0 - 1.0
    with torch.no_grad():
        z_orig = vae.encode(x_base_norm).latent_dist.mean
        clip_orig = F.normalize(get_clip_embedding(x_base), dim=-1)

    adaptive_mask = compute_adaptive_mask_perlin(x_base, use_perlin=True, perlin_scale=perlin_scale, perlin_seed=seed)
    delta = torch.zeros_like(x_base, requires_grad=True, device=device)
    optimizer = torch.optim.Adam([delta], lr=0.01)

    torch.cuda.empty_cache()

    for i in range(iterations):
        optimizer.zero_grad()
        delta_masked = torch.clamp(delta, -adaptive_mask, adaptive_mask)
        x_adv = torch.clamp(x_base + delta_masked, 0, 1)
        x_adv_norm = x_adv * 2.0 - 1.0

        loss_lpips = lpips_fn(x_base_norm, x_adv_norm).mean()
        z_adv = vae.encode(x_adv_norm).latent_dist.mean
        vae_cos_sim = F.cosine_similarity(z_orig.view(1, -1), z_adv.view(1, -1))

        clip_adv = F.normalize(get_clip_embedding(x_adv), dim=-1)
        loss_clip_neg = -F.cosine_similarity(clip_adv, negative_embedding).mean()
        loss_clip_orig = F.cosine_similarity(clip_adv, clip_orig).mean()
        loss_clip_conf = -F.cosine_similarity(clip_adv, target_confusion).mean()

        if loss_lpips > target_lpips:
            total_loss = loss_lpips * 50.0
        else:
            total_loss = vae_cos_sim * 10.0 + loss_clip_neg * 3.0 + loss_clip_orig * 2.0 + loss_clip_conf * 2.0

        total_loss.backward()
        optimizer.step()

        # Periodic memory cleanup
        if i % 10 == 0:
            torch.cuda.empty_cache()

    with torch.no_grad():
        delta_masked = torch.clamp(delta, -adaptive_mask, adaptive_mask)
        x_final = torch.clamp(x_base + delta_masked, 0, 1)

    x_final_resized = F.interpolate(x_final, size=(orig_size[1], orig_size[0]), mode='bilinear', align_corners=False)
    x_final_np = (x_final_resized.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(x_final_np).save(out_path, quality=95)

    # Cleanup
    del vae, clip_model, lpips_fn, delta, optimizer
    torch.cuda.empty_cache()

    volume.commit()
    return {'status': 'processed', 'image': image_name, 'vae_sim': vae_cos_sim.item()}


@app.function(
    image=sap_image,
    volumes={VOLUME_PATH: volume},
    gpu="A100",  # 40GB VRAM
    timeout=600,  # 10分/画像
)
def process_single_image_with_status(
    image_name: str,
    total_images: int = 11,
    iterations: int = 50,
    target_lpips: float = 0.08,
    use_warping: bool = True,
    warp_magnitude: float = 0.01,
    perlin_scale: int = 64,
):
    """1枚の画像にSAP v3 Perlinを適用（ステータス更新付き）"""
    import torch
    import torch.nn.functional as F
    import lpips
    import open_clip
    from PIL import Image
    from pathlib import Path
    from torchvision import transforms
    import numpy as np
    import json
    from datetime import datetime

    def update_batch_status(image_name: str, status: str, result: dict = None):
        """バッチステータスを更新"""
        status_path = Path(VOLUME_PATH) / BATCH_STATUS_FILE
        try:
            if status_path.exists():
                current = json.loads(status_path.read_text())
            else:
                current = {"images": {}, "started_at": datetime.now().isoformat()}
        except:
            current = {"images": {}, "started_at": datetime.now().isoformat()}

        current["images"][image_name] = {
            "status": status,
            "updated_at": datetime.now().isoformat(),
            "result": result,
        }

        # 進捗計算
        completed = sum(1 for v in current["images"].values() if v["status"] in ["processed", "skipped"])
        current["progress"] = f"{completed}/{total_images}"
        current["completed"] = completed
        current["total"] = total_images

        if completed >= total_images:
            current["status"] = "completed"
            current["completed_at"] = datetime.now().isoformat()
        else:
            current["status"] = "running"

        status_path.write_text(json.dumps(current, indent=2, default=str))
        volume.commit()

    device = "cuda"
    torch.cuda.empty_cache()

    # Setup paths
    input_path = Path(VOLUME_PATH) / "train_normal" / image_name
    output_dir = Path(VOLUME_PATH) / "train_sap_v3_variants"
    output_dir.mkdir(exist_ok=True)
    out_path = output_dir / image_name

    # Also copy txt file
    txt_name = image_name.replace('.png', '.txt').replace('.jpg', '.txt')
    txt_input = Path(VOLUME_PATH) / "train_normal" / txt_name
    txt_output = output_dir / txt_name

    if out_path.exists():
        update_batch_status(image_name, "skipped", {"reason": "already exists"})
        return {'status': 'skipped', 'image': image_name}

    update_batch_status(image_name, "processing")

    try:
        # Load models
        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae", torch_dtype=torch.float32).to(device)
        vae.eval()

        clip_model, _, _ = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
        clip_model = clip_model.to(device).eval()
        tokenizer = open_clip.get_tokenizer('ViT-L-14')

        lpips_fn = lpips.LPIPS(net='alex').to(device)

        def get_clip_embedding(img_tensor):
            img_resized = F.interpolate(img_tensor, size=(224, 224), mode='bilinear', align_corners=False)
            img_normalized = (img_resized - torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)) / \
                             torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
            with torch.no_grad():
                return clip_model.encode_image(img_normalized)

        # Precompute embeddings
        with torch.no_grad():
            neg_texts = tokenizer(NEGATIVE_CONCEPTS).to(device)
            neg_embeddings = clip_model.encode_text(neg_texts)
            negative_embedding = F.normalize(neg_embeddings.mean(dim=0, keepdim=True), dim=-1)

            conf_texts = tokenizer(CONFUSION_CONCEPTS).to(device)
            confusion_embeddings = F.normalize(clip_model.encode_text(conf_texts), dim=-1)

        torch.cuda.empty_cache()

        # Process image
        transform = transforms.Compose([transforms.Resize((1024, 1024)), transforms.ToTensor()])
        img = Image.open(input_path).convert("RGB")
        orig_size = img.size
        x_orig = transform(img).unsqueeze(0).to(device)

        seed = get_image_seed(x_orig)
        torch.manual_seed(seed)
        conf_idx = seed % len(CONFUSION_CONCEPTS)
        target_confusion = confusion_embeddings[conf_idx:conf_idx+1]

        if use_warping:
            x_base = apply_micro_warping(x_orig, warp_magnitude, seed)
        else:
            x_base = x_orig

        x_base_norm = x_base * 2.0 - 1.0
        with torch.no_grad():
            z_orig = vae.encode(x_base_norm).latent_dist.mean
            clip_orig = F.normalize(get_clip_embedding(x_base), dim=-1)

        adaptive_mask = compute_adaptive_mask_perlin(x_base, use_perlin=True, perlin_scale=perlin_scale, perlin_seed=seed)
        delta = torch.zeros_like(x_base, requires_grad=True, device=device)
        optimizer = torch.optim.Adam([delta], lr=0.01)

        torch.cuda.empty_cache()

        for i in range(iterations):
            optimizer.zero_grad()
            delta_masked = torch.clamp(delta, -adaptive_mask, adaptive_mask)
            x_adv = torch.clamp(x_base + delta_masked, 0, 1)
            x_adv_norm = x_adv * 2.0 - 1.0

            loss_lpips = lpips_fn(x_base_norm, x_adv_norm).mean()
            z_adv = vae.encode(x_adv_norm).latent_dist.mean
            vae_cos_sim = F.cosine_similarity(z_orig.view(1, -1), z_adv.view(1, -1))

            clip_adv = F.normalize(get_clip_embedding(x_adv), dim=-1)
            loss_clip_neg = -F.cosine_similarity(clip_adv, negative_embedding).mean()
            loss_clip_orig = F.cosine_similarity(clip_adv, clip_orig).mean()
            loss_clip_conf = -F.cosine_similarity(clip_adv, target_confusion).mean()

            if loss_lpips > target_lpips:
                total_loss = loss_lpips * 50.0
            else:
                total_loss = vae_cos_sim * 10.0 + loss_clip_neg * 3.0 + loss_clip_orig * 2.0 + loss_clip_conf * 2.0

            total_loss.backward()
            optimizer.step()

            if i % 10 == 0:
                torch.cuda.empty_cache()

        with torch.no_grad():
            delta_masked = torch.clamp(delta, -adaptive_mask, adaptive_mask)
            x_final = torch.clamp(x_base + delta_masked, 0, 1)

        x_final_resized = F.interpolate(x_final, size=(orig_size[1], orig_size[0]), mode='bilinear', align_corners=False)
        x_final_np = (x_final_resized.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(x_final_np).save(out_path, quality=95)

        # Copy txt file if exists
        if txt_input.exists():
            import shutil
            shutil.copy(txt_input, txt_output)

        # Cleanup
        del vae, clip_model, lpips_fn, delta, optimizer
        torch.cuda.empty_cache()

        result = {'vae_sim': vae_cos_sim.item()}
        update_batch_status(image_name, "processed", result)
        volume.commit()
        return {'status': 'processed', 'image': image_name, 'vae_sim': vae_cos_sim.item()}

    except Exception as e:
        update_batch_status(image_name, "failed", {"error": str(e)})
        volume.commit()
        return {'status': 'failed', 'image': image_name, 'error': str(e)}


@app.function(
    image=sap_image,
    volumes={VOLUME_PATH: volume},
    gpu="A10G",
    timeout=7200,
)
def process_batch(
    iterations: int = 50,
    target_lpips: float = 0.08,
    use_warping: bool = True,
    warp_magnitude: float = 0.01,
    perlin_scale: int = 64,
):
    """train_normalの全画像にSAP v3 Perlinを適用してtrain_sap_v3_variantsに保存"""
    import torch
    import torch.nn.functional as F
    import lpips
    import open_clip
    from PIL import Image
    from pathlib import Path
    from torchvision import transforms
    from tqdm import tqdm
    import numpy as np
    import shutil

    device = "cuda"

    # === Load Models ===
    print("Loading SDXL VAE...")
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sdxl-vae", torch_dtype=torch.float32
    ).to(device)
    vae.eval()

    print("Loading CLIP ViT-L/14...")
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        'ViT-L-14', pretrained='openai'
    )
    clip_model = clip_model.to(device).eval()
    tokenizer = open_clip.get_tokenizer('ViT-L-14')

    print("Loading LPIPS...")
    lpips_fn = lpips.LPIPS(net='alex').to(device)

    # CLIP embedding helper
    def get_clip_embedding(img_tensor):
        img_resized = F.interpolate(img_tensor, size=(224, 224), mode='bilinear', align_corners=False)
        img_normalized = (img_resized - torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)) / \
                         torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
        with torch.no_grad():
            return clip_model.encode_image(img_normalized)

    # Precompute CLIP embeddings
    print("Precomputing CLIP embeddings...")
    with torch.no_grad():
        neg_texts = tokenizer(NEGATIVE_CONCEPTS).to(device)
        neg_embeddings = clip_model.encode_text(neg_texts)
        negative_embedding = F.normalize(neg_embeddings.mean(dim=0, keepdim=True), dim=-1)

        conf_texts = tokenizer(CONFUSION_CONCEPTS).to(device)
        confusion_embeddings = F.normalize(clip_model.encode_text(conf_texts), dim=-1)

    # Setup directories
    input_dir = Path(VOLUME_PATH) / "train_normal"
    output_dir = Path(VOLUME_PATH) / "train_sap_v3_variants"
    output_dir.mkdir(exist_ok=True)

    # Get image list
    image_files = sorted(list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg")))
    print(f"\nFound {len(image_files)} images in {input_dir}")

    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
    ])

    processed = 0
    skipped = 0

    for img_path in tqdm(image_files, desc="Processing"):
        out_path = output_dir / img_path.name

        # Skip if already processed
        if out_path.exists():
            skipped += 1
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            orig_size = img.size
            x_orig = transform(img).unsqueeze(0).to(device)

            seed = get_image_seed(x_orig)
            torch.manual_seed(seed)

            # Select confusion target
            conf_idx = seed % len(CONFUSION_CONCEPTS)
            target_confusion = confusion_embeddings[conf_idx:conf_idx+1]

            # Perlin seed from image hash
            effective_perlin_seed = seed

            # Phase 1: Micro-Warping
            if use_warping:
                x_base = apply_micro_warping(x_orig, warp_magnitude, seed)
            else:
                x_base = x_orig

            # Phase 2: VAE + CLIP Attack
            x_base_norm = x_base * 2.0 - 1.0

            with torch.no_grad():
                z_orig = vae.encode(x_base_norm).latent_dist.mean
                clip_orig = get_clip_embedding(x_base)
                clip_orig = F.normalize(clip_orig, dim=-1)

            # Adaptive mask with Perlin
            adaptive_mask = compute_adaptive_mask_perlin(
                x_base, use_perlin=True, perlin_scale=perlin_scale, perlin_seed=effective_perlin_seed
            )

            delta = torch.zeros_like(x_base, requires_grad=True, device=device)
            optimizer = torch.optim.Adam([delta], lr=0.01)

            for i in range(iterations):
                optimizer.zero_grad()

                delta_masked = torch.clamp(delta, -adaptive_mask, adaptive_mask)
                x_adv = torch.clamp(x_base + delta_masked, 0, 1)
                x_adv_norm = x_adv * 2.0 - 1.0

                loss_lpips = lpips_fn(x_base_norm, x_adv_norm).mean()
                z_adv = vae.encode(x_adv_norm).latent_dist.mean
                vae_cos_sim = F.cosine_similarity(z_orig.view(1, -1), z_adv.view(1, -1))
                loss_vae = vae_cos_sim

                clip_adv = get_clip_embedding(x_adv)
                clip_adv = F.normalize(clip_adv, dim=-1)
                loss_clip_neg = -F.cosine_similarity(clip_adv, negative_embedding).mean()
                loss_clip_orig = F.cosine_similarity(clip_adv, clip_orig).mean()
                loss_clip_conf = -F.cosine_similarity(clip_adv, target_confusion).mean()

                if loss_lpips > target_lpips:
                    total_loss = loss_lpips * 50.0
                else:
                    total_loss = loss_vae * 10.0 + loss_clip_neg * 3.0 + loss_clip_orig * 2.0 + loss_clip_conf * 2.0

                total_loss.backward()
                optimizer.step()

            # Save result
            with torch.no_grad():
                delta_masked = torch.clamp(delta, -adaptive_mask, adaptive_mask)
                x_final = torch.clamp(x_base + delta_masked, 0, 1)

            x_final_resized = F.interpolate(x_final, size=(orig_size[1], orig_size[0]), mode='bilinear', align_corners=False)
            x_final_np = (x_final_resized.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            out_img = Image.fromarray(x_final_np)
            out_img.save(out_path, quality=95)

            processed += 1

            # メモリ解放
            del delta, optimizer, x_orig, x_base, x_adv, x_final, adaptive_mask
            del z_orig, z_adv, clip_orig, clip_adv
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
            torch.cuda.empty_cache()
            continue

    volume.commit()
    return {
        'processed': processed,
        'skipped': skipped,
        'total': len(image_files),
        'output_dir': str(output_dir),
    }


if __name__ == "__main__":
    print("Use: modal run scripts/sap_v3_variants.py --test")
