#!/usr/bin/env python3
"""
SAP v4 - Semantic Adaptive Protection v4

改良点 (v3からの進化):
1. WD14 Tagger攻撃: 毒タグ誘導 (lowres, monochrome, 1boy等)
2. UAP (Universal Adversarial Perturbation): 事前計算で高速化
3. アンサンブル攻撃: ConvNext + SwinV2 両方に効く
4. Geminiの提案: UAPを初期値として使用、軽い重みで維持

Usage:
    # Phase 0: UAP事前生成（1回だけ、数分）
    modal run scripts/sap_v4.py --generate-uap

    # Phase 1: 1枚テスト
    modal run scripts/sap_v4.py --test

    # バッチ攻撃
    modal run scripts/sap_v4.py --attack
"""

import modal
from pathlib import Path

app = modal.App("sap-v4")

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
        "timm",
        "pandas",
    )
)

# === CLIP攻撃用の概念 ===
NEGATIVE_CONCEPTS = [
    "low quality, worst quality, blurry",
    "jpeg artifacts, noise, grainy",
    "text, watermark, signature",
    "error, glitch, corrupted",
    "out of focus, motion blur",
]

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

# === WD14 Tagger用の毒タグ ===
POISON_TAGS = {
    # 品質・除外系
    "lowres": 1.0,
    "bad_anatomy": 1.0,
    "error": 0.9,
    # メタデータ汚染
    "monochrome": 0.9,
    "greyscale": 0.9,
    "sketch": 0.8,
    # コンテンツフラグ
    "text": 0.8,
    "watermark": 0.7,
    # 構造混乱
    "1boy": 0.9,
    "multiple_views": 0.7,
    "comic": 0.6,
}


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
    """適応型マスク: エッジ部5%、平坦部1%（VAE用）"""
    import torch
    import kornia

    grads = kornia.filters.sobel(image_tensor)
    magnitude = torch.sqrt(grads[:, 0:1] ** 2 + grads[:, 1:2] ** 2 + 1e-8)
    magnitude = magnitude / (magnitude.max() + 1e-8)
    mask = 0.01 + (0.04 * torch.sigmoid((magnitude - 0.3) * 20))
    mask = mask.expand(-1, 3, -1, -1)

    return mask


def generate_perlin_noise(shape, scale=8, device='cuda'):
    """
    Perlin-like noise生成（簡易版）
    平坦部のノイズをバラけさせるために使用
    """
    import torch
    import torch.nn.functional as F

    B, C, H, W = shape
    # 低解像度ノイズを生成してアップサンプル
    small_h, small_w = H // scale, W // scale
    noise = torch.randn(B, 1, small_h, small_w, device=device)
    # バイリニア補間で滑らかに拡大
    noise = F.interpolate(noise, size=(H, W), mode='bilinear', align_corners=False)
    # 0-1に正規化
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
    return noise.expand(-1, C, -1, -1)


def compute_dual_masks(image_tensor):
    """
    総予算制のデュアルマスク:
    - 総上限: エッジ5%、平坦1.5%（固定）
    - WD14: 平坦部1%（Perlin noiseでバラけ）
    - VAE: エッジ5%、平坦0.5%（残り予算）

    両方足しても総上限を超えないよう設計
    """
    import torch
    import kornia

    B, C, H, W = image_tensor.shape
    device = image_tensor.device

    grads = kornia.filters.sobel(image_tensor)
    magnitude = torch.sqrt(grads[:, 0:1] ** 2 + grads[:, 1:2] ** 2 + 1e-8)
    magnitude = magnitude / (magnitude.max() + 1e-8)

    # エッジ検出（0=平坦、1=エッジ）
    edge_weight = torch.sigmoid((magnitude - 0.3) * 20)

    # 総マスク: エッジ5%、平坦1.5%
    total_mask = 0.015 + (0.035 * edge_weight)
    total_mask = total_mask.expand(-1, 3, -1, -1)

    # VAE用: エッジ重視（エッジ5%、平坦0.5%）
    mask_vae = 0.005 + (0.045 * edge_weight)
    mask_vae = mask_vae.expand(-1, 3, -1, -1)

    # WD14用: 平坦部のみ（1%）+ Perlin noiseでバラけさせる
    flat_weight = 1.0 - edge_weight
    perlin = generate_perlin_noise((B, C, H, W), scale=16, device=device)
    # 平坦部に1%まで、Perlinでバラけさせる
    mask_wd14 = 0.01 * flat_weight * perlin
    mask_wd14 = mask_wd14.expand(-1, 3, -1, -1)

    return mask_vae, mask_wd14, total_mask


def load_wd14_tagger(model_name, device):
    """WD14 Taggerをロード"""
    import timm
    import pandas as pd
    from huggingface_hub import hf_hub_download

    repo_id = f"SmilingWolf/{model_name}"

    # モデルをロード
    model = timm.create_model(
        f"hf-hub:{repo_id}",
        pretrained=True,
    ).to(device)
    model.eval()

    # タグリストをロード
    csv_path = hf_hub_download(repo_id, "selected_tags.csv")
    tags_df = pd.read_csv(csv_path)
    tag_names = tags_df["name"].tolist()

    return model, tag_names


def get_poison_tag_indices(tag_names, poison_tags):
    """毒タグのインデックスと重みを取得"""
    indices = []
    weights = []
    for tag, weight in poison_tags.items():
        if tag in tag_names:
            indices.append(tag_names.index(tag))
            weights.append(weight)
    return indices, weights


@app.function(
    image=sap_image,
    volumes={VOLUME_PATH: volume},
    gpu="A10G",
    timeout=1800,
)
def generate_uap(
    iterations: int = 300,
    lr: float = 0.01,
    target_size: int = 512,  # 512で生成（メモリ節約）
):
    """
    Phase 0: WD14 Tagger用のUAPを事前生成

    多様な画像で訓練し、汎用的なUAPを生成
    ConvNext + SwinV2の両方に効くアンサンブルUAPを生成
    """
    import torch
    import torch.nn.functional as F
    from torchvision import transforms
    from tqdm import tqdm
    import numpy as np
    from PIL import Image
    import random

    device = torch.device("cuda")

    print("=== SAP v4: UAP Generation (Multi-Image Training) ===")

    # === 訓練画像をロード ===
    train_path = Path(VOLUME_PATH) / "train_normal"
    image_files = sorted(list(train_path.glob("*.png")) + list(train_path.glob("*.jpg")))[:20]  # 最大20枚
    print(f"Loading {len(image_files)} training images...")

    transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
    ])

    train_images = []
    for img_file in image_files:
        img = Image.open(img_file).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)
        train_images.append(img_tensor)

    # グレー画像も追加（多様性のため）
    gray_image = torch.ones(1, 3, target_size, target_size, device=device) * 0.5
    train_images.append(gray_image)
    print(f"Total training images: {len(train_images)} (including gray)")

    # === WD14 Taggerをロード ===
    print("Loading WD14 Taggers...")

    tagger_convnext, tags_convnext = load_wd14_tagger("wd-convnext-tagger-v3", device)
    tagger_swinv2, tags_swinv2 = load_wd14_tagger("wd-swinv2-tagger-v3", device)

    # 毒タグインデックスを取得
    poison_indices_cn, poison_weights_cn = get_poison_tag_indices(tags_convnext, POISON_TAGS)
    poison_indices_sw, poison_weights_sw = get_poison_tag_indices(tags_swinv2, POISON_TAGS)

    print(f"ConvNext poison tags: {len(poison_indices_cn)}")
    print(f"SwinV2 poison tags: {len(poison_indices_sw)}")

    poison_weights_cn = torch.tensor(poison_weights_cn, device=device)
    poison_weights_sw = torch.tensor(poison_weights_sw, device=device)

    # === UAP初期化 ===
    uap = torch.zeros(1, 3, target_size, target_size, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([uap], lr=lr)

    # WD14の入力サイズ
    wd14_size = 448

    # WD14の正規化パラメータ
    wd14_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    wd14_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    # === 最適化 ===
    pbar = tqdm(range(iterations), desc="Generating UAP")
    for i in pbar:
        optimizer.zero_grad()

        # ランダムに画像を選択（各反復で異なる画像）
        base_image = random.choice(train_images)

        # UAPをクランプ（視覚的制約: ±5%）
        uap_clamped = torch.clamp(uap, -0.05, 0.05)

        # 画像にUAPを適用
        x_adv = torch.clamp(base_image + uap_clamped, 0, 1)

        # WD14用にリサイズ・正規化
        x_wd14 = F.interpolate(x_adv, size=(wd14_size, wd14_size), mode="bilinear", align_corners=False)
        x_wd14_norm = (x_wd14 - wd14_mean) / wd14_std

        # ConvNext予測
        logits_cn = tagger_convnext(x_wd14_norm)
        probs_cn = torch.sigmoid(logits_cn)

        # SwinV2予測
        logits_sw = tagger_swinv2(x_wd14_norm)
        probs_sw = torch.sigmoid(logits_sw)

        # 毒タグの確率を最大化（重み付き）
        poison_probs_cn = probs_cn[0, poison_indices_cn]
        poison_probs_sw = probs_sw[0, poison_indices_sw]

        loss_cn = -(poison_probs_cn * poison_weights_cn).mean()
        loss_sw = -(poison_probs_sw * poison_weights_sw).mean()

        # アンサンブルロス
        loss = loss_cn + loss_sw

        loss.backward()
        optimizer.step()

        if i % 30 == 0:
            avg_poison_cn = poison_probs_cn.mean().item()
            avg_poison_sw = poison_probs_sw.mean().item()
            pbar.set_postfix({
                "cn_poison": f"{avg_poison_cn:.3f}",
                "sw_poison": f"{avg_poison_sw:.3f}",
            })

    # === 全画像での最終評価 ===
    print("\n=== UAP Evaluation on All Training Images ===")
    with torch.no_grad():
        uap_final = torch.clamp(uap, -0.05, 0.05)

        all_probs_cn = []
        all_probs_sw = []

        for base_image in train_images:
            x_adv = torch.clamp(base_image + uap_final, 0, 1)
            x_wd14 = F.interpolate(x_adv, size=(wd14_size, wd14_size), mode="bilinear", align_corners=False)
            x_wd14_norm = (x_wd14 - wd14_mean) / wd14_std

            probs_cn = torch.sigmoid(tagger_convnext(x_wd14_norm))
            probs_sw = torch.sigmoid(tagger_swinv2(x_wd14_norm))

            all_probs_cn.append(probs_cn[0, poison_indices_cn])
            all_probs_sw.append(probs_sw[0, poison_indices_sw])

        avg_probs_cn = torch.stack(all_probs_cn).mean(dim=0)
        avg_probs_sw = torch.stack(all_probs_sw).mean(dim=0)

        print("Average ConvNext poison tag probabilities:")
        for tag, prob in zip(POISON_TAGS.keys(), avg_probs_cn):
            print(f"  {tag}: {prob.item():.4f}")

        print("\nAverage SwinV2 poison tag probabilities:")
        for tag, prob in zip(POISON_TAGS.keys(), avg_probs_sw):
            print(f"  {tag}: {prob.item():.4f}")

    # === 保存 ===
    uap_np = uap_final.squeeze(0).cpu().numpy()
    uap_path = Path(VOLUME_PATH) / "uap_wd14_v1.npy"
    np.save(uap_path, uap_np)
    print(f"\nSaved UAP: {uap_path}")

    # 可視化用に保存
    uap_vis = ((uap_final.squeeze(0).permute(1, 2, 0).cpu().numpy() + 0.05) / 0.1 * 255).clip(0, 255).astype(np.uint8)
    uap_vis_path = Path(VOLUME_PATH) / "uap_wd14_v1_vis.png"
    Image.fromarray(uap_vis).save(uap_vis_path)

    volume.commit()

    return {
        "uap_path": str(uap_path),
        "avg_poison_convnext": avg_probs_cn.mean().item(),
        "avg_poison_swinv2": avg_probs_sw.mean().item(),
    }


@app.function(
    image=sap_image,
    volumes={VOLUME_PATH: volume},
    gpu="A10G",
    timeout=1800,
)
def test_sap_v4(
    iterations: int = 50,
    wd14_iterations: int = 30,
    target_lpips: float = 0.08,
    use_warping: bool = True,
    warp_magnitude: float = 0.01,
    vae_weight: float = 10.0,
    clip_neg_weight: float = 3.0,
    clip_conf_weight: float = 2.0,
):
    """
    Phase 1: SAP v4 攻撃テスト（2段階最適化）

    Stage 1: WD14攻撃（poison tags最大化）
    Stage 2: VAE/CLIP攻撃（latent離脱）

    メモリ効率のため、各Stageで必要なモデルのみロード
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
    import gc

    device = torch.device("cuda")

    # 512x512に縮小してメモリ節約（最後に元サイズにリサイズ）
    WORK_SIZE = 512

    print("=== SAP v4: Two-Stage Attack (WD14 → VAE/CLIP) ===")

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
        transforms.Resize((WORK_SIZE, WORK_SIZE)),
        transforms.ToTensor(),
    ])
    x_orig = transform(img).unsqueeze(0).to(device)
    print(f"Working at {WORK_SIZE}x{WORK_SIZE}")

    seed = get_image_seed(x_orig)
    torch.manual_seed(seed)

    # === Phase 0: Micro-Warping ===
    if use_warping:
        print(f"Phase 0: Micro-Warping (magnitude={warp_magnitude})")
        x_base = apply_micro_warping(x_orig, warp_magnitude, seed)
    else:
        x_base = x_orig.clone()

    # Dual masks: VAE→エッジ重視, WD14→平坦部（Perlinでバラけ）, 総予算制
    mask_vae, mask_wd14, total_mask = compute_dual_masks(x_base)
    edge_ratio = (mask_vae > 0.03).float().mean().item()
    flat_ratio = (mask_wd14 > 0.005).float().mean().item()
    print(f"Edge ratio: {edge_ratio*100:.1f}%, WD14 active ratio: {flat_ratio*100:.1f}%")

    # === Initialize delta (no UAP, zero init for cleaner result) ===
    print("Using zero initialization (no UAP for cleaner visuals)")

    # ========================================
    # STAGE 1: WD14 Attack
    # ========================================
    print(f"\n=== Stage 1: WD14 Attack ({wd14_iterations} iterations) ===")

    # Load WD14 Tagger
    print("Loading WD14 Tagger (ConvNext)...")
    tagger_convnext, tags_convnext = load_wd14_tagger("wd-convnext-tagger-v3", device)
    tagger_convnext.eval()  # FP32で勾配計算

    poison_indices_cn, poison_weights_cn = get_poison_tag_indices(tags_convnext, POISON_TAGS)
    poison_weights_cn = torch.tensor(poison_weights_cn, device=device)

    print(f"Poison tags: {len(poison_indices_cn)}")

    # WD14 preprocessing
    wd14_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    wd14_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    print(f"GPU Memory after WD14 load: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    # Initialize delta with zero (cleaner than UAP)
    delta_wd14 = torch.zeros(1, 3, WORK_SIZE, WORK_SIZE, device=device, requires_grad=True)
    optimizer_wd14 = torch.optim.Adam([delta_wd14], lr=0.02)

    pbar = tqdm(range(wd14_iterations), desc="WD14 Attack")
    for i in pbar:
        optimizer_wd14.zero_grad()

        # Apply delta with WD14 mask (平坦部1.5% Perlin、エッジ低)
        delta_masked = torch.clamp(delta_wd14, -mask_wd14, mask_wd14)
        x_adv = torch.clamp(x_base + delta_masked, 0, 1)

        # WD14 forward
        x_wd14 = F.interpolate(x_adv, size=(448, 448), mode='bilinear', align_corners=False)
        x_wd14_norm = (x_wd14 - wd14_mean) / wd14_std

        logits_cn = tagger_convnext(x_wd14_norm)
        probs_cn = torch.sigmoid(logits_cn)

        # Maximize poison tag probabilities
        poison_probs = probs_cn[0, poison_indices_cn]
        loss_wd14 = -(poison_probs * poison_weights_cn).mean()

        loss_wd14.backward()
        optimizer_wd14.step()

        if i % 10 == 0:
            avg_poison = poison_probs.mean().item()
            max_poison = poison_probs.max().item()
            pbar.set_postfix({
                'avg': f'{avg_poison:.3f}',
                'max': f'{max_poison:.3f}',
            })

    # Get WD14 attack result
    with torch.no_grad():
        delta_wd14_final = torch.clamp(delta_wd14, -mask_wd14, mask_wd14)
        x_after_wd14 = torch.clamp(x_base + delta_wd14_final, 0, 1)

        # Evaluate WD14 after Stage 1
        x_wd14 = F.interpolate(x_after_wd14, size=(448, 448), mode='bilinear', align_corners=False)
        x_wd14_norm = (x_wd14 - wd14_mean) / wd14_std
        logits_cn = tagger_convnext(x_wd14_norm)
        probs_cn = torch.sigmoid(logits_cn)

        print("\nWD14 Tags after Stage 1:")
        for tag, idx in zip(POISON_TAGS.keys(), poison_indices_cn):
            print(f"  {tag}: {probs_cn[0, idx].item():.3f}")

    # Free WD14 memory
    del tagger_convnext, optimizer_wd14
    gc.collect()
    torch.cuda.empty_cache()
    print(f"\nGPU Memory after WD14 cleanup: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    # ========================================
    # STAGE 2: VAE/CLIP Attack
    # ========================================
    print(f"\n=== Stage 2: VAE/CLIP Attack ({iterations} iterations) ===")

    # Load VAE, CLIP, LPIPS
    print("Loading SDXL VAE (FP32)...")
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sdxl-vae",
        torch_dtype=torch.float32,
    ).to(device)
    vae.eval()

    print("Loading CLIP ViT-L/14 (FP16)...")
    clip_model, _, _ = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
    clip_model = clip_model.half().to(device)
    clip_model.eval()
    tokenizer = open_clip.get_tokenizer('ViT-L-14')

    print("Loading LPIPS...")
    lpips_fn = lpips.LPIPS(net='alex').to(device)
    lpips_fn.eval()

    print(f"GPU Memory after VAE/CLIP load: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    # Precompute CLIP embeddings
    neg_embeddings = []
    for concept in NEGATIVE_CONCEPTS:
        tokens = tokenizer([concept]).to(device)
        with torch.no_grad():
            emb = clip_model.encode_text(tokens)
            emb = F.normalize(emb, dim=-1)
        neg_embeddings.append(emb)
    negative_embedding = torch.cat(neg_embeddings, dim=0).mean(dim=0, keepdim=True)
    negative_embedding = F.normalize(negative_embedding, dim=-1)

    conf_embeddings = []
    for concept in CONFUSION_CONCEPTS:
        tokens = tokenizer([concept]).to(device)
        with torch.no_grad():
            emb = clip_model.encode_text(tokens)
            emb = F.normalize(emb, dim=-1)
        conf_embeddings.append(emb)
    confusion_embeddings = torch.cat(conf_embeddings, dim=0)

    conf_idx = seed % len(CONFUSION_CONCEPTS)
    target_confusion = confusion_embeddings[conf_idx:conf_idx+1]
    print(f"Confusion target: '{CONFUSION_CONCEPTS[conf_idx]}'")

    # CLIP preprocessing
    clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(device)
    clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(device)

    def get_clip_embedding(img):
        img_resized = F.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False)
        img_norm = (img_resized - clip_mean) / clip_std
        with torch.cuda.amp.autocast():
            return clip_model.encode_image(img_norm.half()).float()

    # Use WD14 attack result as base
    x_stage1 = x_after_wd14.detach()
    x_stage1_norm = x_stage1 * 2.0 - 1.0

    with torch.no_grad():
        z_orig = vae.encode(x_stage1_norm).latent_dist.mean
        clip_orig = get_clip_embedding(x_stage1)
        clip_orig = F.normalize(clip_orig, dim=-1)

    # Fix WD14 delta and only optimize additional VAE/CLIP delta
    # This preserves WD14 effects while adding VAE/CLIP attack
    delta_wd14_fixed = delta_wd14_final.detach()  # Fixed, no gradients

    # Additional delta for VAE/CLIP (starts from zero)
    delta_vae_add = torch.zeros_like(x_base, requires_grad=True, device=device)
    optimizer_vae = torch.optim.Adam([delta_vae_add], lr=0.02)

    # Get original (pre-attack) baselines for comparison
    x_orig_norm = x_base * 2.0 - 1.0
    with torch.no_grad():
        z_base = vae.encode(x_orig_norm).latent_dist.mean
        clip_base = get_clip_embedding(x_base)
        clip_base = F.normalize(clip_base, dim=-1)

    pbar = tqdm(range(iterations), desc="VAE/CLIP Attack")
    for i in pbar:
        optimizer_vae.zero_grad()

        # Combine fixed WD14 delta (平坦部) + trainable VAE delta (エッジ部)
        # 別領域を使うので累積しても干渉は最小限
        delta_vae_masked = torch.clamp(delta_vae_add, -mask_vae, mask_vae)
        combined_delta = delta_wd14_fixed + delta_vae_masked
        x_adv = torch.clamp(x_base + combined_delta, 0, 1)
        x_adv_norm = x_adv * 2.0 - 1.0

        # LPIPS loss (compare to original, not Stage 1 result)
        loss_lpips = lpips_fn(x_orig_norm, x_adv_norm).mean()

        # VAE loss (compare to original)
        z_adv = vae.encode(x_adv_norm).latent_dist.mean
        vae_cos_sim = F.cosine_similarity(z_base.view(1, -1), z_adv.view(1, -1))
        loss_vae = vae_cos_sim

        # CLIP loss (compare to original)
        clip_adv = get_clip_embedding(x_adv)
        clip_adv = F.normalize(clip_adv, dim=-1)
        loss_clip_neg = -F.cosine_similarity(clip_adv, negative_embedding).mean()
        loss_clip_orig = F.cosine_similarity(clip_adv, clip_base).mean()
        loss_clip_conf = -F.cosine_similarity(clip_adv, target_confusion).mean()

        # Combined loss
        if loss_lpips > target_lpips:
            total_loss = loss_lpips * 50.0
        else:
            total_loss = (
                3.0 * loss_lpips
                + vae_weight * loss_vae
                + clip_neg_weight * loss_clip_neg
                + clip_conf_weight * (loss_clip_orig + loss_clip_conf)
            )

        total_loss.backward()
        optimizer_vae.step()

        with torch.no_grad():
            delta_vae_add.data = torch.clamp(delta_vae_add.data, -mask_vae, mask_vae)

        if i % 10 == 0:
            pbar.set_postfix({
                'lpips': f'{loss_lpips.item():.4f}',
                'vae': f'{vae_cos_sim.item():.4f}',
            })

    # === Final result ===
    with torch.no_grad():
        # Combine WD14 delta (平坦部) + VAE delta (エッジ部)
        delta_vae_final = torch.clamp(delta_vae_add, -mask_vae, mask_vae)
        combined_delta_final = delta_wd14_fixed + delta_vae_final
        # 総予算制: 合計deltaを総マスクでクランプ（エッジ5%、平坦2.5%）
        combined_delta_final = torch.clamp(combined_delta_final, -total_mask, total_mask)
        x_protected = torch.clamp(x_base + combined_delta_final, 0, 1)

        x_final = F.interpolate(x_protected, size=(orig_size[1], orig_size[0]), mode='bilinear', align_corners=False)

        x_prot_norm = x_protected * 2.0 - 1.0

        # Compare to original (before any attack)
        x_orig_norm = x_base * 2.0 - 1.0
        final_lpips = lpips_fn(x_orig_norm, x_prot_norm).item()

        z_final = vae.encode(x_prot_norm).latent_dist.mean
        z_base = vae.encode(x_orig_norm).latent_dist.mean
        final_vae_sim = F.cosine_similarity(z_base.view(1, -1), z_final.view(1, -1)).item()

        clip_final = get_clip_embedding(x_protected)
        clip_final = F.normalize(clip_final, dim=-1)
        clip_base = get_clip_embedding(x_base)
        clip_base = F.normalize(clip_base, dim=-1)
        final_clip_orig = F.cosine_similarity(clip_final, clip_base).item()
        final_clip_neg = F.cosine_similarity(clip_final, negative_embedding).item()

    # === WD14 Final Evaluation ===
    print("\nEvaluating final WD14 poison tags...")
    del vae, clip_model, lpips_fn
    gc.collect()
    torch.cuda.empty_cache()

    tagger_convnext, tags_convnext = load_wd14_tagger("wd-convnext-tagger-v3", device)
    tagger_convnext = tagger_convnext.half().eval()
    poison_indices_cn, _ = get_poison_tag_indices(tags_convnext, POISON_TAGS)

    wd14_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    wd14_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    with torch.no_grad():
        x_wd14 = F.interpolate(x_protected, size=(448, 448), mode='bilinear', align_corners=False)
        x_wd14_norm = (x_wd14 - wd14_mean) / wd14_std
        with torch.cuda.amp.autocast():
            logits_cn = tagger_convnext(x_wd14_norm.half())
        probs_cn = torch.sigmoid(logits_cn.float())

    poison_results = {}
    for tag, idx_cn in zip(POISON_TAGS.keys(), poison_indices_cn):
        poison_results[tag] = probs_cn[0, idx_cn].item()

    print(f"\n{'='*60}")
    print(f"=== SAP v4 Results (Two-Stage) ===")
    print(f"{'='*60}")
    print(f"LPIPS: {final_lpips:.4f} (target: {target_lpips})")
    print(f"VAE Cosine Sim: {final_vae_sim:.4f} (lower = better)")
    print(f"CLIP to Original: {final_clip_orig:.4f} (lower = better)")
    print(f"CLIP to Negative: {final_clip_neg:.4f} (higher = better)")
    print(f"\nWD14 Poison Tags (ConvNext):")
    for tag, prob in poison_results.items():
        print(f"  {tag}: {prob:.3f}")

    # Save
    x_final_np = (x_final.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    out_img = Image.fromarray(x_final_np)
    out_path = Path(VOLUME_PATH) / "test_sap_v4.png"
    out_img.save(out_path, quality=95)
    print(f"\nSaved: {out_path}")

    volume.commit()

    return {
        'lpips': final_lpips,
        'vae_cos_sim': final_vae_sim,
        'clip_to_orig': final_clip_orig,
        'clip_to_negative': final_clip_neg,
        'poison_tags': poison_results,
        'size': orig_size,
    }


@app.local_entrypoint()
def main(
    generate_uap: bool = False,
    test: bool = False,
    attack: bool = False,
    iterations: int = 50,
    target_lpips: float = 0.08,
    use_warping: bool = True,
    warp_magnitude: float = 0.01,
):
    """
    SAP v4 Main Entrypoint

    Phase 0: UAP事前生成（--generate-uap）
    Phase 1: テスト/攻撃（--test / --attack）
    """

    if generate_uap:
        print("=== SAP v4: Generating UAP ===")
        result = generate_uap_fn.remote(iterations=200)
        print(f"\nResult: {result}")

    elif test:
        print("=== SAP v4: Single Image Test ===")
        result = test_sap_v4.remote(
            iterations=iterations,
            target_lpips=target_lpips,
            use_warping=use_warping,
            warp_magnitude=warp_magnitude,
        )
        print(f"\nResult: {result}")


# エイリアス（local_entrypointからは関数名でremote呼び出しできないため）
generate_uap_fn = generate_uap


if __name__ == "__main__":
    print("Use: modal run scripts/sap_v4.py --generate-uap")
    print("     modal run scripts/sap_v4.py --test")
