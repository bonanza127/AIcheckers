#!/usr/bin/env python3
"""
Poisoned Anchor Generator for FastProtect

Geminiの提案「Poisoned Anchor Strategy」の実装:
- ターゲット画像（レンガ、布地、森）に対してWD14攻撃を仕掛け
- 見た目はテクスチャのまま、WD14タグは汚染された状態を作る
- FastProtectがこれらをターゲットにすることで、VAE+タグ両方を攻撃

Usage:
    modal run scripts/generate_poisoned_anchors.py --generate
    modal run scripts/generate_poisoned_anchors.py --test  # 効果確認
"""

import modal
from pathlib import Path

app = modal.App("poisoned-anchors")

volume = modal.Volume.from_name("fastprotect-vol", create_if_missing=True)
VOLUME_PATH = "/vol"

anchor_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch==2.1.2",
        "torchvision==0.16.2",
        "huggingface_hub==0.21.4",
        "pillow",
        "numpy<2.0",
        "tqdm",
        "timm",
        "pandas",
    )
)

# 毒タグ: LoRA学習を妨害するタグ
POISON_TAGS = {
    # 品質系（ネガティブプロンプトでよく使われる）
    "lowres": 1.0,
    "bad anatomy": 0.9,
    "worst quality": 1.0,
    "low quality": 1.0,
    "normal quality": 0.7,
    # モノクロ系
    "monochrome": 0.9,
    "greyscale": 0.9,
    # メタ情報汚染
    "text": 0.8,
    "watermark": 0.7,
    "signature": 0.6,
    # 構造混乱
    "error": 0.8,
    "jpeg artifacts": 0.7,
}


def load_wd14_tagger(model_name, device):
    """WD14 Taggerをロード"""
    import timm
    import pandas as pd
    from huggingface_hub import hf_hub_download

    repo_id = f"SmilingWolf/{model_name}"

    model = timm.create_model(
        f"hf-hub:{repo_id}",
        pretrained=True,
    ).to(device)
    model.eval()

    csv_path = hf_hub_download(repo_id, "selected_tags.csv")
    tags_df = pd.read_csv(csv_path)
    tag_names = tags_df["name"].tolist()

    return model, tag_names


def get_poison_tag_indices(tag_names, poison_tags):
    """毒タグのインデックスと重みを取得"""
    indices = []
    weights = []
    found_tags = []

    for tag, weight in poison_tags.items():
        # スペースとアンダースコアの両方を試す
        tag_variants = [tag, tag.replace(" ", "_"), tag.replace("_", " ")]
        for variant in tag_variants:
            if variant in tag_names:
                indices.append(tag_names.index(variant))
                weights.append(weight)
                found_tags.append(variant)
                break

    return indices, weights, found_tags


def generate_base_textures(size=512, device='cuda'):
    """
    ベースとなるテクスチャ画像を生成

    Returns:
        dict: {'y_l': レンガ, 'y_m': 布地, 'y_h': 森}
    """
    import torch

    textures = {}

    # y_l: レンガ/タイルパターン（規則的な人工物）
    y_l = torch.zeros(1, 3, size, size, device=device)
    brick_h, brick_w = 32, 64
    mortar = 4
    colors = [
        [0.6, 0.3, 0.2],
        [0.55, 0.28, 0.18],
        [0.65, 0.32, 0.22],
    ]
    for i in range(0, size, brick_h + mortar):
        offset = (i // (brick_h + mortar)) % 2 * (brick_w // 2)
        for j in range(-brick_w, size + brick_w, brick_w + mortar):
            jj = j + offset
            if 0 <= jj < size:
                color_idx = (i + j) % 3
                c = colors[color_idx]
                i_end = min(i + brick_h, size)
                j_end = min(jj + brick_w, size)
                jj_start = max(0, jj)
                for ch in range(3):
                    y_l[0, ch, i:i_end, jj_start:j_end] = c[ch]
    mask = y_l.sum(dim=1, keepdim=True) == 0
    y_l[:, :, :, :][mask.expand(-1, 3, -1, -1)] = 0.7
    textures['y_l'] = y_l

    # y_m: 布地パターン（中間）
    y_m = torch.zeros(1, 3, size, size, device=device)
    weave_size = 8
    for i in range(size):
        for j in range(size):
            warp = (i // weave_size) % 2
            weft = (j // weave_size) % 2
            if (i + j) % (weave_size * 2) < weave_size:
                val = 0.4 + 0.2 * warp
            else:
                val = 0.5 + 0.2 * weft
            y_m[0, :, i, j] = val
    torch.manual_seed(42)
    y_m += torch.randn_like(y_m) * 0.05
    y_m = torch.clamp(y_m, 0, 1)
    textures['y_m'] = y_m

    # y_h: 森/芝生パターン（フラクタルノイズ）
    torch.manual_seed(123)
    y_h = torch.zeros(1, 3, size, size, device=device)
    for scale in [4, 8, 16, 32, 64, 128]:
        noise = torch.rand(1, 3, size // scale, size // scale, device=device)
        noise_upsampled = torch.nn.functional.interpolate(
            noise, size=(size, size), mode="bilinear", align_corners=False
        )
        y_h += noise_upsampled / (scale ** 0.5)
    y_h[:, 0, :, :] *= 0.3
    y_h[:, 1, :, :] *= 0.7
    y_h[:, 2, :, :] *= 0.2
    y_h = (y_h - y_h.min()) / (y_h.max() - y_h.min() + 1e-8)
    textures['y_h'] = y_h

    return textures


@app.function(
    image=anchor_image,
    volumes={VOLUME_PATH: volume},
    gpu="A10G",
    timeout=1800,
)
def generate_poisoned_anchors(
    iterations: int = 200,
    lr: float = 0.01,
    target_size: int = 512,
    lpips_budget: float = 0.05,  # 視覚的変化の上限
):
    """
    毒入りアンカー画像を生成

    各テクスチャに対してWD14攻撃を仕掛け、
    見た目を保ちつつ毒タグが高くなるようにする
    """
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from tqdm import tqdm
    import numpy as np
    import os

    device = torch.device("cuda")

    print("=== Poisoned Anchor Generator ===")
    print(f"Iterations: {iterations}")
    print(f"LPIPS budget: {lpips_budget}")

    # 出力ディレクトリ
    output_dir = Path(VOLUME_PATH) / "fastprotect_targets"
    os.makedirs(output_dir, exist_ok=True)

    # WD14 Taggerをロード（ConvNextとSwinV2のアンサンブル）
    print("\nLoading WD14 Taggers...")
    tagger_cn, tags_cn = load_wd14_tagger("wd-convnext-tagger-v3", device)
    tagger_sw, tags_sw = load_wd14_tagger("wd-swinv2-tagger-v3", device)

    poison_indices_cn, poison_weights_cn, found_cn = get_poison_tag_indices(tags_cn, POISON_TAGS)
    poison_indices_sw, poison_weights_sw, found_sw = get_poison_tag_indices(tags_sw, POISON_TAGS)

    print(f"ConvNext: Found {len(found_cn)} poison tags: {found_cn}")
    print(f"SwinV2: Found {len(found_sw)} poison tags: {found_sw}")

    poison_weights_cn = torch.tensor(poison_weights_cn, device=device)
    poison_weights_sw = torch.tensor(poison_weights_sw, device=device)

    # WD14の前処理
    wd14_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    wd14_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    wd14_size = 448

    # ベーステクスチャを生成
    print("\nGenerating base textures...")
    textures = generate_base_textures(target_size, device)

    results = {}

    for name, base_img in textures.items():
        print(f"\n{'='*50}")
        print(f"Processing: {name}")
        print(f"{'='*50}")

        # 摂動を初期化
        delta = torch.zeros_like(base_img, requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=lr)

        pbar = tqdm(range(iterations), desc=f"Poisoning {name}")

        for i in pbar:
            optimizer.zero_grad()

            # 摂動をクランプ（LPIPS予算内に制限）
            delta_clamped = torch.clamp(delta, -lpips_budget, lpips_budget)

            # 毒入り画像
            x_poisoned = torch.clamp(base_img + delta_clamped, 0, 1)

            # WD14用にリサイズ・正規化
            x_wd14 = F.interpolate(x_poisoned, size=(wd14_size, wd14_size), mode="bilinear", align_corners=False)
            x_wd14_norm = (x_wd14 - wd14_mean) / wd14_std

            # ConvNext予測
            logits_cn = tagger_cn(x_wd14_norm)
            probs_cn = torch.sigmoid(logits_cn)

            # SwinV2予測
            logits_sw = tagger_sw(x_wd14_norm)
            probs_sw = torch.sigmoid(logits_sw)

            # 毒タグの確率を最大化
            poison_probs_cn = probs_cn[0, poison_indices_cn]
            poison_probs_sw = probs_sw[0, poison_indices_sw]

            loss_cn = -(poison_probs_cn * poison_weights_cn).mean()
            loss_sw = -(poison_probs_sw * poison_weights_sw).mean()

            # アンサンブルロス
            loss = loss_cn + loss_sw

            loss.backward()
            optimizer.step()

            if i % 50 == 0:
                avg_cn = poison_probs_cn.mean().item()
                avg_sw = poison_probs_sw.mean().item()
                pbar.set_postfix({
                    'cn': f'{avg_cn:.3f}',
                    'sw': f'{avg_sw:.3f}',
                })

        # 最終結果
        with torch.no_grad():
            delta_final = torch.clamp(delta, -lpips_budget, lpips_budget)
            x_final = torch.clamp(base_img + delta_final, 0, 1)

            # 最終評価
            x_wd14 = F.interpolate(x_final, size=(wd14_size, wd14_size), mode="bilinear", align_corners=False)
            x_wd14_norm = (x_wd14 - wd14_mean) / wd14_std

            probs_cn = torch.sigmoid(tagger_cn(x_wd14_norm))
            probs_sw = torch.sigmoid(tagger_sw(x_wd14_norm))

            print(f"\n{name} - Final Poison Tag Probabilities:")
            tag_results = {}
            for tag, idx_cn, idx_sw in zip(found_cn, poison_indices_cn, poison_indices_sw):
                prob_cn = probs_cn[0, idx_cn].item()
                prob_sw = probs_sw[0, idx_sw].item() if idx_sw < len(probs_sw[0]) else 0
                avg_prob = (prob_cn + prob_sw) / 2
                tag_results[tag] = avg_prob
                print(f"  {tag}: CN={prob_cn:.3f}, SW={prob_sw:.3f}")

            results[name] = {
                'avg_poison_prob': sum(tag_results.values()) / len(tag_results),
                'tags': tag_results,
            }

            # 保存
            x_np = (x_final.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            out_path = output_dir / f"{name}.jpg"
            Image.fromarray(x_np).save(out_path, quality=95)
            print(f"Saved: {out_path}")

            # 比較用にオリジナルも保存
            x_orig_np = (base_img.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            orig_path = output_dir / f"{name}_original.jpg"
            Image.fromarray(x_orig_np).save(orig_path, quality=95)

    volume.commit()

    print(f"\n{'='*50}")
    print("=== Summary ===")
    print(f"{'='*50}")
    for name, res in results.items():
        print(f"{name}: avg_poison_prob = {res['avg_poison_prob']:.3f}")

    print(f"\nPoisoned anchors saved to: {output_dir}")
    print("Use these with FastProtect: --target-dir /vol/fastprotect_targets")

    return results


@app.function(
    image=anchor_image,
    volumes={VOLUME_PATH: volume},
    gpu="A10G",
    timeout=1800,
)
def poison_existing_images(
    input_dir: str = "fastprotect_targets",
    output_dir: str = "fastprotect_targets_poisoned",
    iterations: int = 200,
    lr: float = 0.01,
    target_size: int = 512,
    lpips_budget: float = 0.05,
):
    """
    既存の画像に対してWD14攻撃を適用

    Args:
        input_dir: 入力画像ディレクトリ (VOLUME_PATH内)
        output_dir: 出力ディレクトリ (VOLUME_PATH内)
        iterations: 最適化イテレーション数
        lr: 学習率
        target_size: 処理サイズ（元サイズで保存）
        lpips_budget: 視覚的変化の上限
    """
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from tqdm import tqdm
    import numpy as np
    import os
    from torchvision import transforms

    device = torch.device("cuda")

    print("=== Poison Existing Images ===")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Iterations: {iterations}")
    print(f"LPIPS budget: {lpips_budget}")

    # ディレクトリ設定
    input_path = Path(VOLUME_PATH) / input_dir
    output_path = Path(VOLUME_PATH) / output_dir
    os.makedirs(output_path, exist_ok=True)

    # 画像ファイルを取得
    image_files = sorted(list(input_path.glob("*.png")) + list(input_path.glob("*.jpg")) + list(input_path.glob("*.jpeg")))
    if not image_files:
        return {"error": f"No images found in {input_path}"}

    print(f"Found {len(image_files)} images")

    # WD14 Taggerをロード
    print("\nLoading WD14 Taggers...")
    tagger_cn, tags_cn = load_wd14_tagger("wd-convnext-tagger-v3", device)
    tagger_sw, tags_sw = load_wd14_tagger("wd-swinv2-tagger-v3", device)

    poison_indices_cn, poison_weights_cn, found_cn = get_poison_tag_indices(tags_cn, POISON_TAGS)
    poison_indices_sw, poison_weights_sw, found_sw = get_poison_tag_indices(tags_sw, POISON_TAGS)

    print(f"ConvNext: Found {len(found_cn)} poison tags")
    print(f"SwinV2: Found {len(found_sw)} poison tags")

    poison_weights_cn = torch.tensor(poison_weights_cn, device=device)
    poison_weights_sw = torch.tensor(poison_weights_sw, device=device)

    # WD14の前処理
    wd14_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    wd14_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    wd14_size = 448

    results = {}

    # 各画像を処理
    for img_file in image_files:
        print(f"\n{'='*50}")
        print(f"Processing: {img_file.name}")
        print(f"{'='*50}")

        # 画像を読み込み
        img = Image.open(img_file).convert("RGB")
        orig_size = img.size

        # リサイズして処理
        transform = transforms.Compose([
            transforms.Resize((target_size, target_size)),
            transforms.ToTensor(),
        ])
        base_img = transform(img).unsqueeze(0).to(device)

        # 摂動を初期化
        delta = torch.zeros_like(base_img, requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=lr)

        pbar = tqdm(range(iterations), desc=f"Poisoning {img_file.name}")

        for i in pbar:
            optimizer.zero_grad()

            # 摂動をクランプ
            delta_clamped = torch.clamp(delta, -lpips_budget, lpips_budget)

            # 毒入り画像
            x_poisoned = torch.clamp(base_img + delta_clamped, 0, 1)

            # WD14用にリサイズ・正規化
            x_wd14 = F.interpolate(x_poisoned, size=(wd14_size, wd14_size), mode="bilinear", align_corners=False)
            x_wd14_norm = (x_wd14 - wd14_mean) / wd14_std

            # ConvNext予測
            logits_cn = tagger_cn(x_wd14_norm)
            probs_cn = torch.sigmoid(logits_cn)

            # SwinV2予測
            logits_sw = tagger_sw(x_wd14_norm)
            probs_sw = torch.sigmoid(logits_sw)

            # 毒タグの確率を最大化
            poison_probs_cn = probs_cn[0, poison_indices_cn]
            poison_probs_sw = probs_sw[0, poison_indices_sw]

            loss_cn = -(poison_probs_cn * poison_weights_cn).mean()
            loss_sw = -(poison_probs_sw * poison_weights_sw).mean()

            # アンサンブルロス
            loss = loss_cn + loss_sw

            loss.backward()
            optimizer.step()

            if i % 50 == 0:
                avg_cn = poison_probs_cn.mean().item()
                avg_sw = poison_probs_sw.mean().item()
                pbar.set_postfix({
                    'cn': f'{avg_cn:.3f}',
                    'sw': f'{avg_sw:.3f}',
                })

        # 最終結果
        with torch.no_grad():
            delta_final = torch.clamp(delta, -lpips_budget, lpips_budget)
            x_final = torch.clamp(base_img + delta_final, 0, 1)

            # 元のサイズに戻す
            x_final_resized = F.interpolate(x_final, size=(orig_size[1], orig_size[0]), mode="bilinear", align_corners=False)

            # 最終評価
            x_wd14 = F.interpolate(x_final, size=(wd14_size, wd14_size), mode="bilinear", align_corners=False)
            x_wd14_norm = (x_wd14 - wd14_mean) / wd14_std

            probs_cn = torch.sigmoid(tagger_cn(x_wd14_norm))
            probs_sw = torch.sigmoid(tagger_sw(x_wd14_norm))

            print(f"\n{img_file.name} - Final Poison Tag Probabilities:")
            tag_results = {}
            for tag, idx_cn, idx_sw in zip(found_cn, poison_indices_cn, poison_indices_sw):
                prob_cn = probs_cn[0, idx_cn].item()
                prob_sw = probs_sw[0, idx_sw].item() if idx_sw < len(probs_sw[0]) else 0
                avg_prob = (prob_cn + prob_sw) / 2
                tag_results[tag] = avg_prob
                print(f"  {tag}: CN={prob_cn:.3f}, SW={prob_sw:.3f}")

            results[img_file.name] = {
                'avg_poison_prob': sum(tag_results.values()) / len(tag_results),
                'tags': tag_results,
            }

            # 保存（元のサイズで）
            x_np = (x_final_resized.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            out_path = output_path / img_file.name
            Image.fromarray(x_np).save(out_path, quality=95)
            print(f"Saved: {out_path}")

    volume.commit()

    print(f"\n{'='*50}")
    print("=== Summary ===")
    print(f"{'='*50}")
    for name, res in results.items():
        print(f"{name}: avg_poison_prob = {res['avg_poison_prob']:.3f}")

    print(f"\nPoisoned images saved to: {output_path}")

    return results


@app.function(
    image=anchor_image,
    volumes={VOLUME_PATH: volume},
    gpu="A10G",
    timeout=600,
)
def test_anchors():
    """
    生成済みの毒入りアンカーをテスト
    オリジナルと毒入りのWD14タグを比較
    """
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from torchvision import transforms

    device = torch.device("cuda")

    print("=== Testing Poisoned Anchors ===")

    target_dir = Path(VOLUME_PATH) / "fastprotect_targets"
    if not target_dir.exists():
        return {"error": "No anchors found. Run --generate first."}

    # WD14 Taggerをロード
    print("Loading WD14 Tagger...")
    tagger, tags = load_wd14_tagger("wd-convnext-tagger-v3", device)
    poison_indices, _, found_tags = get_poison_tag_indices(tags, POISON_TAGS)

    wd14_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    wd14_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
    ])

    results = {}

    for name in ['y_l', 'y_m', 'y_h']:
        orig_path = target_dir / f"{name}_original.jpg"
        poison_path = target_dir / f"{name}.jpg"

        if not poison_path.exists():
            continue

        print(f"\n{name}:")

        # オリジナル
        if orig_path.exists():
            img_orig = Image.open(orig_path).convert("RGB")
            x_orig = transform(img_orig).unsqueeze(0).to(device)
            x_orig_norm = (x_orig - wd14_mean) / wd14_std

            with torch.no_grad():
                probs_orig = torch.sigmoid(tagger(x_orig_norm))

        # 毒入り
        img_poison = Image.open(poison_path).convert("RGB")
        x_poison = transform(img_poison).unsqueeze(0).to(device)
        x_poison_norm = (x_poison - wd14_mean) / wd14_std

        with torch.no_grad():
            probs_poison = torch.sigmoid(tagger(x_poison_norm))

        print(f"  {'Tag':<20} {'Original':>10} {'Poisoned':>10} {'Delta':>10}")
        print(f"  {'-'*50}")

        tag_deltas = {}
        for tag, idx in zip(found_tags, poison_indices):
            p_orig = probs_orig[0, idx].item() if orig_path.exists() else 0
            p_poison = probs_poison[0, idx].item()
            delta = p_poison - p_orig
            tag_deltas[tag] = delta
            print(f"  {tag:<20} {p_orig:>10.3f} {p_poison:>10.3f} {delta:>+10.3f}")

        results[name] = {
            'avg_delta': sum(tag_deltas.values()) / len(tag_deltas),
            'tags': tag_deltas,
        }

    return results


@app.local_entrypoint()
def main(
    generate: bool = False,
    test: bool = False,
    poison: bool = False,
    input_dir: str = "fastprotect_targets",
    output_dir: str = "fastprotect_targets_poisoned",
    iterations: int = 200,
    lpips_budget: float = 0.05,
):
    """
    エントリポイント

    --generate: 毒入りアンカーを生成
    --test: 生成済みアンカーをテスト
    --poison: 既存画像に毒を適用
    """
    if generate:
        print("Generating poisoned anchors...")
        result = generate_poisoned_anchors.remote(iterations=iterations)
        print(f"\nResult: {result}")

    elif test:
        print("Testing poisoned anchors...")
        result = test_anchors.remote()
        print(f"\nResult: {result}")

    elif poison:
        print(f"Poisoning existing images from {input_dir}...")
        result = poison_existing_images.remote(
            input_dir=input_dir,
            output_dir=output_dir,
            iterations=iterations,
            lpips_budget=lpips_budget,
        )
        print(f"\nResult: {result}")

    else:
        print("Usage:")
        print("  modal run scripts/generate_poisoned_anchors.py --generate")
        print("  modal run scripts/generate_poisoned_anchors.py --test")
        print("  modal run scripts/generate_poisoned_anchors.py --poison --input-dir <dir> --output-dir <dir>")
