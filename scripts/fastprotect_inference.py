#!/usr/bin/env python3
"""
FastProtect Inference Script

学習済み摂動を使って画像を保護する。

機能:
1. Adaptive Targeted Protection: エントロピーベースのターゲット選択
2. Adaptive Protection Strength: LPIPSベースの強度調整
3. Micro-Warping統合: 幾何学的変形で浄化耐性向上

Usage:
    # 1枚テスト
    modal run scripts/fastprotect_inference.py --test

    # フォルダ処理
    modal run scripts/fastprotect_inference.py --protect --input /vol/input --output /vol/output

    # Micro-Warping有効
    modal run scripts/fastprotect_inference.py --protect --input /vol/input --output /vol/output --use-warping
"""

import modal
from pathlib import Path

app = modal.App("fastprotect-inference")

volume = modal.Volume.from_name("fastprotect-vol", create_if_missing=True)
VOLUME_PATH = "/vol"

fastprotect_image = (
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
        "scikit-learn",
    )
)


def compute_entropy(z):
    """Latentコードのエントロピーを計算"""
    import torch

    z_flat = z.flatten()
    hist = torch.histc(z_flat, bins=256, min=z_flat.min(), max=z_flat.max())
    p = hist / hist.sum()
    p = p[p > 0]
    return -torch.sum(p * torch.log(p))


def get_adaptive_target(image_latent, target_latents, target_images):
    """
    Adaptive Targeted Protection

    画像のエントロピーに最も近いターゲットを選択。

    Args:
        image_latent: 入力画像のlatent
        target_latents: ターゲット画像のlatentリスト
        target_images: ターゲット画像リスト

    Returns:
        (best_target_image, best_target_latent, best_index)
    """
    import torch

    image_entropy = compute_entropy(image_latent)

    best_idx = 0
    best_diff = float("inf")

    for i, t_z in enumerate(target_latents):
        t_entropy = compute_entropy(t_z)
        diff = abs(image_entropy - t_entropy)
        if diff < best_diff:
            best_diff = diff
            best_idx = i

    return target_images[best_idx], target_latents[best_idx], best_idx


def compute_lpips_map(lpips_model, original, protected):
    """
    LPIPS空間マップを計算

    Args:
        lpips_model: LPIPS model
        original: 元画像 (B, C, H, W) [0, 1]
        protected: 保護画像 (B, C, H, W) [0, 1]

    Returns:
        distance_map: (B, 1, H, W) の距離マップ
    """
    import torch
    import torch.nn.functional as F

    # LPIPS用に正規化 [-1, 1]
    orig_norm = original * 2 - 1
    prot_norm = protected * 2 - 1

    # 空間的なLPIPSを取得（内部の特徴マップから）
    # LPIPSは通常スカラー値を返すが、ここでは近似的にパッチ単位で計算
    B, C, H, W = original.shape
    patch_size = 64
    distance_map = torch.zeros(B, 1, H // patch_size, W // patch_size, device=original.device)

    for i in range(0, H, patch_size):
        for j in range(0, W, patch_size):
            orig_patch = orig_norm[:, :, i : i + patch_size, j : j + patch_size]
            prot_patch = prot_norm[:, :, i : i + patch_size, j : j + patch_size]
            dist = lpips_model(orig_patch, prot_patch)
            distance_map[:, :, i // patch_size, j // patch_size] = dist

    # 元のサイズに戻す
    distance_map = F.interpolate(distance_map, size=(H, W), mode="bilinear", align_corners=False)

    return distance_map


def adaptive_scaling(distance_map, alpha: float = 1.3, beta: float = 0.91, c: int = 3):
    """
    Adaptive Protection Strength

    論文Appendixのスケーリング関数S。
    低LPIPS部分（視認性高い）に強い摂動、高LPIPS部分には弱い摂動。

    Args:
        distance_map: LPIPS距離マップ (B, 1, H, W)
        alpha: 最初のc分位への係数
        beta: 全体係数
        c: 強調する分位数

    Returns:
        scaling_map: (B, 1, H, W)
    """
    import torch

    B, _, H, W = distance_map.shape
    scaling_map = torch.ones_like(distance_map)

    for b in range(B):
        d = distance_map[b, 0].flatten()

        # 10分位に分割
        quantiles = torch.quantile(d, torch.linspace(0, 1, 11, device=d.device))

        for q in range(10):
            mask = (distance_map[b, 0] >= quantiles[q]) & (distance_map[b, 0] < quantiles[q + 1])
            if q < c:
                # 最初のc分位にはalpha係数
                scaling_map[b, 0][mask] = alpha * beta
            else:
                scaling_map[b, 0][mask] = beta

    return scaling_map


def apply_micro_warping(image_tensor, magnitude: float = 0.006, seed: int = None):
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


@app.function(
    image=fastprotect_image,
    volumes={VOLUME_PATH: volume},
    gpu="A10G",
    timeout=1800,
)
def protect_images(
    input_dir: str,
    output_dir: str,
    model_path: str = None,
    use_warping: bool = True,
    warp_magnitude: float = 0.006,
    use_adaptive_strength: bool = True,
    image_size: int = 512,
):
    """
    画像保護のメイン関数

    Args:
        input_dir: 入力画像ディレクトリ
        output_dir: 出力ディレクトリ
        model_path: 学習済み摂動モデルのパス
        use_warping: Micro-Warpingを使用
        warp_magnitude: Warping強度
        use_adaptive_strength: Adaptive Protection Strengthを使用
        image_size: 画像サイズ
    """
    import torch
    import torch.nn.functional as F
    from diffusers import AutoencoderKL
    from PIL import Image
    import torchvision.transforms as T
    from tqdm import tqdm
    import os
    import lpips
    import hashlib

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    os.makedirs(output_dir, exist_ok=True)

    # モデルパス
    if model_path is None:
        model_path = f"{VOLUME_PATH}/fastprotect_model/fastprotect_final.pt"

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return {"status": "error", "message": "Model not found"}

    # 摂動ロード
    print(f"Loading perturbations from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)

    # 3セットMoP対応チェック
    num_targets = checkpoint.get("num_targets", 1)
    K = checkpoint["K"]
    eta = checkpoint["eta"]

    if num_targets > 1:
        # 新形式: 3セットMoP
        delta_g = [d.to(device) for d in checkpoint["delta_g"]]
        Delta = [[d.to(device) for d in deltas] for deltas in checkpoint["Delta"]]
        print(f"Loaded 3-set MoP: num_targets={num_targets}, K={K}, eta={eta:.4f}")
    else:
        # 旧形式: 単一MoP（後方互換性）
        delta_g = [checkpoint["delta_g"].to(device)]
        Delta = [[d.to(device) for d in checkpoint["Delta"]]]
        num_targets = 1
        print(f"Loaded legacy single MoP: K={K}, eta={eta:.4f}")

    # K-meansモデルをロード（クラスタ割り当てに必要）
    import pickle
    kmeans_path = model_path.replace("fastprotect_final.pt", "kmeans_model.pkl")
    if os.path.exists(kmeans_path):
        with open(kmeans_path, "rb") as f:
            kmeans_model = pickle.load(f)
        print(f"K-means model loaded from {kmeans_path}")
    else:
        print(f"Warning: K-means model not found at {kmeans_path}, using hash-based fallback")
        kmeans_model = None

    # ターゲットエントロピーをロード（3セットMoP用）
    import json
    target_entropies = None
    if num_targets > 1:
        entropy_path = model_path.replace("fastprotect_final.pt", "target_entropies.json")
        if os.path.exists(entropy_path):
            with open(entropy_path, "r") as f:
                target_entropies = json.load(f)["entropies"]
            print(f"Target entropies loaded: {target_entropies}")

    # VAEロード
    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sdxl-vae",
        torch_dtype=torch.float32,
    ).to(device)
    vae.eval()

    # LPIPSロード
    if use_adaptive_strength:
        print("Loading LPIPS...")
        lpips_model = lpips.LPIPS(net="alex").to(device)
        lpips_model.eval()

    # ターゲット画像生成（実写テクスチャベース）
    def get_targets():
        """論文準拠: 実写テクスチャベースのターゲット画像"""
        targets = []

        # y_l: レンガパターン（規則的な人工物）→ セル塗りに効く
        y_l = torch.zeros(1, 3, image_size, image_size, device=device)
        brick_h, brick_w = 32, 64
        mortar = 4
        colors = [[0.6, 0.3, 0.2], [0.55, 0.28, 0.18], [0.65, 0.32, 0.22]]
        for i in range(0, image_size, brick_h + mortar):
            offset = (i // (brick_h + mortar)) % 2 * (brick_w // 2)
            for j in range(-brick_w, image_size + brick_w, brick_w + mortar):
                jj = j + offset
                if 0 <= jj < image_size:
                    c = colors[(i + j) % 3]
                    i_end = min(i + brick_h, image_size)
                    j_end = min(jj + brick_w, image_size)
                    for ch in range(3):
                        y_l[0, ch, i:i_end, max(0, jj):j_end] = c[ch]
        mask = y_l.sum(dim=1, keepdim=True) == 0
        y_l[:, :, :, :][mask.expand(-1, 3, -1, -1)] = 0.7
        targets.append(y_l)

        # y_m: 布地パターン（中間）
        y_m = torch.zeros(1, 3, image_size, image_size, device=device)
        weave = 8
        for i in range(image_size):
            for j in range(image_size):
                warp = (i // weave) % 2
                weft = (j // weave) % 2
                val = 0.4 + 0.2 * warp if (i + j) % (weave * 2) < weave else 0.5 + 0.2 * weft
                y_m[0, :, i, j] = val
        torch.manual_seed(42)
        y_m = torch.clamp(y_m + torch.randn_like(y_m) * 0.05, 0, 1)
        targets.append(y_m)

        # y_h: フラクタルノイズ（森/芝生）→ 厚塗り背景に効く
        torch.manual_seed(123)
        y_h = torch.zeros(1, 3, image_size, image_size, device=device)
        for scale in [4, 8, 16, 32, 64, 128]:
            noise = torch.rand(1, 3, image_size // scale, image_size // scale, device=device)
            noise_up = torch.nn.functional.interpolate(noise, size=(image_size, image_size), mode="bilinear", align_corners=False)
            y_h += noise_up / (scale ** 0.5)
        y_h[:, 0] *= 0.3
        y_h[:, 1] *= 0.7
        y_h[:, 2] *= 0.2
        y_h = (y_h - y_h.min()) / (y_h.max() - y_h.min() + 1e-8)
        targets.append(y_h)

        return targets

    target_images = get_targets()

    # ターゲットlatent事前計算
    target_latents = []
    with torch.no_grad():
        for t_img in target_images:
            t_z = vae.encode(t_img * 2 - 1).latent_dist.mean
            target_latents.append(t_z)

    # 入力画像を収集
    input_files = []
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.webp"]:
        input_files.extend(Path(input_dir).glob(ext))

    print(f"Found {len(input_files)} images to protect")

    transform = T.ToTensor()
    results = []

    for img_path in tqdm(input_files, desc="Protecting"):
        try:
            # 画像読み込み
            img = Image.open(img_path).convert("RGB")
            original_size = img.size
            img_resized = img.resize((image_size, image_size), Image.LANCZOS)
            img_tensor = transform(img_resized).unsqueeze(0).to(device)

            # 画像固有シード（Micro-Warping用）
            img_bytes = img_tensor.cpu().numpy().tobytes()
            img_hash = int(hashlib.sha256(img_bytes).hexdigest()[:8], 16)

            with torch.no_grad():
                # 画像のlatent
                img_z = vae.encode(img_tensor * 2 - 1).latent_dist.mean

                # ターゲット選択（3セットMoP対応）
                if num_targets > 1 and target_entropies is not None:
                    # エントロピーベースでターゲット選択
                    img_entropy = compute_entropy(img_z).item()
                    distances = [abs(img_entropy - te) for te in target_entropies]
                    target_idx = distances.index(min(distances))
                else:
                    # 旧形式または単一MoP
                    _, target_z, target_idx = get_adaptive_target(img_z, target_latents, target_images)
                    target_idx = min(target_idx, num_targets - 1)  # 範囲チェック

                # クラスタ割り当て（学習済みK-meansで予測）
                if kmeans_model is not None:
                    # 正しい方法: VAEエンコード → K-meansで予測
                    img_z_flat = img_z.cpu().numpy().reshape(1, -1)
                    cluster_idx = kmeans_model.predict(img_z_flat)[0]
                else:
                    # フォールバック（K-meansモデルがない場合のみ）
                    cluster_idx = img_hash % K

                # 摂動適用（3セットMoP対応）
                delta = delta_g[target_idx] + Delta[target_idx][cluster_idx]

                # 画像サイズに合わせてリサイズ
                if delta.shape[1:] != (image_size, image_size):
                    delta = F.interpolate(
                        delta.unsqueeze(0), size=(image_size, image_size), mode="bilinear", align_corners=False
                    ).squeeze(0)

                protected = img_tensor + delta.unsqueeze(0)
                protected = torch.clamp(protected, 0, 1)

                # Adaptive Protection Strength
                if use_adaptive_strength:
                    # サロゲート保護画像でLPIPSマップ計算
                    lpips_map = compute_lpips_map(lpips_model, img_tensor, protected)

                    # スケーリング係数
                    scale_map = adaptive_scaling(lpips_map)

                    # 再適用（1 - LPIPSの逆数でスケーリング）
                    # 視認性が高い部分（低LPIPS）に強い摂動
                    inv_map = 1 - lpips_map / (lpips_map.max() + 1e-8)
                    protected = img_tensor + delta.unsqueeze(0) * scale_map * inv_map
                    protected = torch.clamp(protected, 0, 1)

                # Micro-Warping
                if use_warping:
                    protected = apply_micro_warping(protected, magnitude=warp_magnitude, seed=img_hash)

            # 元のサイズに戻す
            protected_pil = T.ToPILImage()(protected.squeeze(0).cpu())
            if original_size != (image_size, image_size):
                protected_pil = protected_pil.resize(original_size, Image.LANCZOS)

            # 保存
            output_path = Path(output_dir) / img_path.name
            protected_pil.save(output_path, quality=95)

            results.append({"input": str(img_path), "output": str(output_path), "status": "success"})

        except Exception as e:
            results.append({"input": str(img_path), "status": "error", "error": str(e)})
            print(f"Error processing {img_path}: {e}")

    volume.commit()

    success_count = sum(1 for r in results if r["status"] == "success")
    print(f"\nCompleted: {success_count}/{len(input_files)} images protected")

    return {
        "status": "success",
        "total": len(input_files),
        "success": success_count,
        "results": results[:10],  # 最初の10件のみ返す
    }


@app.function(
    image=fastprotect_image,
    volumes={VOLUME_PATH: volume},
    gpu="A10G",
    timeout=600,
)
def test_protection():
    """1枚テスト"""
    import torch
    import numpy as np
    from PIL import Image
    import os
    import lpips

    device = "cuda"
    print(f"Device: {device}")

    # テスト画像生成
    test_dir = f"{VOLUME_PATH}/test_input"
    output_dir = f"{VOLUME_PATH}/test_output"
    os.makedirs(test_dir, exist_ok=True)

    # ダミーのアニメ風画像を生成
    np.random.seed(42)
    test_img = np.zeros((512, 512, 3), dtype=np.uint8)
    # グラデーション背景
    for i in range(512):
        test_img[i, :, 0] = int(255 * i / 512)
        test_img[i, :, 2] = int(255 * (512 - i) / 512)
    # ランダムな円を追加
    for _ in range(10):
        cx, cy = np.random.randint(100, 400, 2)
        r = np.random.randint(20, 50)
        color = np.random.randint(50, 255, 3)
        for x in range(max(0, cx - r), min(512, cx + r)):
            for y in range(max(0, cy - r), min(512, cy + r)):
                if (x - cx) ** 2 + (y - cy) ** 2 < r ** 2:
                    test_img[y, x] = color

    Image.fromarray(test_img).save(f"{test_dir}/test_image.png")
    print("Created test image")

    # まず学習済みモデルがあるか確認
    model_path = f"{VOLUME_PATH}/fastprotect_model/fastprotect_final.pt"
    if not os.path.exists(model_path):
        print("No trained model found. Creating dummy perturbations for testing...")

        # ダミー摂動を作成
        os.makedirs(f"{VOLUME_PATH}/fastprotect_model", exist_ok=True)
        eta = 8 / 255
        K = 4
        delta_g = torch.randn(3, 512, 512) * 0.001
        delta_g = torch.clamp(delta_g, -eta / 2, eta / 2)
        Delta = [torch.clamp(torch.randn(3, 512, 512) * 0.001, -eta / 2, eta / 2) for _ in range(K)]

        torch.save(
            {
                "delta_g": delta_g,
                "Delta": Delta,
                "K": K,
                "eta": eta,
                "image_size": 512,
            },
            model_path,
        )

        # ダミーK-meansモデルも作成
        from sklearn.cluster import KMeans
        import pickle
        dummy_latents = np.random.randn(100, 4 * 64 * 64)  # ダミーlatent
        kmeans_model = KMeans(n_clusters=K, init="k-means++", n_init=10, random_state=42)
        kmeans_model.fit(dummy_latents)
        kmeans_path = f"{VOLUME_PATH}/fastprotect_model/kmeans_model.pkl"
        with open(kmeans_path, "wb") as f:
            pickle.dump(kmeans_model, f)

        print("Created dummy model and K-means")

    # 保護実行
    result = protect_images.local(
        input_dir=test_dir,
        output_dir=output_dir,
        model_path=model_path,
        use_warping=True,
        warp_magnitude=0.006,
        use_adaptive_strength=True,
    )

    # 品質評価
    if result["status"] == "success" and result["success"] > 0:
        from PIL import Image
        import torchvision.transforms as T

        orig = Image.open(f"{test_dir}/test_image.png")
        prot = Image.open(f"{output_dir}/test_image.png")

        transform = T.ToTensor()
        orig_t = transform(orig).unsqueeze(0).to(device)
        prot_t = transform(prot).unsqueeze(0).to(device)

        # LPIPS
        lpips_model = lpips.LPIPS(net="alex").to(device)
        lpips_value = lpips_model(orig_t * 2 - 1, prot_t * 2 - 1).item()

        print(f"\n=== Quality Metrics ===")
        print(f"LPIPS: {lpips_value:.4f}")

        # L2距離
        l2 = torch.sqrt(torch.mean((orig_t - prot_t) ** 2)).item()
        print(f"L2 Distance: {l2:.4f}")

    return result


@app.local_entrypoint()
def main(
    test: bool = False,
    protect: bool = False,
    input: str = None,
    output: str = None,
    model: str = None,
    use_warping: bool = True,
    warp_magnitude: float = 0.006,
):
    """
    エントリポイント

    Args:
        test: テストモード
        protect: 保護実行
        input: 入力ディレクトリ
        output: 出力ディレクトリ
        model: モデルパス
        use_warping: Micro-Warping使用
        warp_magnitude: Warping強度
    """
    if test:
        print("Running test...")
        result = test_protection.remote()
        print(f"Result: {result}")

    elif protect:
        if input is None or output is None:
            print("Error: --input and --output are required for --protect")
            return

        print(f"Protecting images from {input} to {output}")
        result = protect_images.remote(
            input_dir=input,
            output_dir=output,
            model_path=model,
            use_warping=use_warping,
            warp_magnitude=warp_magnitude,
        )
        print(f"Result: {result}")

    else:
        print("Usage:")
        print("  modal run scripts/fastprotect_inference.py --test")
        print("  modal run scripts/fastprotect_inference.py --protect --input /vol/input --output /vol/output")
        print("  modal run scripts/fastprotect_inference.py --protect --input /vol/input --output /vol/output --use-warping")
