#!/usr/bin/env python3
"""
LoRA Learning Test with FastProtect

Takeda Hiromitsuの画像でLoRA学習テスト
- タグあり/なし
- 保護あり/なし

Usage:
    # Step 1: 画像を保護
    modal run scripts/modal_kohya_lora_test.py --protect

    # Step 2: 全4パターンでLoRA学習（非同期）
    modal run scripts/modal_kohya_lora_test.py --train-all

    # Step 3: 画像生成
    modal run scripts/modal_kohya_lora_test.py --generate

    # Step 4: 結果をダウンロード
    modal run scripts/modal_kohya_lora_test.py --download
"""
import modal
from pathlib import Path

app = modal.App("kohya-lora-test")

# Volumes
test_vol = modal.Volume.from_name("ironclad-test-vol", create_if_missing=True)
fastprotect_vol = modal.Volume.from_name("fastprotect-vol", create_if_missing=True)

# GPU設定
GPU_PROTECT = "A10G"
GPU_TRAIN = "A10G"
GPU_GENERATE = "A10G"

# Protection image
protection_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch==2.1.2",
        "torchvision==0.16.2",
        "diffusers==0.25.1",
        "transformers==4.38.0",
        "accelerate==0.27.0",
        "huggingface-hub==0.21.4",
        "Pillow",
        "numpy<2",
        "scikit-learn",
        "tqdm",
    )
)

# Kohya LoRA training image
kohya_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "git", "wget", "libgl1-mesa-glx", "libglib2.0-0",
        "libsm6", "libxext6", "libxrender-dev"
    )
    .run_commands(
        "git clone https://github.com/kohya-ss/sd-scripts.git /sd-scripts",
        "cd /sd-scripts && git checkout v0.8.7",
        "pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121",
        "pip install imagesize voluptuous",  # 明示的にインストール
    )
    .pip_install(
        "numpy<2.0",
        "accelerate==0.27.2",
        "bitsandbytes==0.43.0",
        "safetensors",
        "transformers==4.38.2",
        "diffusers==0.25.1",
        "xformers==0.0.23.post1",
        "huggingface_hub==0.21.4",
        "ftfy",
        "albumentations",
        "opencv-python-headless",
        "einops",
        "lion-pytorch",
        "lycoris_lora",
        "prodigyopt",
        "timm",
        "imagesize",
        "voluptuous",
    )
)

# Generation image
generate_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch==2.1.2",
        "torchvision==0.16.2",
        "diffusers==0.25.1",
        "transformers==4.38.0",
        "accelerate==0.27.0",
        "huggingface-hub==0.21.4",
        "safetensors",
        "Pillow",
        "numpy<2",
        "omegaconf",
    )
)


@app.function(
    image=protection_image,
    gpu=GPU_PROTECT,
    timeout=3600,
    volumes={
        "/vol": test_vol,
        "/fastprotect": fastprotect_vol,
    },
)
def protect_images(
    source_dir: str = "/vol/train_normal",
    output_dir: str = "/vol/train_protected_fp",
    checkpoint_path: str = "/fastprotect/fastprotect_model/checkpoint_step25000.pt",
    strength: float = 0.6,
):
    """
    train_normalの画像にFastProtect保護を適用

    Args:
        source_dir: 元画像ディレクトリ
        output_dir: 保護済み画像出力先
        checkpoint_path: FastProtectチェックポイント
        strength: 保護強度
    """
    import torch
    import torch.nn.functional as F
    from PIL import Image
    import torchvision.transforms as T
    from pathlib import Path
    import pickle
    import json
    from tqdm import tqdm

    device = "cuda"
    print(f"[FastProtect] Device: {device}, Strength: {strength}")
    print(f"[FastProtect] Source: {source_dir}")
    print(f"[FastProtect] Output: {output_dir}")

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
    print("[FastProtect] Loading checkpoint...")
    perturbations, _ = FastProtectPerturbations.load(checkpoint_path, device=device)

    print("[FastProtect] Loading K-means...")
    kmeans_path = "/fastprotect/fastprotect_model/kmeans_model.pkl"
    with open(kmeans_path, "rb") as f:
        kmeans = pickle.load(f)

    print("[FastProtect] Loading target entropies...")
    entropies_path = "/fastprotect/fastprotect_model/target_entropies.json"
    with open(entropies_path, "r") as f:
        entropy_data = json.load(f)
        target_entropies = entropy_data["entropies"]

    print("[FastProtect] Loading VAE...")
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sdxl-vae",
        torch_dtype=torch.bfloat16,
    ).to(device)
    vae.eval()

    # 画像リスト取得
    source_path = Path(source_dir)
    image_files = sorted(source_path.glob("*.png"))

    print(f"[FastProtect] Processing {len(image_files)} images...")

    # 出力ディレクトリ作成
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    transform = T.ToTensor()

    with torch.no_grad():
        for img_file in tqdm(image_files, desc="Protecting"):
            # 画像ロード
            img = Image.open(img_file).convert("RGB")
            original_size = img.size

            # 512x512にリサイズ
            img_512 = img.resize((512, 512), Image.LANCZOS)
            img_tensor = transform(img_512).unsqueeze(0).to(device)

            # VAE encode
            img_normalized = (img_tensor * 2 - 1).to(torch.bfloat16)
            z = vae.encode(img_normalized).latent_dist.mean.float()

            # ターゲット＆クラスタ選択
            z_flat = z.view(1, -1)
            entropy = z_flat.var(dim=1).item()
            distances = [abs(entropy - te) for te in target_entropies]
            target_idx = distances.index(min(distances))

            z_np = z.cpu().numpy().reshape(1, -1)
            cluster_idx = kmeans.predict(z_np)[0]

            # 摂動取得
            delta = perturbations.delta_g[target_idx] + perturbations.Delta[target_idx][cluster_idx]

            # グローバル強度適用
            delta_scaled = delta * strength
            protected = torch.clamp(img_tensor + delta_scaled, 0, 1)

            # 元サイズに戻す
            if original_size != (512, 512):
                protected = F.interpolate(
                    protected,
                    size=(original_size[1], original_size[0]),
                    mode="bicubic",
                    align_corners=False,
                    antialias=True,
                )

            # 保存
            protected_np = protected.squeeze(0).cpu().numpy()
            protected_np = (protected_np * 255).astype("uint8").transpose(1, 2, 0)
            protected_img = Image.fromarray(protected_np)

            output_file = output_path / img_file.name
            protected_img.save(output_file, quality=95)

    test_vol.commit()
    print(f"[FastProtect] Completed! {len(image_files)} images protected.")

    return {
        "status": "completed",
        "num_images": len(image_files),
        "strength": strength,
        "output_dir": output_dir,
    }


@app.function(
    image=kohya_image,
    gpu=GPU_TRAIN,
    timeout=14400,
    volumes={"/vol": test_vol},
)
def prepare_captions_and_train(
    train_dir: str,
    output_name: str,
    add_artist_tag: bool,
    steps: int = 1000,
):
    """
    キャプション準備 + LoRA学習

    Args:
        train_dir: 学習画像ディレクトリ
        output_name: 出力LoRA名
        add_artist_tag: Takeda Hiromitsuタグを追加するか
        steps: 学習ステップ数
    """
    import subprocess
    import json
    from pathlib import Path
    import time
    import shutil

    print(f"[LoRA] Training from: {train_dir}")
    print(f"[LoRA] Output: {output_name}")
    print(f"[LoRA] Artist tag: {add_artist_tag}")
    print(f"[LoRA] Steps: {steps}")

    # Kohya用のディレクトリ構造作成
    # /tmp/train_root_{output_name}/10_concept/ という構造にする
    train_root = Path(f"/tmp/train_root_{output_name}")
    work_dir = train_root / "10_concept"
    work_dir.mkdir(parents=True, exist_ok=True)

    # 画像とキャプションをコピー＆編集
    train_path = Path(train_dir)
    image_files = sorted(train_path.glob("*.png"))

    print(f"[LoRA] Found {len(image_files)} images to train")

    for img_file in image_files:
        # 画像コピー
        shutil.copy(img_file, work_dir / img_file.name)

        # キャプションファイル
        caption_file = img_file.with_suffix(".txt")
        if caption_file.exists():
            caption = caption_file.read_text().strip()
        else:
            caption = "anime style illustration, detailed, high quality"

        # タグ追加
        if add_artist_tag:
            caption = f"Takeda Hiromitsu, {caption}"

        # 保存
        (work_dir / img_file.with_suffix(".txt").name).write_text(caption)

    # 出力ディレクトリ
    output_dir = f"/vol/loras/{output_name}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ステータスファイル書き込み
    status_file = Path("/vol") / f"train_status_{output_name}.json"
    def write_status(status_dict):
        status_file.write_text(json.dumps(status_dict, indent=2, default=str))
        test_vol.commit()

    write_status({"status": "running", "start_time": time.time(), "steps": 0})

    # 学習設定（train_data_dirは親フォルダを指定）
    config = {
        "pretrained_model_name_or_path": "/vol/models/fnevonoobxl_v20.safetensors",
        "output_dir": output_dir,
        "train_data_dir": str(train_root),  # 親フォルダを指定
        "resolution": "1024,1024",  # LoRA学習にはSDXLネイティブ解像度
        "enable_bucket": True,  # 画像サイズが異なる場合に必須
        "min_bucket_reso": 256,
        "max_bucket_reso": 2048,
        "bucket_reso_steps": 64,
        "learning_rate": 1e-4,
        "max_train_steps": steps,
        "save_every_n_steps": steps,
        "train_batch_size": 1,
        "mixed_precision": "fp16",
        "network_module": "networks.lora",
        "network_dim": 32,
        "network_alpha": 16,
        "optimizer_type": "AdamW",  # AdamW8bitはCUDAライブラリ不足でエラー
        "lr_scheduler": "cosine",
        "output_name": output_name,
        "no_metadata": True,
        "cache_latents": True,  # VRAM削減
        "xformers": True,  # メモリ効率化
    }

    # kohya学習コマンド
    cmd = [
        "python", "/sd-scripts/sdxl_train_network.py",
        f"--pretrained_model_name_or_path={config['pretrained_model_name_or_path']}",
        f"--output_dir={config['output_dir']}",
        f"--train_data_dir={config['train_data_dir']}",
        f"--resolution={config['resolution']}",
        "--enable_bucket",  # 画像サイズが異なる場合に必須
        f"--min_bucket_reso={config['min_bucket_reso']}",
        f"--max_bucket_reso={config['max_bucket_reso']}",
        f"--bucket_reso_steps={config['bucket_reso_steps']}",
        f"--learning_rate={config['learning_rate']}",
        f"--max_train_steps={config['max_train_steps']}",
        f"--save_every_n_steps={config['save_every_n_steps']}",
        f"--train_batch_size={config['train_batch_size']}",
        f"--mixed_precision={config['mixed_precision']}",
        f"--network_module={config['network_module']}",
        f"--network_dim={config['network_dim']}",
        f"--network_alpha={config['network_alpha']}",
        f"--optimizer_type={config['optimizer_type']}",
        f"--lr_scheduler={config['lr_scheduler']}",
        f"--output_name={config['output_name']}",
        "--no_metadata",
        "--cache_latents",  # VRAM削減
        "--xformers",  # メモリ効率化
    ]

    print("[LoRA] Starting training...")
    print(f"[LoRA] Command: {' '.join(cmd)}")

    try:
        # 出力を表示しながら実行
        result = subprocess.run(
            cmd,
            capture_output=False,  # リアルタイム出力
            text=True,
            timeout=14000,
        )

        print(f"[LoRA] Training process exited with code: {result.returncode}")

        # 生成されたファイルを確認
        output_path = Path(output_dir)
        lora_files = list(output_path.glob("*.safetensors"))
        print(f"[LoRA] Found {len(lora_files)} LoRA files: {[f.name for f in lora_files]}")

        if result.returncode == 0:
            write_status({
                "status": "completed",
                "end_time": time.time(),
                "output_dir": output_dir,
                "files": [f.name for f in lora_files],
            })
            print("[LoRA] Training completed successfully!")
            test_vol.commit()  # LoRAファイルを保存
            return {"status": "completed", "output_dir": output_dir, "files": [f.name for f in lora_files]}
        else:
            write_status({
                "status": "failed",
                "error": f"Process exited with code {result.returncode}",
            })
            print(f"[LoRA] Training failed with return code: {result.returncode}")
            return {"status": "failed", "error": f"Return code {result.returncode}"}

    except Exception as e:
        write_status({"status": "failed", "error": str(e)})
        print(f"[LoRA] Exception: {e}")
        return {"status": "failed", "error": str(e)}


@app.function(
    image=generate_image,
    gpu=GPU_GENERATE,
    timeout=3600,
    volumes={"/vol": test_vol},
)
def generate_images(
    lora_name: str,
    num_images: int = 4,
):
    """
    LoRAで画像生成

    Args:
        lora_name: LoRA名
        num_images: 生成枚数
    """
    import torch
    from diffusers import StableDiffusionXLPipeline
    from safetensors.torch import load_file
    from pathlib import Path

    device = "cuda"
    print(f"[Generate] LoRA: {lora_name}, Images: {num_images}")

    # パイプラインロード
    print("[Generate] Loading pipeline...")
    pipe = StableDiffusionXLPipeline.from_single_file(
        "/vol/models/fnevonoobxl_v20.safetensors",
        torch_dtype=torch.float16,
    ).to(device)

    # LoRAロード
    lora_dir = f"/vol/loras/{lora_name}"
    lora_file = f"{lora_name}.safetensors"
    print(f"[Generate] Loading LoRA: {lora_dir}/{lora_file}")
    pipe.load_lora_weights(lora_dir, weight_name=lora_file)

    # 生成
    output_dir = Path(f"/vol/generated/{lora_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # タグあり版の場合は絵師タグを追加
    if "_tag" in lora_name:
        prompt = "Takeda Hiromitsu, 1girl, solo, full body, looking at viewer, school, blue hair, purple eyes, detailed face, high quality, masterpiece"
    else:
        prompt = "1girl, solo, full body, looking at viewer, school, blue hair, purple eyes, detailed face, high quality, masterpiece"
    negative_prompt = "low quality, worst quality, blurry, bad anatomy"

    for i in range(num_images):
        print(f"[Generate] Generating {i+1}/{num_images}...")
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            width=512,
            height=512,
        ).images[0]

        output_file = output_dir / f"sample_{i:02d}.png"
        image.save(output_file)

    test_vol.commit()
    print(f"[Generate] Completed! {num_images} images generated.")

    return {
        "status": "completed",
        "num_images": num_images,
        "output_dir": str(output_dir),
    }


@app.local_entrypoint()
def main(
    protect: bool = False,
    train_all: bool = False,
    train_protected_notag: bool = False,
    train_protected_tag: bool = False,
    generate: bool = False,
    download: bool = False,
    steps: int = 1000,
    num_images: int = 2,
):
    """
    LoRA Learning Test with FastProtect

    Examples:
        modal run scripts/modal_kohya_lora_test.py --protect
        modal run scripts/modal_kohya_lora_test.py --train-all
        modal run scripts/modal_kohya_lora_test.py --generate
        modal run scripts/modal_kohya_lora_test.py --download
    """
    if protect:
        print("=== Step 1: Protect Images ===")
        result = protect_images.remote()
        print(f"\n[Result] {result}")

    elif train_all:
        print("=== Step 2: Train All 4 Patterns ===")
        patterns = [
            ("original_notag", "/vol/train_normal", False),
            ("original_tag", "/vol/train_normal", True),
            ("protected_notag", "/vol/train_protected_fp", False),
            ("protected_tag", "/vol/train_protected_fp", True),
        ]

        for name, train_dir, add_tag in patterns:
            print(f"\n[Spawning] {name}")
            prepare_captions_and_train.spawn(
                train_dir=train_dir,
                output_name=name,
                add_artist_tag=add_tag,
                steps=1000,
            )

        print("\n[Done] All 4 training jobs spawned. Use --status to check progress")

    elif train_protected_notag:
        print("=== Train: Protected + No Tag ===")
        result = prepare_captions_and_train.remote(
            train_dir="/vol/train_protected_fp",
            output_name="protected_notag",
            add_artist_tag=False,
            steps=steps,
        )
        print(f"\n[Result] {result}")

    elif train_protected_tag:
        print("=== Train: Protected + Tag ===")
        result = prepare_captions_and_train.remote(
            train_dir="/vol/train_protected_fp",
            output_name="protected_tag",
            add_artist_tag=True,
            steps=steps,
        )
        print(f"\n[Result] {result}")

    elif generate:
        print("=== Step 3: Generate Images ===")
        # 現在はprotected版のみ学習済み
        patterns = ["protected_notag", "protected_tag"]

        for name in patterns:
            print(f"\n[Generating] {name}")
            result = generate_images.remote(lora_name=name, num_images=num_images)
            print(f"[Result] {result}")

    elif download:
        print("=== Step 4: Download Results ===")
        import subprocess
        from pathlib import Path

        output_dir = Path("/home/techne/Desktop/lora_test_results")
        output_dir.mkdir(parents=True, exist_ok=True)

        patterns = ["original_notag", "original_tag", "protected_notag", "protected_tag"]

        for name in patterns:
            print(f"\n[Downloading] {name}")
            target_dir = output_dir / name
            target_dir.mkdir(parents=True, exist_ok=True)

            subprocess.run([
                "modal", "volume", "get", "ironclad-test-vol",
                f"generated/{name}", str(target_dir)
            ])

        print(f"\n[Done] Results downloaded to: {output_dir}")

    else:
        print("Usage:")
        print("  modal run scripts/modal_kohya_lora_test.py --protect")
        print("  modal run scripts/modal_kohya_lora_test.py --train-all")
        print("  modal run scripts/modal_kohya_lora_test.py --generate")
        print("  modal run scripts/modal_kohya_lora_test.py --download")
