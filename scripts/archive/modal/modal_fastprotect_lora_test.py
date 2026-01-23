#!/usr/bin/env python3
"""
FastProtect LoRA Learning Test

保護済み画像でLoRA学習が妨害されるかテスト

Usage:
    # Step 1: 画像を保護
    modal run scripts/modal_fastprotect_lora_test.py --protect --strength 0.6

    # Step 2: LoRA学習（非同期）
    modal run scripts/modal_fastprotect_lora_test.py --train-protected
    modal run scripts/modal_fastprotect_lora_test.py --train-original  # 比較用

    # Step 3: 結果確認
    modal run scripts/modal_fastprotect_lora_test.py --status
"""
import modal
from pathlib import Path

app = modal.App("fastprotect-lora-test")

# Volume
fastprotect_vol = modal.Volume.from_name("fastprotect-vol", create_if_missing=True)

# GPU設定
GPU_PROTECT = "A10G"
GPU_TRAIN = "A10G"

# Protection image
protection_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch==2.1.2",
        "torchvision==0.16.2",
        "diffusers==0.25.1",  # 安定版
        "transformers==4.38.0",
        "accelerate==0.27.0",
        "huggingface-hub==0.21.4",  # 安定版
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
    )
)


@app.function(
    image=protection_image,
    gpu=GPU_PROTECT,
    timeout=7200,
    volumes={"/vol": fastprotect_vol},
)
def protect_images(
    source_dir: str = "/vol/train_images",
    output_dir: str = "/vol/train_protected",
    checkpoint_path: str = "/vol/fastprotect_model/checkpoint_step25000.pt",
    num_images: int = 100,
    strength: float = 0.6,
):
    """
    train_imagesの一部にFastProtect保護を適用

    Args:
        source_dir: 元画像ディレクトリ
        output_dir: 保護済み画像出力先
        checkpoint_path: FastProtectチェックポイント
        num_images: 保護する画像数
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

    # FastProtectPerturbationsクラス（簡略版）
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
    kmeans_path = "/vol/fastprotect_model/kmeans_model.pkl"
    with open(kmeans_path, "rb") as f:
        kmeans = pickle.load(f)

    print("[FastProtect] Loading target entropies...")
    entropies_path = "/vol/fastprotect_model/target_entropies.json"
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
    image_files = list(source_path.glob("*.jpg")) + list(source_path.glob("*.png"))
    image_files = sorted(image_files)[:num_images]

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

            # 512x512にリサイズ（学習時サイズ）
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

            # グローバル強度適用（簡略版：Adaptive Protectionなし）
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

    fastprotect_vol.commit()
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
    volumes={"/vol": fastprotect_vol},
)
def train_lora(
    train_dir: str,
    output_name: str,
    steps: int = 500,
):
    """
    LoRA学習を実行

    Args:
        train_dir: 学習画像ディレクトリ
        output_name: 出力LoRA名
        steps: 学習ステップ数
    """
    import subprocess
    import json
    from pathlib import Path
    import time

    print(f"[LoRA] Training from: {train_dir}")
    print(f"[LoRA] Output: {output_name}")
    print(f"[LoRA] Steps: {steps}")

    # 出力ディレクトリ
    output_dir = f"/vol/lora_results/{output_name}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ステータスファイル書き込み
    status_file = Path("/vol") / f"train_status_{output_name}.json"
    def write_status(status_dict):
        status_file.write_text(json.dumps(status_dict, indent=2, default=str))
        fastprotect_vol.commit()

    write_status({"status": "running", "start_time": time.time(), "steps": 0})

    # 学習設定
    config = {
        "pretrained_model_name_or_path": "stabilityai/stable-diffusion-xl-base-1.0",
        "output_dir": output_dir,
        "train_data_dir": train_dir,
        "resolution": "512,512",
        "learning_rate": 1e-4,
        "max_train_steps": steps,
        "save_every_n_steps": steps,  # 最後だけ保存
        "train_batch_size": 1,
        "mixed_precision": "fp16",
        "network_module": "networks.lora",
        "network_dim": 32,
        "network_alpha": 16,
        "optimizer_type": "AdamW8bit",
        "lr_scheduler": "cosine",
        "output_name": output_name,
        "no_metadata": True,  # メタデータ不要
    }

    # kohya学習コマンド
    cmd = [
        "python", "/sd-scripts/sdxl_train_network.py",
        f"--pretrained_model_name_or_path={config['pretrained_model_name_or_path']}",
        f"--output_dir={config['output_dir']}",
        f"--train_data_dir={config['train_data_dir']}",
        f"--resolution={config['resolution']}",
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
    ]

    print("[LoRA] Starting training...")
    print(f"[LoRA] Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=14000,
        )

        if result.returncode == 0:
            write_status({
                "status": "completed",
                "end_time": time.time(),
                "output_dir": output_dir,
            })
            print("[LoRA] Training completed successfully!")
            return {"status": "completed", "output_dir": output_dir}
        else:
            error_msg = result.stderr[-2000:]  # 最後2000文字
            write_status({
                "status": "failed",
                "error": error_msg,
            })
            print(f"[LoRA] Training failed: {error_msg}")
            return {"status": "failed", "error": error_msg}

    except Exception as e:
        write_status({"status": "failed", "error": str(e)})
        print(f"[LoRA] Exception: {e}")
        return {"status": "failed", "error": str(e)}


@app.local_entrypoint()
def main(
    protect: bool = False,
    train_protected: bool = False,
    train_original: bool = False,
    status: bool = False,
    strength: float = 0.6,
    num_images: int = 100,
    steps: int = 500,
):
    """
    FastProtect LoRA Learning Test

    Examples:
        # 画像を保護
        modal run scripts/modal_fastprotect_lora_test.py --protect --strength 0.6 --num-images 100

        # LoRA学習（保護済み）
        modal run scripts/modal_fastprotect_lora_test.py --train-protected --steps 500

        # LoRA学習（オリジナル・比較用）
        modal run scripts/modal_fastprotect_lora_test.py --train-original --steps 500

        # ステータス確認
        modal run scripts/modal_fastprotect_lora_test.py --status
    """
    if protect:
        print("=== Step 1: Protect Images ===")
        result = protect_images.remote(
            num_images=num_images,
            strength=strength,
        )
        print(f"\n[Result] {result}")

    elif train_protected:
        print("=== Step 2: Train LoRA (Protected) ===")
        result = train_lora.spawn(
            train_dir="/vol/train_protected",
            output_name=f"protected_s{strength}",
            steps=steps,
        )
        print(f"\n[Spawned] Call ID: {result.object_id}")
        print("Use --status to check progress")

    elif train_original:
        print("=== Step 2: Train LoRA (Original) ===")
        result = train_lora.spawn(
            train_dir="/vol/train_images",
            output_name="original",
            steps=steps,
        )
        print(f"\n[Spawned] Call ID: {result.object_id}")
        print("Use --status to check progress")

    elif status:
        print("=== Status Check ===")
        import subprocess
        import json

        for name in ["protected_s0.6", "original"]:
            status_file = f"train_status_{name}.json"
            temp_path = f"/tmp/{status_file}"

            try:
                result = subprocess.run(
                    ["modal", "volume", "get", "fastprotect-vol", status_file, temp_path],
                    capture_output=True,
                    text=True,
                    timeout=30
                )

                if result.returncode == 0:
                    with open(temp_path, 'r') as f:
                        status_data = json.load(f)
                    print(f"\n[{name}]")
                    print(f"  Status: {status_data.get('status', 'unknown')}")
                    if 'start_time' in status_data:
                        print(f"  Start Time: {status_data['start_time']}")
                    if 'end_time' in status_data:
                        print(f"  End Time: {status_data['end_time']}")
                    if 'error' in status_data:
                        print(f"  Error: {status_data['error'][:200]}")
                else:
                    print(f"\n[{name}] No status file found")

            except Exception as e:
                print(f"\n[{name}] Error reading status: {e}")

    else:
        print("Usage:")
        print("  modal run scripts/modal_fastprotect_lora_test.py --protect --strength 0.6")
        print("  modal run scripts/modal_fastprotect_lora_test.py --train-protected")
        print("  modal run scripts/modal_fastprotect_lora_test.py --train-original")
        print("  modal run scripts/modal_fastprotect_lora_test.py --status")
