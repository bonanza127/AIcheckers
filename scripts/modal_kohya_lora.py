#!/usr/bin/env python3
"""
Kohya sd-scripts を使用した LoRA 学習検証
Modal上でIronclad効果を検証

使用方法:
    modal run scripts/modal_kohya_lora.py --setup      # sd-scripts準備
    modal run scripts/modal_kohya_lora.py --train      # LoRA学習
    modal run scripts/modal_kohya_lora.py --evaluate   # 評価
    modal run scripts/modal_kohya_lora.py --all        # 全て実行
"""

import modal
import os
import io
from pathlib import Path

# Modal App
app = modal.App("kohya-lora-test")

# Volume for persistent storage
volume = modal.Volume.from_name("ironclad-test-vol", create_if_missing=True)
VOLUME_PATH = "/vol"
STATUS_FILE = "job_status.json"


# ==================== Status Management ====================

def write_status(status_data: dict):
    """Write status to Volume (called from within Modal function)"""
    import json
    from pathlib import Path
    try:
        status_path = Path(VOLUME_PATH) / STATUS_FILE
        status_path.write_text(json.dumps(status_data, indent=2, default=str))
        volume.commit()
        print(f"[STATUS] Written: {status_data.get('status', 'unknown')}")
    except Exception as e:
        print(f"[STATUS ERROR] Failed to write status: {e}")


def read_status_local() -> dict:
    """Read status from Volume (called from local entrypoint)"""
    import json
    import subprocess
    import os

    temp_path = "/tmp/modal_job_status.json"

    try:
        # modal volume get で status.json をダウンロード
        result = subprocess.run(
            ["modal", "volume", "get", "ironclad-test-vol", STATUS_FILE, temp_path],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode != 0:
            return {"status": "no_job", "message": f"modal volume get failed: {result.stderr}"}

        if not os.path.exists(temp_path):
            return {"status": "no_job", "message": "Downloaded file not found"}

        with open(temp_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# Kohya sd-scripts用イメージ
kohya_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "git", "wget", "libgl1-mesa-glx", "libglib2.0-0",
        "libsm6", "libxext6", "libxrender-dev"
    )
    .run_commands(
        # sd-scriptsをクローン
        "git clone https://github.com/kohya-ss/sd-scripts.git /sd-scripts",
        "cd /sd-scripts && git checkout v0.8.7",  # 安定版
        # torch系をCUDA 12.1でインストール
        "pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        # numpy固定（sd-scriptsは2.0未満が必要）
        "numpy<2.0",
        # 学習用パッケージ
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
        "pytorch-lightning",
        "omegaconf",
        "lion-pytorch",
        "prodigyopt",
        "toml",
        "voluptuous",
        "open-clip-torch",
        # sd-scripts追加依存
        "imagesize",
        "rich",
        "wandb",
    )
    .run_commands(
        # sd-scriptsの依存関係インストール
        "cd /sd-scripts && pip install -e . --no-deps",
    )
)


# ==================== Debug Test ====================

@app.function(
    image=kohya_image,
    volumes={VOLUME_PATH: volume},
    gpu="A10G",
    timeout=300,
)
def debug_test():
    """デバッグ用テスト関数"""
    from pathlib import Path
    from datetime import datetime

    print("=== DEBUG TEST START ===")

    # 1. ボリューム確認
    vol_path = Path(VOLUME_PATH)
    print(f"Volume path exists: {vol_path.exists()}")
    print(f"Volume contents: {list(vol_path.iterdir())[:10]}")

    # 2. 訓練データ確認
    train_dir = vol_path / "train_sap_v3_variants"
    if train_dir.exists():
        pngs = list(train_dir.glob("*.png"))
        txts = list(train_dir.glob("*.txt"))
        print(f"train_sap_v3_variants: {len(pngs)} PNGs, {len(txts)} TXTs")
    else:
        print(f"train_sap_v3_variants NOT FOUND")

    # 3. ステータス書き込みテスト
    print("Testing write_status...")
    write_status({
        "status": "debug_test",
        "timestamp": datetime.now().isoformat(),
        "message": "Debug test successful"
    })

    print("=== DEBUG TEST END ===")
    return {"status": "ok"}


# ==================== Setup ====================

@app.function(
    image=kohya_image,
    volumes={VOLUME_PATH: volume},
    gpu="A10G",
    timeout=3600,
)
def setup_kohya():
    """Kohya環境の確認とキャッシュ準備"""
    import subprocess
    import torch

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    # accelerate設定
    accelerate_config = """
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: 'NO'
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
"""
    config_dir = Path.home() / ".cache" / "huggingface" / "accelerate"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "default_config.yaml").write_text(accelerate_config)
    print("Accelerate config created")

    # モデルの確認
    model_path = Path(VOLUME_PATH) / "models" / "fnevonoobxl_v20.safetensors"
    if model_path.exists():
        size_gb = model_path.stat().st_size / 1e9
        print(f"Base model found: {model_path} ({size_gb:.2f} GB)")
    else:
        print(f"WARNING: Base model not found at {model_path}")

    # 訓練データの確認
    for data_type in ["train_normal", "train_ironclad"]:
        data_dir = Path(VOLUME_PATH) / data_type
        if data_dir.exists():
            images = list(data_dir.glob("*.png"))
            print(f"{data_type}: {len(images)} images")
        else:
            print(f"WARNING: {data_type} not found")

    volume.commit()
    return {"status": "ready"}


# ==================== Training ====================

@app.function(
    image=kohya_image,
    volumes={VOLUME_PATH: volume},
    gpu="A10G",
    timeout=7200,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def train_lora_kohya(
    training_dir: str,
    output_name: str,
    max_train_epochs: int = 3,
    network_dim: int = 32,
    network_alpha: int = 16,
    learning_rate: float = 2e-4,
):
    """Kohya sd-scriptsでLoRA学習"""
    import subprocess
    import toml
    from pathlib import Path
    from datetime import datetime

    # ステータス: 開始
    write_status({
        "status": "running",
        "output_name": output_name,
        "training_dir": training_dir,
        "started_at": datetime.now().isoformat(),
        "completed_at": None,
        "result": None,
    })

    model_path = Path(VOLUME_PATH) / "models" / "fnevonoobxl_v20.safetensors"
    train_dir = Path(training_dir)
    output_dir = Path(VOLUME_PATH) / "loras" / output_name

    if not model_path.exists():
        write_status({
            "status": "failed",
            "output_name": output_name,
            "training_dir": training_dir,
            "started_at": datetime.now().isoformat(),
            "completed_at": datetime.now().isoformat(),
            "result": {"error": f"Model not found: {model_path}"},
        })
        raise FileNotFoundError(f"Model not found: {model_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # 画像数を確認
    images = list(train_dir.glob("*.png"))
    num_images = len(images)
    print(f"Training images: {num_images}")

    # repeatsを計算（目標: 約2000ステップ）
    target_steps = 2000
    batch_size = 1
    repeats = max(1, (target_steps * batch_size) // (num_images * max_train_epochs))
    print(f"Calculated repeats: {repeats}")

    # データセット設定ファイル
    dataset_config = {
        "general": {
            "shuffle_caption": False,  # cache_text_encoder_outputsと競合するためFalse
            "keep_tokens": 1,
        },
        "datasets": [{
            "resolution": 1024,
            "batch_size": batch_size,
            "enable_bucket": True,
            "min_bucket_reso": 512,
            "max_bucket_reso": 1536,
            "bucket_reso_steps": 64,
            "subsets": [{
                "image_dir": str(train_dir),
                "num_repeats": repeats,
                "caption_extension": ".txt",
            }]
        }]
    }

    dataset_config_path = output_dir / "dataset_config.toml"
    with open(dataset_config_path, "w") as f:
        toml.dump(dataset_config, f)

    # 学習コマンド
    cmd = [
        "accelerate", "launch",
        "--num_cpu_threads_per_process=4",
        "/sd-scripts/sdxl_train_network.py",
        f"--pretrained_model_name_or_path={model_path}",
        f"--dataset_config={dataset_config_path}",
        f"--output_dir={output_dir}",
        f"--output_name={output_name}",
        "--save_model_as=safetensors",
        "--save_precision=fp16",
        f"--max_train_epochs={max_train_epochs}",
        f"--learning_rate={learning_rate}",
        f"--unet_lr={learning_rate}",
        # text_encoder_lr removed: conflicts with cache_text_encoder_outputs
        "--network_train_unet_only",
        "--network_module=networks.lora",
        f"--network_dim={network_dim}",
        f"--network_alpha={network_alpha}",
        "--optimizer_type=AdamW",
        "--lr_scheduler=cosine_with_restarts",
        "--lr_warmup_steps=100",
        "--lr_scheduler_num_cycles=2",
        "--mixed_precision=bf16",
        "--gradient_checkpointing",
        "--xformers",
        "--cache_latents",
        "--cache_latents_to_disk",
        "--cache_text_encoder_outputs",
        "--no_half_vae",
        "--clip_skip=2",
        "--max_token_length=225",
        "--seed=42",
        "--noise_offset=0.05",
        "--min_snr_gamma=5",
        "--logging_dir=/vol/logs",
    ]

    print("Starting training...")
    print(f"Command: {' '.join(cmd)}")
    import sys

    # リアルタイム出力のためPopenを使用
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd="/sd-scripts",
        bufsize=1
    )

    stdout_lines = []
    for line in process.stdout:
        print(line, end='', flush=True)
        stdout_lines.append(line)

    process.wait()
    result_returncode = process.returncode
    result_stdout = ''.join(stdout_lines)

    print(f"\n=== Training finished with return code: {result_returncode} ===")

    # 結果確認
    lora_files = list(output_dir.glob("*.safetensors"))
    if lora_files:
        lora_path = lora_files[0]
        size_mb = lora_path.stat().st_size / 1e6
        print(f"LoRA saved: {lora_path} ({size_mb:.2f} MB)")

        result_data = {
            "status": "success",
            "output_path": str(lora_path),
            "size_mb": size_mb,
        }
        # ステータス: 完了
        write_status({
            "status": "completed",
            "output_name": output_name,
            "training_dir": training_dir,
            "started_at": None,  # 既に書き込み済みなので上書きしない設計もあるが簡略化
            "completed_at": datetime.now().isoformat(),
            "result": result_data,
        })
        return result_data
    else:
        result_data = {
            "status": "failed",
            "returncode": result_returncode,
            "output": result_stdout[-2000:] if result_stdout else "",
        }
        # ステータス: 失敗
        write_status({
            "status": "failed",
            "output_name": output_name,
            "training_dir": training_dir,
            "started_at": None,
            "completed_at": datetime.now().isoformat(),
            "result": result_data,
        })
        return result_data


# ==================== Quick Generation Test ====================

@app.function(
    image=kohya_image,
    volumes={VOLUME_PATH: volume},
    gpu="A10G",
    timeout=1800,
)
def generate_test(lora_name: str = "lora_normal", prompt: str = "", negative: str = ""):
    """単一LoRAで生成テスト"""
    import torch
    from PIL import Image
    from pathlib import Path
    from diffusers import StableDiffusionXLPipeline

    device = torch.device("cuda")
    model_path = Path(VOLUME_PATH) / "models" / "fnevonoobxl_v20.safetensors"
    lora_dir = Path(VOLUME_PATH) / "loras" / lora_name
    lora_files = list(lora_dir.glob("*.safetensors"))

    if not lora_files:
        return {"status": "not_found", "lora_name": lora_name}

    print(f"Loading model and LoRA: {lora_name}")
    pipe = StableDiffusionXLPipeline.from_single_file(
        str(model_path),
        torch_dtype=torch.float16,
    ).to(device)

    pipe.load_lora_weights(str(lora_dir))
    print("LoRA loaded successfully")

    # Default prompts
    if not prompt:
        prompt = "masterpiece, best quality, amazing quality, very aesthetic, absurdres, newest, 1girl, anime style illustration"
    if not negative:
        negative = "lowres, worst quality, bad quality, bad hands, bad feet"

    print(f"Generating: {prompt}")
    print(f"Negative: {negative}")

    images = pipe(
        prompt=prompt,
        negative_prompt=negative,
        num_inference_steps=28,
        guidance_scale=7.0,
        num_images_per_prompt=2,
        generator=torch.Generator(device).manual_seed(42),
    ).images

    save_dir = Path(VOLUME_PATH) / "generated" / lora_name
    save_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(images):
        img.save(save_dir / f"test_{i}.png")
        print(f"Saved: {save_dir / f'test_{i}.png'}")

    volume.commit()
    return {"status": "success", "saved_to": str(save_dir), "count": len(images)}


# ==================== Evaluation ====================

@app.function(
    image=kohya_image,
    volumes={VOLUME_PATH: volume},
    gpu="A10G",
    timeout=3600,
)
def evaluate_loras():
    """両LoRAで生成し、CLIP類似度で比較"""
    import torch
    import numpy as np
    from PIL import Image
    from pathlib import Path
    from diffusers import StableDiffusionXLPipeline
    import open_clip

    device = torch.device("cuda")

    # CLIP
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='laion2b_s34b_b79k'
    )
    clip_model = clip_model.to(device).eval()
    tokenizer = open_clip.get_tokenizer('ViT-B-32')

    # Base model
    model_path = Path(VOLUME_PATH) / "models" / "fnevonoobxl_v20.safetensors"

    prompts = [
        "anime girl, detailed illustration, high quality",
        "anime character, same art style, detailed",
        "illustration in the same style, anime, detailed",
    ]

    results = {}

    for lora_type in ["lora_normal", "lora_hf_stealth"]:
        lora_dir = Path(VOLUME_PATH) / "loras" / lora_type
        lora_files = list(lora_dir.glob("*.safetensors"))

        if not lora_files:
            print(f"LoRA not found for {lora_type}")
            results[lora_type] = {"status": "not_found"}
            continue

        lora_path = lora_files[0]
        print(f"\n=== Evaluating {lora_type} ===")
        print(f"LoRA: {lora_path}")

        # Load pipeline
        pipe = StableDiffusionXLPipeline.from_single_file(
            str(model_path),
            torch_dtype=torch.float16,
        ).to(device)

        # Load LoRA
        try:
            pipe.load_lora_weights(str(lora_dir))
            print("LoRA loaded successfully")
        except Exception as e:
            print(f"Failed to load LoRA: {e}")
            results[lora_type] = {"status": "load_failed", "error": str(e)}
            del pipe
            torch.cuda.empty_cache()
            continue

        clip_scores = []
        for i, prompt in enumerate(prompts):
            print(f"Generating: {prompt[:50]}...")

            images = pipe(
                prompt=prompt,
                negative_prompt="lowres, bad anatomy, bad hands",
                num_inference_steps=25,
                guidance_scale=7.0,
                num_images_per_prompt=2,
                generator=torch.Generator(device).manual_seed(42 + i),
            ).images

            # CLIP score
            for img in images:
                img_tensor = preprocess(img).unsqueeze(0).to(device)
                text_tokens = tokenizer([prompt]).to(device)

                with torch.no_grad():
                    img_features = clip_model.encode_image(img_tensor)
                    text_features = clip_model.encode_text(text_tokens)

                    img_features /= img_features.norm(dim=-1, keepdim=True)
                    text_features /= text_features.norm(dim=-1, keepdim=True)

                    similarity = (img_features @ text_features.T).item()
                    clip_scores.append(similarity)

            # 画像保存
            save_dir = Path(VOLUME_PATH) / "generated" / lora_type
            save_dir.mkdir(parents=True, exist_ok=True)
            for j, img in enumerate(images):
                img.save(save_dir / f"prompt{i}_img{j}.png")

        avg_clip = np.mean(clip_scores) if clip_scores else 0
        results[lora_type] = {
            "status": "success",
            "avg_clip_score": float(avg_clip),
            "clip_scores": clip_scores,
        }
        print(f"Average CLIP score: {avg_clip:.4f}")

        del pipe
        torch.cuda.empty_cache()

    # 比較
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)

    if "lora_normal" in results and "lora_hf_stealth" in results:
        if results["lora_normal"].get("status") == "success" and \
           results["lora_hf_stealth"].get("status") == "success":
            normal_score = results["lora_normal"]["avg_clip_score"]
            hf_stealth_score = results["lora_hf_stealth"]["avg_clip_score"]
            diff = normal_score - hf_stealth_score
            diff_pct = (diff / normal_score) * 100 if normal_score > 0 else 0

            print(f"Normal LoRA CLIP score:     {normal_score:.4f}")
            print(f"HF Stealth LoRA CLIP score: {hf_stealth_score:.4f}")
            print(f"Difference: {diff:.4f} ({diff_pct:.2f}%)")

            if diff > 0.01:
                print("\n🛡️ HF Stealth attack shows learning degradation effect!")
            else:
                print("\n⚠️ No significant difference detected")

    volume.commit()
    return results


# ==================== Main Entry ====================

@app.local_entrypoint()
def main(
    setup: bool = False,
    train: bool = False,
    evaluate: bool = False,
    all: bool = False,
    generate: str = "",
    train_sap_v3: bool = False,
    # === 新規: 非同期実行用フラグ ===
    submit: str = "",       # ジョブ投入: --submit train_sap_v3_variants:lora_sap_v3
    status: bool = False,   # ステータス確認: --status
    debug: bool = False,    # デバッグテスト: --debug
):
    """
    メインエントリーポイント

    非同期実行 (推奨):
        modal run scripts/modal_kohya_lora.py --submit "train_sap_v3_variants:lora_sap_v3"
        modal run scripts/modal_kohya_lora.py --status

    同期実行 (タイムアウトの可能性あり):
        modal run scripts/modal_kohya_lora.py --train-sap-v3

    デバッグ:
        modal run scripts/modal_kohya_lora.py --debug
    """

    # === デバッグテスト ===
    if debug:
        print("\n=== Running Debug Test ===")
        result = debug_test.remote()
        print(f"Result: {result}")
        return

    # === ステータス確認 (即座に戻る) ===
    if status:
        status_data = read_status_local()
        print("\n" + "=" * 60)
        print("JOB STATUS")
        print("=" * 60)
        import json
        print(json.dumps(status_data, indent=2, ensure_ascii=False))
        return

    # === 非同期ジョブ投入 (spawn使用、即座に戻る) ===
    if submit:
        parts = submit.split(":")
        if len(parts) != 2:
            print("ERROR: --submit format: 'training_dir:output_name'")
            print("Example: --submit train_sap_v3_variants:lora_sap_v3")
            return

        training_dir_name, output_name = parts
        training_dir = f"{VOLUME_PATH}/{training_dir_name}"

        print("\n" + "=" * 60)
        print("SUBMITTING JOB (async)")
        print("=" * 60)
        print(f"Training dir: {training_dir}")
        print(f"Output name:  {output_name}")

        # spawn() で非同期実行 - 即座に戻る
        function_call = train_lora_kohya.spawn(
            training_dir=training_dir,
            output_name=output_name,
        )

        print(f"\nJob submitted!")
        print(f"Function call ID: {function_call.object_id}")
        print("\nCheck status with:")
        print("  modal run scripts/modal_kohya_lora.py --status")
        return

    # === 以下、従来の同期実行 (タイムアウトの可能性あり) ===

    if generate:
        print(f"\n=== Generating with {generate} ===")
        result = generate_test.remote(lora_name=generate)
        print(f"Result: {result}")
        return

    # Train SAP v3 variants LoRA only (同期版 - 非推奨)
    if train_sap_v3:
        print("\n" + "=" * 60)
        print("Training SAP v3 Variants LoRA (SYNC - may timeout)")
        print("=" * 60)
        print("WARNING: Use --submit for async execution to avoid timeout")
        result = train_lora_kohya.remote(
            training_dir=f"{VOLUME_PATH}/train_sap_v3_variants",
            output_name="lora_sap_v3",
        )
        print(f"SAP v3 LoRA: {result}")
        return

    if all:
        setup = train = evaluate = True

    if setup:
        print("\n" + "=" * 60)
        print("STEP 1: Setup Kohya Environment")
        print("=" * 60)
        result = setup_kohya.remote()
        print(f"Result: {result}")

    if train:
        print("\n" + "=" * 60)
        print("STEP 2: Train LoRAs")
        print("=" * 60)

        # Train normal LoRA
        print("\n--- Training NORMAL LoRA ---")
        result_normal = train_lora_kohya.remote(
            training_dir=f"{VOLUME_PATH}/train_normal",
            output_name="lora_normal",
        )
        print(f"Normal LoRA: {result_normal}")

        # Train HF Stealth LoRA (エッジ5% + 平坦1%攻撃済み)
        print("\n--- Training HF_STEALTH LoRA ---")
        result_hf_stealth = train_lora_kohya.remote(
            training_dir=f"{VOLUME_PATH}/train_hf_stealth",
            output_name="lora_hf_stealth",
        )
        print(f"HF Stealth LoRA: {result_hf_stealth}")

    if evaluate:
        print("\n" + "=" * 60)
        print("STEP 3: Evaluate")
        print("=" * 60)
        results = evaluate_loras.remote()
        print(f"Results: {results}")


if __name__ == "__main__":
    print("Use: modal run scripts/modal_kohya_lora.py --help")
