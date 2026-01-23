#!/usr/bin/env python3
"""
Ironclad V3.1 学習阻害効果検証
Modal上でLoRA学習を実行し、通常版とIronclad版の品質を比較

使用方法:
    modal run scripts/modal_lora_test.py --setup      # ベースモデルDL
    modal run scripts/modal_lora_test.py --prepare    # 訓練データ準備
    modal run scripts/modal_lora_test.py --train      # LoRA学習
    modal run scripts/modal_lora_test.py --evaluate   # 評価
    modal run scripts/modal_lora_test.py --all        # 全て実行
"""

import modal
import os
import io
import time
from pathlib import Path

# Modal App
app = modal.App("ironclad-lora-test")

# Volume for persistent storage
volume = modal.Volume.from_name("ironclad-test-vol", create_if_missing=True)
VOLUME_PATH = "/vol"

# Base image with ML dependencies
base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0", "git", "wget", "aria2")
    .pip_install(
        "numpy<2",  # NumPy 1.x for torch compatibility
        "torch==2.1.2",
        "torchvision==0.16.2",
        "huggingface_hub==0.21.4",  # Pin for diffusers 0.25.1 compatibility
        "diffusers==0.25.1",
        "transformers==4.38.2",  # Pin compatible version
        "accelerate==0.27.2",
        "peft==0.8.2",
        "safetensors",
        "Pillow",
        "xformers==0.0.23.post1",  # Match torch 2.1.2
        "bitsandbytes",
        "scipy",
        "ftfy",
        "regex",
        "omegaconf",  # Required for SDXL single file loading
        # Ironclad dependencies
        "ptwt",
        "kornia",
        "imagehash",
        # Evaluation
        "open-clip-torch",
    )
)

# ==================== Setup ====================

@app.function(
    image=base_image,
    volumes={VOLUME_PATH: volume},
    timeout=7200,  # 2時間に延長
    secrets=[modal.Secret.from_name("civitai-secret")],
)
def setup_base_model():
    """fnevonoobxl v20をCivitAIからダウンロード"""
    import subprocess

    model_dir = Path(VOLUME_PATH) / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "fnevonoobxl_v20.safetensors"
    expected_size_gb = 6.4  # 期待されるファイルサイズ

    if model_path.exists():
        size_gb = model_path.stat().st_size / 1e9
        print(f"Model exists: {model_path}")
        print(f"Size: {size_gb:.2f} GB")
        # サイズが小さすぎる場合は壊れているので削除
        if size_gb < expected_size_gb * 0.9:
            print(f"File appears corrupted (expected ~{expected_size_gb}GB), removing...")
            model_path.unlink()
        else:
            volume.commit()
            return {"status": "exists", "path": str(model_path), "size_gb": size_gb}

    # CivitAI download URL (fnevonoobxl v20)
    # Model: https://civitai.com/models/1639407?modelVersionId=2138507
    civitai_token = os.environ.get("CIVITAI_API_TOKEN", "")
    civitai_url = f"https://civitai.com/api/download/models/2138507?token={civitai_token}"

    print(f"Downloading fnevonoobxl v20...")
    print(f"URL: https://civitai.com/api/download/models/2138507 (with token)")

    # Use aria2c for faster download with retry
    result = subprocess.run([
        "aria2c", "-x", "16", "-s", "16", "-k", "1M",
        "--max-tries=5", "--retry-wait=10", "--timeout=600",
        "-o", str(model_path),
        civitai_url
    ], capture_output=True, text=True)

    print(f"aria2c stdout: {result.stdout[-2000:] if result.stdout else 'empty'}")
    print(f"aria2c stderr: {result.stderr[-500:] if result.stderr else 'empty'}")

    # aria2c成功判定: 出力に "Download complete" が含まれているか
    aria2c_success = "Download complete" in (result.stdout or "") or result.returncode == 0
    print(f"aria2c returncode: {result.returncode}, success indicator: {aria2c_success}")

    # ファイルシステム同期のために少し待つ
    import time
    time.sleep(2)

    # aria2c成功判定: ファイルが存在し、サイズが十分か
    if model_path.exists():
        size_gb = model_path.stat().st_size / 1e9
        print(f"File exists after aria2c, size: {size_gb:.2f} GB")
        if size_gb >= expected_size_gb * 0.9:
            print(f"aria2c download successful: {size_gb:.2f} GB")
            volume.commit()
            return {"status": "downloaded", "path": str(model_path), "size_gb": size_gb}
        else:
            print(f"aria2c produced incomplete file: {size_gb:.2f} GB, removing...")
            model_path.unlink()
    else:
        print(f"File does not exist after aria2c: {model_path}")

    # aria2c失敗時のみwgetを試行
    if aria2c_success:
        # aria2cが成功したように見えるのにファイルがない場合、パス問題かもしれない
        print(f"WARNING: aria2c reported success but file not found. Checking alternative paths...")
        possible_paths = [
            model_dir / "fnevonoobxl_v20.safetensors",
            Path("/vol/models/fnevonoobxl_v20.safetensors"),
            Path("/root/vol/models/fnevonoobxl_v20.safetensors"),
        ]
        for p in possible_paths:
            if p.exists():
                size_gb = p.stat().st_size / 1e9
                print(f"Found at {p}, size: {size_gb:.2f} GB")
                if size_gb >= expected_size_gb * 0.9:
                    if p != model_path:
                        import shutil
                        shutil.move(str(p), str(model_path))
                    volume.commit()
                    return {"status": "downloaded", "path": str(model_path), "size_gb": size_gb}

    print(f"aria2c failed, trying wget with header auth...")
    result = subprocess.run([
        "wget", "--tries=3", "--timeout=300",
        "--header", f"Authorization: Bearer {civitai_token}",
        "-O", str(model_path),
        "https://civitai.com/api/download/models/2138507"
    ], capture_output=True, text=True)
    print(f"wget stdout: {result.stdout[-500:] if result.stdout else 'empty'}")
    print(f"wget stderr: {result.stderr[-500:] if result.stderr else 'empty'}")

    if model_path.exists():
        size_gb = model_path.stat().st_size / 1e9
        print(f"Download complete: {size_gb:.2f} GB")
        if size_gb < expected_size_gb * 0.9:
            return {"status": "failed", "error": f"Downloaded file too small: {size_gb:.2f}GB"}
        volume.commit()
        return {"status": "downloaded", "path": str(model_path), "size_gb": size_gb}
    else:
        return {"status": "failed", "error": result.stderr}


# ==================== Ironclad Poisoning ====================

@app.function(
    image=base_image,
    gpu="T4",
    timeout=1200,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def apply_ironclad(image_bytes: bytes, strength_mid: float = 0.08, iterations: int = 50) -> bytes:
    """画像にIroncladポイズニングを適用"""
    import hashlib
    import hmac
    from datetime import datetime

    import numpy as np
    import torch
    import torch.nn.functional as F
    import ptwt
    import kornia
    import imagehash
    from PIL import Image
    from diffusers import AutoencoderKL

    device = torch.device("cuda")

    # Load image
    img_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    orig_size = img_pil.size

    # To tensor
    arr = np.array(img_pil).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)

    # Resize to multiple of 8
    _, _, h, w = img_tensor.shape
    new_h = (h // 8) * 8
    new_w = (w // 8) * 8
    if new_h != h or new_w != w:
        img_tensor = F.interpolate(img_tensor, (new_h, new_w), mode='bilinear', align_corners=False)

    # Load VAE
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
        torch_dtype=torch.float32
    ).eval().to(device)

    # YCbCr conversion
    img_ycbcr = kornia.color.rgb_to_ycbcr(img_tensor)
    Y, Cb, Cr = torch.chunk(img_ycbcr, 3, dim=1)

    # Perceptual mask
    edges = kornia.filters.sobel(Y)
    edge_magnitude = edges.abs().mean(dim=1, keepdim=True)
    mask = 0.2 + 0.8 * torch.sigmoid((edge_magnitude - 0.1) * 20)

    # DWT
    WAVELET = 'bior1.3'
    coeffs = ptwt.wavedec2(Y, WAVELET, level=1)
    LL = coeffs[0]
    LH, HL, HH = coeffs[1]

    mask_mid = F.interpolate(mask, size=LH.shape[-2:], mode='bilinear', align_corners=False)

    # Signature
    version = datetime.now().strftime("%Y%m")
    secret_key = f"AICHECKERS_DEFAULT_KEY_{version}".encode()

    Y_np = Y.squeeze(0).squeeze(0).cpu().numpy()
    Y_pil = Image.fromarray((Y_np * 255).clip(0, 255).astype(np.uint8), mode='L')
    phash = imagehash.phash(Y_pil, hash_size=8)
    image_salt = str(phash)

    key = hmac.new(secret_key, image_salt.encode(), hashlib.sha256).digest()
    seed = int.from_bytes(key[:8], 'big')
    rng = torch.Generator().manual_seed(seed)
    sig_pattern = (torch.rand(LH.shape, generator=rng) * 2 - 1).to(device)

    # Untargeted semantic attack
    canonical_size = 512
    strength_low = 0.06

    with torch.no_grad():
        upscaled_orig = F.interpolate(LL, size=(canonical_size, canonical_size), mode='bilinear', align_corners=False)
        rgb_orig = upscaled_orig.repeat(1, 3, 1, 1)
        rgb_orig_scaled = rgb_orig * 2.0 - 1.0
        target_latent = vae.encode(rgb_orig_scaled).latent_dist.mean

    optimized_LL = LL.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([optimized_LL], lr=0.01)

    for i in range(iterations):
        upscaled_opt = F.interpolate(optimized_LL, size=(canonical_size, canonical_size), mode='bilinear', align_corners=False)
        rgb_opt = upscaled_opt.repeat(1, 3, 1, 1)
        rgb_opt_scaled = rgb_opt * 2.0 - 1.0
        current_latent = vae.encode(rgb_opt_scaled).latent_dist.mean

        loss = F.cosine_similarity(current_latent.flatten(), target_latent.flatten(), dim=0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            perturbation = optimized_LL - LL
            perturbation = torch.clamp(perturbation, -strength_low, strength_low)
            optimized_LL.data = LL + perturbation

    LL_poisoned = optimized_LL.detach()

    # Signature embedding
    perturbation_mid = sig_pattern * strength_mid * mask_mid
    LH_poisoned = LH + perturbation_mid
    HL_poisoned = HL + perturbation_mid
    HH_poisoned = HH + (torch.randn_like(HH) * 0.02)

    # Inverse DWT
    Y_poisoned = ptwt.waverec2([LL_poisoned, (LH_poisoned, HL_poisoned, HH_poisoned)], WAVELET)
    if Y_poisoned.shape != Y.shape:
        Y_poisoned = Y_poisoned[:, :, :Y.shape[2], :Y.shape[3]]

    # YCbCr to RGB
    poisoned_ycbcr = torch.cat([Y_poisoned, Cb, Cr], dim=1)
    poisoned_rgb = kornia.color.ycbcr_to_rgb(poisoned_ycbcr)

    # Resize back
    if poisoned_rgb.shape[-2:] != (h, w):
        poisoned_rgb = F.interpolate(poisoned_rgb, (h, w), mode='bilinear', align_corners=False)

    poisoned_rgb = torch.clamp(poisoned_rgb, 0, 1)

    # To PIL
    result_arr = poisoned_rgb.squeeze(0).permute(1, 2, 0).cpu().numpy()
    result_arr = (result_arr * 255).clip(0, 255).astype(np.uint8)
    result_pil = Image.fromarray(result_arr)

    # To bytes
    output_buffer = io.BytesIO()
    result_pil.save(output_buffer, format="PNG")
    return output_buffer.getvalue()


@app.function(
    image=base_image,
    volumes={VOLUME_PATH: volume},
    gpu="T4",
    timeout=1800,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def prepare_training_data(image_bytes_list: list[bytes], apply_poison: bool = False):
    """訓練データを準備してVolumeに保存"""
    from PIL import Image

    output_dir = Path(VOLUME_PATH) / ("train_ironclad" if apply_poison else "train_normal")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clear existing files
    for f in output_dir.glob("*"):
        f.unlink()

    results = []
    for i, img_bytes in enumerate(image_bytes_list):
        print(f"Processing image {i+1}/{len(image_bytes_list)}...")

        if apply_poison:
            # Apply Ironclad
            processed_bytes = apply_ironclad.remote(img_bytes)
        else:
            processed_bytes = img_bytes

        # Save image
        img = Image.open(io.BytesIO(processed_bytes)).convert("RGB")
        img_path = output_dir / f"image_{i:03d}.png"
        img.save(img_path, "PNG")

        # Simple caption (can be improved with BLIP)
        caption_path = output_dir / f"image_{i:03d}.txt"
        caption_path.write_text("anime style illustration, detailed, high quality")

        results.append({
            "image": str(img_path),
            "caption": str(caption_path),
        })
        print(f"  Saved: {img_path}")

    volume.commit()
    return {
        "output_dir": str(output_dir),
        "count": len(results),
        "files": results,
    }


# ==================== LoRA Training ====================

@app.function(
    image=base_image,
    volumes={VOLUME_PATH: volume},
    gpu="A10G",
    timeout=7200,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def train_lora(
    training_dir: str,
    output_name: str,
    train_steps: int = 800,
    learning_rate: float = 5e-5,  # 学習率を下げた
    network_dim: int = 32,
    network_alpha: int = 16,
):
    """diffusers + PEFTでLoRA学習"""
    import torch
    import numpy as np
    from PIL import Image
    from pathlib import Path
    from diffusers import StableDiffusionXLPipeline
    from peft import LoraConfig, get_peft_model
    import torch.nn.functional as F
    from safetensors.torch import save_file

    device = torch.device("cuda")

    # Model path
    model_path = Path(VOLUME_PATH) / "models" / "fnevonoobxl_v20.safetensors"
    if not model_path.exists():
        raise FileNotFoundError(f"Base model not found: {model_path}")

    print(f"Loading base model: {model_path}")

    # Load pipeline
    pipe = StableDiffusionXLPipeline.from_single_file(
        str(model_path),
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).to(device)

    # Setup LoRA
    lora_config = LoraConfig(
        r=network_dim,
        lora_alpha=network_alpha,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_dropout=0.0,
    )

    pipe.unet = get_peft_model(pipe.unet, lora_config)
    pipe.unet.train()

    # Load training data
    train_dir = Path(training_dir)
    image_files = list(train_dir.glob("*.png"))
    print(f"Found {len(image_files)} training images")

    # Simple training loop
    optimizer = torch.optim.AdamW(pipe.unet.parameters(), lr=learning_rate)
    vae = pipe.vae.to(device)

    # テキストエンコーダーからembedding生成
    text_encoders = [pipe.text_encoder, pipe.text_encoder_2]
    tokenizers = [pipe.tokenizer, pipe.tokenizer_2]

    # 固定のプロンプトからembeddingを生成
    prompt = "anime style illustration, detailed, high quality"
    with torch.no_grad():
        # Tokenize
        text_input_ids_1 = tokenizers[0](
            prompt, padding="max_length", max_length=77,
            truncation=True, return_tensors="pt"
        ).input_ids.to(device)
        text_input_ids_2 = tokenizers[1](
            prompt, padding="max_length", max_length=77,
            truncation=True, return_tensors="pt"
        ).input_ids.to(device)

        # Encode
        encoder_output_1 = text_encoders[0](text_input_ids_1, output_hidden_states=True)
        encoder_output_2 = text_encoders[1](text_input_ids_2, output_hidden_states=True)

        # Get embeddings
        text_embeds_1 = encoder_output_1.hidden_states[-2]  # (1, 77, 768)
        text_embeds_2 = encoder_output_2.hidden_states[-2]  # (1, 77, 1280)

        # Concatenate for SDXL
        encoder_hidden_states = torch.cat([text_embeds_1, text_embeds_2], dim=-1)  # (1, 77, 2048)

        # Pooled embedding for text_embeds
        pooled_output = encoder_output_2[0]  # (1, 1280)

    print(f"Starting training for {train_steps} steps...")
    losses = []

    for step in range(train_steps):
        # Random image
        img_path = image_files[step % len(image_files)]
        img = Image.open(img_path).convert("RGB").resize((1024, 1024))

        # To tensor
        img_tensor = torch.from_numpy(
            np.array(img).astype(np.float32) / 255.0
        ).permute(2, 0, 1).unsqueeze(0).to(device, dtype=torch.float16)

        # Normalize to [-1, 1]
        img_tensor = img_tensor * 2.0 - 1.0

        # VAE encode
        with torch.no_grad():
            latents = vae.encode(img_tensor).latent_dist.sample() * 0.18215

        # Add noise
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, 1000, (1,), device=device).long()
        noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

        # Predict noise with proper embeddings
        added_cond_kwargs = {
            "text_embeds": pooled_output.to(dtype=torch.float16),
            "time_ids": torch.tensor([[1024, 1024, 0, 0, 1024, 1024]], device=device, dtype=torch.float16),
        }

        noise_pred = pipe.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states.to(dtype=torch.float16),
            added_cond_kwargs=added_cond_kwargs,
        ).sample

        # Loss
        loss = F.mse_loss(noise_pred.float(), noise.float())

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(pipe.unet.parameters(), max_norm=1.0)

        optimizer.step()

        if (step + 1) % 100 == 0:
            print(f"Step {step+1}/{train_steps}, Loss: {loss.item():.4f}")
            losses.append(loss.item())

    # Save LoRA weights manually (avoid model_card issue)
    output_dir = Path(VOLUME_PATH) / "loras"
    output_dir.mkdir(parents=True, exist_ok=True)
    lora_dir = output_dir / output_name
    lora_dir.mkdir(parents=True, exist_ok=True)

    # Extract LoRA weights
    lora_state_dict = {}
    for name, param in pipe.unet.named_parameters():
        if "lora" in name.lower():
            lora_state_dict[name] = param.detach().cpu()

    # Save as safetensors
    lora_path = lora_dir / "adapter_model.safetensors"
    save_file(lora_state_dict, str(lora_path))
    print(f"Saved LoRA to {lora_path} ({len(lora_state_dict)} tensors)")

    volume.commit()

    avg_loss = sum(losses) / len(losses) if losses else float('nan')
    return {
        "output_path": str(lora_path),
        "steps": train_steps,
        "final_loss": loss.item() if not torch.isnan(loss) else float('nan'),
        "avg_loss": avg_loss,
    }


# ==================== Evaluation ====================

@app.function(
    image=base_image,
    volumes={VOLUME_PATH: volume},
    gpu="A10G",
    timeout=1800,
)
def evaluate_loras(prompts: list[str], num_images: int = 4):
    """両LoRAで生成し比較"""
    import torch
    import numpy as np
    from PIL import Image
    from diffusers import StableDiffusionXLPipeline
    import open_clip

    device = torch.device("cuda")

    # Load CLIP for evaluation
    clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    clip_model = clip_model.to(device).eval()

    # Load base model
    model_path = Path(VOLUME_PATH) / "models" / "fnevonoobxl_v20.safetensors"

    results = {"normal": [], "ironclad": []}

    for lora_type in ["normal", "ironclad"]:
        lora_dir = Path(VOLUME_PATH) / "loras" / f"lora_{lora_type}"

        print(f"\n=== Evaluating {lora_type} LoRA ===")

        # Load pipeline with LoRA
        pipe = StableDiffusionXLPipeline.from_single_file(
            str(model_path),
            torch_dtype=torch.float16,
        ).to(device)

        if lora_dir.exists():
            pipe.load_lora_weights(str(lora_dir))

        for prompt in prompts:
            print(f"Generating: {prompt[:50]}...")

            images = pipe(
                prompt=prompt,
                num_inference_steps=30,
                guidance_scale=7.5,
                num_images_per_prompt=num_images,
                generator=torch.Generator(device).manual_seed(42),
            ).images

            # Calculate CLIP scores
            clip_scores = []
            for img in images:
                img_tensor = preprocess(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    features = clip_model.encode_image(img_tensor)
                    clip_scores.append(features.cpu().numpy())

            results[lora_type].append({
                "prompt": prompt,
                "clip_features": clip_scores,
            })

    # Save results
    output_dir = Path(VOLUME_PATH) / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compare
    print("\n=== Comparison Results ===")
    for i, prompt in enumerate(prompts):
        normal_feats = np.mean(results["normal"][i]["clip_features"], axis=0)
        ironclad_feats = np.mean(results["ironclad"][i]["clip_features"], axis=0)

        # Cosine similarity
        similarity = np.dot(normal_feats.flatten(), ironclad_feats.flatten()) / (
            np.linalg.norm(normal_feats) * np.linalg.norm(ironclad_feats)
        )
        print(f"Prompt {i+1}: Similarity = {similarity:.4f}")

    volume.commit()
    return results


# ==================== Main Entry ====================

@app.local_entrypoint()
def main(
    setup: bool = False,
    prepare: bool = False,
    train: bool = False,
    evaluate: bool = False,
    all: bool = False,
):
    """メインエントリーポイント"""

    if all:
        setup = prepare = train = evaluate = True

    # Load local images
    image_dir = Path("/home/techne/Desktop/memo")
    image_files = list(image_dir.glob("*"))
    image_files = [f for f in image_files if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]]

    print(f"Found {len(image_files)} images in {image_dir}")

    image_bytes_list = []
    for f in image_files:
        with open(f, "rb") as fp:
            image_bytes_list.append(fp.read())

    if setup:
        print("\n" + "="*60)
        print("STEP 1: Setup Base Model")
        print("="*60)
        result = setup_base_model.remote()
        print(f"Result: {result}")

    if prepare:
        print("\n" + "="*60)
        print("STEP 2: Prepare Training Data")
        print("="*60)

        # Normal version
        print("\n--- Preparing NORMAL training data ---")
        result_normal = prepare_training_data.remote(image_bytes_list, apply_poison=False)
        print(f"Normal: {result_normal}")

        # Ironclad version
        print("\n--- Preparing IRONCLAD training data ---")
        result_ironclad = prepare_training_data.remote(image_bytes_list, apply_poison=True)
        print(f"Ironclad: {result_ironclad}")

    if train:
        print("\n" + "="*60)
        print("STEP 3: Train LoRAs")
        print("="*60)

        # Train normal LoRA
        print("\n--- Training NORMAL LoRA ---")
        result_normal = train_lora.remote(
            training_dir=f"{VOLUME_PATH}/train_normal",
            output_name="lora_normal",
        )
        print(f"Normal LoRA: {result_normal}")

        # Train Ironclad LoRA
        print("\n--- Training IRONCLAD LoRA ---")
        result_ironclad = train_lora.remote(
            training_dir=f"{VOLUME_PATH}/train_ironclad",
            output_name="lora_ironclad",
        )
        print(f"Ironclad LoRA: {result_ironclad}")

    if evaluate:
        print("\n" + "="*60)
        print("STEP 4: Evaluate")
        print("="*60)

        test_prompts = [
            "anime girl, detailed illustration, high quality",
            "anime character, same art style, detailed",
            "illustration in the same style, anime, detailed",
        ]

        results = evaluate_loras.remote(prompts=test_prompts)
        print(f"Evaluation complete")


if __name__ == "__main__":
    print("Use: modal run scripts/modal_lora_test.py --help")
