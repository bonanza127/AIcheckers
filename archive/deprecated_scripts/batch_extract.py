#!/usr/bin/env python3
"""
バッチでembedding抽出
複数ディレクトリから指定枚数を抽出
"""
import os
import sys
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel
import random

# 設定
DINOV3_MODEL = "facebook/dinov3-vitb16-pretrain-lvd1689m"
HF_TOKEN = "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"
EMBEDDINGS_DIR = Path("/home/techne/aicheckers/embeddings")
BATCH_SIZE = 32

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    processor = AutoImageProcessor.from_pretrained(DINOV3_MODEL, token=HF_TOKEN)
    model = AutoModel.from_pretrained(DINOV3_MODEL, token=HF_TOKEN)
    model.to(device)
    model.eval()
    return model, processor, device

def get_image_files(directory: Path, limit: int = None):
    """画像ファイル一覧を取得（limit指定時はランダムサンプル）"""
    extensions = {'.jpg', '.jpeg', '.png', '.webp', '.gif'}
    files = [f for f in directory.iterdir() if f.suffix.lower() in extensions]
    if limit and len(files) > limit:
        random.seed(42)
        files = random.sample(files, limit)
    return files

def extract_embeddings(image_files, model, processor, device):
    """画像リストからembedding抽出"""
    embeddings = []
    filenames = []

    for i in tqdm(range(0, len(image_files), BATCH_SIZE), desc="Extracting"):
        batch_files = image_files[i:i+BATCH_SIZE]
        batch_images = []
        batch_names = []

        for f in batch_files:
            try:
                img = Image.open(f).convert("RGB")
                batch_images.append(img)
                batch_names.append(str(f))
            except Exception as e:
                continue

        if not batch_images:
            continue

        inputs = processor(images=batch_images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            batch_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        embeddings.append(batch_emb)
        filenames.extend(batch_names)

    return np.vstack(embeddings), filenames

def main():
    EMBEDDINGS_DIR.mkdir(exist_ok=True)
    model, processor, device = load_model()

    # AI画像の抽出設定
    CIVITAI_BASE = Path("/home/techne/aicheckers/data/animedl2m_dataset_release/civitai_subset/image")

    ai_configs = [
        # (dir_name, output_name, limit)
        ("Pony", "pony_ai", None),              # 19,882枚全部
        ("SDXL 1.0", "sdxl10_ai", None),        # 8,924枚全部
        ("SD 1.5", "sd15_ai", 10000),           # 10,000枚サンプル
        ("Other", "other_ai", None),            # 4,555枚全部
        ("Flux.1 D", "flux1d_ai", None),        # 1,849枚全部
    ]

    for dir_name, output_name, limit in ai_configs:
        emb_path = EMBEDDINGS_DIR / f"{output_name}.npy"
        if emb_path.exists():
            print(f"Skipping {output_name} (already exists)")
            continue

        dir_path = CIVITAI_BASE / dir_name
        if not dir_path.exists():
            print(f"Skipping {dir_name} (not found)")
            continue

        print(f"\n{'='*50}")
        print(f"Processing: {dir_name} -> {output_name}")
        print(f"{'='*50}")

        files = get_image_files(dir_path, limit)
        print(f"Found {len(files)} images")

        embeddings, filenames = extract_embeddings(files, model, processor, device)

        np.save(emb_path, embeddings)
        with open(EMBEDDINGS_DIR / f"{output_name}_files.txt", "w") as f:
            f.write("\n".join(filenames))

        print(f"Saved {len(embeddings)} embeddings to {emb_path}")

    print("\n" + "="*50)
    print("AI extraction complete!")
    print("="*50)

if __name__ == "__main__":
    main()
