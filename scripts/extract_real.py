#!/usr/bin/env python3
"""
Real画像のembedding抽出（5万枚サンプル）
"""
import os
import numpy as np
import torch
import random
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

# 設定
DINOV3_MODEL = "facebook/dinov3-vitb16-pretrain-lvd1689m"
HF_TOKEN = "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"
EMBEDDINGS_DIR = Path("/home/techne/aicheckers/embeddings")
REAL_DIR = Path("/home/techne/aicheckers/data/animedl2m_dataset_release/real_images/images")
BATCH_SIZE = 32
SAMPLE_SIZE = 50000

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    processor = AutoImageProcessor.from_pretrained(DINOV3_MODEL, token=HF_TOKEN)
    model = AutoModel.from_pretrained(DINOV3_MODEL, token=HF_TOKEN)
    model.to(device)
    model.eval()
    return model, processor, device

def main():
    EMBEDDINGS_DIR.mkdir(exist_ok=True)

    output_path = EMBEDDINGS_DIR / "danbooru_real.npy"
    if output_path.exists():
        print(f"Already exists: {output_path}")
        return

    # 画像ファイル収集
    print("Collecting image files...")
    extensions = {'.jpg', '.jpeg', '.png', '.webp', '.gif'}
    all_files = [f for f in REAL_DIR.iterdir()
                 if f.suffix.lower() in extensions]

    print(f"Found {len(all_files)} images")

    # ランダムサンプル
    random.seed(42)
    if len(all_files) > SAMPLE_SIZE:
        files = random.sample(all_files, SAMPLE_SIZE)
    else:
        files = all_files

    print(f"Using {len(files)} images")

    # モデルロード
    model, processor, device = load_model()

    # 抽出
    embeddings = []
    filenames = []

    for i in tqdm(range(0, len(files), BATCH_SIZE), desc="Extracting"):
        batch_files = files[i:i+BATCH_SIZE]
        batch_images = []
        batch_names = []

        for f in batch_files:
            try:
                img = Image.open(f).convert("RGB")
                batch_images.append(img)
                batch_names.append(f.name)
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

    embeddings = np.vstack(embeddings)

    # 保存
    np.save(output_path, embeddings)
    with open(EMBEDDINGS_DIR / "danbooru_real_files.txt", "w") as f:
        f.write("\n".join(filenames))

    print(f"Saved {len(embeddings)} embeddings to {output_path}")

if __name__ == "__main__":
    main()
