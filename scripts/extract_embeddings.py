#!/usr/bin/env python3
"""
DINOv3 embedding抽出スクリプト
抽出した特徴量を保存し、後で再利用可能にする
"""
import os
import sys
import argparse
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

# 設定
DINOV3_MODEL = "facebook/dinov3-vitb16-pretrain-lvd1689m"
HF_TOKEN = "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"
EMBEDDINGS_DIR = Path("/home/techne/aicheckers/embeddings")

def load_model():
    """DINOv3モデルをロード"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    processor = AutoImageProcessor.from_pretrained(DINOV3_MODEL, token=HF_TOKEN)
    model = AutoModel.from_pretrained(DINOV3_MODEL, token=HF_TOKEN)
    model.to(device)
    model.eval()

    return model, processor, device

def extract_embeddings(image_dir: Path, model, processor, device, batch_size=32):
    """ディレクトリ内の全画像からembeddingを抽出"""
    # 画像ファイル一覧
    extensions = {'.jpg', '.jpeg', '.png', '.webp', '.gif'}
    image_files = [f for f in image_dir.iterdir()
                   if f.suffix.lower() in extensions]

    print(f"Found {len(image_files)} images in {image_dir}")

    embeddings = []
    filenames = []

    for i in tqdm(range(0, len(image_files), batch_size), desc="Extracting"):
        batch_files = image_files[i:i+batch_size]
        batch_images = []
        batch_names = []

        for f in batch_files:
            try:
                img = Image.open(f).convert("RGB")
                batch_images.append(img)
                batch_names.append(f.name)
            except Exception as e:
                print(f"Error loading {f}: {e}")
                continue

        if not batch_images:
            continue

        # 前処理
        inputs = processor(images=batch_images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 特徴抽出
        with torch.no_grad():
            outputs = model(**inputs)
            # CLS token
            batch_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        embeddings.append(batch_emb)
        filenames.extend(batch_names)

    embeddings = np.vstack(embeddings)
    return embeddings, filenames

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True, help="Image directory")
    parser.add_argument("--name", type=str, required=True, help="Output name (e.g., 'illustrious_ai')")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    image_dir = Path(args.dir)
    if not image_dir.exists():
        print(f"Error: {image_dir} does not exist")
        sys.exit(1)

    EMBEDDINGS_DIR.mkdir(exist_ok=True)

    # モデルロード
    model, processor, device = load_model()

    # 抽出
    embeddings, filenames = extract_embeddings(
        image_dir, model, processor, device, args.batch_size
    )

    # 保存
    emb_path = EMBEDDINGS_DIR / f"{args.name}.npy"
    names_path = EMBEDDINGS_DIR / f"{args.name}_files.txt"

    np.save(emb_path, embeddings)
    with open(names_path, "w") as f:
        f.write("\n".join(filenames))

    print(f"Saved {len(embeddings)} embeddings to {emb_path}")
    print(f"Saved filenames to {names_path}")

if __name__ == "__main__":
    main()
