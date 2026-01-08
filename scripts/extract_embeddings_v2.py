#!/usr/bin/env python3
"""
DINOv3 embedding抽出スクリプト v2.1
CLSトークン（最終層） + パッチ統計量v2（中間層）を保存

出力:
  - {name}.npy: CLSトークン (N, 768) - 最終層から取得
  - {name}_patch_stats.npy: パッチ統計量v2 (N, 7) - 中間層から取得
    - [0] adj_sim_mean:   隣接パッチ平均コサイン類似度
    - [1] adj_sim_var:    隣接パッチ類似度分散
    - [2] high_sim_ratio: 高類似度率（>0.9）
    - [3] patch_var:      パッチ埋め込み分散
    - [4] anisotropy:     縦横類似度差
    - [5] norm_var:       ノルム分散
    - [6] norm_range:     ノルムレンジ（max - min）

設計原則 (2026-01):
  - 中間層から「分類器を通さない」教師なし統計量を抽出
  - 未知のAIモデルに対する汎化性能を高める
"""
import os
import sys
import argparse
import io
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image, ImageFilter
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

# 共通モジュール
sys.path.insert(0, str(Path(__file__).parent.parent))
from lib.patch_stats import compute_patch_stats_v2_batch

# 設定
DINOV3_MODEL_PATH = Path("/home/techne/aicheckers/models/dinov3-vitb16")
EMBEDDINGS_DIR = Path("/home/techne/aicheckers/embeddings")
MID_LAYER_INDEX = 8  # 中間層のインデックス（0-11、Block 6-8推奨）


def apply_degradation(img: Image.Image) -> Image.Image:
    """
    画像に劣化処理を適用（画質バイアスを除去するため）

    適用される劣化の種類（ランダムに1つ選択）:
    - JPEG圧縮 (quality 30-70)
    - ガウシアンノイズ
    - ダウンサンプリング→アップサンプリング (50-80%)
    """
    degradation_type = random.choice(['jpeg', 'noise', 'downsample'])

    if degradation_type == 'jpeg':
        # JPEG圧縮
        quality = random.randint(30, 70)
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        img = Image.open(buffer).convert('RGB')

    elif degradation_type == 'noise':
        # ガウシアンノイズ
        arr = np.array(img, dtype=np.float32)
        noise_std = random.uniform(5, 25)
        noise = np.random.normal(0, noise_std, arr.shape).astype(np.float32)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    elif degradation_type == 'downsample':
        # ダウンサンプリング→アップサンプリング
        scale = random.uniform(0.5, 0.8)
        w, h = img.size
        small_size = (int(w * scale), int(h * scale))
        img = img.resize(small_size, Image.BILINEAR)
        img = img.resize((w, h), Image.BILINEAR)

    return img


def load_model():
    """DINOv3モデルをロード（ローカルディレクトリから、中間層出力有効）"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not DINOV3_MODEL_PATH.exists():
        raise FileNotFoundError(f"DINOv3 model not found at {DINOV3_MODEL_PATH}")

    # ローカルディレクトリからロード（ネットワーク不要）
    processor = AutoImageProcessor.from_pretrained(str(DINOV3_MODEL_PATH))
    model = AutoModel.from_pretrained(str(DINOV3_MODEL_PATH))
    model.to(device)
    model.eval()

    print(f"[INFO] Loaded DINOv3 from: {DINOV3_MODEL_PATH}")
    print(f"[INFO] Mid-layer extraction enabled: Block {MID_LAYER_INDEX}")
    return model, processor, device


# NOTE: 分類器ロードは不要になりました（v2は教師なし統計量のため）


def extract_embeddings(image_dir: Path, model, processor, device, batch_size=32, degradation_prob=0.0, mid_layer=MID_LAYER_INDEX):
    """ディレクトリ内の全画像からCLS（最終層）とパッチ統計量v2（中間層）を抽出

    Args:
        degradation_prob: 劣化Augmentationを適用する確率 (0.0-1.0)
        mid_layer: パッチ統計量を抽出する中間層のインデックス (0-11)
    """
    extensions = {'.jpg', '.jpeg', '.png', '.webp', '.gif'}
    image_files = [f for f in image_dir.rglob("*")
                   if f.is_file() and f.suffix.lower() in extensions]

    print(f"Found {len(image_files)} images in {image_dir}")
    if degradation_prob > 0:
        print(f"Degradation augmentation enabled: {degradation_prob*100:.0f}% of images")

    cls_embeddings = []
    patch_stats_list = []
    filenames = []
    degradation_count = 0

    for i in tqdm(range(0, len(image_files), batch_size), desc="Extracting"):
        batch_files = image_files[i:i+batch_size]
        batch_images = []
        batch_names = []

        for f in batch_files:
            try:
                img = Image.open(f).convert("RGB")
                # 劣化Augmentation（確率的に適用）
                if degradation_prob > 0 and random.random() < degradation_prob:
                    img = apply_degradation(img)
                    degradation_count += 1
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

        # 特徴抽出（中間層出力を有効化）
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # DINOv3: [CLS(0), REG1-4(1-4), PATCH1-196(5-200)] = 201 tokens
            
            # 最終層からCLSトークン
            final_hidden = outputs.last_hidden_state  # (batch, 201, 768)
            cls_emb = final_hidden[:, 0, :].cpu().numpy()  # (batch, 768)

            # 中間層からパッチトークン（v2統計量用）
            # hidden_states: tuple of (batch, 201, 768) for each layer + initial embedding
            mid_hidden = outputs.hidden_states[mid_layer + 1]  # +1 because index 0 is initial embedding
            mid_patch_emb = mid_hidden[:, 5:5+196, :]  # (batch, 196, 768)

            # パッチ統計量v2（教師なし）
            patch_stats = compute_patch_stats_v2_batch(mid_patch_emb)

        cls_embeddings.append(cls_emb)
        patch_stats_list.append(patch_stats)
        filenames.extend(batch_names)

    cls_embeddings = np.vstack(cls_embeddings)
    patch_stats_all = np.vstack(patch_stats_list)

    if degradation_prob > 0:
        print(f"Applied degradation to {degradation_count}/{len(filenames)} images ({degradation_count/len(filenames)*100:.1f}%)")

    return cls_embeddings, patch_stats_all, filenames


def main():
    parser = argparse.ArgumentParser(description="DINOv3 embedding extraction v2.1 (mid-layer stats)")
    parser.add_argument("--dir", type=str, required=True, help="Image directory")
    parser.add_argument("--name", type=str, required=True, help="Output name (e.g., 'illustrious_ai')")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--mid-layer", type=int, default=MID_LAYER_INDEX,
                        help=f"Mid-layer index for patch stats (0-11, default: {MID_LAYER_INDEX})")
    parser.add_argument("--degradation-prob", type=float, default=0.0,
                        help="Probability of applying degradation augmentation (0.0-1.0, default: 0.0)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of images to process (for testing)")
    args = parser.parse_args()

    image_dir = Path(args.dir)
    if not image_dir.exists():
        print(f"Error: {image_dir} does not exist")
        sys.exit(1)

    EMBEDDINGS_DIR.mkdir(exist_ok=True)

    # モデルロード
    model, processor, device = load_model()

    # 抽出（v2: 分類器不要）
    cls_embeddings, patch_stats, filenames = extract_embeddings(
        image_dir, model, processor, device, args.batch_size, args.degradation_prob, args.mid_layer
    )

    # limitオプション対応
    if args.limit and len(cls_embeddings) > args.limit:
        cls_embeddings = cls_embeddings[:args.limit]
        patch_stats = patch_stats[:args.limit]
        filenames = filenames[:args.limit]

    # 保存（v2は常に同じフォーマット）
    cls_path = EMBEDDINGS_DIR / f"{args.name}.npy"
    stats_path = EMBEDDINGS_DIR / f"{args.name}_patch_stats.npy"
    names_path = EMBEDDINGS_DIR / f"{args.name}_files.txt"

    np.save(cls_path, cls_embeddings)
    np.save(stats_path, patch_stats)
    with open(names_path, "w") as f:
        f.write("\n".join(filenames))

    print(f"\n[DONE] Saved {len(cls_embeddings)} samples")
    print(f"  CLS embeddings (final layer): {cls_path} ({cls_embeddings.shape})")
    print(f"  Patch stats v2 (mid-layer {args.mid_layer}): {stats_path} ({patch_stats.shape})")
    print(f"  Filenames: {names_path}")

    # 統計量サマリー（v2形式）
    print(f"\n[STATS] Patch statistics v2 summary (unsupervised, mid-layer {args.mid_layer}):")
    print(f"  [0] adj_sim_mean:   {patch_stats[:, 0].mean():.4f} ± {patch_stats[:, 0].std():.4f}")
    print(f"  [1] adj_sim_var:    {patch_stats[:, 1].mean():.6f} ± {patch_stats[:, 1].std():.6f}")
    print(f"  [2] high_sim_ratio: {patch_stats[:, 2].mean():.4f} ± {patch_stats[:, 2].std():.4f}")
    print(f"  [3] patch_var:      {patch_stats[:, 3].mean():.4f} ± {patch_stats[:, 3].std():.4f}")
    print(f"  [4] anisotropy:     {patch_stats[:, 4].mean():.6f} ± {patch_stats[:, 4].std():.6f}")
    print(f"  [5] norm_var:       {patch_stats[:, 5].mean():.4f} ± {patch_stats[:, 5].std():.4f}")
    print(f"  [6] norm_range:     {patch_stats[:, 6].mean():.4f} ± {patch_stats[:, 6].std():.4f}")


if __name__ == "__main__":
    main()
