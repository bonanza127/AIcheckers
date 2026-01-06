#!/usr/bin/env python3
"""
DINOv3 embedding抽出スクリプト v2
CLSトークン + パッチ統計量を保存

出力:
  - {name}.npy: CLSトークン (N, 768)
  - {name}_patch_stats.npy: パッチ統計量 (N, 7)
    - [0] patch_mean: パッチスコアの平均
    - [1] patch_max: パッチスコアの最大
    - [2] patch_var: パッチスコアの分散
    - [3] max_minus_mean: 最大 - 平均（局所的突出度）
    - [4] embed_var_mean: 次元ごとの分散の平均（パッチ多様性）
    - [5] count_high_score: スコア≥0.8のパッチ数 / 196（高スコア領域の割合）
    - [6] v_high_sim_85: 垂直方向に隣接するパッチ間で類似度>0.85の割合

注意:
  - 分類器がない場合は _sim_based サフィックスが付く
  - 分類器ベースと類似度ベースの数値は意味が異なるため混合禁止
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
from lib.patch_stats import compute_patch_stats_batch

# 設定
DINOV3_MODEL = "facebook/dinov3-vitb16-pretrain-lvd1689m"
HF_TOKEN = os.getenv("HF_TOKEN", "")
EMBEDDINGS_DIR = Path("/home/techne/aicheckers/embeddings")
CLASSIFIER_PATH = Path("/home/techne/aicheckers/models/dinov3_classifier.pt")


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
    """DINOv3モデルをロード"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    processor = AutoImageProcessor.from_pretrained(DINOV3_MODEL, token=HF_TOKEN)
    model = AutoModel.from_pretrained(DINOV3_MODEL, token=HF_TOKEN)
    model.to(device)
    model.eval()

    return model, processor, device


def load_classifier(device):
    """分類器をロード（パッチスコア計算用）

    重要: backend/main.py と同じ方法でパッチ統計を計算するため、
    775d分類器の先頭768dを使用する。これにより学習と推論の一貫性を保つ。
    """
    if CLASSIFIER_PATH.exists():
        checkpoint = torch.load(CLASSIFIER_PATH, map_location=device, weights_only=True)
        input_dim = checkpoint.get("input_dim", 768)

        classifier = nn.Linear(input_dim, 2)
        classifier.load_state_dict(checkpoint["classifier"])
        classifier.to(device)
        classifier.eval()

        if input_dim > 768:
            print(f"[INFO] Loaded {input_dim}d classifier, using first 768d for patch scoring")
        else:
            print(f"[INFO] Loaded {input_dim}d classifier from {CLASSIFIER_PATH}")
        return classifier
    else:
        print(f"[WARN] No classifier found at {CLASSIFIER_PATH}")
        print("[WARN] Patch statistics will use embedding-based metrics only")
        return None


def compute_patch_stats(patch_embeddings: torch.Tensor, classifier: nn.Linear = None) -> np.ndarray:
    """パッチ統計量計算（共通モジュールへの委譲）"""
    return compute_patch_stats_batch(patch_embeddings, classifier)


def extract_embeddings(image_dir: Path, model, processor, device, classifier=None, batch_size=32, degradation_prob=0.0):
    """ディレクトリ内の全画像からembeddingとパッチ統計量を抽出

    Args:
        degradation_prob: 劣化Augmentationを適用する確率 (0.0-1.0)
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

        # 特徴抽出
        with torch.no_grad():
            outputs = model(**inputs)
            # DINOv3: [CLS(0), REG1-4(1-4), PATCH1-196(5-200)] = 201 tokens
            hidden_states = outputs.last_hidden_state  # (batch, 201, 768)

            # CLSトークン
            cls_emb = hidden_states[:, 0, :].cpu().numpy()  # (batch, 768)

            # パッチトークン（REGトークンをスキップ）
            patch_emb = hidden_states[:, 5:5+196, :]  # (batch, 196, 768)

            # パッチ統計量
            patch_stats = compute_patch_stats(patch_emb, classifier)

        cls_embeddings.append(cls_emb)
        patch_stats_list.append(patch_stats)
        filenames.extend(batch_names)

    cls_embeddings = np.vstack(cls_embeddings)
    patch_stats_all = np.vstack(patch_stats_list)

    if degradation_prob > 0:
        print(f"Applied degradation to {degradation_count}/{len(filenames)} images ({degradation_count/len(filenames)*100:.1f}%)")

    return cls_embeddings, patch_stats_all, filenames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True, help="Image directory")
    parser.add_argument("--name", type=str, required=True, help="Output name (e.g., 'illustrious_ai')")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--no-classifier", action="store_true",
                        help="Skip classifier-based patch stats (use embedding-based only)")
    parser.add_argument("--degradation-prob", type=float, default=0.0,
                        help="Probability of applying degradation augmentation (0.0-1.0, default: 0.0)")
    args = parser.parse_args()

    image_dir = Path(args.dir)
    if not image_dir.exists():
        print(f"Error: {image_dir} does not exist")
        sys.exit(1)

    EMBEDDINGS_DIR.mkdir(exist_ok=True)

    # モデルロード
    model, processor, device = load_model()

    # 分類器ロード（オプション）
    classifier = None
    if not args.no_classifier:
        classifier = load_classifier(device)

    # 抽出
    cls_embeddings, patch_stats, filenames = extract_embeddings(
        image_dir, model, processor, device, classifier, args.batch_size, args.degradation_prob
    )

    # 保存（分類器がない場合は _sim_based サフィックス）
    stats_suffix = "_patch_stats" if classifier else "_patch_stats_sim_based"

    cls_path = EMBEDDINGS_DIR / f"{args.name}.npy"
    stats_path = EMBEDDINGS_DIR / f"{args.name}{stats_suffix}.npy"
    names_path = EMBEDDINGS_DIR / f"{args.name}_files.txt"

    np.save(cls_path, cls_embeddings)
    np.save(stats_path, patch_stats)
    with open(names_path, "w") as f:
        f.write("\n".join(filenames))

    stats_type = "classifier-based" if classifier else "similarity-based (⚠️ different meaning)"
    print(f"\n[DONE] Saved {len(cls_embeddings)} samples")
    print(f"  CLS embeddings: {cls_path} ({cls_embeddings.shape})")
    print(f"  Patch stats:    {stats_path} ({patch_stats.shape}) [{stats_type}]")
    print(f"  Filenames:      {names_path}")

    # 統計量サマリー
    print(f"\n[STATS] Patch statistics summary ({stats_type}):")
    print(f"  [0] patch_mean:       {patch_stats[:, 0].mean():.4f} ± {patch_stats[:, 0].std():.4f}")
    print(f"  [1] patch_max:        {patch_stats[:, 1].mean():.4f} ± {patch_stats[:, 1].std():.4f}")
    print(f"  [2] patch_var:        {patch_stats[:, 2].mean():.4f} ± {patch_stats[:, 2].std():.4f}")
    print(f"  [3] max_minus_mean:   {patch_stats[:, 3].mean():.4f} ± {patch_stats[:, 3].std():.4f}")
    print(f"  [4] embed_var_mean:   {patch_stats[:, 4].mean():.4f} ± {patch_stats[:, 4].std():.4f}")
    print(f"  [5] count_high_score: {patch_stats[:, 5].mean():.4f} ± {patch_stats[:, 5].std():.4f}")
    print(f"  [6] v_high_sim_85:    {patch_stats[:, 6].mean():.4f} ± {patch_stats[:, 6].std():.4f}")


if __name__ == "__main__":
    main()
