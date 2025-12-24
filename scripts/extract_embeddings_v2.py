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
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

# 設定
DINOV3_MODEL = "facebook/dinov3-vitb16-pretrain-lvd1689m"
HF_TOKEN = "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"
EMBEDDINGS_DIR = Path("/home/techne/aicheckers/embeddings")
CLASSIFIER_PATH = Path("/home/techne/aicheckers/models/dinov3_classifier.pt")


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
    """
    パッチ統計量を計算

    Args:
        patch_embeddings: (batch, 196, 768) パッチのembedding
        classifier: 分類器（Noneの場合はembedding-based統計のみ）

    Returns:
        stats: (batch, 7) パッチ統計量
    """
    batch_size = patch_embeddings.shape[0]
    num_patches = patch_embeddings.shape[1]  # 196
    grid_size = 14  # 14x14 patches
    stats = np.zeros((batch_size, 7), dtype=np.float32)

    HIGH_SCORE_THRESHOLD = 0.8
    HIGH_SIM_THRESHOLD = 0.85

    with torch.no_grad():
        if classifier is not None:
            # 分類器を通してパッチごとのAIスコアを計算
            # 775次元分類器の場合は先頭768次元のみ使用（backend/main.pyと同じ）
            # (batch, 196, 768) -> (batch * 196, 768)
            flat_patches = patch_embeddings.reshape(-1, 768)
            weight = classifier.weight[:, :768]  # (2, 768)
            bias = classifier.bias
            logits = torch.mm(flat_patches, weight.t()) + bias  # (batch * 196, 2)
            probs = F.softmax(logits, dim=1)
            ai_scores = probs[:, 1].reshape(batch_size, -1)  # (batch, 196)

            # スコアベースの統計量
            for i in range(batch_size):
                scores = ai_scores[i].cpu().numpy()
                stats[i, 0] = np.mean(scores)                    # patch_mean
                stats[i, 1] = np.max(scores)                     # patch_max
                stats[i, 2] = np.var(scores)                     # patch_var
                stats[i, 3] = stats[i, 1] - stats[i, 0]          # max_minus_mean
                stats[i, 5] = np.sum(scores >= HIGH_SCORE_THRESHOLD) / num_patches  # count_high_score (ratio)
        else:
            # 分類器がない場合はembedding-based統計（コサイン類似度ベース）
            for i in range(batch_size):
                patch_emb = patch_embeddings[i].cpu().numpy()  # (196, 768)
                # パッチ間のコサイン類似度
                norms = np.linalg.norm(patch_emb, axis=1, keepdims=True)
                normalized = patch_emb / (norms + 1e-8)
                mean_patch = normalized.mean(axis=0)
                cos_sims = normalized @ mean_patch  # 各パッチと平均の類似度
                stats[i, 0] = np.mean(cos_sims)
                stats[i, 1] = np.max(cos_sims)
                stats[i, 2] = np.var(cos_sims)
                stats[i, 3] = stats[i, 1] - stats[i, 0]
                # 類似度ベースでは0.8以上を「高類似度」とみなす
                stats[i, 5] = np.sum(cos_sims >= HIGH_SCORE_THRESHOLD) / num_patches

        # embedding空間でのパッチ多様性 + 垂直方向の高類似度パッチ比率（分類器有無に関わらず計算）
        for i in range(batch_size):
            patch_emb = patch_embeddings[i]  # (196, 768) - keep as tensor
            # 次元ごとの分散の平均 = パッチ間の多様性
            stats[i, 4] = patch_emb.cpu().numpy().var(axis=0).mean()  # embed_var_mean

            # [6] v_high_sim_85: 垂直方向の高類似度パッチ比率
            patches_grid = patch_emb.reshape(grid_size, grid_size, -1)  # (14, 14, 768)
            v_sims = []
            for row in range(grid_size - 1):
                for col in range(grid_size):
                    current = patches_grid[row, col]
                    down = patches_grid[row + 1, col]
                    sim = F.cosine_similarity(current.unsqueeze(0), down.unsqueeze(0)).item()
                    v_sims.append(sim)
            stats[i, 6] = np.sum(np.array(v_sims) > HIGH_SIM_THRESHOLD) / len(v_sims)  # v_high_sim_85

    return stats


def extract_embeddings(image_dir: Path, model, processor, device, classifier=None, batch_size=32):
    """ディレクトリ内の全画像からembeddingとパッチ統計量を抽出"""
    extensions = {'.jpg', '.jpeg', '.png', '.webp', '.gif'}
    image_files = [f for f in image_dir.rglob("*")
                   if f.is_file() and f.suffix.lower() in extensions]

    print(f"Found {len(image_files)} images in {image_dir}")

    cls_embeddings = []
    patch_stats_list = []
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

    return cls_embeddings, patch_stats_all, filenames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True, help="Image directory")
    parser.add_argument("--name", type=str, required=True, help="Output name (e.g., 'illustrious_ai')")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--no-classifier", action="store_true",
                        help="Skip classifier-based patch stats (use embedding-based only)")
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
        image_dir, model, processor, device, classifier, args.batch_size
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
