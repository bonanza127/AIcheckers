#!/usr/bin/env python3
"""
既存のembeddingsに対応するパッチ統計量のみを抽出
CLSトークンは既存のものを使用し、patch_statsだけ追加

使い方:
  python scripts/extract_patch_stats_only.py --name illustrious_ai --image-dir data/animedl2m_dataset_release/civitai_subset/image/Illustrious
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
    """分類器をロード（パッチスコア計算用）- 768d CLS-only版を使用"""
    # パッチスコア計算には768次元分類器が必要
    cls_only_path = CLASSIFIER_PATH.with_name("dinov3_classifier_cls_only.pt")

    if cls_only_path.exists():
        checkpoint = torch.load(cls_only_path, map_location=device, weights_only=True)
        input_dim = checkpoint.get("input_dim", 768)
        classifier = nn.Linear(input_dim, 2)
        classifier.load_state_dict(checkpoint["classifier"])
        classifier.to(device)
        classifier.eval()
        print(f"[INFO] Loaded 768d classifier from {cls_only_path}")
        return classifier
    elif CLASSIFIER_PATH.exists():
        # フォールバック: 774d分類器があってもパッチには使えないので警告
        checkpoint = torch.load(CLASSIFIER_PATH, map_location=device, weights_only=True)
        input_dim = checkpoint.get("input_dim", 768)
        if input_dim != 768:
            print(f"[ERROR] Only 774d classifier found, cannot use for patch scoring")
            return None
        classifier = nn.Linear(768, 2)
        classifier.load_state_dict(checkpoint["classifier"])
        classifier.to(device)
        classifier.eval()
        print(f"[INFO] Loaded 768d classifier from {CLASSIFIER_PATH}")
        return classifier
    else:
        print(f"[ERROR] No classifier found")
        return None


def compute_patch_stats(patch_embeddings: torch.Tensor, classifier: nn.Linear) -> np.ndarray:
    """パッチ統計量を計算"""
    batch_size = patch_embeddings.shape[0]
    num_patches = patch_embeddings.shape[1]  # 196
    grid_size = 14  # 14x14 patches
    stats = np.zeros((batch_size, 7), dtype=np.float32)
    HIGH_SCORE_THRESHOLD = 0.8
    HIGH_SIM_THRESHOLD = 0.85

    with torch.no_grad():
        # 分類器を通してパッチごとのAIスコアを計算
        flat_patches = patch_embeddings.reshape(-1, 768)
        logits = classifier(flat_patches)
        probs = F.softmax(logits, dim=1)
        ai_scores = probs[:, 1].reshape(batch_size, -1)

        for i in range(batch_size):
            scores = ai_scores[i].cpu().numpy()
            patch_emb = patch_embeddings[i]  # (196, 768)

            stats[i, 0] = np.mean(scores)                                    # patch_mean
            stats[i, 1] = np.max(scores)                                     # patch_max
            stats[i, 2] = np.var(scores)                                     # patch_var
            stats[i, 3] = stats[i, 1] - stats[i, 0]                          # max_minus_mean
            stats[i, 4] = patch_emb.cpu().numpy().var(axis=0).mean()         # embed_var_mean
            stats[i, 5] = np.sum(scores >= HIGH_SCORE_THRESHOLD) / num_patches  # count_high_score

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


def extract_patch_stats(file_list: list, image_dir: Path, model, processor, device, classifier, batch_size=32):
    """ファイルリストに基づいてパッチ統計量を抽出（順序維持）"""
    patch_stats_list = []
    not_found = []

    for i in tqdm(range(0, len(file_list), batch_size), desc="Extracting"):
        batch_files = file_list[i:i+batch_size]
        batch_images = []
        batch_indices = []

        for j, fname in enumerate(batch_files):
            # 画像パスを探す（ディレクトリ直下またはサブディレクトリ）
            img_path = image_dir / fname
            if not img_path.exists():
                # サブディレクトリを検索
                found = list(image_dir.rglob(fname))
                if found:
                    img_path = found[0]
                else:
                    not_found.append(fname)
                    # プレースホルダー（後で処理）
                    continue

            try:
                img = Image.open(img_path).convert("RGB")
                batch_images.append(img)
                batch_indices.append(i + j)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                not_found.append(fname)
                continue

        if not batch_images:
            # バッチ全体がスキップされた場合
            patch_stats_list.extend([np.zeros(6, dtype=np.float32)] * len(batch_files))
            continue

        # 前処理
        inputs = processor(images=batch_images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 特徴抽出
        with torch.no_grad():
            outputs = model(**inputs)
            # DINOv3: [CLS(0), REG1-4(1-4), PATCH1-196(5-200)] = 201 tokens
            hidden_states = outputs.last_hidden_state  # (batch, 201, 768)
            patch_emb = hidden_states[:, 5:5+196, :]  # (batch, 196, 768)
            patch_stats = compute_patch_stats(patch_emb, classifier)

        # 結果を正しい位置に挿入
        stats_idx = 0
        for j, fname in enumerate(batch_files):
            if fname in not_found:
                patch_stats_list.append(np.zeros(6, dtype=np.float32))
            else:
                patch_stats_list.append(patch_stats[stats_idx])
                stats_idx += 1

    if not_found:
        print(f"[WARN] {len(not_found)} files not found")

    return np.array(patch_stats_list)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True, help="Embedding name (e.g., 'illustrious_ai')")
    parser.add_argument("--image-dir", type=str, required=True, help="Image directory")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    # ファイルリスト読み込み
    files_path = EMBEDDINGS_DIR / f"{args.name}_files.txt"
    if not files_path.exists():
        print(f"Error: {files_path} does not exist")
        sys.exit(1)

    with open(files_path) as f:
        file_list = [line.strip() for line in f if line.strip()]

    print(f"Found {len(file_list)} files in list")

    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        print(f"Error: {image_dir} does not exist")
        sys.exit(1)

    # モデルロード
    model, processor, device = load_model()
    classifier = load_classifier(device)
    if classifier is None:
        print("Error: Classifier is required for patch stats")
        sys.exit(1)

    # 抽出
    patch_stats = extract_patch_stats(file_list, image_dir, model, processor, device, classifier, args.batch_size)

    # 保存
    stats_path = EMBEDDINGS_DIR / f"{args.name}_patch_stats.npy"
    np.save(stats_path, patch_stats)

    print(f"\n[DONE] Saved {len(patch_stats)} samples")
    print(f"  Patch stats: {stats_path} ({patch_stats.shape})")

    # 統計量サマリー
    print(f"\n[STATS] Patch statistics summary:")
    print(f"  [0] patch_mean:       {patch_stats[:, 0].mean():.4f} ± {patch_stats[:, 0].std():.4f}")
    print(f"  [1] patch_max:        {patch_stats[:, 1].mean():.4f} ± {patch_stats[:, 1].std():.4f}")
    print(f"  [2] patch_var:        {patch_stats[:, 2].mean():.4f} ± {patch_stats[:, 2].std():.4f}")
    print(f"  [3] max_minus_mean:   {patch_stats[:, 3].mean():.4f} ± {patch_stats[:, 3].std():.4f}")
    print(f"  [4] embed_var_mean:   {patch_stats[:, 4].mean():.4f} ± {patch_stats[:, 4].std():.4f}")
    print(f"  [5] count_high_score: {patch_stats[:, 5].mean():.4f} ± {patch_stats[:, 5].std():.4f}")


if __name__ == "__main__":
    main()
