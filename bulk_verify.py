#!/usr/bin/env python3
"""
大量のNovelAI画像を検証し、Human判定されたものを分離
"""
import os
import shutil
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
import json
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

# 設定
MODEL_PATH = Path("models/dinov3_classifier.pt")
DINOV3_MODEL = "facebook/dinov3-vitb16-pretrain-lvd1689m"
HF_TOKEN = "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"

# 検証対象ディレクトリ
TEST_DIRS = [
    "data/novelai_combined",
    "data/novelai",
    "data/twitter_novelai_filtered",
]

# 出力先
OUTPUT_DIR = Path("misclassified_analysis")
HUMAN_JUDGED_DIR = OUTPUT_DIR / "human_judged_images"
RESULTS_JSON = OUTPUT_DIR / "bulk_verification_results.json"

def load_classifier(device):
    """分類器をロード"""
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    input_dim = checkpoint.get("input_dim", 768)

    classifier = nn.Linear(input_dim, 2)
    classifier.load_state_dict(checkpoint["classifier"])
    classifier.to(device)
    classifier.eval()

    return classifier, input_dim

def compute_patch_stats(patch_embeddings: torch.Tensor, classifier: nn.Linear, device):
    """パッチ統計量を計算（backend/main.pyと同じ）"""
    import torch.nn.functional as F

    HIGH_SCORE_THRESHOLD = 0.8
    HIGH_SIM_THRESHOLD = 0.85
    grid_size = 14

    with torch.no_grad():
        # パッチごとのAIスコアを計算（775d分類器の先頭768dを使用）
        weight = classifier.weight[:, :768]  # (2, 768)
        bias = classifier.bias
        flat_patches = patch_embeddings.reshape(-1, 768)  # (196, 768)
        logits = torch.mm(flat_patches, weight.t()) + bias  # (196, 2)
        probs = torch.softmax(logits, dim=1)
        ai_scores = probs[:, 1]  # (196,)

        # 統計量計算
        patch_mean = ai_scores.mean()
        patch_max = ai_scores.max()
        patch_var = ai_scores.var()
        max_minus_mean = patch_max - patch_mean
        embed_var_mean = patch_embeddings[0].var(dim=0).mean()
        count_high_score = (ai_scores >= HIGH_SCORE_THRESHOLD).float().mean()

        # v_high_sim_85: 垂直方向の高類似度パッチ比率
        patch_emb = patch_embeddings[0]  # (196, 768)
        patches_grid = patch_emb.reshape(grid_size, grid_size, -1)  # (14, 14, 768)
        v_sims = []
        for row in range(grid_size - 1):
            for col in range(grid_size):
                current = patches_grid[row, col]
                down = patches_grid[row + 1, col]
                sim = F.cosine_similarity(current.unsqueeze(0), down.unsqueeze(0)).item()
                v_sims.append(sim)
        v_high_sim = sum(1 for s in v_sims if s > HIGH_SIM_THRESHOLD) / len(v_sims)

        stats = torch.stack([
            patch_mean,
            patch_max,
            patch_var,
            max_minus_mean,
            embed_var_mean,
            count_high_score,
            torch.tensor(v_high_sim, device=device)
        ]).unsqueeze(0)  # (1, 7)

    return stats

def get_image_score(image_path: Path, model, processor, classifier, device):
    """画像のAIスコアを計算"""
    try:
        # 画像読み込み
        image = Image.open(image_path).convert('RGB')

        # DINOv3で埋め込み抽出
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

            # CLS token
            cls_token = outputs.last_hidden_state[:, 0, :]  # (1, 768)

            # パッチ埋め込み（レジスタトークンを除く）
            patch_embeddings = outputs.last_hidden_state[:, 5:5+196, :]  # (1, 196, 768)

            # パッチ統計量
            patch_stats = compute_patch_stats(patch_embeddings, classifier, device)

            # 最終特徴量
            features = torch.cat([cls_token, patch_stats], dim=1)  # (1, 775)

            # スコア計算
            logits = classifier(features)
            probs = torch.softmax(logits, dim=1)
            ai_score = probs[0, 1].item()

        return ai_score, True
    except Exception as e:
        print(f"ERROR {image_path.name}: {e}")
        return None, False

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # 出力ディレクトリ作成
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    HUMAN_JUDGED_DIR.mkdir(exist_ok=True, parents=True)

    # モデルロード
    print("Loading models...")
    processor = AutoImageProcessor.from_pretrained(DINOV3_MODEL, token=HF_TOKEN)
    model = AutoModel.from_pretrained(DINOV3_MODEL, token=HF_TOKEN)
    model.to(device)
    model.eval()

    classifier, input_dim = load_classifier(device)
    print(f"Classifier dimension: {input_dim}d\n")

    # 全ディレクトリの画像を収集
    all_images = []
    for test_dir in TEST_DIRS:
        dir_path = Path(test_dir)
        if dir_path.exists():
            images = list(dir_path.glob("*.jpg")) + list(dir_path.glob("*.png"))
            all_images.extend(images)
            print(f"Found {len(images)} images in {test_dir}")

    print(f"\nTotal images to verify: {len(all_images)}\n")

    # 検証実行
    results = []
    human_judged_count = 0
    error_count = 0

    for img_path in tqdm(all_images, desc="Verifying"):
        score, success = get_image_score(img_path, model, processor, classifier, device)

        if success:
            is_human_judged = score < 0.5
            results.append({
                "path": str(img_path),
                "filename": img_path.name,
                "source_dir": str(img_path.parent),
                "score": float(score),
                "human_judged": is_human_judged
            })

            # Human判定された画像をコピー
            if is_human_judged:
                dest_path = HUMAN_JUDGED_DIR / f"{score:.4f}_{img_path.name}"
                shutil.copy2(img_path, dest_path)
                human_judged_count += 1
        else:
            error_count += 1

    # 結果を保存
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)

    # サマリー出力
    print(f"\n{'='*60}")
    print(f"VERIFICATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total verified: {len(results)}")
    print(f"Correct (AI judged): {len(results) - human_judged_count} ({(len(results) - human_judged_count)/len(results)*100:.1f}%)")
    print(f"Human judged (misclassified): {human_judged_count} ({human_judged_count/len(results)*100:.1f}%)")
    print(f"Errors: {error_count}")
    print(f"\nHuman judged images saved to: {HUMAN_JUDGED_DIR}")
    print(f"Results JSON saved to: {RESULTS_JSON}")

    # スコア分布
    human_judged = [r for r in results if r["human_judged"]]
    if human_judged:
        scores = [r["score"] for r in human_judged]
        print(f"\n{'='*60}")
        print(f"HUMAN JUDGED SCORE DISTRIBUTION")
        print(f"{'='*60}")
        print(f"Min score: {min(scores):.4f}")
        print(f"Max score: {max(scores):.4f}")
        print(f"Mean score: {np.mean(scores):.4f}")
        print(f"Median score: {np.median(scores):.4f}")

        # スコア範囲別
        ranges = [
            (0.0, 0.1, "Extreme Human (0.0-0.1)"),
            (0.1, 0.2, "Strong Human (0.1-0.2)"),
            (0.2, 0.3, "Moderate Human (0.2-0.3)"),
            (0.3, 0.4, "Weak Human (0.3-0.4)"),
            (0.4, 0.5, "Borderline (0.4-0.5)"),
        ]
        print(f"\nScore range breakdown:")
        for low, high, label in ranges:
            count = sum(1 for s in scores if low <= s < high)
            print(f"  {label}: {count} ({count/len(scores)*100:.1f}%)")

if __name__ == "__main__":
    main()
