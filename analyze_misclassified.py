#!/usr/bin/env python3
"""
NovelAI Combined 50枚の誤判定画像を特定・分析
"""
import os
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
import json

# 設定
MODEL_PATH = Path("models/dinov3_classifier.pt")
TEST_DIR = Path("data/novelai_test_new")  # NovelAI Combined テスト画像

def load_classifier(device):
    """分類器をロード"""
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    input_dim = checkpoint.get("input_dim", 768)

    classifier = nn.Linear(input_dim, 2)
    classifier.load_state_dict(checkpoint["classifier"])
    classifier.to(device)
    classifier.eval()

    return classifier, input_dim

def get_image_score(image_path: Path, model, processor, classifier, device):
    """画像のAIスコアを計算"""
    from transformers import AutoImageProcessor, AutoModel

    # 画像読み込み
    image = Image.open(image_path).convert('RGB')

    # DINOv3で埋め込み抽出
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

        # CLS token
        cls_token = outputs.last_hidden_state[:, 0, :]  # (1, 768)

        # パッチ統計量計算（backend/main.pyと同じ方法）
        # DINOv3はレジスタトークン（4個）を含むため、パッチは5番目から
        patch_embeddings = outputs.last_hidden_state[:, 5:5+196, :]  # (1, 196, 768)

        # 分類器でパッチスコア計算（775次元分類器の先頭768次元を使用）
        weight = classifier.weight[:, :768]  # (2, 768)
        bias = classifier.bias
        flat_patches = patch_embeddings.reshape(-1, 768)  # (196, 768)
        patch_logits = torch.mm(flat_patches, weight.t()) + bias  # (196, 2)
        patch_probs = torch.softmax(patch_logits, dim=1)
        patch_ai_scores = patch_probs[:, 1]  # (196,) AI確率

        # パッチ統計量
        patch_mean = patch_ai_scores.mean()
        patch_max = patch_ai_scores.max()
        patch_var = patch_ai_scores.var()
        max_minus_mean = patch_max - patch_mean

        embed_var = patch_embeddings[0].var(dim=0).mean()
        count_high = (patch_ai_scores >= 0.8).float().mean()

        # 垂直方向の類似度
        import torch.nn.functional as F
        grid_size = 14
        patch_emb = patch_embeddings[0]  # (196, 768)
        patches_grid = patch_emb.reshape(grid_size, grid_size, -1)  # (14, 14, 768)
        v_sims = []
        for row in range(grid_size - 1):
            for col in range(grid_size):
                current = patches_grid[row, col]
                down = patches_grid[row + 1, col]
                sim = F.cosine_similarity(current.unsqueeze(0), down.unsqueeze(0)).item()
                v_sims.append(sim)
        v_high_sim = sum(1 for s in v_sims if s > 0.85) / len(v_sims)

        patch_stats = torch.stack([
            patch_mean,
            patch_max,
            patch_var,
            max_minus_mean,
            embed_var,
            count_high,
            torch.tensor(v_high_sim, device=device)
        ]).unsqueeze(0)  # (1, 7)

        # 最終特徴量
        features = torch.cat([cls_token, patch_stats], dim=1)  # (1, 775)

        # スコア計算
        logits = classifier(features)
        probs = torch.softmax(logits, dim=1)
        ai_score = probs[0, 1].item()

    return ai_score, patch_stats.cpu().numpy()[0]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # モデルロード
    print("Loading models...")
    from transformers import AutoImageProcessor, AutoModel
    HF_TOKEN = "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"
    DINOV3_MODEL = "facebook/dinov3-vitb16-pretrain-lvd1689m"

    processor = AutoImageProcessor.from_pretrained(DINOV3_MODEL, token=HF_TOKEN)
    model = AutoModel.from_pretrained(DINOV3_MODEL, token=HF_TOKEN)
    model.to(device)
    model.eval()

    classifier, input_dim = load_classifier(device)
    print(f"Classifier dimension: {input_dim}d\n")

    # テスト画像を取得
    image_files = sorted(list(TEST_DIR.glob("*.jpg")) + list(TEST_DIR.glob("*.png")))[:50]
    print(f"Found {len(image_files)} test images\n")

    if len(image_files) == 0:
        print(f"ERROR: No images found in {TEST_DIR}")
        return

    # 各画像のスコアを計算
    results = []
    print("Computing scores...")
    for img_path in image_files:
        try:
            score, patch_stats = get_image_score(img_path, model, processor, classifier, device)
            results.append({
                "path": str(img_path),
                "filename": img_path.name,
                "score": float(score),
                "patch_stats": patch_stats.tolist(),
                "misclassified": score < 0.5  # AI画像なのにHumanと判定
            })
            status = "❌ MISS" if score < 0.5 else "✓"
            print(f"{status} {img_path.name}: {score:.4f}")
        except Exception as e:
            print(f"ERROR {img_path.name}: {e}")

    # 結果を保存
    output_path = Path("analysis_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # サマリー
    misclassified = [r for r in results if r["misclassified"]]
    print(f"\n{'='*60}")
    print(f"Total: {len(results)} images")
    print(f"Correct: {len(results) - len(misclassified)} ({(len(results) - len(misclassified))/len(results)*100:.1f}%)")
    print(f"Misclassified: {len(misclassified)} ({len(misclassified)/len(results)*100:.1f}%)")
    print(f"\nMisclassified images:")
    for r in misclassified:
        print(f"  - {r['filename']}: {r['score']:.4f}")

    print(f"\nResults saved to: {output_path}")

    # パッチ統計量の分析
    if len(misclassified) > 0:
        print(f"\n{'='*60}")
        print("PATCH STATISTICS ANALYSIS")
        print(f"{'='*60}")

        correct = [r for r in results if not r["misclassified"]]

        miss_stats = np.array([r["patch_stats"] for r in misclassified])
        correct_stats = np.array([r["patch_stats"] for r in correct])

        stat_names = [
            "patch_mean", "patch_max", "patch_var",
            "max_minus_mean", "embed_var_mean",
            "count_high_score", "v_high_sim_85"
        ]

        print(f"\n{'Statistic':<20} {'Misclassified':<15} {'Correct':<15} {'Diff':<10}")
        print("-" * 60)
        for i, name in enumerate(stat_names):
            miss_mean = miss_stats[:, i].mean()
            correct_mean = correct_stats[:, i].mean()
            diff = miss_mean - correct_mean
            print(f"{name:<20} {miss_mean:<15.4f} {correct_mean:<15.4f} {diff:<10.4f}")

if __name__ == "__main__":
    main()
