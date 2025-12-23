#!/usr/bin/env python3
"""
パッチ別AI度の統計量分析
DINOv3の196パッチそれぞれのスコアを分析し、
LoRA重ね合わせ検出に有効な特徴量を探る
"""

import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from PIL import Image

def skew(x):
    """歪度を計算（scipy不要版）"""
    n = len(x)
    mean = np.mean(x)
    std = np.std(x)
    if std == 0:
        return 0.0
    return np.mean(((x - mean) / std) ** 3)

# パス設定
EMBEDDINGS_DIR = Path("/home/techne/aicheckers/embeddings")
MODEL_PATH = Path("/home/techne/aicheckers/models/dinov3_classifier.pt")

def load_classifier():
    """分類器をロード"""
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    classifier = nn.Linear(768, 2)
    classifier.load_state_dict(checkpoint["classifier"])
    classifier.eval()
    return classifier

def compute_patch_stats(patch_scores: np.ndarray) -> dict:
    """
    パッチスコアから統計量を計算

    Args:
        patch_scores: 各パッチのAIスコア (196,)

    Returns:
        統計量の辞書
    """
    mean_score = np.mean(patch_scores)
    max_score = np.max(patch_scores)
    min_score = np.min(patch_scores)
    variance = np.var(patch_scores)
    std = np.std(patch_scores)

    # Max - Mean: 一部だけ異常にAIっぽいケースを検出
    max_minus_mean = max_score - mean_score

    # Skewness（歪度）: LoRA顔が突出するケースに強い
    skewness = skew(patch_scores)

    # Top-k mean - global mean: 上位10%パッチだけ異常か
    k = max(1, len(patch_scores) // 10)  # 上位10%
    top_k_indices = np.argsort(patch_scores)[-k:]
    top_k_mean = np.mean(patch_scores[top_k_indices])
    top_k_minus_mean = top_k_mean - mean_score

    # 実務的指標: (max - mean) + variance
    practical_score = max_minus_mean + variance

    return {
        "mean": mean_score,
        "max": max_score,
        "min": min_score,
        "variance": variance,
        "std": std,
        "max_minus_mean": max_minus_mean,
        "skewness": skewness,
        "top_k_minus_mean": top_k_minus_mean,
        "practical_score": practical_score,
    }

def analyze_embeddings():
    """既存embeddingsをパッチ単位で分析"""

    print("=" * 60)
    print("パッチ別AI度 統計量分析")
    print("=" * 60)
    print()

    # 分類器ロード
    classifier = load_classifier()

    # 各カテゴリを分析
    categories = [
        ("novelai_ai", "NovelAI"),
        ("pony_ai", "Pony"),
        ("sdxl10_ai", "SDXL"),
        ("illustrious_ai", "Illustrious"),
        ("danbooru_real", "Human (Danbooru)"),
    ]

    results = {}

    for filename, label in categories:
        emb_path = EMBEDDINGS_DIR / f"{filename}.npy"
        if not emb_path.exists():
            print(f"[SKIP] {filename} not found")
            continue

        embeddings = np.load(emb_path)
        n_samples = min(500, len(embeddings))  # 最大500サンプル

        # ランダムサンプリング
        indices = np.random.choice(len(embeddings), n_samples, replace=False)
        sampled = embeddings[indices]

        # 全体スコア（従来方式）
        with torch.no_grad():
            tensor = torch.tensor(sampled, dtype=torch.float32)
            logits = classifier(tensor)
            probs = torch.softmax(logits, dim=1)
            ai_scores = probs[:, 1].numpy()  # class 1 = AI

        # 統計量計算（サンプル全体の平均）
        mean_ai_score = np.mean(ai_scores)

        # 注意: 現在のembeddingsは[CLS]トークンのみ（768次元）
        # パッチ別分析には、全パッチの特徴量が必要
        # ここでは代わりに、サンプル間のばらつきを分析

        sample_stats = {
            "mean": np.mean(ai_scores),
            "std": np.std(ai_scores),
            "max": np.max(ai_scores),
            "min": np.min(ai_scores),
            "max_minus_mean": np.max(ai_scores) - np.mean(ai_scores),
            "skewness": skew(ai_scores),
        }

        results[label] = sample_stats

        print(f"📊 {label} ({n_samples}サンプル)")
        print(f"   Mean AI Score: {sample_stats['mean']*100:.1f}%")
        print(f"   Std:           {sample_stats['std']*100:.1f}%")
        print(f"   Max:           {sample_stats['max']*100:.1f}%")
        print(f"   Min:           {sample_stats['min']*100:.1f}%")
        print(f"   Max-Mean:      {sample_stats['max_minus_mean']*100:.1f}%")
        print(f"   Skewness:      {sample_stats['skewness']:.3f}")
        print()

    print("=" * 60)
    print("⚠️  現在のembeddingsは[CLS]トークンのみ（768次元）")
    print("   パッチ別分析には、全197トークンの特徴量が必要")
    print("   → extract_patch_embeddings.py で再抽出が必要")
    print("=" * 60)

def main():
    analyze_embeddings()

if __name__ == "__main__":
    main()
