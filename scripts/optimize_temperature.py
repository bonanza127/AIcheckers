#!/usr/bin/env python3
"""
Temperature Scaling最適化

Expected Calibration Error (ECE) を最小化するTemperature値を探索
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

EMBEDDINGS_DIR = Path("/home/techne/aicheckers/embeddings")
MODEL_PATH = Path("/home/techne/aicheckers/models/dinov3_classifier.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_embeddings(name: str):
    """CLSトークン + パッチ統計量を結合してロード"""
    cls = np.load(EMBEDDINGS_DIR / f"{name}.npy")
    stats = np.load(EMBEDDINGS_DIR / f"{name}_patch_stats.npy")
    return np.concatenate([cls, stats], axis=1)


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """
    Expected Calibration Error を計算

    Args:
        probs: 予測確率 (N,) - AI確率
        labels: 正解ラベル (N,) - 1=AI, 0=Human
        n_bins: ビン数

    Returns:
        ECE値 (0が最良)
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (probs > bin_lower) & (probs <= bin_upper)

        if in_bin.sum() > 0:
            bin_accuracy = labels[in_bin].mean()
            bin_confidence = probs[in_bin].mean()
            bin_weight = in_bin.sum() / len(probs)
            ece += bin_weight * abs(bin_accuracy - bin_confidence)

    return ece


def compute_metrics(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5):
    """精度指標を計算"""
    preds = (probs > threshold).astype(int)
    accuracy = (preds == labels).mean()

    # AI画像の検出率
    ai_mask = labels == 1
    ai_recall = preds[ai_mask].mean() if ai_mask.sum() > 0 else 0

    # Human画像の正解率
    human_mask = labels == 0
    human_acc = (1 - preds[human_mask]).mean() if human_mask.sum() > 0 else 0

    return accuracy, ai_recall, human_acc


def main():
    print("=" * 60)
    print("Temperature Scaling 最適化")
    print("=" * 60)

    # モデルロード
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    input_dim = checkpoint.get("input_dim", 768)
    classifier = nn.Linear(input_dim, 2).to(DEVICE)
    classifier.load_state_dict(checkpoint["classifier"])
    classifier.eval()
    print(f"Loaded classifier: {input_dim}d → 2")

    # 検証データ準備
    print("\n[1] 検証データ読み込み...")

    # AI画像（テスト用）
    ai_data = load_embeddings("novelai_ai")  # 学習に使っていないデータ
    print(f"  AI (novelai_ai): {len(ai_data)}")

    # Human画像（一部を検証用に）
    human_full = load_embeddings("danbooru_real")
    np.random.seed(42)
    indices = np.random.permutation(len(human_full))
    human_data = human_full[indices[-5000:]]  # 最後の5000枚を検証用
    print(f"  Human (danbooru_real validation): {len(human_data)}")

    # 結合
    X = np.vstack([ai_data, human_data])
    y = np.array([1] * len(ai_data) + [0] * len(human_data))

    X_tensor = torch.tensor(X, dtype=torch.float32, device=DEVICE)

    # 生のlogitsを取得
    print("\n[2] Logits計算中...")
    with torch.no_grad():
        logits = classifier(X_tensor)  # (N, 2)

    logits_np = logits.cpu().numpy()

    # 各Temperature値で評価
    print("\n[3] Temperature探索...")
    print("-" * 70)
    print(f"{'T':>6} | {'ECE':>8} | {'Accuracy':>8} | {'AI検出率':>8} | {'Human正解率':>10}")
    print("-" * 70)

    temperatures = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0]
    results = []

    for T in temperatures:
        # Temperature適用
        scaled_logits = logits_np / T
        probs = np.exp(scaled_logits) / np.exp(scaled_logits).sum(axis=1, keepdims=True)
        ai_probs = probs[:, 1]

        # 評価
        ece = compute_ece(ai_probs, y)
        accuracy, ai_recall, human_acc = compute_metrics(ai_probs, y)

        results.append({
            'T': T,
            'ece': ece,
            'accuracy': accuracy,
            'ai_recall': ai_recall,
            'human_acc': human_acc
        })

        print(f"{T:>6.1f} | {ece:>8.4f} | {accuracy*100:>7.2f}% | {ai_recall*100:>7.2f}% | {human_acc*100:>9.2f}%")

    print("-" * 70)

    # 最適値を見つける
    best_ece = min(results, key=lambda x: x['ece'])
    best_balanced = max(results, key=lambda x: x['ai_recall'] * 0.6 + x['human_acc'] * 0.4)  # AI検出重視

    print(f"\n[4] 結果:")
    print(f"  ECE最小: T={best_ece['T']:.1f} (ECE={best_ece['ece']:.4f})")
    print(f"  バランス最良: T={best_balanced['T']:.1f} (AI検出={best_balanced['ai_recall']*100:.1f}%, Human={best_balanced['human_acc']*100:.1f}%)")
    print(f"  現在の設定: T=1.5")

    # 推奨
    print(f"\n[5] 推奨:")
    if best_ece['T'] < 1.5:
        print(f"  → T={best_ece['T']:.1f} に変更を推奨（ECEが改善）")
    else:
        print(f"  → 現在のT=1.5を維持")


if __name__ == "__main__":
    main()
