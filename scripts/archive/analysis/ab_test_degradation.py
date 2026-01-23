#!/usr/bin/env python3
"""
劣化Augmentation A/Bテスト

比較:
- A: 劣化なし (novelai_combined_ai)
- B: 劣化あり (novelai_combined_ai_degraded)

両方ともdanbooru_realと組み合わせて学習し、テストデータで精度比較
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

EMBEDDINGS_DIR = Path("/home/techne/aicheckers/embeddings")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_embeddings(name: str):
    """CLSトークン + パッチ統計量を結合してロード"""
    cls = np.load(EMBEDDINGS_DIR / f"{name}.npy")
    stats = np.load(EMBEDDINGS_DIR / f"{name}_patch_stats.npy")
    return np.concatenate([cls, stats], axis=1)  # (N, 775)


def train_classifier(ai_data, human_data, epochs=50, lr=0.001):
    """シンプルなLinear Probeを学習"""
    # ラベル作成
    X = np.vstack([ai_data, human_data])
    y = np.array([1] * len(ai_data) + [0] * len(human_data))

    # Train/Val分割（手動実装）
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    split_idx = int(len(X) * 0.9)
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Tensor化
    X_train = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
    y_train = torch.tensor(y_train, dtype=torch.long, device=DEVICE)
    X_val = torch.tensor(X_val, dtype=torch.float32, device=DEVICE)
    y_val = torch.tensor(y_val, dtype=torch.long, device=DEVICE)

    # モデル
    model = nn.Linear(X_train.shape[1], 2).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_preds = val_logits.argmax(dim=1)
            val_acc = (val_preds == y_val).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()

    model.load_state_dict(best_state)
    return model, best_val_acc


def test_on_dataset(model, test_data, label, temp=1.5):
    """テストデータで評価"""
    model.eval()
    X = torch.tensor(test_data, dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        logits = model(X) / temp
        probs = F.softmax(logits, dim=1)
        preds = probs[:, 1] > 0.5  # AI確率 > 0.5

        if label == 1:  # AI
            acc = preds.float().mean().item()
        else:  # Human
            acc = (~preds).float().mean().item()

    return acc


def main():
    print("=" * 60)
    print("劣化Augmentation A/Bテスト")
    print("=" * 60)

    # Human画像（共通）
    print("\n[1] データ読み込み...")
    human_data = load_embeddings("danbooru_real")
    print(f"  Human (danbooru_real): {len(human_data)}")

    # AI画像（劣化なし/あり）
    ai_normal = load_embeddings("novelai_combined_ai")
    ai_degraded = load_embeddings("novelai_combined_ai_degraded")
    print(f"  AI (劣化なし): {len(ai_normal)}")
    print(f"  AI (劣化あり): {len(ai_degraded)}")

    # テストデータ準備（Human側も一部を使う）
    # 学習にはHumanの90%を使用、テストに10%
    np.random.seed(42)
    human_indices = np.random.permutation(len(human_data))
    human_train = human_data[human_indices[:int(len(human_data) * 0.9)]]
    human_test = human_data[human_indices[int(len(human_data) * 0.9):]]

    print(f"  Human学習用: {len(human_train)}, テスト用: {len(human_test)}")

    # テスト用AI画像（学習に使わないもの）
    # novelai_test_new があれば使う
    test_ai_path = EMBEDDINGS_DIR / "novelai_test_new.npy"
    if test_ai_path.exists():
        test_ai = load_embeddings("novelai_test_new")
        print(f"  AIテスト用 (novelai_test_new): {len(test_ai)}")
    else:
        # なければnovelaiを使う
        test_ai = load_embeddings("novelai_ai")
        print(f"  AIテスト用 (novelai_ai): {len(test_ai)}")

    # A: 劣化なしで学習
    print("\n[2] モデルA（劣化なし）学習中...")
    model_a, val_acc_a = train_classifier(ai_normal, human_train)
    print(f"  Validation Accuracy: {val_acc_a*100:.2f}%")

    # B: 劣化ありで学習
    print("\n[3] モデルB（劣化あり）学習中...")
    model_b, val_acc_b = train_classifier(ai_degraded, human_train)
    print(f"  Validation Accuracy: {val_acc_b*100:.2f}%")

    # テスト
    print("\n[4] テスト結果:")
    print("-" * 60)
    print(f"{'データセット':<30} {'モデルA(劣化なし)':<18} {'モデルB(劣化あり)':<18}")
    print("-" * 60)

    # AI検出率
    acc_a_ai = test_on_dataset(model_a, test_ai, label=1)
    acc_b_ai = test_on_dataset(model_b, test_ai, label=1)
    diff_ai = acc_b_ai - acc_a_ai
    print(f"{'AI検出率 (テストAI)':<30} {acc_a_ai*100:>6.2f}%          {acc_b_ai*100:>6.2f}%  ({diff_ai*100:+.2f}%)")

    # Human正解率
    acc_a_human = test_on_dataset(model_a, human_test, label=0)
    acc_b_human = test_on_dataset(model_b, human_test, label=0)
    diff_human = acc_b_human - acc_a_human
    print(f"{'Human正解率 (テストHuman)':<30} {acc_a_human*100:>6.2f}%          {acc_b_human*100:>6.2f}%  ({diff_human*100:+.2f}%)")

    print("-" * 60)

    # 総合判定
    print("\n[5] 結論:")
    if diff_ai > 0.01 and diff_human > -0.02:
        print("  ✅ 劣化Augmentationは効果あり。全データ再抽出を推奨。")
    elif diff_ai > 0 and diff_human >= 0:
        print("  ⚠️ 若干の改善あり。さらなる検証を推奨。")
    elif diff_ai < 0:
        print("  ❌ 劣化AugmentationはAI検出率を下げている。採用非推奨。")
    else:
        print("  ➖ 有意な差なし。他の改善を優先すべき。")


if __name__ == "__main__":
    main()
