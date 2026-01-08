#!/usr/bin/env python3
"""
Radial Energy特徴量のA/Bテスト

比較:
- A: CLS + Patch Stats (775次元)
- B: CLS + Patch Stats + Radial Energy (778次元)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from tqdm import tqdm

EMBEDDINGS_DIR = Path("/home/techne/aicheckers/embeddings")
DATA_DIR = Path("/home/techne/aicheckers/data")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_radial_energy(image: Image.Image) -> np.ndarray:
    """
    画像からRadial Energy特徴量を抽出（3次元）

    Returns:
        [mid_freq_ratio, high_freq_ratio, high_low_ratio]
    """
    # グレースケール化
    gray = np.array(image.convert('L'), dtype=np.float32)

    # FFT
    f = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f)
    magnitude = np.log1p(np.abs(f_shift))

    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    max_r = min(cy, cx)

    # パーセンタイルで周波数帯を定義（サイズ非依存）
    low_r = int(max_r * 0.1)    # 0-10%: 低周波
    mid_r = int(max_r * 0.3)    # 10-30%: 中周波
    high_r = int(max_r * 0.5)   # 30-50%: 高周波

    # 距離マップ
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx)**2 + (y - cy)**2)

    # 各周波数帯のエネルギー
    low_mask = r < low_r
    mid_mask = (r >= low_r) & (r < mid_r)
    high_mask = (r >= mid_r) & (r < high_r)

    low_energy = magnitude[low_mask].mean() if low_mask.sum() > 0 else 0
    mid_energy = magnitude[mid_mask].mean() if mid_mask.sum() > 0 else 0
    high_energy = magnitude[high_mask].mean() if high_mask.sum() > 0 else 0

    total = low_energy + mid_energy + high_energy + 1e-8

    return np.array([
        mid_energy / total,                    # mid_freq_ratio
        high_energy / total,                   # high_freq_ratio
        high_energy / (low_energy + 1e-8)      # high_low_ratio
    ], dtype=np.float32)


def extract_radial_features(image_dir: Path, filenames: list) -> np.ndarray:
    """ディレクトリ内の画像からRadial Energy特徴量を抽出"""
    features = []

    for fname in tqdm(filenames, desc="Extracting Radial Energy"):
        img_path = image_dir / fname
        if not img_path.exists():
            # サブディレクトリを探す
            for subdir in image_dir.iterdir():
                if subdir.is_dir():
                    alt_path = subdir / fname
                    if alt_path.exists():
                        img_path = alt_path
                        break

        if img_path.exists():
            try:
                img = Image.open(img_path).convert('RGB')
                feat = compute_radial_energy(img)
                features.append(feat)
            except Exception as e:
                # エラー時はゼロベクトル
                features.append(np.zeros(3, dtype=np.float32))
        else:
            features.append(np.zeros(3, dtype=np.float32))

    return np.array(features)


def load_embeddings(name: str):
    """CLSトークン + パッチ統計量を結合してロード"""
    cls = np.load(EMBEDDINGS_DIR / f"{name}.npy")
    stats = np.load(EMBEDDINGS_DIR / f"{name}_patch_stats.npy")
    return np.concatenate([cls, stats], axis=1)


def load_filenames(name: str) -> list:
    """ファイル名リストをロード"""
    with open(EMBEDDINGS_DIR / f"{name}_files.txt") as f:
        return [line.strip() for line in f]


def train_classifier(X, y, epochs=50, lr=0.001):
    """分類器を学習"""
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    split_idx = int(len(X) * 0.9)
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    X_train = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
    y_train = torch.tensor(y_train, dtype=torch.long, device=DEVICE)
    X_val = torch.tensor(X_val, dtype=torch.float32, device=DEVICE)
    y_val = torch.tensor(y_val, dtype=torch.long, device=DEVICE)

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


def test_model(model, X, y, temp=1.1):
    """テスト"""
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        logits = model(X_tensor) / temp
        probs = F.softmax(logits, dim=1)
        preds = probs[:, 1] > 0.5

        ai_mask = y == 1
        human_mask = y == 0

        ai_recall = preds[ai_mask].float().mean().item() if ai_mask.sum() > 0 else 0
        human_acc = (~preds[human_mask]).float().mean().item() if human_mask.sum() > 0 else 0

    return ai_recall, human_acc


def main():
    print("=" * 60)
    print("Radial Energy A/Bテスト")
    print("=" * 60)

    # データ読み込み
    print("\n[1] 既存Embedding読み込み...")
    ai_emb = load_embeddings("novelai_combined_ai")
    human_emb = load_embeddings("danbooru_real")

    # 小規模テスト用にサンプリング
    np.random.seed(42)
    ai_idx = np.random.permutation(len(ai_emb))[:2000]
    human_idx = np.random.permutation(len(human_emb))[:5000]

    ai_emb = ai_emb[ai_idx]
    human_emb = human_emb[human_idx]

    print(f"  AI: {len(ai_emb)}, Human: {len(human_emb)}")

    # ファイル名読み込み
    print("\n[2] Radial Energy抽出...")
    ai_files = load_filenames("novelai_combined_ai")
    human_files = load_filenames("danbooru_real")

    ai_files = [ai_files[i] for i in ai_idx]
    human_files = [human_files[i] for i in human_idx]

    # Radial Energy抽出
    ai_radial = extract_radial_features(DATA_DIR / "novelai_combined", ai_files)
    # danbooru_realは空なのでanimedl2mのreal_imagesを使用
    human_radial = extract_radial_features(
        DATA_DIR / "animedl2m_dataset_release" / "real_images" / "images",
        human_files
    )

    print(f"  AI Radial: {ai_radial.shape}, Human Radial: {human_radial.shape}")

    # 統計表示
    print("\n[3] Radial Energy統計:")
    print(f"  AI   - mid: {ai_radial[:, 0].mean():.4f}, high: {ai_radial[:, 1].mean():.4f}, ratio: {ai_radial[:, 2].mean():.4f}")
    print(f"  Human- mid: {human_radial[:, 0].mean():.4f}, high: {human_radial[:, 1].mean():.4f}, ratio: {human_radial[:, 2].mean():.4f}")

    # データ準備
    X_a = np.vstack([ai_emb, human_emb])  # 775次元
    X_b = np.vstack([
        np.concatenate([ai_emb, ai_radial], axis=1),
        np.concatenate([human_emb, human_radial], axis=1)
    ])  # 778次元
    y = np.array([1] * len(ai_emb) + [0] * len(human_emb))

    print(f"\n  A (baseline): {X_a.shape}")
    print(f"  B (+radial):  {X_b.shape}")

    # テストデータ分離
    np.random.seed(123)
    indices = np.random.permutation(len(X_a))
    train_idx = indices[:int(len(indices) * 0.8)]
    test_idx = indices[int(len(indices) * 0.8):]

    X_a_train, X_a_test = X_a[train_idx], X_a[test_idx]
    X_b_train, X_b_test = X_b[train_idx], X_b[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # 学習
    print("\n[4] モデルA（baseline）学習中...")
    model_a, val_acc_a = train_classifier(X_a_train, y_train)
    print(f"  Validation Accuracy: {val_acc_a*100:.2f}%")

    print("\n[5] モデルB（+Radial Energy）学習中...")
    model_b, val_acc_b = train_classifier(X_b_train, y_train)
    print(f"  Validation Accuracy: {val_acc_b*100:.2f}%")

    # テスト
    print("\n[6] テスト結果:")
    print("-" * 60)

    ai_a, human_a = test_model(model_a, X_a_test, y_test)
    ai_b, human_b = test_model(model_b, X_b_test, y_test)

    print(f"{'指標':<20} {'A (baseline)':<15} {'B (+radial)':<15} {'差分':<10}")
    print("-" * 60)
    print(f"{'AI検出率':<20} {ai_a*100:>6.2f}%        {ai_b*100:>6.2f}%        {(ai_b-ai_a)*100:+.2f}%")
    print(f"{'Human正解率':<20} {human_a*100:>6.2f}%        {human_b*100:>6.2f}%        {(human_b-human_a)*100:+.2f}%")
    print("-" * 60)

    # 結論
    print("\n[7] 結論:")
    if (ai_b - ai_a) > 0.01 and (human_b - human_a) > -0.02:
        print("  ✅ Radial Energyは効果あり。導入推奨。")
    elif (ai_b - ai_a) > 0 and (human_b - human_a) >= 0:
        print("  ⚠️ 若干の改善あり。さらなる検証を推奨。")
    elif (ai_b - ai_a) < 0:
        print("  ❌ Radial EnergyはAI検出率を下げている。導入非推奨。")
    else:
        print("  ➖ 有意な差なし。導入の優先度は低い。")


if __name__ == "__main__":
    main()
