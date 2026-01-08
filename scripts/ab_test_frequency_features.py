#!/usr/bin/env python3
"""
高周波特徴量 A/Bテスト

比較:
- A: CLS + Patch Stats (775次元) - ベースライン
- B1: A + DCT高周波比率 (776次元)
- B2: A + FFT Phase Variance (776次元)
- B3: A + Nyquist Spike (776次元)
- B4: A + 全部 (778次元)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from scipy.fft import dct
import warnings
warnings.filterwarnings('ignore')

EMBEDDINGS_DIR = Path("/home/techne/aicheckers/embeddings")
DATA_DIR = Path("/home/techne/aicheckers/data")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_dct_high_freq_ratio(image: Image.Image) -> float:
    """
    DCT高周波比率を計算

    高周波成分のエネルギー / 全体エネルギー
    AI画像は高周波が弱い傾向
    """
    gray = np.array(image.convert('L'), dtype=np.float32)

    # 2D DCT
    dct_coeffs = dct(dct(gray, axis=0, norm='ortho'), axis=1, norm='ortho')

    h, w = dct_coeffs.shape

    # 高周波領域のマスク（右下1/4）
    high_freq_mask = np.zeros((h, w), dtype=bool)
    high_freq_mask[h//2:, w//2:] = True

    # エネルギー計算
    total_energy = np.sum(dct_coeffs**2)
    high_freq_energy = np.sum(dct_coeffs[high_freq_mask]**2)

    ratio = high_freq_energy / (total_energy + 1e-8)
    return ratio


def compute_fft_phase_variance(image: Image.Image) -> float:
    """
    FFT位相の分散を計算

    AI画像は位相パターンが均一になりやすい（分散が低い）
    """
    gray = np.array(image.convert('L'), dtype=np.float32)

    # FFT
    f = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f)

    # 位相を取得
    phase = np.angle(f_shift)

    # 中心付近を除外（DC成分周辺）
    h, w = phase.shape
    cy, cx = h // 2, w // 2
    mask = np.ones((h, w), dtype=bool)
    mask[cy-5:cy+5, cx-5:cx+5] = False

    # 位相の分散
    phase_var = np.var(phase[mask])
    return phase_var


def compute_nyquist_spike(image: Image.Image) -> float:
    """
    Nyquist周波数付近のスパイク検出

    アップサンプリングによるアーティファクトは
    Nyquist周波数（端）に現れやすい
    """
    gray = np.array(image.convert('L'), dtype=np.float32)

    # FFT
    f = np.fft.fft2(gray)
    magnitude = np.abs(f)

    h, w = magnitude.shape

    # Nyquist周波数付近（端の5%）のエネルギー
    edge_size_h = max(1, h // 20)
    edge_size_w = max(1, w // 20)

    # 4辺のエッジ領域
    top = magnitude[:edge_size_h, :].mean()
    bottom = magnitude[-edge_size_h:, :].mean()
    left = magnitude[:, :edge_size_w].mean()
    right = magnitude[:, -edge_size_w:].mean()

    edge_energy = (top + bottom + left + right) / 4
    center_energy = magnitude[h//4:3*h//4, w//4:3*w//4].mean()

    # エッジ/中心比率（高いほどスパイクあり）
    spike_ratio = edge_energy / (center_energy + 1e-8)
    return spike_ratio


def extract_frequency_features(image_dir: Path, filenames: list) -> dict:
    """ディレクトリ内の画像から周波数特徴量を抽出"""
    dct_features = []
    phase_features = []
    nyquist_features = []

    for fname in tqdm(filenames, desc="Extracting frequency features"):
        img_path = image_dir / fname

        # ファイルが見つからない場合はサブディレクトリを探す
        if not img_path.exists():
            for subdir in image_dir.iterdir():
                if subdir.is_dir():
                    alt_path = subdir / fname
                    if alt_path.exists():
                        img_path = alt_path
                        break

        if img_path.exists():
            try:
                img = Image.open(img_path).convert('RGB')
                dct_features.append(compute_dct_high_freq_ratio(img))
                phase_features.append(compute_fft_phase_variance(img))
                nyquist_features.append(compute_nyquist_spike(img))
            except Exception as e:
                dct_features.append(0.0)
                phase_features.append(0.0)
                nyquist_features.append(0.0)
        else:
            dct_features.append(0.0)
            phase_features.append(0.0)
            nyquist_features.append(0.0)

    return {
        'dct': np.array(dct_features, dtype=np.float32).reshape(-1, 1),
        'phase': np.array(phase_features, dtype=np.float32).reshape(-1, 1),
        'nyquist': np.array(nyquist_features, dtype=np.float32).reshape(-1, 1),
    }


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
    print("=" * 70)
    print("高周波特徴量 A/Bテスト")
    print("=" * 70)

    # データ読み込み
    print("\n[1] 既存Embedding読み込み...")
    ai_emb = load_embeddings("novelai_artist_tagged_ai")
    human_emb = load_embeddings("danbooru_real")

    # Humanをサンプリング（バランス調整）
    np.random.seed(42)
    human_idx = np.random.permutation(len(human_emb))[:5000]
    human_emb = human_emb[human_idx]

    print(f"  AI (novelai_artist_tagged): {len(ai_emb)}")
    print(f"  Human (danbooru_real sample): {len(human_emb)}")

    # ファイル名読み込み
    print("\n[2] 周波数特徴量抽出...")
    ai_files = load_filenames("novelai_artist_tagged_ai")
    human_files = load_filenames("danbooru_real")
    human_files = [human_files[i] for i in human_idx]

    # 周波数特徴量抽出
    print("  AI画像から抽出中...")
    ai_freq = extract_frequency_features(DATA_DIR / "novelai_artist_tagged", ai_files)

    print("  Human画像から抽出中...")
    human_freq = extract_frequency_features(
        DATA_DIR / "animedl2m_dataset_release" / "real_images" / "images",
        human_files
    )

    # 統計表示
    print("\n[3] 周波数特徴量統計:")
    print(f"{'特徴量':<20} {'AI平均':>12} {'Human平均':>12} {'差分':>12}")
    print("-" * 60)
    for name in ['dct', 'phase', 'nyquist']:
        ai_mean = ai_freq[name].mean()
        human_mean = human_freq[name].mean()
        diff = ai_mean - human_mean
        print(f"{name:<20} {ai_mean:>12.6f} {human_mean:>12.6f} {diff:>+12.6f}")
    print("-" * 60)

    # データ準備
    X_base = np.vstack([ai_emb, human_emb])  # 775次元
    y = np.array([1] * len(ai_emb) + [0] * len(human_emb))

    # 各特徴量を結合
    ai_dct = ai_freq['dct']
    ai_phase = ai_freq['phase']
    ai_nyquist = ai_freq['nyquist']
    human_dct = human_freq['dct']
    human_phase = human_freq['phase']
    human_nyquist = human_freq['nyquist']

    X_dct = np.vstack([
        np.concatenate([ai_emb, ai_dct], axis=1),
        np.concatenate([human_emb, human_dct], axis=1)
    ])
    X_phase = np.vstack([
        np.concatenate([ai_emb, ai_phase], axis=1),
        np.concatenate([human_emb, human_phase], axis=1)
    ])
    X_nyquist = np.vstack([
        np.concatenate([ai_emb, ai_nyquist], axis=1),
        np.concatenate([human_emb, human_nyquist], axis=1)
    ])
    X_all = np.vstack([
        np.concatenate([ai_emb, ai_dct, ai_phase, ai_nyquist], axis=1),
        np.concatenate([human_emb, human_dct, human_phase, human_nyquist], axis=1)
    ])

    print(f"\n  A (baseline):    {X_base.shape}")
    print(f"  B1 (+DCT):       {X_dct.shape}")
    print(f"  B2 (+Phase):     {X_phase.shape}")
    print(f"  B3 (+Nyquist):   {X_nyquist.shape}")
    print(f"  B4 (+All):       {X_all.shape}")

    # テストデータ分離
    np.random.seed(123)
    indices = np.random.permutation(len(X_base))
    train_idx = indices[:int(len(indices) * 0.8)]
    test_idx = indices[int(len(indices) * 0.8):]

    datasets = {
        'A (baseline)': X_base,
        'B1 (+DCT)': X_dct,
        'B2 (+Phase)': X_phase,
        'B3 (+Nyquist)': X_nyquist,
        'B4 (+All)': X_all,
    }

    results = {}

    print("\n[4] 各モデル学習・テスト中...")
    print("-" * 70)

    for name, X in datasets.items():
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model, val_acc = train_classifier(X_train, y_train)
        ai_recall, human_acc = test_model(model, X_test, y_test)

        results[name] = {
            'val_acc': val_acc,
            'ai_recall': ai_recall,
            'human_acc': human_acc,
        }
        print(f"  {name:<15} Val={val_acc*100:.2f}%  AI検出={ai_recall*100:.2f}%  Human正解={human_acc*100:.2f}%")

    print("-" * 70)

    # 結果比較
    print("\n[5] ベースラインとの比較:")
    print("-" * 70)
    print(f"{'モデル':<15} {'AI検出率':>12} {'Human正解率':>12} {'AI差分':>10} {'Human差分':>10}")
    print("-" * 70)

    base_ai = results['A (baseline)']['ai_recall']
    base_human = results['A (baseline)']['human_acc']

    for name, r in results.items():
        ai_diff = r['ai_recall'] - base_ai
        human_diff = r['human_acc'] - base_human
        print(f"{name:<15} {r['ai_recall']*100:>11.2f}% {r['human_acc']*100:>11.2f}% {ai_diff*100:>+9.2f}% {human_diff*100:>+9.2f}%")

    print("-" * 70)

    # 結論
    print("\n[6] 結論:")
    best_improvement = None
    best_name = None

    for name, r in results.items():
        if name == 'A (baseline)':
            continue
        ai_diff = r['ai_recall'] - base_ai
        human_diff = r['human_acc'] - base_human

        # AI検出率が上がり、Human正解率が大きく下がらない
        if ai_diff > 0.01 and human_diff > -0.02:
            if best_improvement is None or ai_diff > best_improvement:
                best_improvement = ai_diff
                best_name = name

    if best_name:
        print(f"  ✅ {best_name} が最も効果的。AI検出率 +{best_improvement*100:.2f}%")
    else:
        # 小さな改善でもチェック
        for name, r in results.items():
            if name == 'A (baseline)':
                continue
            ai_diff = r['ai_recall'] - base_ai
            human_diff = r['human_acc'] - base_human
            if ai_diff > 0 and human_diff >= -0.01:
                print(f"  ⚠️ {name} に小さな改善あり（AI +{ai_diff*100:.2f}%, Human {human_diff*100:+.2f}%）")
                break
        else:
            print("  ❌ 有効な改善なし。周波数特徴量の効果は限定的。")


if __name__ == "__main__":
    main()
