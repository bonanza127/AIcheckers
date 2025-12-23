#!/usr/bin/env python3
"""
AIcheckers モデル精度診断ツール
カテゴリ別のAI検出率、改善推奨、バックエンド状態を一括表示
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# パス設定
PROJECT_ROOT = Path("/home/techne/aicheckers")
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
MODEL_PATH = PROJECT_ROOT / "models" / "dinov3_classifier.pt"

# カテゴリ定義
AI_CATEGORIES = [
    "illustrious_ai",
    "pony_ai",
    "sdxl10_ai",
    "sd15_ai",
    "other_ai",
    "flux1d_ai",
    "novelai_ai",
    "pixai_ai",
]

REAL_CATEGORIES = [
    "danbooru_real",
]

# 推奨サンプル数
RECOMMENDED_SAMPLES = 3000


def load_model(device):
    """分類器をロード"""
    if not MODEL_PATH.exists():
        return None, None

    checkpoint = torch.load(MODEL_PATH, map_location=device)

    if isinstance(checkpoint, dict) and "classifier" in checkpoint:
        state_dict = checkpoint["classifier"]
        val_acc = checkpoint.get("val_acc", None)
    else:
        state_dict = checkpoint
        val_acc = None

    model = nn.Linear(768, 2).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    return model, val_acc


def get_detection_rate(model, embeddings, device):
    """AI検出率を計算"""
    X = torch.FloatTensor(embeddings).to(device)

    with torch.no_grad():
        logits = model(X)
        preds = logits.argmax(dim=1)
        ai_rate = (preds == 1).float().mean().item() * 100

    return ai_rate


def make_bar(rate, width=20):
    """プログレスバーを生成"""
    filled = int(rate / 100 * width)
    return "█" * filled + "░" * (width - filled)


def check_backend():
    """バックエンド状態を確認"""
    try:
        result = subprocess.run(
            ["systemctl", "--user", "is-active", "aicheckers-backend"],
            capture_output=True, text=True, timeout=5
        )
        is_active = result.stdout.strip() == "active"

        if is_active:
            # 応答速度テスト
            start = time.time()
            health = subprocess.run(
                ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
                 "http://localhost:8000/health"],
                capture_output=True, text=True, timeout=10
            )
            latency_ms = (time.time() - start) * 1000

            if health.stdout.strip() == "200":
                return True, latency_ms
            else:
                return False, None
        else:
            return False, None
    except Exception:
        return False, None


def run_diagnosis(verbose=False, quick=False):
    """メイン診断実行"""
    print()
    print("🔍 Moonlight V1.3 診断レポート")
    print("━" * 50)

    # クイックモード: バックエンドのみ
    if quick:
        print()
        is_running, latency = check_backend()
        if is_running:
            print(f"🏥 バックエンド: ✓ 稼働中 (応答 {latency:.0f}ms)")
        else:
            print("🏥 バックエンド: ✗ 停止中")
        print()
        return 0

    # モデルロード
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, val_acc = load_model(device)

    if model is None:
        print("❌ モデルが見つかりません")
        return 1

    # カテゴリ別検出率
    print()
    print("📊 カテゴリ別AI検出率")

    results = []
    issues = []

    for cat in AI_CATEGORIES:
        npy_path = EMBEDDINGS_DIR / f"{cat}.npy"
        if not npy_path.exists():
            print(f"  {cat}: NOT FOUND")
            continue

        emb = np.load(npy_path)
        rate = get_detection_rate(model, emb, device)
        count = len(emb)

        bar = make_bar(rate)
        status = "✓" if rate >= 90 else "⚠️"

        if verbose:
            print(f"  {cat:18} {bar} {rate:5.1f}%  {status} ({count:,}件)")
        else:
            print(f"  {cat:18} {bar} {rate:5.1f}%  {status}")

        results.append((cat, rate, count))

        if rate < 90:
            issues.append((cat, rate, count))

    # Real検出率（参考）
    if verbose:
        print()
        print("📊 Real判定率（参考）")
        for cat in REAL_CATEGORIES:
            npy_path = EMBEDDINGS_DIR / f"{cat}.npy"
            if not npy_path.exists():
                continue

            emb = np.load(npy_path)
            X = torch.FloatTensor(emb).to(device)

            with torch.no_grad():
                logits = model(X)
                preds = logits.argmax(dim=1)
                real_rate = (preds == 0).float().mean().item() * 100

            bar = make_bar(real_rate)
            status = "✓" if real_rate >= 90 else "⚠️"
            print(f"  {cat:18} {bar} {real_rate:5.1f}%  {status} ({len(emb):,}件)")

    # 全体精度
    print()
    if val_acc:
        print(f"📈 全体精度: {val_acc*100:.2f}%")

    # 改善推奨
    if issues:
        print()
        print("⚠️ 改善推奨:")
        for cat, rate, count in issues:
            if count < RECOMMENDED_SAMPLES:
                print(f"  • {cat}: {count:,}件 → {RECOMMENDED_SAMPLES:,}件以上推奨")
            else:
                print(f"  • {cat}: 精度{rate:.1f}% - データ品質確認推奨")

    # バックエンド状態
    print()
    is_running, latency = check_backend()
    if is_running:
        print(f"🏥 バックエンド: ✓ 稼働中 (応答 {latency:.0f}ms)")
    else:
        print("🏥 バックエンド: ✗ 停止中")

    print()
    return 0


def main():
    parser = argparse.ArgumentParser(description="AIcheckers モデル精度診断")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="詳細統計を表示")
    parser.add_argument("--quick", "-q", action="store_true",
                        help="バックエンド状態のみ確認")
    args = parser.parse_args()

    sys.exit(run_diagnosis(verbose=args.verbose, quick=args.quick))


if __name__ == "__main__":
    main()
