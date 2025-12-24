#!/usr/bin/env python3
"""
AIcheckers モデル精度診断ツール
カテゴリ別のAI検出率、実画像テスト、改善推奨、バックエンド状態を一括表示
"""
import argparse
import subprocess
import sys
import time
import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# パス設定
PROJECT_ROOT = Path("/home/techne/aicheckers")
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
MODEL_PATH = PROJECT_ROOT / "models" / "dinov3_classifier.pt"

# カテゴリ定義（学習に使用）
AI_CATEGORIES = [
    "illustrious_ai",
    "pony_ai",
    "sdxl10_ai",
    "sd15_ai",
    "other_ai",
    "flux1d_ai",
    "novelai_ai",
    "novelai_aibooru_ai",
    "novelai_combined_ai",
    "pixai_ai",
    "pixiv_novelai_v2_ai",
    "twitter_novelai_v2_ai",
]

REAL_CATEGORIES = [
    "danbooru_real",
]

# 実画像テスト用フォルダ
TEST_FOLDERS = {
    # AI画像（正しい場所）
    "Illustrious": ("data/animedl2m_dataset_release/civitai_subset/image/Illustrious/", True),
    "Pony": ("data/animedl2m_dataset_release/civitai_subset/image/Pony/", True),
    "SDXL 1.0": ("data/animedl2m_dataset_release/civitai_subset/image/SDXL 1.0/", True),
    "SD 1.5": ("data/animedl2m_dataset_release/civitai_subset/image/SD 1.5/", True),
    "Flux.1 D": ("data/animedl2m_dataset_release/civitai_subset/image/Flux.1 D/", True),
    "NovelAI (AIBooru)": ("data/novelai/", True),
    "NovelAI Combined": ("data/novelai_combined/", True),
    # テスト用（data/test_images/）
    "test/illustrious": ("data/test_images/illustrious/", True),
    "test/pony": ("data/test_images/pony/", True),
    "test/flux": ("data/test_images/flux/", True),
    "test/noobai": ("data/test_images/noobai/", True),
    # Human画像
    "Human (Danbooru)": ("data/animedl2m_dataset_release/real_images/images/", False),
    "test/real": ("data/test_images/real/", False),
}

# 推奨サンプル数
RECOMMENDED_SAMPLES = 3000


def load_model(device):
    """分類器をロード"""
    if not MODEL_PATH.exists():
        return None, None, None

    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)

    if isinstance(checkpoint, dict) and "classifier" in checkpoint:
        state_dict = checkpoint["classifier"]
        val_acc = checkpoint.get("val_acc", None)
        input_dim = checkpoint.get("input_dim", 768)
    else:
        state_dict = checkpoint
        val_acc = None
        input_dim = 768

    model = nn.Linear(input_dim, 2).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    return model, val_acc, input_dim


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


def test_real_images(device, input_dim, max_images=50):
    """実画像でテスト（DINOv3 + 分類器で推論）"""
    try:
        from PIL import Image
        from transformers import AutoImageProcessor, AutoModel
    except ImportError:
        print("  ⚠️ transformers/PIL未インストール、スキップ")
        return {}

    HF_TOKEN = "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"
    MODEL_NAME = "facebook/dinov3-vitb16-pretrain-lvd1689m"

    # モデルロード
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    backbone = AutoModel.from_pretrained(
        MODEL_NAME, token=HF_TOKEN, attn_implementation="eager"
    )
    backbone.to(device)
    backbone.eval()

    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    classifier = nn.Linear(input_dim, 2).to(device)
    classifier.load_state_dict(checkpoint["classifier"])
    classifier.eval()
    use_patch_stats = input_dim > 768

    def compute_patch_stats(patch_embeddings):
        import torch.nn.functional as F
        with torch.no_grad():
            weight = classifier.weight[:, :768]
            bias = classifier.bias
            flat_patches = patch_embeddings.reshape(-1, 768)
            logits = torch.mm(flat_patches, weight.t()) + bias
            probs = torch.softmax(logits, dim=1)
            ai_scores = probs[:, 1]

            patch_mean = ai_scores.mean()
            patch_max = ai_scores.max()
            patch_var = ai_scores.var()
            max_minus_mean = patch_max - patch_mean
            embed_var_mean = patch_embeddings[0].var(dim=0).mean()
            count_high_score = (ai_scores >= 0.8).float().mean()

            patch_emb = patch_embeddings[0]
            patches_grid = patch_emb.reshape(14, 14, -1)
            v_sims = []
            for row in range(13):
                for col in range(14):
                    sim = F.cosine_similarity(
                        patches_grid[row, col].unsqueeze(0),
                        patches_grid[row + 1, col].unsqueeze(0)
                    ).item()
                    v_sims.append(sim)
            v_high_sim_85 = torch.tensor(
                sum(1 for s in v_sims if s > 0.85) / len(v_sims),
                device=device
            )

            return torch.stack([
                patch_mean, patch_max, patch_var, max_minus_mean,
                embed_var_mean, count_high_score, v_high_sim_85
            ]).unsqueeze(0)

    def analyze_image(img_path):
        image = Image.open(img_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = backbone(**inputs)
            hidden_states = outputs.last_hidden_state
            features = hidden_states[:, 0, :]

            if use_patch_stats:
                patch_embeddings = hidden_states[:, 5:5+196, :]
                patch_stats = compute_patch_stats(patch_embeddings)
                features = torch.cat([features, patch_stats], dim=1)

            logits = classifier(features)
            probs = torch.softmax(logits / 1.5, dim=1)[0]
            return probs[1].item() * 100

    results = {}
    for name, (folder, is_ai) in TEST_FOLDERS.items():
        folder_path = PROJECT_ROOT / folder
        if not folder_path.exists():
            continue

        images = list(glob.glob(str(folder_path / "*.jpg")))[:max_images//2]
        images += list(glob.glob(str(folder_path / "*.png")))[:max_images//2]

        if not images:
            continue

        scores = []
        for img_path in images[:max_images]:
            try:
                score = analyze_image(img_path)
                scores.append(score)
            except:
                pass

        if scores:
            avg = np.mean(scores)
            if is_ai:
                detected = len([s for s in scores if s >= 50])
                results[name] = {
                    "is_ai": True,
                    "detected": detected,
                    "total": len(scores),
                    "avg": avg
                }
            else:
                correct = len([s for s in scores if s < 50])
                fp = len([s for s in scores if s >= 50])
                results[name] = {
                    "is_ai": False,
                    "correct": correct,
                    "total": len(scores),
                    "fp": fp,
                    "avg": avg
                }

    return results


def run_diagnosis(verbose=False, quick=False, test_images=False):
    """メイン診断実行"""
    print()
    print("🔍 Moonlight 診断レポート")
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
    model, val_acc, input_dim = load_model(device)

    if model is None:
        print("❌ モデルが見つかりません")
        return 1

    print(f"📐 モデル次元: {input_dim}d")
    if val_acc:
        print(f"📈 Validation精度: {val_acc*100:.2f}%")

    # カテゴリ別検出率（Embeddingベース）
    print()
    print("📊 カテゴリ別AI検出率（Embedding）")

    results = []
    issues = []

    for cat in AI_CATEGORIES:
        npy_path = EMBEDDINGS_DIR / f"{cat}.npy"
        stats_path = EMBEDDINGS_DIR / f"{cat}_patch_stats.npy"
        if not npy_path.exists():
            continue

        emb = np.load(npy_path)
        if input_dim == 775 and stats_path.exists():
            stats = np.load(stats_path)
            emb = np.concatenate([emb, stats], axis=1)
        rate = get_detection_rate(model, emb, device)
        count = len(emb)

        bar = make_bar(rate)
        status = "✓" if rate >= 90 else "⚠️"

        if verbose:
            print(f"  {cat:22} {bar} {rate:5.1f}%  {status} ({count:,}件)")
        else:
            print(f"  {cat:22} {bar} {rate:5.1f}%  {status}")

        results.append((cat, rate, count))
        if rate < 90:
            issues.append((cat, rate, count))

    # Real検出率（参考）
    if verbose:
        print()
        print("📊 Real判定率（参考）")
        for cat in REAL_CATEGORIES:
            npy_path = EMBEDDINGS_DIR / f"{cat}.npy"
            stats_path = EMBEDDINGS_DIR / f"{cat}_patch_stats.npy"
            if not npy_path.exists():
                continue

            emb = np.load(npy_path)
            if input_dim == 775 and stats_path.exists():
                stats = np.load(stats_path)
                emb = np.concatenate([emb, stats], axis=1)
            X = torch.FloatTensor(emb).to(device)

            with torch.no_grad():
                logits = model(X)
                preds = logits.argmax(dim=1)
                real_rate = (preds == 0).float().mean().item() * 100

            bar = make_bar(real_rate)
            status = "✓" if real_rate >= 90 else "⚠️"
            print(f"  {cat:22} {bar} {real_rate:5.1f}%  {status} ({len(emb):,}件)")

    # 実画像テスト
    if test_images:
        print()
        print("🖼️ 実画像テスト（50枚/カテゴリ）")
        img_results = test_real_images(device, input_dim, max_images=50)

        for name, data in img_results.items():
            if data["is_ai"]:
                rate = data["detected"] / data["total"] * 100
                bar = make_bar(rate)
                status = "✓" if rate >= 85 else "⚠️"
                print(f"  {name:22} {bar} {data['detected']}/{data['total']} ({data['avg']:.1f}%) {status}")
            else:
                rate = data["correct"] / data["total"] * 100
                bar = make_bar(rate)
                status = "✓" if rate >= 95 else "⚠️"
                print(f"  {name:22} {bar} {data['correct']}/{data['total']} ok, {data['fp']} FP ({data['avg']:.1f}%) {status}")

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
    parser.add_argument("--test-images", "-t", action="store_true",
                        help="実画像でテスト（時間がかかる）")
    args = parser.parse_args()

    sys.exit(run_diagnosis(
        verbose=args.verbose,
        quick=args.quick,
        test_images=args.test_images
    ))


if __name__ == "__main__":
    main()
