#!/usr/bin/env python3
"""
モデルテストスクリプト
学習後に必ず実行して、実世界での検出率を確認する

使用方法:
    python3 scripts/test_model.py --model models/dinov3_classifier.pt
"""
import argparse
import glob
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from pathlib import Path

# 設定
HF_TOKEN = "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"
MODEL_NAME = "facebook/dinov3-vitb16-pretrain-lvd1689m"
TEMPERATURE = 1.5

# テスト用フォルダ
TEST_FOLDERS = {
    "NovelAI (AIBooru)": "data/novelai/",
    "NovelAI Combined": "data/novelai_combined/",
    "Human (Danbooru)": "data/animedl2m_dataset_release/real_images/images/",
}


def compute_patch_stats(patch_embeddings, classifier, device):
    """パッチ統計量を計算"""
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


def load_model(model_path: str, device: torch.device):
    """DINOv3と分類器をロード"""
    print(f"Loading DINOv3...")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    backbone = AutoModel.from_pretrained(
        MODEL_NAME, token=HF_TOKEN, attn_implementation="eager"
    )
    backbone.to(device)
    backbone.eval()

    print(f"Loading classifier from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    input_dim = checkpoint.get("input_dim", 768)
    use_patch_stats = checkpoint.get("use_patch_stats", input_dim > 768)
    val_acc = checkpoint.get("val_acc", "N/A")

    classifier = nn.Linear(input_dim, 2).to(device)
    classifier.load_state_dict(checkpoint["classifier"])
    classifier.eval()

    print(f"  input_dim: {input_dim}")
    print(f"  use_patch_stats: {use_patch_stats}")
    print(f"  val_acc: {val_acc if isinstance(val_acc, str) else f'{val_acc*100:.2f}%'}")

    return processor, backbone, classifier, use_patch_stats


def analyze_image(
    image_path: str,
    processor,
    backbone,
    classifier,
    use_patch_stats: bool,
    device: torch.device
) -> float:
    """画像を分析してAIスコアを返す"""
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = backbone(**inputs)
        hidden_states = outputs.last_hidden_state
        features = hidden_states[:, 0, :]  # CLS token

        if use_patch_stats:
            patch_embeddings = hidden_states[:, 5:5+196, :]
            patch_stats = compute_patch_stats(patch_embeddings, classifier, device)
            features = torch.cat([features, patch_stats], dim=1)

        logits = classifier(features)
        probs = torch.softmax(logits / TEMPERATURE, dim=1)[0]
        ai_prob = probs[1].item()

    return ai_prob * 100


def test_folder(
    folder_path: str,
    label: str,
    processor,
    backbone,
    classifier,
    use_patch_stats: bool,
    device: torch.device,
    max_images: int = 100
):
    """フォルダ内の画像をテスト"""
    images = glob.glob(f"{folder_path}/*.jpg")[:max_images//2]
    images += glob.glob(f"{folder_path}/*.png")[:max_images//2]

    if not images:
        print(f"{label}: フォルダが空または存在しない")
        return None

    scores = []
    for img_path in images[:max_images]:
        try:
            score = analyze_image(
                img_path, processor, backbone, classifier,
                use_patch_stats, device
            )
            scores.append(score)
        except Exception as e:
            print(f"  Error: {Path(img_path).name}: {e}")

    return scores


def main():
    parser = argparse.ArgumentParser(description="Test model accuracy")
    parser.add_argument("--model", type=str, default="models/dinov3_classifier.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--max-images", type=int, default=100,
                        help="Max images per category")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # モデルロード
    processor, backbone, classifier, use_patch_stats = load_model(
        args.model, device
    )

    print("\n" + "=" * 50)
    print("カテゴリ別テスト結果")
    print("=" * 50 + "\n")

    for label, folder in TEST_FOLDERS.items():
        is_ai = "Human" not in label
        scores = test_folder(
            folder, label, processor, backbone, classifier,
            use_patch_stats, device, args.max_images
        )

        if scores is None:
            continue

        avg_score = np.mean(scores)

        if is_ai:
            detected = len([s for s in scores if s >= 50])
            print(f"{label}: {detected}/{len(scores)} detected ({avg_score:.1f}% avg)")
        else:
            correct = len([s for s in scores if s < 50])
            false_positives = len([s for s in scores if s >= 50])
            high_fps = len([s for s in scores if s >= 80])
            print(f"{label}: {correct}/{len(scores)} correct ({avg_score:.1f}% avg)")
            print(f"  誤検知 (AI>=50%): {false_positives}")
            print(f"  高誤検知 (AI>=80%): {high_fps}")

    print("\n" + "=" * 50)
    print("目標値: AI検出率88%+, Human正解率98%+")
    print("=" * 50)


if __name__ == "__main__":
    main()
