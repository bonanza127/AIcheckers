#!/usr/bin/env python3
"""
Smart Scraper v2 - ダウンロード→評価→フィルタリング→embedding保存の自動パイプライン

使い方:
  # 高スコアAI画像を収集（通常モード）
  python scripts/smart_scraper.py --url "https://x.com/search?q=%23NovelAI" --limit 500

  # 低スコアAI画像を収集（Hard Negative Mining用）
  python scripts/smart_scraper.py --url "https://x.com/search?q=%23NovelAI" --limit 500 --hard-negatives

  # 既存ディレクトリをフィルタリング
  python scripts/smart_scraper.py --dir data/novelai_raw --min-score 60

  # 既存ディレクトリからHard Negativeを抽出
  python scripts/smart_scraper.py --dir data/novelai_raw --hard-negatives --max-score 50

特徴:
  - 774d/768d分類器を自動検出
  - Hard Negative Mining: 検出困難なAI画像を優先収集
  - embedding自動保存: 再学習用にCLS + patch_statsを保存
"""
import argparse
import subprocess
import tempfile
import shutil
import time
from pathlib import Path
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import numpy as np

# 設定
MODEL_PATH_774 = Path("/home/techne/aicheckers/models/dinov3_classifier.pt")  # 774d (CLS + patch_stats)
MODEL_PATH_768 = Path("/home/techne/aicheckers/models/dinov3_classifier_cls_only.pt")  # 768d CLS-only
OUTPUT_DIR = Path("/home/techne/aicheckers/data/scraped")
EMBEDDINGS_DIR = Path("/home/techne/aicheckers/embeddings")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DINOv3モデル（グローバルで1回だけロード）
processor = None
dino_model = None
classifier = None
classifier_768 = None  # パッチスコア計算用
input_dim = 768


def load_models():
    """モデルをロード（774d優先、なければ768d）"""
    global processor, dino_model, classifier, classifier_768, input_dim
    import torch.nn.functional as F

    import os
    HF_TOKEN = os.environ.get("HF_TOKEN")

    print("Loading DINOv3...")
    processor = AutoImageProcessor.from_pretrained(
        "facebook/dinov3-vitb16-pretrain-lvd1689m",
        token=HF_TOKEN,
        use_fast=True
    )
    dino_model = AutoModel.from_pretrained(
        "facebook/dinov3-vitb16-pretrain-lvd1689m",
        token=HF_TOKEN
    ).to(DEVICE).eval()

    # 774d分類器を優先ロード
    if MODEL_PATH_774.exists():
        print("Loading 774d classifier (CLS + patch_stats)...")
        checkpoint = torch.load(MODEL_PATH_774, map_location=DEVICE, weights_only=True)
        input_dim = checkpoint.get("input_dim", 774)
        classifier = nn.Linear(input_dim, 2).to(DEVICE)
        classifier.load_state_dict(checkpoint["classifier"])
        classifier.eval()

        # 768d分類器もロード（パッチスコア計算用）
        if MODEL_PATH_768.exists():
            checkpoint_768 = torch.load(MODEL_PATH_768, map_location=DEVICE, weights_only=True)
            classifier_768 = nn.Linear(768, 2).to(DEVICE)
            classifier_768.load_state_dict(checkpoint_768["classifier"])
            classifier_768.eval()
            print("  Also loaded 768d classifier for patch scoring")
    elif MODEL_PATH_768.exists():
        print("Loading 768d classifier (CLS only)...")
        checkpoint = torch.load(MODEL_PATH_768, map_location=DEVICE, weights_only=True)
        input_dim = 768
        classifier = nn.Linear(768, 2).to(DEVICE)
        classifier.load_state_dict(checkpoint["classifier"])
        classifier.eval()
        classifier_768 = classifier  # 同じものを使う
    else:
        raise FileNotFoundError("No classifier found!")

    print(f"Models loaded. Classifier: {input_dim}d, Device: {DEVICE}")
    return input_dim


def compute_patch_stats(patch_embeddings: torch.Tensor) -> np.ndarray:
    """パッチ統計量を計算（6次元）"""
    import torch.nn.functional as F
    HIGH_SCORE_THRESHOLD = 0.8

    with torch.no_grad():
        # 分類器を通してパッチごとのAIスコアを計算
        flat_patches = patch_embeddings.reshape(-1, 768)  # (196, 768)
        logits = classifier_768(flat_patches)
        probs = F.softmax(logits, dim=1)
        scores = probs[:, 1].cpu().numpy()  # (196,)
        patch_emb = patch_embeddings[0].cpu().numpy()  # (196, 768)

        stats = np.zeros(6, dtype=np.float32)
        stats[0] = np.mean(scores)                                    # patch_mean
        stats[1] = np.max(scores)                                     # patch_max
        stats[2] = np.var(scores)                                     # patch_var
        stats[3] = stats[1] - stats[0]                                # max_minus_mean
        stats[4] = patch_emb.var(axis=0).mean()                       # embed_var_mean
        stats[5] = np.sum(scores >= HIGH_SCORE_THRESHOLD) / 196       # count_high_score

    return stats


def evaluate_image(image_path: Path) -> dict:
    """画像を評価してAIスコア + embeddingを返す"""
    import torch.nn.functional as F

    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            outputs = dino_model(**inputs)
            hidden_states = outputs.last_hidden_state  # (1, 201, 768)

            # CLSトークン
            cls_token = hidden_states[:, 0, :]  # (1, 768)
            cls_np = cls_token.cpu().numpy().flatten()  # (768,)

            # パッチトークン（REGをスキップ）
            patch_emb = hidden_states[:, 5:5+196, :]  # (1, 196, 768)

            # パッチ統計量を計算
            patch_stats = compute_patch_stats(patch_emb)

            # 774d or 768d で分類
            if input_dim == 774:
                features = torch.cat([cls_token, torch.tensor(patch_stats).unsqueeze(0).to(DEVICE)], dim=1)
            else:
                features = cls_token

            logits = classifier(features)
            probs = F.softmax(logits, dim=1)
            ai_score = probs[0, 1].item() * 100

            return {
                "ai_score": ai_score,
                "cls_embedding": cls_np,
                "patch_stats": patch_stats
            }
    except Exception as e:
        print(f"  Error evaluating {image_path.name}: {e}")
        return None


def download_and_filter(url: str, limit: int, min_score: float = 60.0, max_score: float = 50.0,
                        hard_negatives: bool = False, save_embeddings: bool = True,
                        output_name: str = None):
    """
    ダウンロード→評価→フィルタリング→embedding保存

    Args:
        url: gallery-dl対応URL
        limit: ダウンロード上限
        min_score: 最小AIスコア（通常モード）
        max_score: 最大AIスコア（Hard Negativeモード）
        hard_negatives: True = 低スコア画像を収集
        save_embeddings: True = CLS + patch_statsを保存
        output_name: embedding保存時のカテゴリ名
    """
    output_dir = OUTPUT_DIR / ("hard_negatives" if hard_negatives else "scraped")
    output_dir.mkdir(parents=True, exist_ok=True)
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    load_models()

    # embedding保存用
    cls_embeddings = []
    patch_stats_list = []
    filenames = []

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        print(f"\nDownloading from: {url}")
        print(f"Temp dir: {temp_path}")
        if hard_negatives:
            print(f"[HARD NEGATIVE MODE] Filter: AI score <= {max_score}%")
        else:
            print(f"Filter: AI score >= {min_score}%")
        print(f"Save embeddings: {save_embeddings}")
        print("-" * 50)

        # gallery-dlでダウンロード（バッチで処理）
        batch_size = 50
        total_downloaded = 0
        total_kept = 0
        cursor = None

        while total_downloaded < limit:
            # gallery-dlコマンド構築
            cmd = [
                "gallery-dl",
                "-d", str(temp_path),
                "--filter", "extension in ('jpg', 'jpeg', 'png', 'webp')",
                "--sleep", "1-2",
                "--sleep-request", "0.5-1",
                "--range", f"1-{batch_size}",
            ]
            if cursor:
                cmd.extend(["-o", f"cursor={cursor}"])
            cmd.append(url)

            # ダウンロード実行
            result = subprocess.run(cmd, capture_output=True, text=True)

            # cursorを抽出
            for line in result.stderr.split('\n'):
                if "cursor=" in line:
                    cursor = line.split("cursor=")[1].strip().rstrip("'")
                    break

            # ダウンロードされた画像を評価
            images = list(temp_path.rglob("*.jpg")) + list(temp_path.rglob("*.png")) + list(temp_path.rglob("*.webp"))

            if not images:
                print("No more images to download")
                break

            for img_path in images:
                total_downloaded += 1

                # 評価
                eval_result = evaluate_image(img_path)

                if eval_result is None:
                    img_path.unlink()
                    continue

                ai_score = eval_result["ai_score"]

                # フィルタリング条件
                if hard_negatives:
                    should_keep = ai_score <= max_score
                else:
                    should_keep = ai_score >= min_score

                if should_keep:
                    # 合格 → 本番ディレクトリに移動
                    dest = output_dir / img_path.name
                    # 重複回避
                    if dest.exists():
                        dest = output_dir / f"{img_path.stem}_{int(time.time())}{img_path.suffix}"
                    shutil.move(str(img_path), str(dest))
                    total_kept += 1

                    # embedding保存
                    if save_embeddings:
                        cls_embeddings.append(eval_result["cls_embedding"])
                        patch_stats_list.append(eval_result["patch_stats"])
                        filenames.append(dest.name)

                    status = "HARD" if hard_negatives else "KEEP"
                    print(f"  [{status}] {img_path.name}: {ai_score:.1f}%")
                else:
                    # 不合格 → 削除
                    img_path.unlink()
                    # 静かにスキップ

                if total_downloaded % 100 == 0:
                    print(f"\n--- Progress: {total_downloaded}/{limit}, Kept: {total_kept} ({total_kept/total_downloaded*100:.1f}%) ---\n")

            # 次のバッチへ
            if total_downloaded >= limit:
                break

    # embedding保存
    if save_embeddings and cls_embeddings:
        name = output_name or f"scraped_{int(time.time())}"
        if hard_negatives:
            name += "_hard_neg"

        cls_path = EMBEDDINGS_DIR / f"{name}.npy"
        stats_path = EMBEDDINGS_DIR / f"{name}_patch_stats.npy"
        files_path = EMBEDDINGS_DIR / f"{name}_files.txt"

        np.save(cls_path, np.array(cls_embeddings))
        np.save(stats_path, np.array(patch_stats_list))
        with open(files_path, "w") as f:
            f.write("\n".join(filenames))

        print(f"\n[EMBEDDINGS SAVED]")
        print(f"  CLS: {cls_path} ({len(cls_embeddings)} samples)")
        print(f"  Stats: {stats_path}")
        print(f"  Files: {files_path}")

    print("\n" + "=" * 50)
    print(f"Complete!")
    print(f"Downloaded: {total_downloaded}")
    print(f"Kept: {total_kept} ({total_kept/total_downloaded*100:.1f}%)" if total_downloaded > 0 else "Kept: 0")
    print(f"Output: {output_dir}")


def filter_existing(input_dir: Path, min_score: float = 60.0, max_score: float = 100.0,
                    hard_negatives: bool = False, delete_rejects: bool = False,
                    save_embeddings: bool = True, output_name: str = None):
    """
    既存ファイルをフィルタリング

    Args:
        input_dir: 入力ディレクトリ
        min_score: 最小AIスコア（通常モード）
        max_score: 最大AIスコア（Hard Negativeモード）
        hard_negatives: True = 低スコア画像を収集
        delete_rejects: True = 不合格画像を削除
        save_embeddings: True = CLS + patch_statsを保存
        output_name: embedding保存時のカテゴリ名
    """
    output_dir = OUTPUT_DIR / ("hard_negatives" if hard_negatives else "filtered")
    output_dir.mkdir(parents=True, exist_ok=True)
    load_models()

    images = list(input_dir.rglob("*.jpg")) + list(input_dir.rglob("*.jpeg")) + \
             list(input_dir.rglob("*.png")) + list(input_dir.rglob("*.webp"))

    if hard_negatives:
        print(f"\n[HARD NEGATIVE MODE] Collecting AI images with low detection scores")
        print(f"Filter: AI score <= {max_score}%")
    else:
        print(f"\n[NORMAL MODE] Filtering high-confidence AI images")
        print(f"Filter: AI score >= {min_score}%")

    print(f"Input: {input_dir} ({len(images)} images)")
    print(f"Output: {output_dir}")
    print(f"Delete rejects: {delete_rejects}")
    print(f"Save embeddings: {save_embeddings}")
    print("-" * 50)

    total = len(images)
    kept = 0
    dropped = 0

    # embedding保存用
    cls_embeddings = []
    patch_stats_list = []
    filenames = []

    for i, img_path in enumerate(images, 1):
        result = evaluate_image(img_path)

        if result is None:
            if delete_rejects:
                img_path.unlink()
            dropped += 1
            continue

        ai_score = result["ai_score"]

        # フィルタリング条件
        if hard_negatives:
            should_keep = ai_score <= max_score
        else:
            should_keep = ai_score >= min_score

        if should_keep:
            # 合格 → 出力ディレクトリにコピー
            dest = output_dir / img_path.name
            if dest.exists():
                dest = output_dir / f"{img_path.stem}_{int(time.time())}{img_path.suffix}"
            shutil.copy2(str(img_path), str(dest))
            kept += 1

            # embedding保存
            if save_embeddings:
                cls_embeddings.append(result["cls_embedding"])
                patch_stats_list.append(result["patch_stats"])
                filenames.append(img_path.name)

            status = "HARD" if hard_negatives else "KEEP"
            print(f"[{i}/{total}] {status} {img_path.name}: {ai_score:.1f}%")
        else:
            if delete_rejects:
                img_path.unlink()
            dropped += 1
            # 静かにスキップ（大量出力を避ける）

        if i % 100 == 0:
            print(f"\n--- Progress: {i}/{total}, Kept: {kept} ({kept/i*100:.1f}%) ---\n")

    # embedding保存
    if save_embeddings and cls_embeddings:
        name = output_name or f"scraped_{int(time.time())}"
        if hard_negatives:
            name += "_hard_neg"

        cls_path = EMBEDDINGS_DIR / f"{name}.npy"
        stats_path = EMBEDDINGS_DIR / f"{name}_patch_stats.npy"
        files_path = EMBEDDINGS_DIR / f"{name}_files.txt"

        np.save(cls_path, np.array(cls_embeddings))
        np.save(stats_path, np.array(patch_stats_list))
        with open(files_path, "w") as f:
            f.write("\n".join(filenames))

        print(f"\n[EMBEDDINGS SAVED]")
        print(f"  CLS: {cls_path} ({len(cls_embeddings)} samples)")
        print(f"  Stats: {stats_path}")
        print(f"  Files: {files_path}")

    print("\n" + "=" * 50)
    print(f"Complete!")
    print(f"Total: {total}")
    print(f"Kept: {kept} ({kept/total*100:.1f}%)" if total > 0 else "Kept: 0")
    print(f"Dropped: {dropped}")
    print(f"Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Smart Scraper v2 - AI画像フィルタリング & embedding保存",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  # 高スコアAI画像を収集
  python scripts/smart_scraper.py --url "https://x.com/search?q=%23NovelAI" --limit 500

  # 低スコアAI画像を収集（Hard Negative Mining）
  python scripts/smart_scraper.py --url "https://x.com/search?q=%23NovelAI" --limit 500 --hard-negatives

  # 既存ディレクトリをフィルタリング
  python scripts/smart_scraper.py --dir data/novelai_raw --min-score 60

  # Hard Negative抽出（検出困難なAI画像）
  python scripts/smart_scraper.py --dir data/novelai_raw --hard-negatives --max-score 50
        """
    )
    parser.add_argument("--url", help="URL to scrape (gallery-dl)")
    parser.add_argument("--dir", type=Path, help="Existing directory to filter")
    parser.add_argument("--limit", type=int, default=500, help="Max images to download")

    # フィルタリング条件
    parser.add_argument("--min-score", type=float, default=60.0,
                        help="Minimum AI score to keep (normal mode)")
    parser.add_argument("--max-score", type=float, default=50.0,
                        help="Maximum AI score to keep (hard-negative mode)")
    parser.add_argument("--hard-negatives", action="store_true",
                        help="Collect low-score AI images for hard negative mining")

    # 出力オプション
    parser.add_argument("--delete", action="store_true",
                        help="Delete rejected images (for --dir mode)")
    parser.add_argument("--no-embeddings", action="store_true",
                        help="Don't save embeddings (faster)")
    parser.add_argument("--output-name", type=str,
                        help="Category name for saved embeddings")

    args = parser.parse_args()

    if args.dir:
        filter_existing(
            input_dir=args.dir,
            min_score=args.min_score,
            max_score=args.max_score,
            hard_negatives=args.hard_negatives,
            delete_rejects=args.delete,
            save_embeddings=not args.no_embeddings,
            output_name=args.output_name
        )
    elif args.url:
        download_and_filter(
            url=args.url,
            limit=args.limit,
            min_score=args.min_score,
            max_score=args.max_score,
            hard_negatives=args.hard_negatives,
            save_embeddings=not args.no_embeddings,
            output_name=args.output_name
        )
    else:
        parser.error("Either --url or --dir is required")


if __name__ == "__main__":
    main()
