#!/usr/bin/env python3
"""
patch_stats_v3 のみを高速抽出するスクリプト
既存のfiles.txtを使用し、GPU統計(34d)のみを抽出
"""
import sys
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel
from concurrent.futures import ThreadPoolExecutor
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))
from lib.patch_stats import compute_patch_stats_v3

DINOV3_MODEL_PATH = Path("/home/techne/aicheckers/models/dinov3-vitb16")
EMB_DIR = Path("/home/techne/aicheckers/embeddings")
DATA_DIR = Path("/home/techne/aicheckers/data")
MID_LAYER_INDEX = 6


def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(str(DINOV3_MODEL_PATH))
    model = AutoModel.from_pretrained(str(DINOV3_MODEL_PATH))
    model.to(device)
    model.eval()
    print(f"[INFO] DINOv3 loaded on {device}")
    return model, processor, device


def load_image(path):
    try:
        return Image.open(path).convert("RGB")
    except:
        return None


def extract_for_category(cat, model, processor, device, batch_size=32, base_dirs=None, checkpoint_interval=50):
    """カテゴリのpatch_stats_v3を抽出（チェックポイント対応）"""
    files_path = EMB_DIR / f"{cat}_files.txt"
    output_path = EMB_DIR / f"{cat}_patch_stats_v3.npy"
    checkpoint_path = EMB_DIR / f"{cat}_patch_stats_v3_checkpoint.npy"

    if not files_path.exists():
        print(f"[SKIP] {cat}: files.txt not found")
        return

    if output_path.exists():
        print(f"[SKIP] {cat}: patch_stats_v3 already exists")
        return

    # ファイルリスト読み込み
    with open(files_path) as f:
        filenames = [line.strip() for line in f if line.strip()]

    print(f"\n[{cat}] {len(filenames)} files")

    # チェックポイントから再開
    start_idx = 0
    all_stats = []
    if checkpoint_path.exists():
        checkpoint_data = np.load(checkpoint_path)
        all_stats = list(checkpoint_data)
        start_idx = len(all_stats)
        print(f"  Resuming from checkpoint: {start_idx}/{len(filenames)} completed")

    # 画像パス解決
    image_paths = []
    for fn in filenames:
        found = False
        for base in base_dirs:
            candidate = base / fn
            if candidate.exists():
                image_paths.append(candidate)
                found = True
                break
        if not found:
            # ファイル名だけで検索
            for base in base_dirs:
                for p in base.rglob(Path(fn).name):
                    image_paths.append(p)
                    found = True
                    break
                if found:
                    break
        if not found:
            image_paths.append(None)

    valid_count = sum(1 for p in image_paths if p is not None)
    print(f"  Found {valid_count}/{len(filenames)} files")

    # 抽出（チェックポイントから再開）
    batch_count = 0
    total_batches = (len(image_paths) - start_idx + batch_size - 1) // batch_size

    with ThreadPoolExecutor(max_workers=8) as executor:
        for i in tqdm(range(start_idx, len(image_paths), batch_size), desc=f"  {cat}", initial=start_idx//batch_size, total=(len(image_paths)+batch_size-1)//batch_size):
            batch_paths = image_paths[i:i+batch_size]

            # 並列画像読み込み
            images = list(executor.map(load_image, batch_paths))
            valid_images = [img for img in images if img is not None]

            if not valid_images:
                # 無効な画像はゼロで埋める
                all_stats.extend([np.zeros(34, dtype=np.float32) for _ in batch_paths])
            else:
                # 処理
                inputs = processor(images=valid_images, return_tensors="pt").to(device)

                with torch.no_grad():
                    outputs = model(**inputs, output_hidden_states=True)
                    mid_hidden = outputs.hidden_states[MID_LAYER_INDEX]
                    patch_tokens = mid_hidden[:, 5:, :]  # register tokens skip
                    cls_tokens = outputs.last_hidden_state[:, 0, :]  # CLS token

                    stats = compute_patch_stats_v3(patch_tokens, cls_tokens).cpu().numpy()

                # 結果をマッピング
                valid_idx = 0
                for img in images:
                    if img is not None:
                        all_stats.append(stats[valid_idx])
                        valid_idx += 1
                    else:
                        all_stats.append(np.zeros(34, dtype=np.float32))

            batch_count += 1
            # チェックポイント保存（50バッチごと）
            if batch_count % checkpoint_interval == 0:
                checkpoint = np.array(all_stats, dtype=np.float32)
                np.save(checkpoint_path, checkpoint)

    # 最終保存
    result = np.array(all_stats, dtype=np.float32)
    np.save(output_path, result)
    print(f"  Saved: {output_path} ({result.shape})")

    # チェックポイント削除
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"  Removed checkpoint")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories", nargs="+", default=[
        "novelai_aibooru_ai", "pixiv_novelai_v2_ai", "twitter_novelai_v2_ai"
    ])
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    # 画像検索用ディレクトリ
    base_dirs = [
        DATA_DIR / "aibooru_new",
        DATA_DIR / "novelai",
        DATA_DIR / "novelai_combined",
        DATA_DIR / "twitter" / "twitter",
        DATA_DIR / "twitter" / "twitter_novelai_filtered",
        DATA_DIR / "twitter" / "duplicates",
        DATA_DIR / "pixiv" / "images",
        DATA_DIR,
    ]

    model, processor, device = load_model()

    for cat in args.categories:
        extract_for_category(cat, model, processor, device, args.batch_size, base_dirs)

    print("\n[DONE]")


if __name__ == "__main__":
    main()
