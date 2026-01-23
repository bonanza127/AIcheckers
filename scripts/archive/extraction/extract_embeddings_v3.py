#!/usr/bin/env python3
"""
v3パッチ統計量抽出スクリプト（ローカル最適化版）
33次元のGPU only統計量を高速抽出（pca_lowrank + band features）

Usage:
    # 全カテゴリ抽出
    python scripts/extract_embeddings_v3.py --all

    # 単一カテゴリ
    python scripts/extract_embeddings_v3.py --category novelai_ai

    # 進捗確認
    python scripts/extract_embeddings_v3.py --status
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from lib.patch_stats import compute_patch_stats_v3, V3_STAT_NAMES

# ============================================================================
# Configuration
# ============================================================================

EMBEDDINGS_DIR = Path("/home/techne/aicheckers/embeddings")
MODEL_DIR = Path("/home/techne/aicheckers/models/dinov3-vitb16")
MID_LAYER_INDEX = 6

# GTX 1660 (6GB) に最適化
BATCH_SIZE = 8
NUM_WORKERS = 4

# カテゴリと画像パスのマッピング
DATA_ROOT = Path("/home/techne/aicheckers/data")
ANIMEDL_ROOT = DATA_ROOT / "animedl2m_dataset_release"

CATEGORY_PATHS = {
    # AI Categories (CLAUDE.md 2026-01-10 確認済み)
    "illustrious_ai": ANIMEDL_ROOT / "civitai_subset/image/Illustrious",
    "pony_ai": ANIMEDL_ROOT / "civitai_subset/image/Pony",
    "sdxl10_ai": ANIMEDL_ROOT / "civitai_subset/image/SDXL 1.0",
    "sd15_ai": ANIMEDL_ROOT / "civitai_subset/image/SD 1.5",
    "other_ai": ANIMEDL_ROOT / "civitai_subset/image/Other",
    "flux1d_ai": ANIMEDL_ROOT / "civitai_subset/image/Flux.1 D",
    "novelai_ai": DATA_ROOT / "novelai",
    "pixai_ai": DATA_ROOT / "pixai",
    "novelai_combined_ai": DATA_ROOT / "novelai_combined",
    "novelai_artist_tagged_ai": DATA_ROOT / "novelai_artist_tagged",
    # Real Categories
    "danbooru_real": ANIMEDL_ROOT / "real_images/images",
}

# 学習に使うカテゴリ
AI_CATEGORIES = [
    "illustrious_ai", "pony_ai", "sdxl10_ai", "sd15_ai", "other_ai",
    "flux1d_ai", "novelai_ai", "pixai_ai",
    "novelai_combined_ai", "novelai_artist_tagged_ai",
]
REAL_CATEGORIES = ["danbooru_real"]


# ============================================================================
# Dataset
# ============================================================================

class ImageDataset(Dataset):
    def __init__(self, image_paths, processor):
        self.image_paths = image_paths
        self.processor = processor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            img = Image.open(path).convert("RGB")
            inputs = self.processor(images=img, return_tensors="pt")
            pixel_values = inputs["pixel_values"].squeeze(0)
            return pixel_values, path.name, True
        except Exception as e:
            # Return dummy on error
            dummy = torch.zeros(3, 224, 224)
            return dummy, path.name, False


def collate_fn(batch):
    pixels, names, valids = zip(*batch)
    valid_idx = [i for i, v in enumerate(valids) if v]
    if not valid_idx:
        return None, [], []
    pixels = torch.stack([pixels[i] for i in valid_idx])
    names = [names[i] for i in valid_idx]
    return pixels, names, valid_idx


# ============================================================================
# Extraction
# ============================================================================

def _save_checkpoint(checkpoint_file, cls_list, stats_list, files_list, prev_checkpoint=None):
    """チェックポイントを保存"""
    cls_arr = np.concatenate(cls_list, axis=0) if cls_list else np.array([])
    stats_arr = np.concatenate(stats_list, axis=0) if stats_list else np.array([])

    # 以前のチェックポイントとマージ
    if prev_checkpoint is not None:
        cls_arr = np.concatenate([prev_checkpoint["cls"], cls_arr], axis=0)
        stats_arr = np.concatenate([prev_checkpoint["stats"], stats_arr], axis=0)
        files_list = prev_checkpoint["files"].tolist() + files_list

    np.savez(
        checkpoint_file,
        cls=cls_arr.astype(np.float32),
        stats=stats_arr.astype(np.float32),
        files=np.array(files_list),
    )


def extract_category(
    category: str,
    model,
    processor,
    device: torch.device,
    limit: int = 0,
    skip_existing: bool = True,
    batch_size: int = BATCH_SIZE,
) -> dict:
    """単一カテゴリのv3統計量を抽出（チェックポイント対応）"""

    # 出力パス
    out_cls = EMBEDDINGS_DIR / f"{category}.npy"
    out_stats = EMBEDDINGS_DIR / f"{category}_patch_stats_v3.npy"
    out_files = EMBEDDINGS_DIR / f"{category}_files.txt"
    checkpoint_file = EMBEDDINGS_DIR / f"{category}_v3_checkpoint.npz"

    # 既存チェック（完了済み）
    if skip_existing and out_stats.exists():
        existing = np.load(out_stats)
        print(f"[SKIP] {category}: v3 already exists ({len(existing)} samples)")
        return {"status": "skipped", "samples": len(existing)}

    # チェックポイントから再開
    checkpoint_data = None
    processed_files = set()
    if checkpoint_file.exists():
        checkpoint_data = np.load(checkpoint_file, allow_pickle=True)
        processed_files = set(checkpoint_data["files"].tolist())
        print(f"[RESUME] {category}: {len(processed_files)} samples already processed")

    # 画像パス取得
    img_dir = CATEGORY_PATHS.get(category)
    if img_dir is None or not img_dir.exists():
        print(f"[ERROR] {category}: directory not found")
        return {"status": "error", "error": "dir_not_found"}

    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    image_paths = sorted([
        p for p in img_dir.rglob("*")
        if p.suffix.lower() in extensions
    ])

    # 処理済みファイルを除外
    if processed_files:
        image_paths = [p for p in image_paths if p.name not in processed_files]
        print(f"[INFO] {category}: {len(image_paths)} remaining after checkpoint")

    if limit > 0:
        image_paths = image_paths[:limit]

    if not image_paths:
        if processed_files:
            print(f"[DONE] {category}: all images already processed")
            return {"status": "completed", "samples": len(processed_files)}
        print(f"[ERROR] {category}: no images found")
        return {"status": "error", "error": "no_images"}

    print(f"[START] {category}: {len(image_paths)} images")

    # DataLoader
    dataset = ImageDataset(image_paths, processor)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=2,
    )

    # 抽出
    cls_list = []
    stats_list = []
    files_list = []

    start_time = time.time()

    with torch.no_grad():
        for batch_idx, (pixels, names, _) in enumerate(tqdm(loader, desc=category)):
            if pixels is None:
                continue

            pixels = pixels.to(device, non_blocking=True)

            # Forward
            outputs = model(pixels, output_hidden_states=True)

            # CLS (final layer)
            cls = outputs.last_hidden_state[:, 0, :]  # (B, 768)

            # Mid-layer patches (skip CLS + 4 register tokens)
            # +1 because index 0 is initial embedding
            hidden_states = outputs.hidden_states
            mid_layer = hidden_states[MID_LAYER_INDEX + 1]  # (B, 201, 768)
            mid_cls = mid_layer[:, 0, :]  # (B, 768)
            mid_patches = mid_layer[:, 5:, :]  # (B, 196, 768) - skip CLS + 4 registers

            # v3 stats
            stats = compute_patch_stats_v3(mid_patches, mid_cls)  # (B, 24)

            cls_list.append(cls.cpu().numpy())
            stats_list.append(stats.cpu().numpy())
            files_list.extend(names)

            # Progress & checkpoint every 500 batches
            if (batch_idx + 1) % 100 == 0:
                elapsed = time.time() - start_time
                speed = len(files_list) / elapsed
                eta = (len(image_paths) - len(files_list)) / speed
                tqdm.write(f"  {len(files_list)}/{len(image_paths)} | {speed:.1f} img/s | ETA {eta/60:.1f} min")

            # Save checkpoint every 500 batches
            if (batch_idx + 1) % 500 == 0:
                _save_checkpoint(checkpoint_file, cls_list, stats_list, files_list, checkpoint_data)
                tqdm.write(f"  [CHECKPOINT] saved {len(files_list)} samples")

    # マージ with checkpoint data if exists
    if checkpoint_data is not None:
        cls_list.insert(0, checkpoint_data["cls"])
        stats_list.insert(0, checkpoint_data["stats"])
        files_list = checkpoint_data["files"].tolist() + files_list

    # 保存
    cls_arr = np.concatenate(cls_list, axis=0).astype(np.float32)
    stats_arr = np.concatenate(stats_list, axis=0).astype(np.float32)

    np.save(out_cls, cls_arr)
    np.save(out_stats, stats_arr)
    with open(out_files, "w") as f:
        f.write("\n".join(files_list))

    # チェックポイント削除
    if checkpoint_file.exists():
        checkpoint_file.unlink()

    elapsed = time.time() - start_time
    speed = len(files_list) / elapsed

    print(f"[DONE] {category}: {len(files_list)} samples in {elapsed/60:.1f} min ({speed:.1f} img/s)")
    print(f"  CLS: {out_cls} ({cls_arr.shape})")
    print(f"  Stats v3: {out_stats} ({stats_arr.shape})")

    return {
        "status": "success",
        "samples": len(files_list),
        "time_sec": elapsed,
        "speed": speed,
    }


def show_status():
    """抽出状況を表示"""
    print("=== v3 Extraction Status ===\n")

    all_cats = AI_CATEGORIES + REAL_CATEGORIES

    total_done = 0
    total_pending = 0

    for cat in all_cats:
        v3_path = EMBEDDINGS_DIR / f"{cat}_patch_stats_v3.npy"
        v2_path = EMBEDDINGS_DIR / f"{cat}_patch_stats.npy"

        if v3_path.exists():
            n = len(np.load(v3_path))
            print(f"  [✓] {cat}: {n:,} samples (v3)")
            total_done += n
        elif v2_path.exists():
            n = len(np.load(v2_path))
            print(f"  [ ] {cat}: {n:,} samples (v2 only)")
            total_pending += n
        else:
            print(f"  [?] {cat}: not found")

    print(f"\nv3 done: {total_done:,} | pending: {total_pending:,}")


def main():
    parser = argparse.ArgumentParser(description="v3 patch stats extraction")
    parser.add_argument("--category", type=str, help="Single category to extract")
    parser.add_argument("--all", action="store_true", help="Extract all categories")
    parser.add_argument("--status", action="store_true", help="Show extraction status")
    parser.add_argument("--limit", type=int, default=0, help="Limit images per category")
    parser.add_argument("--force", action="store_true", help="Force re-extraction")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    args = parser.parse_args()

    if args.status:
        show_status()
        return

    if not args.category and not args.all:
        parser.print_help()
        return

    # GPU check (require CUDA)
    if not torch.cuda.is_available():
        print("[ERROR] CUDA not available. This script requires GPU.")
        return
    device = torch.device("cuda")
    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load model
    print(f"\nLoading DINOv3 from {MODEL_DIR}...")
    from transformers import AutoImageProcessor, AutoModel
    processor = AutoImageProcessor.from_pretrained(MODEL_DIR)
    model = AutoModel.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()
    print("Model loaded.\n")

    # Update batch size if specified
    batch_size = args.batch_size

    # Categories to process
    if args.all:
        categories = AI_CATEGORIES + REAL_CATEGORIES
    else:
        categories = [args.category]

    # Extract
    results = []
    total_start = time.time()

    for cat in categories:
        result = extract_category(
            cat, model, processor, device,
            limit=args.limit,
            skip_existing=not args.force,
            batch_size=batch_size,
        )
        results.append((cat, result))

    # Summary
    total_time = time.time() - total_start
    print("\n" + "=" * 60)
    print("=== Summary ===")
    print("=" * 60)

    total_samples = 0
    for cat, result in results:
        status = result.get("status", "unknown")
        samples = result.get("samples", 0)
        if status == "success":
            total_samples += samples
            print(f"  {cat}: {samples:,} samples")
        else:
            print(f"  {cat}: {status}")

    print(f"\nTotal: {total_samples:,} samples in {total_time/60:.1f} min")
    print(f"Average speed: {total_samples / total_time:.1f} img/s")


if __name__ == "__main__":
    main()
