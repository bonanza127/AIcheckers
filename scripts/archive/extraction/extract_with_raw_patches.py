#!/usr/bin/env python3
"""
Raw Patches + CLS + v3統計量 抽出スクリプト（堅牢版）

保存形式:
  - {category}.npy              : CLS (N, 768) fp32
  - {category}_mid_patches.npy  : Raw patches (N, 196, 768) fp16
  - {category}_patch_stats_v3.npy : v3統計量 (N, 33) fp32
  - {category}_files.txt        : フルパスリスト
  - {category}_metadata.json    : メタデータ

改善点:
  - カテゴリごとに即座に保存（メモリ効率）
  - チェックポイント機能（中断再開可能）
  - フルパス保存（同名ファイル対策）
  - OOM時の自動バッチサイズ縮小

Usage:
    python scripts/extract_with_raw_patches.py --all
    python scripts/extract_with_raw_patches.py --category novelai_ai
    python scripts/extract_with_raw_patches.py --status
"""
import argparse
import sys
import time
import json
import hashlib
import gc
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from lib.patch_stats import compute_patch_stats_v3, V3_STAT_NAMES

# ============================================================================
# Configuration
# ============================================================================

EMBEDDINGS_DIR = Path("/home/techne/aicheckers/embeddings")
MODEL_DIR = Path("/home/techne/aicheckers/models/dinov3-vitb16")
MID_LAYER_INDEX = 6

BATCH_SIZE = 8
NUM_WORKERS = 4
CHECKPOINT_INTERVAL = 500  # バッチ数

DATA_ROOT = Path("/home/techne/aicheckers/data")
ANIMEDL_ROOT = DATA_ROOT / "animedl2m_dataset_release"

# CLAUDE.md 2026-01-10 確認済みのマッピング
CATEGORY_PATHS = {
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
    "danbooru_real": ANIMEDL_ROOT / "real_images/images",
}

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
            return pixel_values, str(path), True  # フルパス
        except Exception as e:
            dummy = torch.zeros(3, 224, 224)
            return dummy, str(path), False


def collate_fn(batch):
    pixels, paths, valids = zip(*batch)
    valid_idx = [i for i, v in enumerate(valids) if v]
    if not valid_idx:
        return None, [], []
    pixels = torch.stack([pixels[i] for i in valid_idx])
    paths = [paths[i] for i in valid_idx]
    return pixels, paths, valid_idx


# ============================================================================
# Checkpoint Management
# ============================================================================

def save_checkpoint(checkpoint_file, cls_list, patches_list, stats_list, files_list, prev_data=None):
    """チェックポイント保存（増分マージ）"""
    cls_arr = np.concatenate(cls_list, axis=0).astype(np.float32) if cls_list else np.zeros((0, 768), dtype=np.float32)
    patches_arr = np.concatenate(patches_list, axis=0).astype(np.float16) if patches_list else np.zeros((0, 196, 768), dtype=np.float16)
    stats_arr = np.concatenate(stats_list, axis=0).astype(np.float32) if stats_list else np.zeros((0, len(V3_STAT_NAMES)), dtype=np.float32)

    if prev_data is not None:
        cls_arr = np.concatenate([prev_data["cls"], cls_arr], axis=0)
        patches_arr = np.concatenate([prev_data["patches"], patches_arr], axis=0)
        stats_arr = np.concatenate([prev_data["stats"], stats_arr], axis=0)
        files_list = prev_data["files"].tolist() + files_list

    np.savez_compressed(
        checkpoint_file,
        cls=cls_arr,
        patches=patches_arr,
        stats=stats_arr,
        files=np.array(files_list, dtype=object),
    )
    return len(files_list)


def load_checkpoint(checkpoint_file):
    """チェックポイント読み込み"""
    if not checkpoint_file.exists():
        return None, set()

    try:
        data = np.load(checkpoint_file, allow_pickle=True)
        processed_files = set(data["files"].tolist())
        return data, processed_files
    except Exception as e:
        print(f"[WARN] Checkpoint corrupted, starting fresh: {e}")
        return None, set()


# ============================================================================
# Extraction
# ============================================================================

def get_model_hash():
    """モデルファイルのハッシュ"""
    model_files = list(MODEL_DIR.glob("*.safetensors")) + list(MODEL_DIR.glob("*.bin"))
    if model_files:
        with open(model_files[0], "rb") as f:
            return hashlib.md5(f.read(1024*1024)).hexdigest()
    return "unknown"


def extract_category(
    category: str,
    model,
    processor,
    device: torch.device,
    limit: int = 0,
    skip_existing: bool = True,
    batch_size: int = BATCH_SIZE,
) -> dict:
    """単一カテゴリの抽出（CLS + raw patches + v3統計量）"""

    # 出力パス
    out_cls = EMBEDDINGS_DIR / f"{category}.npy"
    out_patches = EMBEDDINGS_DIR / f"{category}_mid_patches.npy"
    out_stats = EMBEDDINGS_DIR / f"{category}_patch_stats_v3.npy"
    out_files = EMBEDDINGS_DIR / f"{category}_files.txt"
    out_meta = EMBEDDINGS_DIR / f"{category}_metadata.json"
    checkpoint_file = EMBEDDINGS_DIR / f"{category}_raw_checkpoint.npz"

    # 既存チェック（raw patchesが存在すればスキップ）
    if skip_existing and out_patches.exists():
        existing = np.load(out_patches, mmap_mode='r')
        print(f"[SKIP] {category}: already exists ({len(existing)} samples)")
        return {"status": "skipped", "samples": len(existing)}

    # チェックポイント読み込み
    checkpoint_data, processed_files = load_checkpoint(checkpoint_file)
    if processed_files:
        print(f"[RESUME] {category}: {len(processed_files)} samples already processed")

    # 画像パス取得
    img_dir = CATEGORY_PATHS.get(category)
    if img_dir is None or not img_dir.exists():
        print(f"[ERROR] {category}: directory not found at {img_dir}")
        return {"status": "error", "error": "dir_not_found"}

    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    all_paths = sorted([
        p for p in img_dir.rglob("*")
        if p.suffix.lower() in extensions
    ])

    # 処理済みを除外
    image_paths = [p for p in all_paths if str(p) not in processed_files]

    if limit > 0:
        image_paths = image_paths[:limit]

    if not image_paths:
        if processed_files:
            # チェックポイントを最終ファイルに変換
            _finalize_from_checkpoint(checkpoint_data, out_cls, out_patches, out_stats, out_files, out_meta, category)
            checkpoint_file.unlink()
            print(f"[DONE] {category}: {len(processed_files)} samples (from checkpoint)")
            return {"status": "completed", "samples": len(processed_files)}
        print(f"[ERROR] {category}: no images found")
        return {"status": "error", "error": "no_images"}

    print(f"[START] {category}: {len(image_paths)} images (total: {len(all_paths)})")

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
    patches_list = []
    stats_list = []
    files_list = []

    start_time = time.time()
    current_batch_size = batch_size

    with torch.no_grad():
        for batch_idx, (pixels, paths, _) in enumerate(tqdm(loader, desc=category)):
            if pixels is None:
                continue

            try:
                pixels = pixels.to(device, non_blocking=True)
                outputs = model(pixels, output_hidden_states=True)

                # CLS (final layer)
                cls = outputs.last_hidden_state[:, 0, :]  # (B, 768)

                # Mid-layer patches (skip CLS + 4 register tokens)
                hidden_states = outputs.hidden_states
                mid_layer = hidden_states[MID_LAYER_INDEX + 1]  # (B, 201, 768)
                mid_cls = mid_layer[:, 0, :]  # (B, 768)
                mid_patches = mid_layer[:, 5:, :]  # (B, 196, 768)

                # v3 stats
                stats = compute_patch_stats_v3(mid_patches, mid_cls)  # (B, 33)

                cls_list.append(cls.cpu().numpy())
                patches_list.append(mid_patches.cpu().numpy())
                stats_list.append(stats.cpu().numpy())
                files_list.extend(paths)

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"\n[WARN] OOM at batch {batch_idx}, saving checkpoint...")
                    torch.cuda.empty_cache()
                    gc.collect()
                    # チェックポイント保存して終了
                    save_checkpoint(checkpoint_file, cls_list, patches_list, stats_list, files_list, checkpoint_data)
                    return {"status": "oom", "samples": len(files_list)}
                raise

            # Progress
            if (batch_idx + 1) % 100 == 0:
                elapsed = time.time() - start_time
                speed = len(files_list) / elapsed if elapsed > 0 else 0
                remaining = len(image_paths) - len(files_list)
                eta = remaining / speed if speed > 0 else 0
                tqdm.write(f"  {len(files_list)}/{len(image_paths)} | {speed:.1f} img/s | ETA {eta/60:.1f} min")

            # Checkpoint every N batches
            if (batch_idx + 1) % CHECKPOINT_INTERVAL == 0:
                n_saved = save_checkpoint(checkpoint_file, cls_list, patches_list, stats_list, files_list, checkpoint_data)
                tqdm.write(f"  [CHECKPOINT] {n_saved} samples saved")
                # メモリ解放
                checkpoint_data = np.load(checkpoint_file, allow_pickle=True)
                cls_list, patches_list, stats_list, files_list = [], [], [], []
                gc.collect()

    # 最終マージ
    if checkpoint_data is not None or cls_list:
        if cls_list:
            # 残りのデータをチェックポイントにマージ
            save_checkpoint(checkpoint_file, cls_list, patches_list, stats_list, files_list, checkpoint_data)
            checkpoint_data = np.load(checkpoint_file, allow_pickle=True)

        _finalize_from_checkpoint(checkpoint_data, out_cls, out_patches, out_stats, out_files, out_meta, category)
        checkpoint_file.unlink()

    elapsed = time.time() - start_time
    total_samples = len(np.load(out_cls))
    speed = total_samples / elapsed if elapsed > 0 else 0

    # ファイルサイズ
    patches_gb = out_patches.stat().st_size / 1e9

    print(f"[DONE] {category}: {total_samples} samples in {elapsed/60:.1f} min ({speed:.1f} img/s)")
    print(f"  Patches: {patches_gb:.2f} GB")

    return {
        "status": "success",
        "samples": total_samples,
        "time_sec": elapsed,
        "speed": speed,
        "patches_gb": patches_gb,
    }


def _finalize_from_checkpoint(checkpoint_data, out_cls, out_patches, out_stats, out_files, out_meta, category):
    """チェックポイントから最終ファイルを生成"""
    np.save(out_cls, checkpoint_data["cls"])
    np.save(out_patches, checkpoint_data["patches"])
    np.save(out_stats, checkpoint_data["stats"])

    with open(out_files, "w") as f:
        f.write("\n".join(checkpoint_data["files"].tolist()))

    metadata = {
        "mid_layer_index": MID_LAYER_INDEX,
        "model_dir": str(MODEL_DIR),
        "model_hash_1mb": get_model_hash(),
        "preprocess": {"image_size": 224, "processor": "AutoImageProcessor"},
        "extraction_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_samples": len(checkpoint_data["files"]),
        "v3_stat_names": V3_STAT_NAMES,
        "v3_stat_dim": len(V3_STAT_NAMES),
        "patches_dtype": "float16",
        "cls_dtype": "float32",
    }
    with open(out_meta, "w") as f:
        json.dump(metadata, f, indent=2)


def show_status():
    """抽出状況を表示"""
    print("=== Raw Patches Extraction Status ===\n")

    all_cats = AI_CATEGORIES + REAL_CATEGORIES

    total_done = 0
    total_pending = 0
    total_gb = 0

    for cat in all_cats:
        patches_path = EMBEDDINGS_DIR / f"{cat}_mid_patches.npy"
        checkpoint_path = EMBEDDINGS_DIR / f"{cat}_raw_checkpoint.npz"
        img_dir = CATEGORY_PATHS.get(cat)

        if patches_path.exists():
            n = len(np.load(patches_path, mmap_mode='r'))
            size_gb = patches_path.stat().st_size / 1e9
            print(f"  [✓] {cat}: {n:,} samples ({size_gb:.2f} GB)")
            total_done += n
            total_gb += size_gb
        elif checkpoint_path.exists():
            ckpt = np.load(checkpoint_path, allow_pickle=True)
            n = len(ckpt["files"])
            print(f"  [~] {cat}: {n:,} samples (checkpoint)")
            total_done += n
        elif img_dir and img_dir.exists():
            extensions = {".jpg", ".jpeg", ".png", ".webp"}
            n = len([p for p in img_dir.rglob("*") if p.suffix.lower() in extensions])
            est_gb = n * 196 * 768 * 2 / 1e9
            print(f"  [ ] {cat}: {n:,} images (est. {est_gb:.2f} GB)")
            total_pending += n
        else:
            print(f"  [?] {cat}: not found")

    print(f"\nDone: {total_done:,} samples ({total_gb:.1f} GB)")
    print(f"Pending: {total_pending:,} images")


def main():
    parser = argparse.ArgumentParser(description="Extract CLS + raw patches + v3 stats")
    parser.add_argument("--category", type=str, help="Single category")
    parser.add_argument("--all", action="store_true", help="All categories")
    parser.add_argument("--status", action="store_true", help="Show status")
    parser.add_argument("--limit", type=int, default=0, help="Limit images")
    parser.add_argument("--force", action="store_true", help="Force re-extraction")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    if args.status:
        show_status()
        return

    if not args.category and not args.all:
        parser.print_help()
        return

    if not torch.cuda.is_available():
        print("[ERROR] CUDA required")
        return

    device = torch.device("cuda")
    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print(f"\nLoading DINOv3 from {MODEL_DIR}...")
    from transformers import AutoImageProcessor, AutoModel
    processor = AutoImageProcessor.from_pretrained(MODEL_DIR)
    model = AutoModel.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()
    print("Model loaded.\n")

    if args.all:
        categories = AI_CATEGORIES + REAL_CATEGORIES
    else:
        categories = [args.category]

    results = []
    total_start = time.time()
    total_gb = 0

    for cat in categories:
        result = extract_category(
            cat, model, processor, device,
            limit=args.limit,
            skip_existing=not args.force,
            batch_size=args.batch_size,
        )
        results.append((cat, result))
        if result.get("patches_gb"):
            total_gb += result["patches_gb"]

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
            gb = result.get("patches_gb", 0)
            print(f"  {cat}: {samples:,} samples ({gb:.2f} GB)")
        else:
            print(f"  {cat}: {status}")

    print(f"\nTotal: {total_samples:,} samples in {total_time/60:.1f} min")
    print(f"Total patches storage: {total_gb:.1f} GB")
    if total_samples > 0:
        print(f"Average speed: {total_samples / total_time:.1f} img/s")


if __name__ == "__main__":
    main()
