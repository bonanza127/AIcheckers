#!/usr/bin/env python3
"""
Hard Negatives + 代表サブセット用 Raw Patches抽出

対象:
  - Hard negatives (境界例、誤分類例)
  - 各カテゴリから1-2万枚の代表サンプル

保存形式:
  - embeddings/raw_patches/{category}_patches.npy  : (N, 196, 768) fp16
  - embeddings/raw_patches/{category}_cls.npy     : (N, 768) fp16
  - embeddings/raw_patches/{category}_files.txt   : ファイル名
  - embeddings/raw_patches/{category}_meta.json   : メタデータ

Usage:
    # 代表サブセット抽出（各カテゴリ最大N枚）
    python scripts/extract_raw_patches_subset.py --all --max-per-category 10000

    # 特定カテゴリ
    python scripts/extract_raw_patches_subset.py --category novelai_ai --max-per-category 5000

    # Hard negativeリストから抽出
    python scripts/extract_raw_patches_subset.py --from-list logs/hard_negatives.csv

    # ステータス確認
    python scripts/extract_raw_patches_subset.py --status
"""
import argparse
import sys
import time
import json
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================================
# Configuration
# ============================================================================

EMBEDDINGS_DIR = Path("/home/techne/aicheckers/embeddings")
RAW_PATCHES_DIR = EMBEDDINGS_DIR / "raw_patches"
MODEL_DIR = Path("/home/techne/aicheckers/models/dinov3-vitb16")
MID_LAYER_INDEX = 6

BATCH_SIZE = 8
NUM_WORKERS = 4

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
            return pixel_values, str(path), True
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
# Extraction
# ============================================================================

def get_model_hash():
    """モデルファイルのハッシュを取得"""
    model_files = list(MODEL_DIR.glob("*.safetensors")) + list(MODEL_DIR.glob("*.bin"))
    if model_files:
        with open(model_files[0], "rb") as f:
            return hashlib.md5(f.read(1024*1024)).hexdigest()
    return "unknown"


def extract_raw_patches(
    image_paths: list,
    output_name: str,
    model,
    processor,
    device: torch.device,
    batch_size: int = BATCH_SIZE,
) -> dict:
    """画像リストからraw patchesを抽出"""

    RAW_PATCHES_DIR.mkdir(exist_ok=True)

    out_patches = RAW_PATCHES_DIR / f"{output_name}_patches.npy"
    out_cls = RAW_PATCHES_DIR / f"{output_name}_cls.npy"
    out_files = RAW_PATCHES_DIR / f"{output_name}_files.txt"
    out_meta = RAW_PATCHES_DIR / f"{output_name}_meta.json"

    if not image_paths:
        print(f"[ERROR] {output_name}: no images")
        return {"status": "error", "error": "no_images"}

    print(f"[START] {output_name}: {len(image_paths)} images")

    dataset = ImageDataset(image_paths, processor)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=2,
    )

    cls_list = []
    patches_list = []
    files_list = []

    start_time = time.time()

    with torch.no_grad():
        for batch_idx, (pixels, paths, _) in enumerate(tqdm(loader, desc=output_name)):
            if pixels is None:
                continue

            pixels = pixels.to(device, non_blocking=True)
            outputs = model(pixels, output_hidden_states=True)

            cls = outputs.last_hidden_state[:, 0, :]  # (B, 768)
            hidden_states = outputs.hidden_states
            mid_layer = hidden_states[MID_LAYER_INDEX + 1]
            mid_patches = mid_layer[:, 5:, :]  # (B, 196, 768)

            cls_list.append(cls.cpu().numpy())
            patches_list.append(mid_patches.cpu().numpy())
            files_list.extend(paths)

    # 保存 (fp16)
    cls_arr = np.concatenate(cls_list, axis=0).astype(np.float16)
    patches_arr = np.concatenate(patches_list, axis=0).astype(np.float16)

    np.save(out_cls, cls_arr)
    np.save(out_patches, patches_arr)
    with open(out_files, "w") as f:
        f.write("\n".join(files_list))

    # メタデータ
    metadata = {
        "mid_layer_index": MID_LAYER_INDEX,
        "model_dir": str(MODEL_DIR),
        "model_hash_1mb": get_model_hash(),
        "preprocess": {"image_size": 224, "processor": "AutoImageProcessor"},
        "extraction_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_samples": len(files_list),
        "dtype": "float16",
        "normalization": "raw",  # 正規化なし
    }
    with open(out_meta, "w") as f:
        json.dump(metadata, f, indent=2)

    elapsed = time.time() - start_time
    speed = len(files_list) / elapsed
    patches_gb = out_patches.stat().st_size / 1e9

    print(f"[DONE] {output_name}: {len(files_list)} samples in {elapsed/60:.1f} min")
    print(f"  Patches: {patches_arr.shape} ({patches_gb:.2f} GB)")

    return {
        "status": "success",
        "samples": len(files_list),
        "patches_gb": patches_gb,
    }


def extract_category_subset(
    category: str,
    model,
    processor,
    device: torch.device,
    max_samples: int = 10000,
    skip_existing: bool = True,
) -> dict:
    """カテゴリから代表サブセットを抽出"""

    out_patches = RAW_PATCHES_DIR / f"{category}_patches.npy"
    if skip_existing and out_patches.exists():
        n = len(np.load(out_patches, mmap_mode='r'))
        print(f"[SKIP] {category}: already exists ({n} samples)")
        return {"status": "skipped", "samples": n}

    img_dir = CATEGORY_PATHS.get(category)
    if img_dir is None or not img_dir.exists():
        print(f"[ERROR] {category}: directory not found")
        return {"status": "error", "error": "dir_not_found"}

    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    all_paths = sorted([
        p for p in img_dir.rglob("*")
        if p.suffix.lower() in extensions
    ])

    # 均等サンプリング
    if len(all_paths) > max_samples:
        step = len(all_paths) / max_samples
        image_paths = [all_paths[int(i * step)] for i in range(max_samples)]
    else:
        image_paths = all_paths

    return extract_raw_patches(image_paths, category, model, processor, device)


def extract_from_list(
    list_file: str,
    model,
    processor,
    device: torch.device,
) -> dict:
    """ファイルリスト（CSV/TXT）から抽出"""
    import csv

    list_path = Path(list_file)
    if not list_path.exists():
        print(f"[ERROR] List file not found: {list_file}")
        return {"status": "error"}

    image_paths = []
    if list_path.suffix == ".csv":
        with open(list_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                # "path" or "file" or "image" column
                for col in ["path", "file", "image", "filepath"]:
                    if col in row:
                        p = Path(row[col])
                        if p.exists():
                            image_paths.append(p)
                        break
    else:
        with open(list_path) as f:
            for line in f:
                p = Path(line.strip())
                if p.exists():
                    image_paths.append(p)

    output_name = f"hard_negatives_{list_path.stem}"
    return extract_raw_patches(image_paths, output_name, model, processor, device)


def show_status():
    """ステータス表示"""
    print("=== Raw Patches Status ===\n")

    if not RAW_PATCHES_DIR.exists():
        print("No raw patches extracted yet.")
        return

    total_samples = 0
    total_gb = 0

    for f in sorted(RAW_PATCHES_DIR.glob("*_patches.npy")):
        name = f.stem.replace("_patches", "")
        n = len(np.load(f, mmap_mode='r'))
        gb = f.stat().st_size / 1e9
        print(f"  {name}: {n:,} samples ({gb:.2f} GB)")
        total_samples += n
        total_gb += gb

    print(f"\nTotal: {total_samples:,} samples ({total_gb:.1f} GB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, help="Single category")
    parser.add_argument("--all", action="store_true", help="All categories")
    parser.add_argument("--from-list", type=str, help="Extract from file list (CSV/TXT)")
    parser.add_argument("--max-per-category", type=int, default=10000)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.status:
        show_status()
        return

    if not (args.category or args.all or args.from_list):
        parser.print_help()
        return

    if not torch.cuda.is_available():
        print("[ERROR] CUDA required")
        return

    device = torch.device("cuda")
    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    from transformers import AutoImageProcessor, AutoModel
    processor = AutoImageProcessor.from_pretrained(MODEL_DIR)
    model = AutoModel.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()
    print("Model loaded.\n")

    if args.from_list:
        extract_from_list(args.from_list, model, processor, device)
    elif args.all:
        for cat in AI_CATEGORIES + REAL_CATEGORIES:
            extract_category_subset(
                cat, model, processor, device,
                max_samples=args.max_per_category,
                skip_existing=not args.force,
            )
    elif args.category:
        extract_category_subset(
            args.category, model, processor, device,
            max_samples=args.max_per_category,
            skip_existing=not args.force,
        )


if __name__ == "__main__":
    main()
