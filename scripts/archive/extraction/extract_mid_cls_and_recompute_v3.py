#!/usr/bin/env python3
"""
中間層CLSを抽出し、既存raw_patchesと合わせてv3統計を再計算

OOM対策: raw_patchesは保存済みなので再保存しない
チェックポイント機能付き
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

from lib.patch_stats import compute_patch_stats_v3, V3_STAT_NAMES, KNN_K

EMBEDDINGS_DIR = Path("/home/techne/aicheckers/embeddings")
MODEL_PATH = "/home/techne/aicheckers/models/dinov3-vitb16"

CATEGORIES = {
    "illustrious_ai": "/home/techne/aicheckers/data/animedl2m_dataset_release/civitai_subset/image/Illustrious",
    "pony_ai": "/home/techne/aicheckers/data/animedl2m_dataset_release/civitai_subset/image/Pony",
    "sdxl10_ai": "/home/techne/aicheckers/data/animedl2m_dataset_release/civitai_subset/image/SDXL 1.0",
    "sd15_ai": "/home/techne/aicheckers/data/animedl2m_dataset_release/civitai_subset/image/SD 1.5",
    "other_ai": "/home/techne/aicheckers/data/animedl2m_dataset_release/civitai_subset/image/Other",
    "flux1d_ai": "/home/techne/aicheckers/data/animedl2m_dataset_release/civitai_subset/image/Flux.1 D",
    "novelai_ai": "/home/techne/aicheckers/data/novelai",
    "pixai_ai": "/home/techne/aicheckers/data/pixai",
    "novelai_combined_ai": "/home/techne/aicheckers/data/novelai_combined",
    "novelai_artist_tagged_ai": "/home/techne/aicheckers/data/novelai_artist_tagged",
    "danbooru_real": "/home/techne/aicheckers/data/animedl2m_dataset_release/real_images/images",
}

# サンプル数制限
SAMPLE_LIMITS = {
    "sd15_ai": 10000,
}

MID_LAYER = 6
BATCH_SIZE = 8
CHECKPOINT_INTERVAL = 1000  # サンプルごとにチェックポイント


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
            return inputs["pixel_values"].squeeze(0), str(path), idx, True
        except Exception as e:
            # 失敗時はダミー
            dummy = torch.zeros(3, 518, 518)
            return dummy, str(path), idx, False


def extract_mid_cls_only(model, pixel_values, device):
    """中間層CLSのみを抽出（パッチは既存を使う）"""
    with torch.no_grad():
        pixel_values = pixel_values.to(device)
        outputs = model(pixel_values, output_hidden_states=True)
        # hidden_states[0]は埋め込み層なので+1
        mid_layer = outputs.hidden_states[MID_LAYER + 1]  # (B, 197, 768)
        mid_cls = mid_layer[:, 0, :]  # (B, 768)
    return mid_cls


def save_checkpoint(checkpoint_file, cls_list, stats_list, files_list, processed_indices):
    """チェックポイント保存"""
    np.savez_compressed(
        checkpoint_file,
        cls=np.array(cls_list, dtype=np.float32),
        stats=np.array(stats_list, dtype=np.float32),
        files=np.array(files_list),
        processed_indices=np.array(processed_indices),
    )


def load_checkpoint(checkpoint_file):
    """チェックポイント読み込み"""
    if checkpoint_file.exists():
        data = np.load(checkpoint_file, allow_pickle=True)
        return {
            "cls": data["cls"].tolist(),
            "stats": data["stats"].tolist(),
            "files": data["files"].tolist(),
            "processed_indices": set(data["processed_indices"].tolist()),
        }
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, help="Single category")
    parser.add_argument("--all", action="store_true", help="All categories")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="Ignore checkpoint")
    parser.add_argument(
        "--allow-mismatch",
        action="store_true",
        help="Allow length mismatch by truncating to min length (unsafe)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"MID_LAYER: {MID_LAYER}")
    print(f"KNN_K: {KNN_K}")
    print(f"V3 stats: {len(V3_STAT_NAMES)}d")

    # モデルロード
    print(f"\nLoading DINOv3 from {MODEL_PATH}...")
    processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
    model = AutoModel.from_pretrained(MODEL_PATH).to(device).eval()
    print("Model loaded.")

    # カテゴリ選択
    if args.category:
        categories = {args.category: CATEGORIES[args.category]}
    elif args.all:
        categories = CATEGORIES
    else:
        print("Specify --category or --all")
        return

    for cat, img_dir in categories.items():
        patches_path = EMBEDDINGS_DIR / f"{cat}_mid_patches.npy"
        checkpoint_file = EMBEDDINGS_DIR / f"{cat}_v3_checkpoint.npz"
        out_cls = EMBEDDINGS_DIR / f"{cat}_mid_cls.npy"
        out_v3 = EMBEDDINGS_DIR / f"{cat}_patch_stats_v3.npy"
        out_files = EMBEDDINGS_DIR / f"{cat}_files.txt"

        if not patches_path.exists():
            print(f"\n[SKIP] {cat}: no mid_patches found")
            continue

        # 既存のファイルリストを読む（raw_patchesと同じ順序を保証）
        existing_files_path = EMBEDDINGS_DIR / f"{cat}_files.txt"
        fallback_files_path = EMBEDDINGS_DIR / f"{cat}_image_list.txt"
        if not existing_files_path.exists() and not fallback_files_path.exists():
            print(f"\n[SKIP] {cat}: no files list found (needed for ordering)")
            continue

        files_path = existing_files_path if existing_files_path.exists() else fallback_files_path
        if files_path == fallback_files_path:
            print(f"  [WARN] {cat}: using fallback list {files_path.name}")

        with open(files_path, "r") as f:
            raw_lines = [line.strip() for line in f if line.strip()]

        # ファイル名のみの場合はディレクトリと結合
        img_dir_path = Path(img_dir)
        image_files = []
        for line in raw_lines:
            p = Path(line)
            if p.is_absolute():
                image_files.append(p)
            else:
                # 相対パス/ファイル名のみの場合はディレクトリと結合
                image_files.append(img_dir_path / p)

        # missing files are not allowed because raw_patches were created by skipping invalids
        missing_count = sum(1 for p in image_files if not p.exists())
        if missing_count:
            print(f"  [ERROR] {cat}: {missing_count} missing files in list; cannot recompute safely")
            print("          Restore the original files list or re-extract raw patches.")
            continue

        # サンプル数制限
        limit = SAMPLE_LIMITS.get(cat)
        if limit and len(image_files) > limit:
            image_files = image_files[:limit]

        # 既存パッチ読み込み（メモリマップ）
        patches = np.load(patches_path, mmap_mode='r')
        n_existing = len(patches)

        # 画像数とパッチ数を合わせる（不一致は危険なので原則NG）
        if len(image_files) != n_existing:
            print(f"  [ERROR] {cat}: files={len(image_files)} vs patches={n_existing} mismatch")
            if not args.allow_mismatch:
                print("          Use --allow-mismatch to truncate, or fix the files list.")
                continue
            min_len = min(len(image_files), n_existing)
            image_files = image_files[:min_len]
            print(f"  [WARN] {cat}: truncating to {min_len} (unsafe)")

        print(f"\n[START] {cat}: {len(image_files)} images, {n_existing} patches")

        # チェックポイント読み込み
        checkpoint_data = None
        if not args.force and checkpoint_file.exists():
            checkpoint_data = load_checkpoint(checkpoint_file)
            if checkpoint_data:
                n_done = len(checkpoint_data["processed_indices"])
                print(f"  [RESUME] {n_done} samples already processed")

        # 初期化
        if checkpoint_data:
            cls_list = checkpoint_data["cls"]
            stats_list = checkpoint_data["stats"]
            files_list = checkpoint_data["files"]
            processed_indices = checkpoint_data["processed_indices"]
        else:
            cls_list = []
            stats_list = []
            files_list = []
            processed_indices = set()

        dataset = ImageDataset(image_files, processor)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=False,
            shuffle=False
        )

        samples_since_checkpoint = 0

        for batch_pixels, batch_paths, batch_indices, batch_valids in tqdm(dataloader, desc=cat):
            # Any invalids mean the file list is not aligned with raw_patches.
            if not all(batch_valids):
                bad_paths = [p for p, v in zip(batch_paths, batch_valids) if not v][:5]
                raise RuntimeError(
                    f"{cat}: invalid images encountered; example(s): {bad_paths}. "
                    "This breaks alignment with raw_patches."
                )

            # 既に処理済みのインデックスをスキップ
            valid_indices = [
                idx.item()
                for idx in batch_indices
                if idx.item() not in processed_indices
            ]
            if not valid_indices:
                continue

            valid_mask = torch.tensor(
                [idx.item() in valid_indices for idx in batch_indices],
                dtype=torch.bool
            )
            valid_pixels = batch_pixels[valid_mask]
            valid_paths = [p for p, m in zip(batch_paths, valid_mask) if m]

            # 中間層CLS抽出
            mid_cls = extract_mid_cls_only(model, valid_pixels, device)

            # 対応するパッチを取得
            batch_patches = torch.tensor(
                np.array([patches[idx] for idx in valid_indices]),
                device=device, dtype=torch.float32
            )

            # v3統計計算
            with torch.no_grad():
                stats = compute_patch_stats_v3(batch_patches, mid_cls)

            # 結果を追加
            for i, idx in enumerate(valid_indices):
                cls_list.append(mid_cls[i].cpu().numpy())
                stats_list.append(stats[i].cpu().numpy())
                files_list.append(valid_paths[i])
                processed_indices.add(idx)
            samples_since_checkpoint += len(valid_indices)

            # メモリ解放
            del mid_cls, batch_patches, stats
            if device.type == 'cuda':
                torch.cuda.empty_cache()

            # チェックポイント保存
            if samples_since_checkpoint >= CHECKPOINT_INTERVAL:
                save_checkpoint(checkpoint_file, cls_list, stats_list, files_list, list(processed_indices))
                tqdm.write(f"  [CHECKPOINT] {len(processed_indices)} samples saved")
                samples_since_checkpoint = 0

        # 最終保存
        if cls_list:
            cls_array = np.stack(cls_list).astype(np.float32)
            stats_array = np.stack(stats_list).astype(np.float32)

            np.save(out_cls, cls_array)
            np.save(out_v3, stats_array)

            with open(out_files, 'w') as f:
                f.write('\n'.join(files_list))

            # チェックポイント削除
            if checkpoint_file.exists():
                checkpoint_file.unlink()

            print(f"[DONE] {cat}: {len(cls_array)} samples")
            print(f"  Saved: {out_cls.name} ({cls_array.shape})")
            print(f"  Saved: {out_v3.name} ({stats_array.shape})")
        else:
            print(f"[WARN] {cat}: no valid samples")

    print("\nDone!")


if __name__ == "__main__":
    main()
