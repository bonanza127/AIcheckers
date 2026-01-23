#!/usr/bin/env python3
"""
DINOv3 embedding抽出スクリプト v2.1 (SS-VAT用)
CLSトークン（最終層） + パッチ統計量v2（中間層） + 中間層パッチ埋め込みを保存

出力:
  - {name}.npy: CLSトークン (N, 768) - 最終層
  - {name}_patch_stats.npy: パッチ統計量v2 (N, 7) - 中間層
  - {name}_mid_patches.npy: 中間層パッチ埋め込み (N, 196, 768)

設計原則 (2026-01):
  - 中間層から「分類器を通さない」教師なし統計量を抽出
  - SS-VAT向けに中間層パッチ埋め込みも保持
"""
import os
import sys
import argparse
import io
import json
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image, ImageFilter
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

# 共通モジュール
sys.path.insert(0, str(Path(__file__).parent.parent))
from lib.patch_stats import compute_patch_stats_v2_batch

# 設定
DINOV3_MODEL_PATH = Path("/home/techne/aicheckers/models/dinov3-vitb16")
EMBEDDINGS_DIR = Path("/home/techne/aicheckers/embeddings")
MID_LAYER_INDEX = 6  # SS-VAT用の中間層（0-11）
CIVITAI_BASE = "/home/techne/aicheckers/data/animedl2m_dataset_release/civitai_subset/image"

CATEGORY_SOURCES = {
    "pony_ai": f"{CIVITAI_BASE}/Pony",
    "illustrious_ai": f"{CIVITAI_BASE}/Illustrious",
    "sdxl10_ai": f"{CIVITAI_BASE}/SDXL 1.0",
    "sd15_ai": f"{CIVITAI_BASE}/SD 1.5",
    "flux1d_ai": f"{CIVITAI_BASE}/Flux.1 D",
    "other_ai": f"{CIVITAI_BASE}/Other",
    "novelai_ai": "/home/techne/aicheckers/data/novelai",
    "novelai_combined_ai": "/home/techne/aicheckers/data/novelai_combined",
    "novelai_artist_tagged_ai": "/home/techne/aicheckers/data/novelai_artist_tagged",
    "pixai_ai": "/home/techne/aicheckers/data/pixai",
    "danbooru_real": "/home/techne/aicheckers/data/animedl2m_dataset_release/real_images/images",
}


def apply_degradation(img: Image.Image) -> Image.Image:
    """
    画像に劣化処理を適用（画質バイアスを除去するため）

    適用される劣化の種類（ランダムに1つ選択）:
    - JPEG圧縮 (quality 30-70)
    - ガウシアンノイズ
    - ダウンサンプリング→アップサンプリング (50-80%)
    """
    degradation_type = random.choice(['jpeg', 'noise', 'downsample'])

    if degradation_type == 'jpeg':
        # JPEG圧縮
        quality = random.randint(30, 70)
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        img = Image.open(buffer).convert('RGB')

    elif degradation_type == 'noise':
        # ガウシアンノイズ
        arr = np.array(img, dtype=np.float32)
        noise_std = random.uniform(5, 25)
        noise = np.random.normal(0, noise_std, arr.shape).astype(np.float32)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    elif degradation_type == 'downsample':
        # ダウンサンプリング→アップサンプリング
        scale = random.uniform(0.5, 0.8)
        w, h = img.size
        small_size = (int(w * scale), int(h * scale))
        img = img.resize(small_size, Image.BILINEAR)
        img = img.resize((w, h), Image.BILINEAR)

    return img


def load_model():
    """DINOv3モデルをロード（ローカルディレクトリから、中間層出力有効）"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not DINOV3_MODEL_PATH.exists():
        raise FileNotFoundError(f"DINOv3 model not found at {DINOV3_MODEL_PATH}")

    # ローカルディレクトリからロード（ネットワーク不要）
    processor = AutoImageProcessor.from_pretrained(str(DINOV3_MODEL_PATH))
    model = AutoModel.from_pretrained(str(DINOV3_MODEL_PATH))
    model.to(device)
    model.eval()

    print(f"[INFO] Loaded DINOv3 from: {DINOV3_MODEL_PATH}")
    print(f"[INFO] Mid-layer extraction enabled: Block {MID_LAYER_INDEX}")
    return model, processor, device


# NOTE: 分類器ロードは不要になりました（v2は教師なし統計量のため）


def extract_embeddings(image_dir: Path, output_name: str, model, processor, device, batch_size=32, degradation_prob=0.0, mid_layer=MID_LAYER_INDEX, limit=None, mid_dtype="float16", seed=42, resume=False, log_every=50):
    """ディレクトリ内の全画像からCLS（最終層）・パッチ統計量v2（中間層）・中間層パッチ埋め込みを抽出
    
    Args:
        degradation_prob: 劣化Augmentationを適用する確率 (0.0-1.0)
        mid_layer: パッチ統計量を抽出する中間層のインデックス (0-11)
        limit: 処理する最大画像数 (Noneの場合は全件)
        mid_dtype: 中間層パッチ埋め込みの保存dtype ("float16" or "float32")
    """
    extensions = {'.jpg', '.jpeg', '.png', '.webp', '.gif'}
    image_list_path = EMBEDDINGS_DIR / f"{output_name}_image_list.txt"
    progress_path = EMBEDDINGS_DIR / f"{output_name}_progress.json"
    bad_files_path = EMBEDDINGS_DIR / f"{output_name}_bad_files.txt"

    if bad_files_path.exists():
        with open(bad_files_path, "r") as f:
            bad_files = {line.strip() for line in f if line.strip()}
    else:
        bad_files = set()

    if resume:
        if not image_list_path.exists():
            raise FileNotFoundError(f"Resume requested but image list missing: {image_list_path}")
        with open(image_list_path, "r") as f:
            rel_paths = [line.strip() for line in f if line.strip()]
        image_files = [image_dir / rel for rel in rel_paths if rel not in bad_files]
    else:
        image_files = [f for f in image_dir.rglob("*")
                       if f.is_file() and f.suffix.lower() in extensions]
        if bad_files:
            image_files = [f for f in image_files if str(f.relative_to(image_dir)) not in bad_files]

    total_images = len(image_files)
    if limit:
        rng = random.Random(seed)
        rng.shuffle(image_files)
        total_images = min(total_images, limit)
        image_files = image_files[:total_images]

    print(f"Found {total_images} images in {image_dir}")
    if degradation_prob > 0:
        print(f"Degradation augmentation enabled: {degradation_prob*100:.0f}% of images")

    filenames = []
    degradation_count = 0
    processed_count = 0
    start_time = time.time()

    cls_dtype = np.float32
    stats_dtype = np.float32
    if mid_dtype not in ("float16", "float32"):
        raise ValueError("mid_dtype must be 'float16' or 'float32'")
    mid_dtype_np = np.float16 if mid_dtype == "float16" else np.float32

    cls_path = EMBEDDINGS_DIR / f"{output_name}.npy"
    stats_path = EMBEDDINGS_DIR / f"{output_name}_patch_stats.npy"
    mid_path = EMBEDDINGS_DIR / f"{output_name}_mid_patches.npy"
    names_path = EMBEDDINGS_DIR / f"{output_name}_files.txt"

    if resume:
        required_paths = [cls_path, stats_path, mid_path, names_path, progress_path]
        missing = [str(p) for p in required_paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Resume requested but missing files: {', '.join(missing)}")

        with open(progress_path, "r") as f:
            progress = json.load(f)
        processed_count = int(progress.get("processed", 0))
        if processed_count > total_images:
            raise ValueError(f"Progress exceeds total images: {processed_count} > {total_images}")
    else:
        # Save ordered image list for reproducibility/resume.
        rel_paths = [str(p.relative_to(image_dir)) for p in image_files]
        with open(image_list_path, "w") as f:
            f.write("\n".join(rel_paths))

    memmap_mode = "r+" if resume else "w+"
    cls_map = np.lib.format.open_memmap(
        cls_path, mode=memmap_mode, dtype=cls_dtype, shape=(total_images, 768)
    )
    stats_map = np.lib.format.open_memmap(
        stats_path, mode=memmap_mode, dtype=stats_dtype, shape=(total_images, 7)
    )
    mid_map = np.lib.format.open_memmap(
        mid_path, mode=memmap_mode, dtype=mid_dtype_np, shape=(total_images, 196, 768)
    )

    if resume and processed_count > 0:
        with open(names_path, "r") as f:
            filenames = [line.strip() for line in f if line.strip()]
        if len(filenames) != processed_count:
            raise ValueError("files.txt count does not match processed count")

    for i in tqdm(range(processed_count, total_images, batch_size), desc="Extracting"):
        batch_files = image_files[i:i+batch_size]
        batch_images = []
        batch_names = []

        for f in batch_files:
            try:
                img = Image.open(f).convert("RGB")
                # 劣化Augmentation（確率的に適用）
                if degradation_prob > 0 and random.random() < degradation_prob:
                    img = apply_degradation(img)
                    degradation_count += 1
                batch_images.append(img)
                batch_names.append(str(f.relative_to(image_dir)))
            except Exception as e:
                rel_path = str(f.relative_to(image_dir))
                if rel_path not in bad_files:
                    with open(bad_files_path, "a") as bf:
                        bf.write(rel_path + "\n")
                    bad_files.add(rel_path)
                print(f"[WARN] Skipping unreadable image: {rel_path} ({e})")
                continue

        if not batch_images:
            raise RuntimeError("No images loaded in batch; aborting to avoid partial outputs.")

        # 前処理
        inputs = processor(images=batch_images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 特徴抽出（中間層出力を有効化）
        with torch.inference_mode():
            outputs = model(**inputs, output_hidden_states=True)
            # DINOv3: [CLS(0), REG1-4(1-4), PATCH1-196(5-200)] = 201 tokens
            
            # 最終層からCLSトークン
            final_hidden = outputs.last_hidden_state  # (batch, 201, 768)
            cls_emb = final_hidden[:, 0, :].cpu().numpy()  # (batch, 768)

            # 中間層からパッチトークン（v2統計量用）
            # hidden_states: tuple of (batch, 201, 768) for each layer + initial embedding
            mid_hidden = outputs.hidden_states[mid_layer + 1]  # +1 because index 0 is initial embedding
            mid_patch_emb = mid_hidden[:, 5:5+196, :]  # (batch, 196, 768)

            # パッチ統計量v2（教師なし）
            patch_stats = compute_patch_stats_v2_batch(mid_patch_emb)

        batch_count = len(batch_names)
        cls_map[processed_count:processed_count + batch_count] = cls_emb
        stats_map[processed_count:processed_count + batch_count] = patch_stats
        mid_map[processed_count:processed_count + batch_count] = (
            mid_patch_emb.cpu().numpy().astype(mid_dtype_np, copy=False)
        )
        filenames.extend(batch_names)
        processed_count += batch_count

        if log_every and (processed_count // batch_size) % log_every == 0:
            elapsed = time.time() - start_time
            rate = processed_count / max(elapsed, 1e-6)
            eta = (total_images - processed_count) / max(rate, 1e-6)
            print(f"[PROGRESS] {processed_count}/{total_images} | {rate:.2f} img/s | ETA {eta/60:.1f} min")

        with open(progress_path, "w") as f:
            json.dump({
                "processed": processed_count,
                "total": total_images,
                "mid_layer": mid_layer,
                "mid_dtype": mid_dtype,
                "seed": seed,
                "batch_size": batch_size,
                "degradation_prob": degradation_prob,
            }, f)
        
        if limit and processed_count >= limit:
            print(f"Limit reached ({limit}), stopping extraction.")
            break

    if processed_count != total_images:
        new_total = processed_count
        print(f"[WARN] Processed {processed_count}/{total_images}. Compacting outputs to {new_total}.")
        for src_path, shape, dtype in (
            (cls_path, (new_total, 768), cls_dtype),
            (stats_path, (new_total, 7), stats_dtype),
            (mid_path, (new_total, 196, 768), mid_dtype_np),
        ):
            tmp_path = src_path.with_suffix(".tmp.npy")
            tmp_map = np.lib.format.open_memmap(tmp_path, mode="w+", dtype=dtype, shape=shape)
            tmp_map[:] = np.lib.format.open_memmap(src_path, mode="r", dtype=dtype, shape=(total_images,) + shape[1:])[:new_total]
            tmp_map.flush()
            os.replace(tmp_path, src_path)
        image_list_path = EMBEDDINGS_DIR / f"{output_name}_image_list.txt"
        with open(image_list_path, "r") as f:
            rel_paths = [line.strip() for line in f if line.strip()]
        rel_paths = [p for p in rel_paths if p not in bad_files]
        with open(image_list_path, "w") as f:
            f.write("\n".join(rel_paths[:new_total]))

    if degradation_prob > 0:
        print(f"Applied degradation to {degradation_count}/{len(filenames)} images ({degradation_count/len(filenames)*100:.1f}%)")

    return filenames, cls_path, stats_path, mid_path, names_path


def main():
    parser = argparse.ArgumentParser(description="DINOv3 embedding extraction v2.1 (SS-VAT, mid patches saved)")
    parser.add_argument("--dir", type=str, help="Image directory")
    parser.add_argument("--name", type=str, help="Output name (e.g., 'illustrious_ai')")
    parser.add_argument("--all", action="store_true",
                        help="Run extraction for all categories (uses built-in CATEGORY_SOURCES)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--mid-layer", type=int, default=MID_LAYER_INDEX,
                        help=f"Mid-layer index for patch stats (0-11, default: {MID_LAYER_INDEX})")
    parser.add_argument("--degradation-prob", type=float, default=0.0,
                        help="Probability of applying degradation augmentation (0.0-1.0, default: 0.0)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of images to process (for testing)")
    parser.add_argument("--mid-dtype", type=str, default="float16",
                        help="Dtype for mid-patch embeddings: float16 or float32 (default: float16)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed used when limit is set (default: 42)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing outputs and progress file")
    parser.add_argument("--log-every", type=int, default=50,
                        help="Log progress every N batches (default: 50)")
    args = parser.parse_args()

    EMBEDDINGS_DIR.mkdir(exist_ok=True)

    if args.all:
        all_lock = EMBEDDINGS_DIR / "ssvat_extract_all.lock"
        try:
            fd = os.open(all_lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
        except FileExistsError:
            print(f"Error: lock exists ({all_lock}). Another run may be active.")
            sys.exit(1)

        try:
            for name, source_dir in CATEGORY_SOURCES.items():
                if not Path(source_dir).exists():
                    print(f"❌ {name}: Source directory not found: {source_dir}")
                    continue
                cmd = [
                    sys.executable, __file__,
                    "--dir", source_dir,
                    "--name", name,
                    "--batch-size", str(args.batch_size),
                    "--mid-layer", str(args.mid_layer),
                    "--mid-dtype", args.mid_dtype,
                    "--seed", str(args.seed),
                    "--log-every", str(args.log_every),
                ]
                if args.degradation_prob > 0:
                    cmd += ["--degradation-prob", str(args.degradation_prob)]
                if args.resume:
                    cmd += ["--resume"]
                if args.limit is not None:
                    cmd += ["--limit", str(args.limit)]
                if name == "sd15_ai":
                    cmd += ["--limit", "10000"]
                print(f"Running: {name}")
                result = os.spawnv(os.P_WAIT, sys.executable, cmd)
                if result != 0:
                    print(f"❌ Failed: {name}")
                    break
        finally:
            if all_lock.exists():
                try:
                    all_lock.unlink()
                except OSError:
                    pass
        return

    if not args.dir or not args.name:
        print("Error: --dir and --name are required unless --all is set.")
        sys.exit(1)

    image_dir = Path(args.dir)
    if not image_dir.exists():
        print(f"Error: {image_dir} does not exist")
        sys.exit(1)

    lock_path = EMBEDDINGS_DIR / f"{args.name}.lock"
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        print(f"Error: lock exists for {args.name} ({lock_path}). Another run may be active.")
        sys.exit(1)

    try:
        model, processor, device = load_model()
        filenames, cls_path, stats_path, mid_path, names_path = extract_embeddings(
            image_dir,
            args.name,
            model,
            processor,
            device,
            args.batch_size,
            args.degradation_prob,
            args.mid_layer,
            args.limit,
            args.mid_dtype,
            args.seed,
            args.resume,
            args.log_every,
        )
    except Exception:
        for path in (
            EMBEDDINGS_DIR / f"{args.name}.npy",
            EMBEDDINGS_DIR / f"{args.name}_patch_stats.npy",
            EMBEDDINGS_DIR / f"{args.name}_mid_patches.npy",
            EMBEDDINGS_DIR / f"{args.name}_files.txt",
        ):
            if path.exists():
                try:
                    path.unlink()
                except OSError:
                    pass
        raise
    finally:
        if lock_path.exists():
            try:
                lock_path.unlink()
            except OSError:
                pass

    with open(names_path, "w") as f:
        f.write("\n".join(filenames))

    print(f"\n[DONE] Saved {len(filenames)} samples")
    print(f"  CLS embeddings (final layer): {cls_path}")
    print(f"  Patch stats v2 (mid-layer {args.mid_layer}): {stats_path}")
    print(f"  Mid patches (mid-layer {args.mid_layer}, {args.mid_dtype}): {mid_path}")
    print(f"  Filenames: {names_path}")


if __name__ == "__main__":
    main()
