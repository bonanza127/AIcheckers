"""
Modal v3 パッチ統計量抽出スクリプト
24次元のGPU only統計量を並列抽出

使い方:
  # ローカルからModalボリュームに画像をアップロード
  modal volume put aicheckers-images /path/to/images /images/category_name

  # 抽出実行
  modal run scripts/modal_extract_v3.py --category novelai_ai

  # 結果をダウンロード
  modal volume get aicheckers-embeddings-v3 /embeddings ./embeddings_v3/
"""
import os
import sys
import modal
import numpy as np
from pathlib import Path

# Modal App
app = modal.App("aicheckers-v3-extractor")

# Volumes
images_vol = modal.Volume.from_name("aicheckers-images", create_if_missing=True)
embeddings_vol = modal.Volume.from_name("aicheckers-embeddings-v3", create_if_missing=True)

# Image definition
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.1.0",
        "torchvision==0.16.0",
        "transformers==4.36.0",
        "pillow",
        "numpy",
        "tqdm",
        "accelerate",
    )
    .add_local_file("lib/patch_stats.py", "/root/lib/patch_stats.py")
    .add_local_dir("models/dinov3-vitb16", "/models/dinov3-vitb16")
)

MODEL_DIR = "/models/dinov3-vitb16"
MID_LAYER_INDEX = 6
BATCH_SIZE = 32


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/images": images_vol, "/embeddings": embeddings_vol},
    timeout=7200,
    memory=32768,
)
def extract_category(category: str, limit: int = 0):
    """単一カテゴリのv3統計量を抽出"""
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel
    from tqdm import tqdm

    sys.path.insert(0, "/root")
    from lib.patch_stats import compute_patch_stats_v3, V3_STAT_NAMES

    device = torch.device("cuda")

    # 出力パス確認
    out_cls = Path(f"/embeddings/{category}.npy")
    out_stats = Path(f"/embeddings/{category}_patch_stats_v3.npy")
    out_files = Path(f"/embeddings/{category}_files.txt")

    if out_cls.exists() and out_stats.exists():
        print(f"[SKIP] {category} already exists")
        return {"status": "skipped", "category": category}

    # 画像ディレクトリ
    img_dir = Path(f"/images/{category}")
    if not img_dir.exists():
        print(f"[ERROR] Image directory not found: {img_dir}")
        return {"status": "error", "category": category, "error": "dir_not_found"}

    # 画像リスト
    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    image_paths = sorted([
        p for p in img_dir.rglob("*")
        if p.suffix.lower() in extensions
    ])

    if limit > 0:
        image_paths = image_paths[:limit]

    print(f"[INFO] Found {len(image_paths)} images in {category}")

    if len(image_paths) == 0:
        return {"status": "error", "category": category, "error": "no_images"}

    # モデルロード
    print("[INFO] Loading DINOv3...")
    processor = AutoImageProcessor.from_pretrained(MODEL_DIR)
    model = AutoModel.from_pretrained(MODEL_DIR, output_hidden_states=True)
    model.to(device)
    model.eval()

    # 抽出
    cls_embeddings = []
    patch_stats_list = []
    valid_files = []

    for i in tqdm(range(0, len(image_paths), BATCH_SIZE), desc=category):
        batch_paths = image_paths[i:i+BATCH_SIZE]
        batch_images = []
        batch_valid = []

        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                batch_images.append(img)
                batch_valid.append(p.name)
            except Exception as e:
                print(f"[WARN] Failed to load {p}: {e}")
                continue

        if not batch_images:
            continue

        # 前処理
        inputs = processor(images=batch_images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

            # CLS（最終層）
            cls = outputs.last_hidden_state[:, 0, :]  # (B, 768)

            # 中間層パッチ
            hidden_states = outputs.hidden_states
            mid_layer = hidden_states[MID_LAYER_INDEX + 1]  # (B, 201, 768)
            mid_cls = mid_layer[:, 0, :]  # (B, 768)
            mid_patches = mid_layer[:, 5:, :]  # (B, 196, 768) - skip CLS + 4 registers

            # v3統計量
            stats = compute_patch_stats_v3(mid_patches, mid_cls)  # (B, 33)

        cls_embeddings.append(cls.cpu().numpy())
        patch_stats_list.append(stats.cpu().numpy())
        valid_files.extend(batch_valid)

    # 結合・保存
    cls_embeddings = np.concatenate(cls_embeddings, axis=0)
    patch_stats = np.concatenate(patch_stats_list, axis=0)

    out_cls.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_cls, cls_embeddings.astype(np.float32))
    np.save(out_stats, patch_stats.astype(np.float32))
    with open(out_files, "w") as f:
        f.write("\n".join(valid_files))

    # ボリュームをコミット
    embeddings_vol.commit()

    print(f"[DONE] {category}: {len(valid_files)} samples")
    print(f"  CLS: {cls_embeddings.shape}")
    print(f"  Stats v3: {patch_stats.shape}")

    # 統計サマリー
    print("\n[STATS] v3 statistics summary:")
    for idx, name in enumerate(V3_STAT_NAMES):
        mean = patch_stats[:, idx].mean()
        std = patch_stats[:, idx].std()
        print(f"  [{idx:2d}] {name:20s}: {mean:.6f} ± {std:.6f}")

    return {
        "status": "success",
        "category": category,
        "samples": len(valid_files),
        "cls_shape": cls_embeddings.shape,
        "stats_shape": patch_stats.shape,
    }


@app.function(
    image=image,
    volumes={"/images": images_vol, "/embeddings": embeddings_vol},
    timeout=300,
)
def list_categories():
    """利用可能なカテゴリ一覧"""
    img_root = Path("/images")
    categories = sorted([d.name for d in img_root.iterdir() if d.is_dir()])

    emb_root = Path("/embeddings")
    done = set()
    if emb_root.exists():
        done = {p.stem.replace("_patch_stats_v3", "") for p in emb_root.glob("*_patch_stats_v3.npy")}

    print("=== Categories ===")
    for cat in categories:
        status = "✓" if cat in done else " "
        print(f"  [{status}] {cat}")

    return {"categories": categories, "done": list(done)}


@app.local_entrypoint()
def main(
    category: str = "",
    all_categories: bool = False,
    list_only: bool = False,
    limit: int = 0,
):
    """
    Modal v3抽出のエントリーポイント

    Examples:
        modal run scripts/modal_extract_v3.py --list-only
        modal run scripts/modal_extract_v3.py --category novelai_ai
        modal run scripts/modal_extract_v3.py --all-categories
    """
    if list_only:
        result = list_categories.remote()
        return

    if all_categories:
        result = list_categories.remote()
        categories = result["categories"]
        done = set(result["done"])

        pending = [c for c in categories if c not in done]
        print(f"Processing {len(pending)} categories...")

        # 並列実行
        results = list(extract_category.map(pending, kwargs={"limit": limit}))

        for r in results:
            print(f"  {r['category']}: {r['status']}")
        return

    if category:
        result = extract_category.remote(category, limit=limit)
        print(result)
        return

    print("Usage:")
    print("  modal run scripts/modal_extract_v3.py --list-only")
    print("  modal run scripts/modal_extract_v3.py --category <name>")
    print("  modal run scripts/modal_extract_v3.py --all-categories")
