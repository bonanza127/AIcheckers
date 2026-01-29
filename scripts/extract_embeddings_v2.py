#!/usr/bin/env python3
"""
DINOv3 embedding抽出スクリプト v2.4
CLSトークン（最終層） + パッチ統計量v2（中間層）を保存

出力:
  - {name}.npy: CLSトークン (N, 768) - 最終層から取得
  - {name}_patch_stats.npy: パッチ統計量v2 (N, 7) - 中間層から取得
    ※ 保存ファイル名は歴史的経緯で_patch_stats_v3.npyだが中身はv2フォーマット（7次元）
    - [0] adj_sim_mean:   隣接パッチ平均コサイン類似度
    - [1] adj_sim_var:    隣接パッチ類似度分散
    - [2] high_sim_ratio: 高類似度率（>0.9）
    - [3] patch_var:      パッチ埋め込み分散
    - [4] anisotropy:     縦横類似度差
    - [5] norm_var:       ノルム分散
    - [6] norm_range:     ノルムレンジ（max - min）

設計原則 (2026-01):
  - 中間層から「分類器を通さない」教師なし統計量を抽出
  - 未知のAIモデルに対する汎化性能を高める
"""
import os
import sys
import argparse
import io
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image, ImageFilter, ImageOps
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

# 共通モジュール
sys.path.insert(0, str(Path(__file__).parent.parent))
from lib.patch_stats import compute_patch_stats_v2_batch

# 設定
DINOV3_MODEL_PATH = Path("/home/techne/aicheckers/models/dinov3-vitb16")
EMBEDDINGS_DIR = Path("/home/techne/aicheckers/embeddings")
MID_LAYER_INDEX = 8  # 中間層のインデックス（0-11、Block 6-8推奨）


def apply_degradation(img: Image.Image) -> tuple[Image.Image, int]:
    """
    画像に劣化処理を適用（画質バイアスを除去するため）

    v2.3: jpeg + resizeのみ、強制適用廃止
    - 各劣化を独立確率で選択（合計1.0 = 平均1種類/枚）
    - 最大2つまで重ねがけ（過剰劣化防止）
    - 何も選択されなかった場合はクリーン画像を返す
    """
    degradations = []
    if random.random() < 0.50:
        degradations.append('jpeg')
    if random.random() < 0.50:
        degradations.append('resize')

    if len(degradations) > 2:
        degradations = random.sample(degradations, 2)

    if not degradations:
        return img, 0

    random.shuffle(degradations)

    for deg in degradations:
        if deg == 'jpeg':
            quality = random.randint(55, 85)
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=quality)
            buffer.seek(0)
            img = Image.open(buffer).convert('RGB')
        elif deg == 'resize':
            scale = random.uniform(0.6, 0.9)
            w, h = img.size
            img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
            img = img.resize((w, h), Image.BILINEAR)

    return img, len(degradations)


def apply_scale_tta(img: Image.Image, scale: float = 0.85) -> Image.Image:
    """推論時TTA整合性のためのスケール変換（縮小→元サイズに戻す）"""
    w, h = img.size
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    img = img.resize((new_w, new_h), Image.LANCZOS)
    img = img.resize((w, h), Image.LANCZOS)
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

    num_register_tokens = getattr(model.config, "num_register_tokens", 4)
    patch_start_idx = 1 + num_register_tokens
    image_size = getattr(model.config, "image_size", 224)
    patch_size = getattr(model.config, "patch_size", 16)
    num_patches = (image_size // patch_size) ** 2

    print(f"[INFO] Loaded DINOv3 from: {DINOV3_MODEL_PATH}")
    print(f"[INFO] Mid-layer extraction enabled: Block {MID_LAYER_INDEX}")
    print(f"[INFO] Token layout: CLS(1) + REG({num_register_tokens}) + PATCH({num_patches}) = {1 + num_register_tokens + num_patches} tokens")
    return model, processor, device, patch_start_idx, num_patches


# NOTE: 分類器ロードは不要になりました（v2は教師なし統計量のため）


def extract_embeddings(image_dir: Path, model, processor, device, batch_size=32,
                        degradation_prob=0.3, flip_prob=0.5, scale_prob=0.5,
                        mid_layer=MID_LAYER_INDEX, num_workers=8, file_list=None,
                        patch_start_idx=5, num_patches=196):
    """ディレクトリ内の全画像からCLS（最終層）とパッチ統計量（中間層）を抽出

    Args:
        degradation_prob: 劣化Augmentationを適用する確率 (0.0-1.0, default: 0.3)
                          jpeg + resizeのみ、未選択時はクリーン画像
        flip_prob: 水平反転Augmentationを適用する確率 (0.0-1.0, default: 0.5)
        scale_prob: スケールAugmentationを適用する確率 (0.0-1.0, default: 0.5)
        mid_layer: パッチ統計量を抽出する中間層のインデックス (0-11)
        num_workers: 画像読み込み・Augmentation用の並列ワーカー数
        file_list: 処理対象のファイルパスリスト
        patch_start_idx: パッチトークン開始インデックス (CLS + REG)
        num_patches: パッチトークン数
    """
    if file_list is not None:
        image_files = []
        for f in file_list:
            p = Path(f)
            if not p.is_absolute():
                p = image_dir / p
            if p.exists():
                image_files.append(p)
        print(f"Using file list: {len(image_files)} files (from {len(file_list)} entries)")
    else:
        extensions = {'.jpg', '.jpeg', '.png', '.webp', '.gif'}
        image_files = [f for f in image_dir.rglob("*")
                       if f.is_file() and f.suffix.lower() in extensions]

    print(f"Found {len(image_files)} images in {image_dir}")
    if degradation_prob > 0:
        print(f"Degradation augmentation: {degradation_prob*100:.0f}% (avg 1 type/image)")
    if flip_prob > 0:
        print(f"Flip augmentation: {flip_prob*100:.0f}%")
    if scale_prob > 0:
        print(f"Scale augmentation: {scale_prob*100:.0f}% (scale=0.85, TTA consistency)")
    print(f"Using {num_workers} workers for parallel image loading")

    cls_embeddings = []
    patch_stats_list = []
    filenames = []
    degradation_count = 0
    flip_count = 0
    scale_count = 0

    def load_and_augment(f):
        try:
            img = Image.open(f).convert("RGB")
            applied_flip = False
            applied_deg = False
            applied_scale = False
            if flip_prob > 0 and random.random() < flip_prob:
                img = ImageOps.mirror(img)
                applied_flip = True
            if scale_prob > 0 and random.random() < scale_prob:
                img = apply_scale_tta(img, scale=0.85)
                applied_scale = True
            if degradation_prob > 0 and random.random() < degradation_prob:
                img, deg_n = apply_degradation(img)
                applied_deg = deg_n > 0
            return (img, f.name, applied_flip, applied_deg, applied_scale)
        except Exception as e:
            return (None, f.name, False, False, False)

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for i in tqdm(range(0, len(image_files), batch_size), desc="Extracting"):
            batch_files = image_files[i:i+batch_size]
            results = list(executor.map(load_and_augment, batch_files))

            batch_images = []
            batch_names = []
            for img, name, did_flip, did_deg, did_scale in results:
                if img is not None:
                    batch_images.append(img)
                    batch_names.append(name)
                    if did_flip: flip_count += 1
                    if did_deg: degradation_count += 1
                    if did_scale: scale_count += 1

            if not batch_images:
                continue

            inputs = processor(images=batch_images, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                final_hidden = outputs.last_hidden_state
                cls_emb = final_hidden[:, 0, :].cpu().numpy()
                mid_hidden = outputs.hidden_states[mid_layer + 1]
                mid_patch_emb = mid_hidden[:, patch_start_idx:patch_start_idx+num_patches, :]
                patch_stats = compute_patch_stats_v2_batch(mid_patch_emb)

            cls_embeddings.append(cls_emb)
            patch_stats_list.append(patch_stats)
            filenames.extend(batch_names)

    cls_embeddings = np.vstack(cls_embeddings)
    patch_stats_all = np.vstack(patch_stats_list)

    if flip_prob > 0:
        print(f"Applied flip to {flip_count}/{len(filenames)} images ({flip_count/len(filenames)*100:.1f}%)")
    if scale_prob > 0:
        print(f"Applied scale to {scale_count}/{len(filenames)} images ({scale_count/len(filenames)*100:.1f}%)")
    if degradation_prob > 0:
        print(f"Applied degradation to {degradation_count}/{len(filenames)} images ({degradation_count/len(filenames)*100:.1f}%)")

    return cls_embeddings, patch_stats_all, filenames


def main():
    parser = argparse.ArgumentParser(description="DINOv3 embedding extraction v2.4 (mid-layer stats)")
    parser.add_argument("--dir", type=str, required=True, help="Image directory")
    parser.add_argument("--name", type=str, required=True, help="Output name (e.g., 'illustrious_ai')")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--mid-layer", type=int, default=MID_LAYER_INDEX,
                        help=f"Mid-layer index for patch stats (0-11, default: {MID_LAYER_INDEX})")
    parser.add_argument("--degradation-prob", type=float, default=0.3,
                        help="Probability of applying degradation augmentation (0.0-1.0, default: 0.3)")
    parser.add_argument("--flip-prob", type=float, default=0.5)
    parser.add_argument("--scale-prob", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--no-aug", action="store_true", help="Disable all augmentations")
    parser.add_argument("--file-list", type=str, default=None, help="Text file with image paths")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of images to process (for testing)")
    args = parser.parse_args()

    image_dir = Path(args.dir)
    if not image_dir.exists():
        print(f"Error: {image_dir} does not exist")
        sys.exit(1)

    EMBEDDINGS_DIR.mkdir(exist_ok=True)

    # モデルロード
    model, processor, device, patch_start_idx, num_patches = load_model()

    # Augmentation設定
    degradation_prob = 0.0 if args.no_aug else args.degradation_prob
    flip_prob = 0.0 if args.no_aug else args.flip_prob
    scale_prob = 0.0 if args.no_aug else args.scale_prob

    # file_list処理
    file_list = None
    if args.file_list:
        file_list_path = Path(args.file_list)
        if file_list_path.exists():
            file_list = [line.strip() for line in file_list_path.read_text().strip().split('\n') if line.strip()]

    # 抽出（v2: 分類器不要）
    cls_embeddings, patch_stats, filenames = extract_embeddings(
        image_dir, model, processor, device, args.batch_size,
        degradation_prob, flip_prob, scale_prob, args.mid_layer, args.num_workers,
        file_list=file_list, patch_start_idx=patch_start_idx, num_patches=num_patches
    )

    # limitオプション対応
    if args.limit and len(cls_embeddings) > args.limit:
        cls_embeddings = cls_embeddings[:args.limit]
        patch_stats = patch_stats[:args.limit]
        filenames = filenames[:args.limit]

    # 保存（v2は常に同じフォーマット）
    cls_path = EMBEDDINGS_DIR / f"{args.name}.npy"
    stats_path = EMBEDDINGS_DIR / f"{args.name}_patch_stats.npy"
    names_path = EMBEDDINGS_DIR / f"{args.name}_files.txt"

    np.save(cls_path, cls_embeddings)
    np.save(stats_path, patch_stats)
    with open(names_path, "w") as f:
        f.write("\n".join(filenames))

    print(f"\n[DONE] Saved {len(cls_embeddings)} samples")
    print(f"  CLS embeddings (final layer): {cls_path} ({cls_embeddings.shape})")
    print(f"  Patch stats v2 (mid-layer {args.mid_layer}): {stats_path} ({patch_stats.shape})")
    print(f"  Filenames: {names_path}")

    # 統計量サマリー（v2形式）
    print(f"\n[STATS] Patch statistics v2 summary (unsupervised, mid-layer {args.mid_layer}):")
    print(f"  [0] adj_sim_mean:   {patch_stats[:, 0].mean():.4f} ± {patch_stats[:, 0].std():.4f}")
    print(f"  [1] adj_sim_var:    {patch_stats[:, 1].mean():.6f} ± {patch_stats[:, 1].std():.6f}")
    print(f"  [2] high_sim_ratio: {patch_stats[:, 2].mean():.4f} ± {patch_stats[:, 2].std():.4f}")
    print(f"  [3] patch_var:      {patch_stats[:, 3].mean():.4f} ± {patch_stats[:, 3].std():.4f}")
    print(f"  [4] anisotropy:     {patch_stats[:, 4].mean():.6f} ± {patch_stats[:, 4].std():.6f}")
    print(f"  [5] norm_var:       {patch_stats[:, 5].mean():.4f} ± {patch_stats[:, 5].std():.4f}")
    print(f"  [6] norm_range:     {patch_stats[:, 6].mean():.4f} ± {patch_stats[:, 6].std():.4f}")


if __name__ == "__main__":
    main()
