#!/usr/bin/env python3
"""
Boundary Stats 特徴抽出スクリプト (5次元)
- fractal_dim_1x: 境界のフラクタル次元（通常スケール）[メイン]
- fractal_dim_05x: 境界のフラクタル次元（0.5xスケール）[メイン]
- fractal_dim_diff: スケール間差分（1x - 0.5x）[メイン・Hard Neg対策]
- rank_entropy: flat領域のランクエントロピー [メイン]
- curvature_var: 境界曲率の分散 [アブレーション用]

使用法:
    python scripts/extract_boundary_stats.py --all --resume
    python scripts/extract_boundary_stats.py --name pony_ai --dir /path/to/images
"""
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
import argparse
from tqdm import tqdm

# 設定
EMBEDDINGS_DIR = Path("/home/techne/aicheckers/embeddings")
CHECKPOINT_INTERVAL = 1000

# Flat領域判定の相対閾値
FLAT_PERCENTILE = 15

# Box-counting用サイズ（固定）
BOX_SIZES_1X = [2, 4, 8, 16, 32]
BOX_SIZES_05X = [2, 4, 8, 16]  # 1段減らす


def get_flat_mask(img_gray):
    """勾配が小さい領域のマスクを取得"""
    grad_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    threshold = np.percentile(grad_mag, FLAT_PERCENTILE)
    flat_mask = (grad_mag <= threshold).astype(np.uint8)
    return flat_mask


def get_edge_map(img_gray):
    """Cannyエッジマップを取得"""
    edges = cv2.Canny(img_gray, 50, 150)
    return edges


def box_counting_dimension(binary_map, box_sizes):
    """Box-counting法でフラクタル次元を計算（高速化版）"""
    counts = []
    h, w = binary_map.shape

    for box_size in box_sizes:
        if box_size > min(h, w):
            continue

        # 端を切り落として box_size で割り切れるサイズに
        h_trim = (h // box_size) * box_size
        w_trim = (w // box_size) * box_size
        trimmed = binary_map[:h_trim, :w_trim]

        # reshape して各ボックスの max を取る（高速）
        n_rows = h_trim // box_size
        n_cols = w_trim // box_size
        reshaped = trimmed.reshape(n_rows, box_size, n_cols, box_size)
        # 各ボックスに1つでもエッジがあるか
        box_has_edge = reshaped.any(axis=(1, 3))
        count = box_has_edge.sum()
        counts.append((box_size, count))

    if len(counts) < 2:
        return 0.0

    # log-log回帰でフラクタル次元を推定
    log_sizes = np.log([1.0 / s for s, _ in counts])
    log_counts = np.log([max(c, 1) for _, c in counts])  # log(0)防止

    try:
        slope, _ = np.polyfit(log_sizes, log_counts, 1)
        return slope
    except:
        return 0.0


def compute_fractal_dim_multiscale(edge_map):
    """2スケールでフラクタル次元を計算"""
    # スケール1.0
    fd_1x = box_counting_dimension(edge_map, BOX_SIZES_1X)

    # スケール0.5（リサイズ + 再二値化）
    edge_half = cv2.resize(
        edge_map,
        None,
        fx=0.5,
        fy=0.5,
        interpolation=cv2.INTER_AREA
    )
    edge_half = (edge_half > 127).astype(np.uint8) * 255

    fd_05x = box_counting_dimension(edge_half, BOX_SIZES_05X)

    # 差分
    fd_diff = fd_1x - fd_05x

    return fd_1x, fd_05x, fd_diff


def compute_boundary_features(flat_mask):
    """Flat領域境界の曲率分散を計算"""
    contours, _ = cv2.findContours(flat_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    all_curvatures = []

    for contour in contours:
        if len(contour) < 10:
            continue

        area = cv2.contourArea(contour)
        if area < 100:
            continue

        contour = contour.squeeze()
        if len(contour.shape) == 1:
            continue

        for i in range(len(contour)):
            p1 = contour[i - 2]
            p2 = contour[i - 1]
            p3 = contour[i]

            v1 = p2 - p1
            v2 = p3 - p2

            cross = v1[0] * v2[1] - v1[1] * v2[0]
            dot = v1[0] * v2[0] + v1[1] * v2[1]
            angle = np.arctan2(cross, dot)
            all_curvatures.append(abs(angle))

    curvature_var = np.var(all_curvatures) if all_curvatures else 0

    return curvature_var


def compute_rank_entropy(img_gray, flat_mask):
    """Flat領域のランクエントロピーを計算"""
    flat_values = img_gray[flat_mask > 0]

    if len(flat_values) < 100:
        return 0.0

    hist, _ = np.histogram(flat_values, bins=256, range=(0, 256))
    hist = hist / hist.sum()
    hist = hist[hist > 0]

    entropy = -np.sum(hist * np.log2(hist + 1e-10))

    return entropy


def process_image(img_path):
    """画像から5次元特徴を抽出
    [fractal_dim_1x, fractal_dim_05x, fractal_dim_diff, rank_entropy, curvature_var]
    """
    try:
        img = Image.open(img_path).convert('RGB')
        img = img.resize((512, 512), Image.LANCZOS)
        img_np = np.array(img)
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        # Edge map（1回だけ計算）
        edge_map = get_edge_map(img_gray)

        # Flat mask（rank_entropyとcurvature_var用）
        flat_mask = get_flat_mask(img_gray)

        # Fractal dim (2スケール + 差分)
        fd_1x, fd_05x, fd_diff = compute_fractal_dim_multiscale(edge_map)

        # Rank entropy
        rank_entropy = compute_rank_entropy(img_gray, flat_mask)

        # Curvature var (アブレーション用)
        curvature_var = compute_boundary_features(flat_mask)

        # 順序: [fd_1x, fd_05x, fd_diff, rank_entropy, curvature_var]
        return np.array([fd_1x, fd_05x, fd_diff, rank_entropy, curvature_var], dtype=np.float32)

    except Exception as e:
        return None


def extract_category(name: str, image_dir: Path, resume: bool = False):
    """カテゴリの特徴を抽出"""
    output_path = EMBEDDINGS_DIR / f"{name}_boundary_stats.npy"
    checkpoint_path = EMBEDDINGS_DIR / f"{name}_boundary_stats_checkpoint.npz"
    files_path = EMBEDDINGS_DIR / f"{name}_boundary_files.txt"

    # 既存の_files.txtがあればそれを使う（embedding抽出時と同じファイルセット）
    existing_files_path = EMBEDDINGS_DIR / f"{name}_files.txt"
    if existing_files_path.exists():
        with open(existing_files_path) as f:
            file_names = [line.strip() for line in f if line.strip()]
        images = [image_dir / fn for fn in file_names if (image_dir / fn).exists()]
        print(f"[INFO] Using existing file list: {existing_files_path} ({len(images)} files)")
    else:
        # フォールバック: ディレクトリ全体をglob
        images = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.jpeg")) + list(image_dir.glob("*.png"))

    if not images:
        print(f"[WARN] No images found in {image_dir}")
        return

    print(f"\n=== Extracting {name} ===")
    print(f"  Images: {len(images)}")
    print(f"  Output: {output_path}")
    print(f"  Features: [fd_1x, fd_05x, fd_diff, rank_entropy, curvature_var]")

    # Resume check
    stats_list = []
    files_list = []
    start_idx = 0

    if resume and checkpoint_path.exists():
        try:
            ckpt = np.load(checkpoint_path, allow_pickle=True)
            stats_list = list(ckpt['stats'])
            files_list = list(ckpt['files'])
            start_idx = int(ckpt['next_idx'])
            print(f"[RESUME] Loaded checkpoint: {start_idx}/{len(images)} completed")
        except Exception as e:
            print(f"[WARN] Failed to load checkpoint: {e}")

    if resume and output_path.exists() and not checkpoint_path.exists():
        print(f"[SKIP] Already completed: {output_path}")
        return

    # 抽出
    remaining_images = images[start_idx:]

    for batch_start in range(0, len(remaining_images), CHECKPOINT_INTERVAL):
        batch_end = min(batch_start + CHECKPOINT_INTERVAL, len(remaining_images))
        batch_images = remaining_images[batch_start:batch_end]
        current_idx = start_idx + batch_start

        desc = f"Extracting {name} [{current_idx}-{current_idx + len(batch_images)}/{len(images)}]"

        for img_path in tqdm(batch_images, desc=desc):
            feat = process_image(img_path)
            if feat is not None:
                stats_list.append(feat)
                files_list.append(img_path.name)

        # Checkpoint保存
        np.savez(
            checkpoint_path,
            stats=np.array(stats_list),
            files=np.array(files_list),
            next_idx=start_idx + batch_end
        )
        print(f"[CHECKPOINT] Saved at {start_idx + batch_end}/{len(images)} ({len(stats_list)} valid)")

    # 最終保存
    stats_array = np.array(stats_list, dtype=np.float32)
    np.save(output_path, stats_array)

    with open(files_path, 'w') as f:
        for fname in files_list:
            f.write(fname + '\n')

    # Checkpoint削除
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"[CLEANUP] Removed checkpoint")

    print(f"[DONE] Saved {len(stats_list)} samples to {output_path}")
    print(f"  Shape: {stats_array.shape}")


def main():
    parser = argparse.ArgumentParser(description="Extract boundary stats (5d)")
    parser.add_argument("--name", type=str, help="Category name")
    parser.add_argument("--dir", type=str, help="Image directory")
    parser.add_argument("--all", action="store_true", help="Extract all categories")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()

    if args.all:
        # 全カテゴリ定義
        categories = {
            # AI
            "pony_ai": Path("/home/techne/aicheckers/data/animedl2m_dataset_release/civitai_subset/image/Pony"),
            "illustrious_ai": Path("/home/techne/aicheckers/data/animedl2m_dataset_release/civitai_subset/image/Illustrious"),
            "sdxl10_ai": Path("/home/techne/aicheckers/data/animedl2m_dataset_release/civitai_subset/image/SDXL 1.0"),
            "sd15_ai": Path("/home/techne/aicheckers/data/animedl2m_dataset_release/civitai_subset/image/SD 1.5"),
            "flux1d_ai": Path("/home/techne/aicheckers/data/animedl2m_dataset_release/civitai_subset/image/Flux.1 D"),
            "other_ai": Path("/home/techne/aicheckers/data/animedl2m_dataset_release/civitai_subset/image/Other"),
            "novelai_ai": Path("/home/techne/aicheckers/data/novelai"),
            "novelai_combined_ai": Path("/home/techne/aicheckers/data/novelai_combined"),
            "novelai_artist_tagged_ai": Path("/home/techne/aicheckers/data/novelai_artist_tagged"),
            "pixai_ai": Path("/home/techne/aicheckers/data/pixai"),
            # 追加カテゴリ (2026-01-11)
            "novelai_aibooru_ai": Path("/home/techne/aicheckers/data/novelai"),  # files.txtで指定
            "pixiv_novelai_v2_ai": Path("/home/techne/aicheckers/data/novelai_combined"),  # files.txtで指定
            "twitter_novelai_v2_ai": Path("/home/techne/aicheckers/data/novelai_combined"),  # files.txtで指定
            # Real
            "danbooru_real": Path("/home/techne/aicheckers/data/animedl2m_dataset_release/real_images/images"),
        }

        for name, dir_path in categories.items():
            if dir_path.exists():
                extract_category(name, dir_path, resume=args.resume)
            else:
                print(f"[SKIP] Directory not found: {dir_path}")

    elif args.name and args.dir:
        extract_category(args.name, Path(args.dir), resume=args.resume)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
