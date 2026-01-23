#!/usr/bin/env python3
"""
CbCr統計 + 線統計を抽出して保存

使い方:
  python scripts/extract_extra_stats.py --dir /path/to/images --name category_name
  python scripts/extract_extra_stats.py --all  # 全カテゴリを処理

出力:
  embeddings/{name}_extra_stats.npy  # (N, 15) の統計量

次元構成:
  [0] cbcr_var_global      - 全体CbCr分散 (log1p適用、※ablation推奨)
  [1] cbcr_var_flat        - Flat領域限定CbCr分散 (log1p適用)
  [2] flat_cluster_median  - Flatクラスタ中央値（典型的塊サイズ）
  [3] flat_cluster_mean    - Flatクラスタ平均サイズ ★Medium〜Large
  [4] flat_cluster_count   - Flatクラスタ数（√面積正規化）★Large
  [5] cbcr_autocorr        - 空間自己相関
  [6] flat_ratio           - Flat領域の割合 ★最強
  [7] flat_cluster_size    - Flat領域の最大クラスタサイズ（4近傍連結）
  [8] edge_direction_entropy - エッジ方向エントロピー
  [9] edge_length_mean     - エッジ連結長平均 (閾値依存注意)
  [10] edge_length_var     - エッジ連結長分散
  [11] edge_length_gini    - エッジ連結長ジニ係数 ★Medium
  [12] micro_loop_ratio    - 微小ループ率
  [13] patch_edge_density_var - patch間エッジ密度分散
  [14] flat_ratio_variance - パッチ間flat_ratio分散
"""

import argparse
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from scipy import ndimage
from multiprocessing import Pool, TimeoutError as MPTimeoutError
import warnings
warnings.filterwarnings('ignore')

EMBEDDINGS_DIR = Path("/home/techne/aicheckers/embeddings")

# Flat領域判定の相対閾値（上位N%を平坦とみなす）
FLAT_PERCENTILE = 15  # 勾配が小さい上位15%（感度分析で最適と判定）
# Flatクラスタとして扱う最小面積（画像全体に対する比率）
CLUSTER_MIN_FRAC = 0.005  # 0.5%未満はノイズとして除外


def rgb_to_ycbcr(img_rgb):
    """RGB -> YCbCr変換"""
    img = img_rgb.astype(np.float32)
    y = 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]
    cb = 128 - 0.168736 * img[:,:,0] - 0.331264 * img[:,:,1] + 0.5 * img[:,:,2]
    cr = 128 + 0.5 * img[:,:,0] - 0.418688 * img[:,:,1] - 0.081312 * img[:,:,2]
    return y, cb, cr


def get_flat_mask(img_gray, percentile=FLAT_PERCENTILE):
    """勾配が小さい領域（Flat領域）のマスクを取得"""
    # Sobelで勾配計算
    grad_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    # 相対閾値で判定（絶対値を避ける）
    threshold = np.percentile(grad_mag, percentile)
    flat_mask = grad_mag <= threshold

    return flat_mask, grad_mag


def compute_flat_cluster_stats(flat_mask):
    """Flat領域のクラスタ統計を計算（max, median, mean, count）"""
    # 連結成分分析
    flat_uint8 = flat_mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(flat_uint8, connectivity=4)

    img_size = flat_mask.shape[0] * flat_mask.shape[1]

    if num_labels <= 1:  # 背景のみ
        return 0.0, 0.0, 0.0, 0.0

    # 背景(label=0)を除いた統計
    cluster_sizes = stats[1:, cv2.CC_STAT_AREA].astype(np.float32) / img_size
    cluster_sizes = cluster_sizes[cluster_sizes >= CLUSTER_MIN_FRAC]
    if cluster_sizes.size == 0:
        return 0.0, 0.0, 0.0, 0.0

    max_cluster_size = float(cluster_sizes.max())
    median_cluster_size = float(np.median(cluster_sizes))
    mean_cluster_size = float(np.mean(cluster_sizes))
    cluster_count = float(cluster_sizes.size)

    return (
        max_cluster_size,            # max: 0-1スケール
        median_cluster_size,         # median: 0-1スケール
        mean_cluster_size,           # mean: 0-1スケール
        cluster_count / np.sqrt(img_size) * 100  # count: √面積正規化
    )


def compute_flat_ratio_variance(img_gray, patch_size=32):
    """パッチごとのflat_ratioの分散を計算（分布のムラ）"""
    h, w = img_gray.shape

    # Sobelで勾配計算
    grad_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    # 全体の閾値（相対）
    threshold = np.percentile(grad_mag, FLAT_PERCENTILE)

    # パッチごとのflat_ratioを計算
    patch_flat_ratios = []
    for i in range(0, h - patch_size, patch_size):
        for j in range(0, w - patch_size, patch_size):
            patch_grad = grad_mag[i:i+patch_size, j:j+patch_size]
            flat_ratio = (patch_grad <= threshold).sum() / (patch_size * patch_size)
            patch_flat_ratios.append(flat_ratio)

    if len(patch_flat_ratios) > 1:
        return np.var(patch_flat_ratios)
    return 0.0


def compute_cbcr_stats(img_rgb, flat_mask):
    """CbCr統計 + Flat統計を計算（8次元）"""
    y, cb, cr = rgb_to_ycbcr(img_rgb)

    # CbCrを結合（色差成分）
    cbcr = np.stack([cb, cr], axis=-1)

    # [0] 全体CbCr分散（※ショートカット注意）
    cbcr_var_global = cbcr.var()

    # [1] Flat領域限定CbCr分散
    if flat_mask.sum() > 100:
        cbcr_flat = cbcr[flat_mask]
        cbcr_var_flat = cbcr_flat.var()
    else:
        cbcr_var_flat = 0.0

    # Flatクラスタ統計（max, median, mean, count）
    flat_cluster_size, flat_cluster_median, flat_cluster_mean, flat_cluster_count = compute_flat_cluster_stats(flat_mask)

    # [5] 空間自己相関（ラグ1）
    cb_shifted = np.roll(cb, 1, axis=1)
    cr_shifted = np.roll(cr, 1, axis=1)

    cb_corr = np.corrcoef(cb.flatten(), cb_shifted.flatten())[0, 1]
    cr_corr = np.corrcoef(cr.flatten(), cr_shifted.flatten())[0, 1]
    cbcr_autocorr = (cb_corr + cr_corr) / 2
    if np.isnan(cbcr_autocorr):
        cbcr_autocorr = 0.0

    # [6] Flat領域の割合
    flat_ratio = flat_mask.sum() / flat_mask.size

    # log1p で分散の裾を潰す（解像度・圧縮依存のショートカット回避）
    return np.array([
        np.log1p(cbcr_var_global),  # [0]
        np.log1p(cbcr_var_flat),    # [1]
        flat_cluster_median,         # [2]
        flat_cluster_mean,           # [3] ★新規
        flat_cluster_count,          # [4] ★新規
        cbcr_autocorr,               # [5]
        flat_ratio,                  # [6]
        flat_cluster_size            # [7]
    ], dtype=np.float32)


def compute_gini(values):
    """ジニ係数を計算（0=完全均等, 1=完全不均等）"""
    if len(values) == 0:
        return 0.0
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    cumsum = np.cumsum(sorted_vals)
    return (2 * np.sum((np.arange(1, n+1) * sorted_vals)) / (n * cumsum[-1] + 1e-8)) - (n + 1) / n


def compute_line_stats(img_gray):
    """線統計を計算（6次元）"""
    # Cannyエッジ検出
    edges = cv2.Canny(img_gray, 50, 150)

    if edges.sum() < 100:
        # エッジがほぼない場合
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    # [8] エッジ方向エントロピー
    grad_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)

    # エッジ上の勾配方向のみ
    edge_mask = edges > 0
    angles = np.arctan2(grad_y[edge_mask], grad_x[edge_mask])

    # 方向を8ビンにヒストグラム化
    hist, _ = np.histogram(angles, bins=8, range=(-np.pi, np.pi))
    hist = hist / (hist.sum() + 1e-8)

    # エントロピー計算
    edge_direction_entropy = -np.sum(hist * np.log(hist + 1e-8)) / np.log(8)

    # [9,10,11] エッジ連結長の平均・分散・ジニ係数
    labeled, num_features = ndimage.label(edges)
    if num_features > 0:
        lengths = ndimage.sum(edges, labeled, range(1, num_features + 1))
        edge_length_mean = np.mean(lengths)
        edge_length_var = np.var(lengths) if len(lengths) > 1 else 0.0
        edge_length_gini = compute_gini(lengths)
    else:
        edge_length_mean = 0.0
        edge_length_var = 0.0
        edge_length_gini = 0.0

    # 正規化（画像サイズに依存しないように）
    img_size = img_gray.shape[0] * img_gray.shape[1]
    edge_length_mean = edge_length_mean / np.sqrt(img_size) * 100
    edge_length_var = edge_length_var / img_size * 10000

    # [8] 微小ループ率
    # 小さな閉じた輪郭を検出
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    micro_loop_count = 0
    total_contours = len(contours)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)

        # 微小ループ: 面積が小さく、閉じている
        if perimeter > 0 and area > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if area < 100 and circularity > 0.3:  # 小さくて丸っぽい
                micro_loop_count += 1

    micro_loop_ratio = micro_loop_count / (total_contours + 1)

    # [9] patch間エッジ密度分散
    # 画像を16x16パッチに分割し、各パッチのエッジ密度の分散を計算
    h, w = img_gray.shape
    patch_size = 32  # 32x32ピクセルのパッチ
    patch_densities = []

    for i in range(0, h - patch_size, patch_size):
        for j in range(0, w - patch_size, patch_size):
            patch = edges[i:i+patch_size, j:j+patch_size]
            density = patch.sum() / (patch_size * patch_size * 255)  # 0-1に正規化
            patch_densities.append(density)

    if len(patch_densities) > 1:
        patch_edge_density_var = np.var(patch_densities)
    else:
        patch_edge_density_var = 0.0

    return np.array([
        edge_direction_entropy,  # [8]
        edge_length_mean,        # [9]
        edge_length_var,         # [10]
        edge_length_gini,        # [11] ★新規
        micro_loop_ratio,        # [12]
        patch_edge_density_var   # [13]
    ], dtype=np.float32)


def extract_extra_stats(img_path):
    """画像から追加統計量を抽出（15次元）"""
    try:
        # ファイルサイズチェック（100MB以上は異常）
        file_size = Path(img_path).stat().st_size
        if file_size > 100 * 1024 * 1024:
            print(f"[SKIP] File too large: {img_path} ({file_size // 1024 // 1024}MB)")
            return None

        img = Image.open(img_path).convert("RGB")

        # 解像度チェック（32K以上は異常）
        if img.width > 32000 or img.height > 32000:
            print(f"[SKIP] Image too large: {img_path} ({img.width}x{img.height})")
            return None

        img_rgb = np.array(img)
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        # Flat領域マスク取得
        flat_mask, _ = get_flat_mask(img_gray)

        # CbCr + Flat統計（8d）
        cbcr_stats = compute_cbcr_stats(img_rgb, flat_mask)

        # 線統計（6d: edge_dir_entropy, edge_len_mean/var/gini, micro_loop, patch_edge_var）
        line_stats = compute_line_stats(img_gray)

        # flat_ratio_variance（1d）
        flat_ratio_var = compute_flat_ratio_variance(img_gray)

        # 結合（15d）
        return np.concatenate([cbcr_stats, line_stats, [flat_ratio_var]])

    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None


CHECKPOINT_INTERVAL = 1000  # チェックポイント保存間隔


def process_directory(img_dir, name, files_list=None, workers=6, resume=False):
    """ディレクトリ内の画像から統計量を抽出（並列処理対応、チェックポイント付き）"""
    img_dir = Path(img_dir)
    output_path = EMBEDDINGS_DIR / f"{name}_extra_stats.npy"
    checkpoint_path = EMBEDDINGS_DIR / f"{name}_extra_stats_checkpoint.npz"

    # 完了済みスキップ（resumeモード）
    if resume and output_path.exists():
        print(f"[SKIP] {name}: already completed ({output_path})")
        return

    # ファイルリストがあればそれを使用（順序を保持）
    if files_list and Path(files_list).exists():
        with open(files_list, 'r') as f:
            filenames = [line.strip() for line in f if line.strip()]
        images = [img_dir / fn for fn in filenames if (img_dir / fn).exists()]
        print(f"Using file list: {len(images)} images")
    else:
        images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.jpeg")) + list(img_dir.glob("*.png")) + list(img_dir.glob("*.webp"))
        print(f"Found {len(images)} images in directory")

    if not images:
        print(f"No images found in {img_dir}")
        return

    # チェックポイントからの再開
    start_idx = 0
    stats_list = []
    valid_files = []

    if checkpoint_path.exists():
        try:
            ckpt = np.load(checkpoint_path, allow_pickle=True)
            stats_arr = ckpt['stats']  # (N, 15) の2D配列
            stats_list = [stats_arr[i] for i in range(len(stats_arr))]
            valid_files = list(ckpt['files'])
            start_idx = int(ckpt['next_idx'])
            print(f"[RESUME] Loaded checkpoint: {start_idx}/{len(images)} completed, {len(stats_list)} valid")
        except Exception as e:
            print(f"[WARN] Failed to load checkpoint: {e}, starting fresh")
            start_idx = 0
            stats_list = []
            valid_files = []

    remaining_images = images[start_idx:]

    if not remaining_images:
        print(f"All images already processed")
    else:
        print(f"Processing {len(remaining_images)} images with {workers} workers...")
        print(f"Checkpoint every {CHECKPOINT_INTERVAL} images")

        # バッチ処理（チェックポイント対応）
        for batch_start in range(0, len(remaining_images), CHECKPOINT_INTERVAL):
            batch_end = min(batch_start + CHECKPOINT_INTERVAL, len(remaining_images))
            batch_images = remaining_images[batch_start:batch_end]

            # 並列処理（maxtasksperchildでメモリリーク防止）
            if workers > 1:
                with Pool(workers, maxtasksperchild=100) as pool:
                    results = list(tqdm(
                        pool.imap(extract_extra_stats, batch_images, chunksize=10),
                        total=len(batch_images),
                        desc=f"Extracting {name} [{start_idx + batch_start}-{start_idx + batch_end}/{len(images)}]"
                    ))
            else:
                results = [extract_extra_stats(p) for p in tqdm(batch_images, desc=f"Extracting {name}")]

            # 結果をフィルタリング（Noneを除外、順序保持）
            for img_path, stats in zip(batch_images, results):
                if stats is not None:
                    stats_list.append(stats)
                    valid_files.append(img_path.name)

            # チェックポイント保存（2D配列として保存）
            current_idx = start_idx + batch_end
            np.savez(
                checkpoint_path,
                stats=np.stack(stats_list) if stats_list else np.zeros((0, 15), dtype=np.float32),
                files=np.array(valid_files, dtype=str),
                next_idx=current_idx
            )
            print(f"[CHECKPOINT] Saved at {current_idx}/{len(images)} ({len(stats_list)} valid)")

    if not stats_list:
        print("No valid images processed")
        return

    stats_array = np.stack(stats_list)

    # 最終保存
    np.save(output_path, stats_array)

    # チェックポイント削除
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"[CLEANUP] Removed checkpoint file")

    print(f"\nSaved: {output_path}")
    print(f"Shape: {stats_array.shape}")
    print(f"Stats summary (15d):")
    print(f"  [0]  cbcr_var_global:     mean={stats_array[:,0].mean():.4f}, std={stats_array[:,0].std():.4f}")
    print(f"  [1]  cbcr_var_flat:       mean={stats_array[:,1].mean():.4f}, std={stats_array[:,1].std():.4f}")
    print(f"  [2]  flat_cluster_median: mean={stats_array[:,2].mean():.4f}, std={stats_array[:,2].std():.4f}")
    print(f"  [3]  flat_cluster_mean:   mean={stats_array[:,3].mean():.4f}, std={stats_array[:,3].std():.4f}")
    print(f"  [4]  flat_cluster_count:  mean={stats_array[:,4].mean():.4f}, std={stats_array[:,4].std():.4f}")
    print(f"  [5]  cbcr_autocorr:       mean={stats_array[:,5].mean():.4f}, std={stats_array[:,5].std():.4f}")
    print(f"  [6]  flat_ratio:          mean={stats_array[:,6].mean():.4f}, std={stats_array[:,6].std():.4f}")
    print(f"  [7]  flat_cluster_size:   mean={stats_array[:,7].mean():.4f}, std={stats_array[:,7].std():.4f}")
    print(f"  [8]  edge_dir_entropy:    mean={stats_array[:,8].mean():.4f}, std={stats_array[:,8].std():.4f}")
    print(f"  [9]  edge_len_mean:       mean={stats_array[:,9].mean():.4f}, std={stats_array[:,9].std():.4f}")
    print(f"  [10] edge_len_var:        mean={stats_array[:,10].mean():.4f}, std={stats_array[:,10].std():.4f}")
    print(f"  [11] edge_len_gini:       mean={stats_array[:,11].mean():.4f}, std={stats_array[:,11].std():.4f}")
    print(f"  [12] micro_loop_ratio:    mean={stats_array[:,12].mean():.4f}, std={stats_array[:,12].std():.4f}")
    print(f"  [13] patch_edge_var:      mean={stats_array[:,13].mean():.4f}, std={stats_array[:,13].std():.4f}")
    print(f"  [14] flat_ratio_var:      mean={stats_array[:,14].mean():.4f}, std={stats_array[:,14].std():.4f}")


def main():
    parser = argparse.ArgumentParser(description="Extract CbCr + line statistics")
    parser.add_argument("--dir", type=str, help="Image directory")
    parser.add_argument("--name", type=str, help="Category name for output")
    parser.add_argument("--files", type=str, help="File list (to match existing embeddings order)")
    parser.add_argument("--all", action="store_true", help="Process all categories with existing embeddings")
    parser.add_argument("--workers", type=int, default=6, help="Number of parallel workers (default: 6)")
    parser.add_argument("--resume", action="store_true", help="Skip completed categories and resume from checkpoint")
    args = parser.parse_args()

    if args.all:
        # 既存のembeddingsに対応するカテゴリを処理
        # パスマッピング (2026-01-10 確認済み)
        CIVITAI_BASE = "/home/techne/aicheckers/data/animedl2m_dataset_release/civitai_subset/image"
        categories = {
            "illustrious_ai": f"{CIVITAI_BASE}/Illustrious",
            "pony_ai": f"{CIVITAI_BASE}/Pony",
            "sdxl10_ai": f"{CIVITAI_BASE}/SDXL 1.0",
            "sd15_ai": f"{CIVITAI_BASE}/SD 1.5",
            "other_ai": f"{CIVITAI_BASE}/Other",
            "flux1d_ai": f"{CIVITAI_BASE}/Flux.1 D",
            "novelai_ai": "/home/techne/aicheckers/data/novelai",
            "novelai_combined_ai": "/home/techne/aicheckers/data/novelai_combined",
            "novelai_artist_tagged_ai": "/home/techne/aicheckers/data/novelai_artist_tagged",
            "danbooru_real": "/home/techne/aicheckers/data/animedl2m_dataset_release/real_images/images",
            # 追加カテゴリ (2026-01-11)
            "pixai_ai": "/home/techne/aicheckers/data/pixai",
            "novelai_aibooru_ai": "/home/techne/aicheckers/data/novelai",  # files.txtで指定
            "pixiv_novelai_v2_ai": "/home/techne/aicheckers/data/novelai_combined",  # files.txtで指定
            "twitter_novelai_v2_ai": "/home/techne/aicheckers/data/novelai_combined",  # files.txtで指定
        }

        for name, dir_path in categories.items():
            files_list = EMBEDDINGS_DIR / f"{name}_files.txt"
            if not Path(dir_path).exists():
                print(f"Skipping {name}: directory not found")
                continue
            if not (EMBEDDINGS_DIR / f"{name}.npy").exists():
                print(f"Skipping {name}: no existing embeddings")
                continue

            print(f"\n{'='*60}")
            print(f"Processing: {name}")
            print(f"{'='*60}")
            process_directory(dir_path, name, files_list if files_list.exists() else None,
                            workers=args.workers, resume=args.resume)

    elif args.dir and args.name:
        process_directory(args.dir, args.name, args.files, workers=args.workers, resume=args.resume)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
