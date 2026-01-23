#!/usr/bin/env python3
"""
Extract CPU features v3 (Batch系特徴量 - 画像1から)

Output:
  embeddings/{category}_cpu_stats_v3.npy
  embeddings/{category}_cpu_stats_v3_files.txt
  embeddings/{category}_cpu_stats_v3_meta.json

Features (13d):
  1. histogram_flatness     - ヒストグラム平坦度 (Cohen's d: -1.361)
  2. histogram_modality     - ヒストグラムモード数 (+1.153)
  3. color_palette_entropy  - 色パレットエントロピー (+1.124)
  4. luminance_layer_count  - 輝度層数 (+0.892)
  5. edge_sharpness         - エッジシャープネス (-0.867)
  6. chroma_spatial_entropy - 彩度空間エントロピー (+0.846)
  7. lbp_uniformity         - LBP一様性 (-0.802)
  8. luminance_skewness     - 輝度歪度 (+0.776)
  9. frequency_band_ratio_var - 周波数帯域比分散 (-0.728)
  10. value_bimodality      - 輝度二峰性 (-0.705)
  11. multiscale_variance_ratio - マルチスケール分散比 (-0.684)
  12. gradient_magnitude_entropy - 勾配強度エントロピー (+0.680)
  13. noise_spectrum_slope  - ノイズスペクトル傾斜 (-0.631)
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.signal import find_peaks
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

EMBEDDINGS_DIR = Path("/home/techne/aicheckers/embeddings")
DATA_ROOT = Path("/home/techne/aicheckers/data")
ANIMEDL_ROOT = DATA_ROOT / "animedl2m_dataset_release"

CATEGORY_PATHS = {
    "aibooru_new_ai": DATA_ROOT / "aibooru_new",
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

FEATURE_NAMES = [
    "histogram_flatness",
    "histogram_modality",
    "color_palette_entropy",
    "luminance_layer_count",
    "edge_sharpness",
    "chroma_spatial_entropy",
    "lbp_uniformity",
    "luminance_skewness",
    "frequency_band_ratio_var",
    "value_bimodality",
    "multiscale_variance_ratio",
    "gradient_magnitude_entropy",
    "noise_spectrum_slope",
]

CHECKPOINT_INTERVAL = 1000


def histogram_flatness(gray):
    """ヒストグラムの平坦度（エントロピー / log(bins)）"""
    hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
    hist = hist.astype(np.float64)
    hist = hist / (hist.sum() + 1e-10)
    entropy = -np.sum(hist * np.log2(hist + 1e-10))
    return float(entropy / np.log2(256))


def histogram_modality(gray):
    """ヒストグラムのモード数（ピーク数）"""
    hist, _ = np.histogram(gray.ravel(), bins=64, range=(0, 256))
    hist_smooth = ndimage.gaussian_filter1d(hist.astype(np.float64), sigma=2)
    peaks, _ = find_peaks(hist_smooth, height=hist_smooth.max() * 0.05, distance=5)
    return float(len(peaks))


def color_palette_entropy(img_rgb):
    """色パレットのエントロピー（量子化した色の分布）"""
    # 32レベルに量子化
    quantized = (img_rgb // 8).astype(np.uint8)
    # 色をフラット化してユニークカウント
    colors = quantized.reshape(-1, 3)
    color_ids = colors[:, 0].astype(np.int32) * 1024 + colors[:, 1] * 32 + colors[:, 2]
    unique, counts = np.unique(color_ids, return_counts=True)
    probs = counts.astype(np.float64) / counts.sum()
    entropy = -np.sum(probs * np.log2(probs + 1e-10))
    return float(entropy)


def luminance_layer_count(gray):
    """輝度層の数（連続した輝度領域の数）"""
    # 16レベルに量子化
    quantized = (gray // 16).astype(np.uint8)
    unique_levels = len(np.unique(quantized))
    return float(unique_levels)


def edge_sharpness(gray):
    """エッジのシャープネス（勾配の鋭さ）"""
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(sobelx**2 + sobely**2)
    # 上位5%の勾配平均 / 全体平均
    threshold = np.percentile(gradient, 95)
    high_grad = gradient[gradient > threshold]
    if len(high_grad) == 0 or gradient.mean() < 1e-6:
        return 0.0
    return float(high_grad.mean() / (gradient.mean() + 1e-6))


def chroma_spatial_entropy(img_rgb):
    """彩度の空間エントロピー"""
    # LAB色空間に変換
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    a, b = lab[:, :, 1], lab[:, :, 2]
    chroma = np.sqrt(a.astype(np.float64)**2 + b.astype(np.float64)**2)
    # 空間的なエントロピー（16x16タイル）
    h, w = chroma.shape
    tiles = 16
    tile_h, tile_w = h // tiles, w // tiles
    if tile_h == 0 or tile_w == 0:
        return 0.0
    means = []
    for i in range(tiles):
        for j in range(tiles):
            tile = chroma[i*tile_h:(i+1)*tile_h, j*tile_w:(j+1)*tile_w]
            means.append(tile.mean())
    means = np.array(means)
    hist, _ = np.histogram(means, bins=32)
    hist = hist.astype(np.float64) / (hist.sum() + 1e-10)
    entropy = -np.sum(hist * np.log2(hist + 1e-10))
    return float(entropy)


def lbp_uniformity(gray, radius=1, n_points=8):
    """LBP（Local Binary Pattern）の一様性"""
    h, w = gray.shape
    lbp = np.zeros((h, w), dtype=np.uint8)

    for i in range(n_points):
        angle = 2 * np.pi * i / n_points
        dy = -radius * np.cos(angle)
        dx = radius * np.sin(angle)

        y = np.arange(h).reshape(-1, 1) + dy
        x = np.arange(w).reshape(1, -1) + dx

        y = np.clip(y, 0, h - 1).astype(int)
        x = np.clip(x, 0, w - 1).astype(int)

        neighbor = gray[y, x]
        lbp += ((neighbor >= gray).astype(np.uint8) << i)

    # 一様パターン（0-1遷移が2以下）のカウント
    def count_transitions(val):
        bits = [(val >> i) & 1 for i in range(n_points)]
        bits.append(bits[0])
        return sum(1 for i in range(n_points) if bits[i] != bits[i+1])

    uniform_count = 0
    for val in lbp.ravel():
        if count_transitions(val) <= 2:
            uniform_count += 1

    return float(uniform_count / lbp.size)


def luminance_skewness(gray):
    """輝度の歪度"""
    vals = gray.astype(np.float64).ravel()
    mean = vals.mean()
    std = vals.std()
    if std < 1e-6:
        return 0.0
    skewness = ((vals - mean) ** 3).mean() / (std ** 3 + 1e-10)
    return float(skewness)


def frequency_band_ratio_var(gray):
    """周波数帯域比の分散"""
    f = np.fft.fft2(gray.astype(np.float64))
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)

    h, w = gray.shape
    cy, cx = h // 2, w // 2
    max_r = min(cy, cx)

    # 4つの周波数帯域
    bands = [0, max_r // 4, max_r // 2, 3 * max_r // 4, max_r]
    band_powers = []

    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    for i in range(len(bands) - 1):
        mask = (r >= bands[i]) & (r < bands[i + 1])
        power = mag[mask].mean() if mask.sum() > 0 else 0.0
        band_powers.append(power)

    total = sum(band_powers) + 1e-10
    ratios = [p / total for p in band_powers]
    return float(np.var(ratios))


def value_bimodality(gray):
    """輝度の二峰性（Bimodality Coefficient）"""
    vals = gray.astype(np.float64).ravel()
    n = len(vals)
    if n < 3:
        return 0.0

    mean = vals.mean()
    std = vals.std()
    if std < 1e-6:
        return 0.0

    skewness = ((vals - mean) ** 3).mean() / (std ** 3 + 1e-10)
    kurtosis = ((vals - mean) ** 4).mean() / (std ** 4 + 1e-10) - 3

    # Bimodality coefficient
    bc = (skewness ** 2 + 1) / (kurtosis + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)) + 1e-10)
    return float(bc)


def multiscale_variance_ratio(gray):
    """マルチスケール分散比"""
    variances = []
    for scale in [1, 2, 4, 8]:
        if gray.shape[0] // scale < 8 or gray.shape[1] // scale < 8:
            break
        resized = cv2.resize(gray, (gray.shape[1] // scale, gray.shape[0] // scale))
        variances.append(resized.var())

    if len(variances) < 2:
        return 0.0

    # 細かいスケール / 粗いスケールの分散比
    ratios = [variances[i] / (variances[i + 1] + 1e-10) for i in range(len(variances) - 1)]
    return float(np.mean(ratios))


def gradient_magnitude_entropy(gray):
    """勾配強度のエントロピー"""
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(sobelx**2 + sobely**2)

    # 正規化して量子化
    if gradient.max() < 1e-6:
        return 0.0
    gradient_norm = (gradient / (gradient.max() + 1e-10) * 255).astype(np.uint8)

    hist, _ = np.histogram(gradient_norm.ravel(), bins=64, range=(0, 256))
    hist = hist.astype(np.float64) / (hist.sum() + 1e-10)
    entropy = -np.sum(hist * np.log2(hist + 1e-10))
    return float(entropy)


def noise_spectrum_slope(gray):
    """ノイズスペクトル傾斜（高周波成分の傾き）"""
    # ガウシアンでローパス
    low = cv2.GaussianBlur(gray.astype(np.float64), (5, 5), 1.5)
    noise = gray.astype(np.float64) - low

    # ノイズのFFT
    f = np.fft.fft2(noise)
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift) + 1e-10

    h, w = gray.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(np.int32)
    max_r = r.max()

    # ラジアル平均
    radial_mean = np.bincount(r.ravel(), mag.ravel(), minlength=max_r + 1)
    radial_cnt = np.bincount(r.ravel(), minlength=max_r + 1)
    radial_mean = radial_mean / (radial_cnt + 1e-6)

    # 高周波領域（半径の50%以上）でフィット
    start_r = max_r // 2
    xs = np.log(np.arange(start_r, max_r) + 1)
    ys = np.log(radial_mean[start_r:max_r] + 1e-10)

    if len(xs) < 2:
        return 0.0

    slope, _ = np.polyfit(xs, ys, 1)
    return float(slope)


def extract_features(img_rgb):
    """全特徴量を抽出"""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    feats = [
        histogram_flatness(gray),
        histogram_modality(gray),
        color_palette_entropy(img_rgb),
        luminance_layer_count(gray),
        edge_sharpness(gray),
        chroma_spatial_entropy(img_rgb),
        lbp_uniformity(gray),
        luminance_skewness(gray),
        frequency_band_ratio_var(gray),
        value_bimodality(gray),
        multiscale_variance_ratio(gray),
        gradient_magnitude_entropy(gray),
        noise_spectrum_slope(gray),
    ]
    return np.array(feats, dtype=np.float32)


def load_image(path, target_size=512):
    """画像を読み込み、正方形にリサイズ"""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = target_size / max(h, w)
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))
    if (nw, nh) != img.size:
        img = img.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGB", (target_size, target_size), (128, 128, 128))
    x0 = (target_size - nw) // 2
    y0 = (target_size - nh) // 2
    canvas.paste(img, (x0, y0))
    return np.array(canvas)


def _save_checkpoint(path, stats_list, files_list):
    stats_arr = (
        np.concatenate(stats_list, axis=0).astype(np.float32)
        if stats_list
        else np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)
    )
    np.savez_compressed(path, stats=stats_arr, files=np.array(files_list, dtype=object))


def _load_checkpoint(path):
    if not path.exists():
        return None, []
    data = np.load(path, allow_pickle=True)
    return data["stats"], data["files"].tolist()


def extract_category(name, img_dir, limit=0):
    img_dir = Path(img_dir)
    if not img_dir.exists():
        print(f"[SKIP] {name}: missing dir {img_dir}")
        return

    out_stats = EMBEDDINGS_DIR / f"{name}_cpu_stats_v3.npy"
    out_files = EMBEDDINGS_DIR / f"{name}_cpu_stats_v3_files.txt"
    out_meta = EMBEDDINGS_DIR / f"{name}_cpu_stats_v3_meta.json"
    checkpoint_file = EMBEDDINGS_DIR / f"{name}_cpu_stats_v3_ckpt.npz"

    # ファイルリストを取得
    files_list_path = EMBEDDINGS_DIR / f"{name}_files.txt"
    paths = []
    if files_list_path.exists():
        raw_lines = [line.strip() for line in files_list_path.read_text().splitlines() if line.strip()]
        for line in raw_lines:
            p = Path(line)
            if not p.is_absolute():
                p = img_dir / line
            paths.append(p)
        print(f"[INFO] {name}: using existing files list ({len(paths)} files)")
    else:
        exts = {".jpg", ".jpeg", ".png", ".webp"}
        paths = sorted([p for p in img_dir.rglob("*") if p.suffix.lower() in exts])
        print(f"[INFO] {name}: scanned dir ({len(paths)} files)")

    if limit > 0:
        paths = paths[:limit]

    stats_list = []
    files_list = []
    start_index = 0

    # チェックポイント復元
    if checkpoint_file.exists():
        ckpt_stats, ckpt_files = _load_checkpoint(checkpoint_file)
        if ckpt_files:
            if len(ckpt_files) <= len(paths) and all(str(paths[i]) == ckpt_files[i] for i in range(len(ckpt_files))):
                stats_list.append(ckpt_stats)
                files_list.extend(ckpt_files)
                start_index = len(ckpt_files)
                print(f"[RESUME] {name}: {start_index} samples from checkpoint")

    zero_feats = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    error_count = 0

    for p in tqdm(paths[start_index:], desc=name):
        try:
            img = load_image(p)
            feats = extract_features(img)
            stats_list.append(feats[None, :])
            files_list.append(str(p))
        except Exception as e:
            stats_list.append(zero_feats[None, :])
            files_list.append(str(p))
            error_count += 1

        if len(files_list) % CHECKPOINT_INTERVAL == 0:
            _save_checkpoint(checkpoint_file, stats_list, files_list)

    if not stats_list:
        print(f"[WARN] {name}: no samples")
        return

    stats_arr = np.concatenate(stats_list, axis=0).astype(np.float32)
    np.save(out_stats, stats_arr)
    out_files.write_text("\n".join(files_list) + "\n")
    out_meta.write_text(json.dumps({
        "features": FEATURE_NAMES,
        "samples": len(stats_arr),
        "dtype": "float32",
    }, indent=2))

    if error_count:
        print(f"[WARN] {name}: {error_count} failures (zero-filled)")
    if checkpoint_file.exists():
        checkpoint_file.unlink()
    print(f"[DONE] {name}: {stats_arr.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    if args.category:
        if args.category not in CATEGORY_PATHS:
            raise SystemExit(f"Unknown category: {args.category}")
        extract_category(args.category, CATEGORY_PATHS[args.category], limit=args.limit)
        return

    if args.all:
        for name, path in CATEGORY_PATHS.items():
            extract_category(name, path, limit=args.limit)
        return

    print("Use --category or --all")


if __name__ == "__main__":
    main()
