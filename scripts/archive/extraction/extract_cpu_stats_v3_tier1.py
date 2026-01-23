#!/usr/bin/env python3
"""
Extract CPU features v3 Tier1 (12d).

Output:
  embeddings/{category}_cpu_stats_v3_tier1.npy
  embeddings/{category}_cpu_stats_v3_tier1_files.txt
  embeddings/{category}_cpu_stats_v3_tier1_meta.json

Features (12d):
  1. quantization_step_count
  2. piecewise_constant_ratio
  3. noise_floor_variance
  4. highlight_clipping_ratio
  5. band_entropy
  6. band_energy_gini
  7. radial_spectrum_slope_patch_gap
  8. color_banding_score
  9. color_palette_entropy
  10. compression_artifact_pattern
  11. edge_continuity_ratio
  12. edge_endpoint_density
"""
import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

EMBEDDINGS_DIR = Path("/home/techne/aicheckers/embeddings")
DATA_ROOT = Path("/home/techne/aicheckers/data")
ANIMEDL_ROOT = DATA_ROOT / "animedl2m_dataset_release"

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

FEATURE_NAMES = [
    "quantization_step_count",
    "piecewise_constant_ratio",
    "noise_floor_variance",
    "highlight_clipping_ratio",
    "band_entropy",
    "band_energy_gini",
    "radial_spectrum_slope_patch_gap",
    "color_banding_score",
    "color_palette_entropy",
    "compression_artifact_pattern",
    "edge_continuity_ratio",
    "edge_endpoint_density",
]

CHECKPOINT_INTERVAL = 1000


def _safe_entropy(probs):
    probs = probs / (probs.sum() + 1e-10)
    return float(-np.sum(probs * np.log2(probs + 1e-10)))


def quantization_step_count(gray):
    vals = np.unique(gray)
    if len(vals) < 2:
        return 0.0
    diffs = np.diff(vals).astype(np.float64)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return 0.0
    step = np.median(diffs)
    if step <= 0:
        return 0.0
    return float(min(255.0 / step, 255.0))


def piecewise_constant_ratio(gray, grad_thresh=2.0):
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(sobelx**2 + sobely**2)
    return float((gradient <= grad_thresh).mean())


def noise_floor_variance(gray):
    low = cv2.GaussianBlur(gray.astype(np.float64), (5, 5), 1.5)
    residual = gray.astype(np.float64) - low
    abs_res = np.abs(residual)
    thr = np.percentile(abs_res, 25)
    mask = abs_res <= thr
    if mask.sum() == 0:
        return 0.0
    return float(residual[mask].var())


def highlight_clipping_ratio(gray, clip_val=250):
    return float((gray >= clip_val).mean())


def _band_powers(gray, bands=4):
    f = np.fft.fft2(gray.astype(np.float64))
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)

    h, w = gray.shape
    cy, cx = h // 2, w // 2
    max_r = min(cy, cx)
    band_edges = np.linspace(0, max_r, bands + 1, dtype=np.int32)

    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    powers = []
    for i in range(bands):
        mask = (r >= band_edges[i]) & (r < band_edges[i + 1])
        power = mag[mask].mean() if mask.sum() > 0 else 0.0
        powers.append(power)
    return np.array(powers, dtype=np.float64)


def band_entropy(gray):
    powers = _band_powers(gray, bands=4)
    return _safe_entropy(powers)


def band_energy_gini(gray):
    powers = _band_powers(gray, bands=4)
    if np.all(powers == 0):
        return 0.0
    sorted_p = np.sort(powers)
    n = len(sorted_p)
    cum = np.cumsum(sorted_p)
    gini = (n + 1 - 2 * np.sum(cum) / (cum[-1] + 1e-10)) / n
    return float(gini)


def _radial_spectrum_slope(gray):
    f = np.fft.fft2(gray.astype(np.float64))
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift) + 1e-10

    h, w = gray.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(np.int32)
    max_r = r.max()

    radial_mean = np.bincount(r.ravel(), mag.ravel(), minlength=max_r + 1)
    radial_cnt = np.bincount(r.ravel(), minlength=max_r + 1)
    radial_mean = radial_mean / (radial_cnt + 1e-6)

    start_r = max_r // 4
    xs = np.log(np.arange(start_r, max_r) + 1)
    ys = np.log(radial_mean[start_r:max_r] + 1e-10)
    if len(xs) < 2:
        return 0.0
    slope, _ = np.polyfit(xs, ys, 1)
    return float(slope)


def radial_spectrum_slope_patch_gap(gray, tiles=4):
    h, w = gray.shape
    if h < tiles or w < tiles:
        return 0.0
    global_slope = _radial_spectrum_slope(gray)
    tile_h = h // tiles
    tile_w = w // tiles
    slopes = []
    for i in range(tiles):
        for j in range(tiles):
            tile = gray[i * tile_h:(i + 1) * tile_h, j * tile_w:(j + 1) * tile_w]
            slopes.append(_radial_spectrum_slope(tile))
    if not slopes:
        return 0.0
    return float(global_slope - np.mean(slopes))


def color_banding_score(img_rgb):
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    a = lab[:, :, 1].astype(np.int16)
    b = lab[:, :, 2].astype(np.int16)
    chroma = np.sqrt(a.astype(np.float64) ** 2 + b.astype(np.float64) ** 2)
    quant = np.clip((chroma / (chroma.max() + 1e-10) * 31).astype(np.uint8), 0, 31)
    unique = np.unique(quant).size
    return float(1.0 - unique / 32.0)


def color_palette_entropy(img_rgb):
    quantized = (img_rgb // 8).astype(np.uint8)
    colors = quantized.reshape(-1, 3)
    color_ids = colors[:, 0].astype(np.int32) * 1024 + colors[:, 1] * 32 + colors[:, 2]
    _, counts = np.unique(color_ids, return_counts=True)
    probs = counts.astype(np.float64) / counts.sum()
    return _safe_entropy(probs)


def compression_artifact_pattern(gray):
    gray = gray.astype(np.float64)
    h, w = gray.shape
    if h < 8 or w < 8:
        return 0.0
    v_idx = np.arange(8, w, 8)
    h_idx = np.arange(8, h, 8)
    vert = np.abs(gray[:, v_idx] - gray[:, v_idx - 1]).mean() if v_idx.size else 0.0
    horiz = np.abs(gray[h_idx, :] - gray[h_idx - 1, :]).mean() if h_idx.size else 0.0
    overall = np.mean(np.abs(np.diff(gray, axis=1))) + np.mean(np.abs(np.diff(gray, axis=0)))
    return float((vert + horiz) / (overall + 1e-6))


def edge_continuity_ratio(gray, min_size=10):
    edges = cv2.Canny(gray, 100, 200)
    if edges.sum() == 0:
        return 0.0
    num_labels, labels = cv2.connectedComponents((edges > 0).astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return 0.0
    counts = np.bincount(labels.ravel())
    large = counts >= min_size
    large[0] = False
    return float(counts[large].sum() / (edges.sum() + 1e-10))


def edge_endpoint_density(gray):
    edges = (cv2.Canny(gray, 100, 200) > 0).astype(np.uint8)
    if edges.sum() == 0:
        return 0.0
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    neighbors = cv2.filter2D(edges, -1, kernel)
    endpoints = (edges == 1) & (neighbors == 1)
    return float(endpoints.sum() / (edges.sum() + 1e-10))


def extract_features(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    feats = [
        quantization_step_count(gray),
        piecewise_constant_ratio(gray),
        noise_floor_variance(gray),
        highlight_clipping_ratio(gray),
        band_entropy(gray),
        band_energy_gini(gray),
        radial_spectrum_slope_patch_gap(gray),
        color_banding_score(img_rgb),
        color_palette_entropy(img_rgb),
        compression_artifact_pattern(gray),
        edge_continuity_ratio(gray),
        edge_endpoint_density(gray),
    ]
    return np.array(feats, dtype=np.float32)


def load_image(path, target_size=512):
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


def _process_one(path_str):
    try:
        img = load_image(path_str)
        feats = extract_features(img)
        return path_str, feats, False
    except Exception:
        return path_str, None, True


def extract_category(name, img_dir, limit=0, workers=1):
    img_dir = Path(img_dir)
    if not img_dir.exists():
        print(f"[SKIP] {name}: missing dir {img_dir}")
        return

    out_stats = EMBEDDINGS_DIR / f"{name}_cpu_stats_v3_tier1.npy"
    out_files = EMBEDDINGS_DIR / f"{name}_cpu_stats_v3_tier1_files.txt"
    out_meta = EMBEDDINGS_DIR / f"{name}_cpu_stats_v3_tier1_meta.json"
    checkpoint_file = EMBEDDINGS_DIR / f"{name}_cpu_stats_v3_tier1_ckpt.npz"

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
    remaining = [str(p) for p in paths[start_index:]]

    if workers > 1:
        with mp.Pool(processes=workers) as pool:
            for path_str, feats, failed in tqdm(
                pool.imap(_process_one, remaining, chunksize=8),
                total=len(remaining),
                desc=name,
            ):
                files_list.append(path_str)
                if failed or feats is None:
                    stats_list.append(zero_feats[None, :])
                    error_count += 1
                else:
                    stats_list.append(feats[None, :])

                if len(files_list) % CHECKPOINT_INTERVAL == 0:
                    _save_checkpoint(checkpoint_file, stats_list, files_list)
    else:
        for p in tqdm(remaining, desc=name):
            try:
                img = load_image(p)
                feats = extract_features(img)
                stats_list.append(feats[None, :])
                files_list.append(str(p))
            except Exception:
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
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    if args.category:
        if args.category not in CATEGORY_PATHS:
            raise SystemExit(f"Unknown category: {args.category}")
        extract_category(args.category, CATEGORY_PATHS[args.category], limit=args.limit, workers=args.workers)
        return

    if args.all:
        for name, path in CATEGORY_PATHS.items():
            extract_category(name, path, limit=args.limit, workers=args.workers)
        return

    print("Use --category or --all")


if __name__ == "__main__":
    main()
