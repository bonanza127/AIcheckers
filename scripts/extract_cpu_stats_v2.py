#!/usr/bin/env python3
"""
Extract CPU features (512-resized) and save per category.

Output:
  embeddings/{category}_cpu_stats_v2.npy
  embeddings/{category}_cpu_stats_v2_files.txt
  embeddings/{category}_cpu_stats_v2_meta.json
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from lib.extra_stats import FLAT_PERCENTILE

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
    "niji7_twitter_ai": DATA_ROOT / "niji7_twitter",
    "danbooru_real": ANIMEDL_ROOT / "real_images/images",
}

PATCH = 32
TILE = 64
MIN_PATCH_COVERAGE = 0.5
CHECKPOINT_INTERVAL = 1000

FEATURE_NAMES = [
    "banding_score",
    "radial_spectrum_slope",
    "stroke_width_proxy",
    "text_area_ratio",
    "fractal_dim_edge_512",
    "patchwise_edge_density",
    "st_aniso_mean",
    "st_aniso_var",
    "st_aniso_spatial_gradient",
    "flat_boundary_peri_area",
    "stroke_p90",
    "flat_hole_ratio",
    "highfreq_spatial_autocorr",
    "patch_vs_global_rank_entropy_gap",
    "flat_ratio",
    "flat_ratio_variance_across_tiles",
    "patch_vs_global_st_aniso_gap",
    "patch_vs_global_spectrum_slope_gap",
]


def _save_checkpoint(path, stats_list, files_list):
    stats_arr = (
        np.concatenate(stats_list, axis=0).astype(np.float16)
        if stats_list
        else np.zeros((0, len(FEATURE_NAMES)), dtype=np.float16)
    )
    np.savez_compressed(
        path,
        stats=stats_arr,
        files=np.array(files_list, dtype=object),
    )


def _load_checkpoint(path):
    if not path.exists():
        return None, []
    data = np.load(path, allow_pickle=True)
    stats = data["stats"]
    files = data["files"].tolist()
    return stats, files


def banding_score(img_gray, mask):
    q = (img_gray // 8).astype(np.uint8)
    m = mask[:, 1:] & mask[:, :-1]
    if m.sum() == 0:
        return 0.0
    diffs = np.abs(q[:, 1:].astype(np.int16) - q[:, :-1].astype(np.int16))
    return float((diffs[m] == 0).mean())


def _fill_mask_mean(img_gray, mask):
    vals = img_gray[mask]
    if vals.size == 0:
        return img_gray
    filled = img_gray.copy()
    filled[~mask] = float(vals.mean())
    return filled


def radial_spectrum_slope(img_gray):
    f = np.fft.fft2(img_gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)
    h, w = img_gray.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(np.int32)
    max_r = r.max()
    radial_mean = np.bincount(r.ravel(), mag.ravel(), minlength=max_r + 1)
    radial_cnt = np.bincount(r.ravel(), minlength=max_r + 1)
    radial_mean = radial_mean / (radial_cnt + 1e-6)
    xs = np.log(np.arange(1, max_r + 1))
    ys = np.log(radial_mean[1:max_r + 1] + 1e-6)
    slope, _ = np.polyfit(xs, ys, 1)
    return float(slope)


def stroke_width_proxy(edges):
    # Handle edge case: if no edges detected, return 0
    if np.count_nonzero(edges) == 0:
        return 0.0
    inv = (edges == 0).astype(np.uint8)
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    return float(np.percentile(dist, 90))


def text_area_ratio(img_gray, mask):
    th = cv2.adaptiveThreshold(
        img_gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        15,
        5,
    )
    th[~mask] = 0
    _, _, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    small = areas[(areas >= 10) & (areas <= 200)]
    return float(small.sum() / (mask.sum() + 1e-6))


def fractal_dim_edge_512(edges):
    edge = (edges > 0).astype(np.uint8)
    sizes = [2, 4, 8, 16, 32]
    counts = []
    for s in sizes:
        h, w = edge.shape
        hh = h // s
        ww = w // s
        if hh == 0 or ww == 0:
            continue
        blocks = edge[:hh * s, :ww * s].reshape(hh, s, ww, s)
        count = np.count_nonzero(blocks.max(axis=(1, 3)))
        counts.append(count)
    if len(counts) < 2:
        return 0.0
    xs = np.log(np.array(sizes[:len(counts)]))
    ys = np.log(np.array(counts) + 1e-6)
    slope, _ = np.polyfit(xs, ys, 1)
    return float(-slope)


def patchwise_edge_density(edges, mask):
    h, w = edges.shape
    dens = []
    for i in range(0, h - PATCH, PATCH):
        for j in range(0, w - PATCH, PATCH):
            m = mask[i:i + PATCH, j:j + PATCH]
            if m.mean() < MIN_PATCH_COVERAGE:
                continue
            patch = edges[i:i + PATCH, j:j + PATCH]
            dens.append(patch.mean() / 255.0)
    return float(np.var(dens)) if len(dens) > 1 else 0.0


def st_anisotropy(gray):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    jxx = cv2.GaussianBlur(gx * gx, (3, 3), 1.0)
    jyy = cv2.GaussianBlur(gy * gy, (3, 3), 1.0)
    jxy = cv2.GaussianBlur(gx * gy, (3, 3), 1.0)
    trace = jxx + jyy
    det = jxx * jyy - jxy * jxy
    tmp = np.sqrt(np.maximum(trace * trace / 4 - det, 0))
    l1 = trace / 2 + tmp
    l2 = trace / 2 - tmp
    aniso = (l1 - l2) / (l1 + l2 + 1e-6)
    return aniso, float(aniso.mean()), float(aniso.var())


def st_aniso_spatial_gradient(aniso):
    gx = cv2.Sobel(aniso, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(aniso, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    return float(mag.mean())


def flat_boundary_peri_area(flat_mask):
    contours, _ = cv2.findContours(
        flat_mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    per = 0.0
    area = 0.0
    for c in contours:
        a = cv2.contourArea(c)
        if a < 10:
            continue
        area += a
        per += cv2.arcLength(c, True)
    return float(per / (area + 1e-6))


def flat_hole_ratio(flat_mask):
    inv = (~flat_mask).astype(np.uint8)
    num, _, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    if num <= 1:
        return 0.0
    h, w = flat_mask.shape
    holes_area = 0
    for i in range(1, num):
        x, y, ww, hh, area = stats[i]
        if x == 0 or y == 0 or x + ww >= w or y + hh >= h:
            continue
        holes_area += area
    return float(holes_area / (flat_mask.sum() + 1e-6))


def rank_entropy(vals):
    if len(vals) < 20:
        return 0.0
    hist, _ = np.histogram(vals, bins=256, range=(0, 256))
    if hist.sum() <= 0:
        return 0.0
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist + 1e-10)))


def patch_vs_global_rank_entropy_gap(gray, flat_mask, mask):
    flat_vals = gray[flat_mask]
    global_ent = rank_entropy(flat_vals)
    h, w = gray.shape
    ents = []
    for i in range(0, h - PATCH, PATCH):
        for j in range(0, w - PATCH, PATCH):
            m = mask[i:i + PATCH, j:j + PATCH]
            if m.mean() < MIN_PATCH_COVERAGE:
                continue
            patch = gray[i:i + PATCH, j:j + PATCH]
            patch_flat = flat_mask[i:i + PATCH, j:j + PATCH]
            ents.append(rank_entropy(patch[patch_flat]))
    patch_mean = float(np.mean(ents)) if ents else 0.0
    return global_ent - patch_mean


def highfreq_spatial_autocorr(gray):
    low = cv2.GaussianBlur(gray.astype(np.float32), (3, 3), 1.0)
    res = gray.astype(np.float32) - low
    a = res[:-1, :-1].flatten()
    b = res[1:, 1:].flatten()
    if a.size < 2:
        return 0.0
    corr = np.corrcoef(a, b)[0, 1]
    return float(0.0 if np.isnan(corr) else corr)


def patch_vs_global_spectrum_slope_gap(gray, mask):
    global_slope = radial_spectrum_slope(gray)
    h, w = gray.shape
    slopes = []
    for i in range(0, h - TILE + 1, TILE):
        for j in range(0, w - TILE + 1, TILE):
            m = mask[i:i + TILE, j:j + TILE]
            if m.mean() < MIN_PATCH_COVERAGE:
                continue
            tile = gray[i:i + TILE, j:j + TILE]
            slopes.append(radial_spectrum_slope(tile))
    tile_mean = float(np.mean(slopes)) if slopes else 0.0
    return global_slope - tile_mean


def patch_vs_global_st_aniso_gap(aniso, mask):
    global_mean = float(aniso.mean())
    h, w = aniso.shape
    vals = []
    for i in range(0, h - PATCH, PATCH):
        for j in range(0, w - PATCH, PATCH):
            m = mask[i:i + PATCH, j:j + PATCH]
            if m.mean() < MIN_PATCH_COVERAGE:
                continue
            patch = aniso[i:i + PATCH, j:j + PATCH]
            vals.append(float(patch.mean()))
    patch_mean = float(np.mean(vals)) if vals else 0.0
    return global_mean - patch_mean


def flat_ratio_variance_across_tiles(flat_mask, mask, tiles=4):
    h, w = flat_mask.shape
    th = h // tiles
    tw = w // tiles
    ratios = []
    for i in range(tiles):
        for j in range(tiles):
            m = flat_mask[i * th:(i + 1) * th, j * tw:(j + 1) * tw]
            mm = mask[i * th:(i + 1) * th, j * tw:(j + 1) * tw]
            if mm.mean() < MIN_PATCH_COVERAGE:
                continue
            ratios.append(m.mean())
    return float(np.var(ratios)) if len(ratios) > 1 else 0.0


def compute_flat_mask(gray, mask):
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    vals = grad_mag[mask]
    if vals.size == 0:
        return np.zeros_like(gray, dtype=bool)
    threshold = np.percentile(vals, FLAT_PERCENTILE)
    return (grad_mag <= threshold) & mask


def extract_features(img_rgb, mask):
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    filled_gray = _fill_mask_mean(img_gray, mask)
    flat_mask = compute_flat_mask(filled_gray, mask)
    edges = cv2.Canny(filled_gray, 50, 150)
    edges[~mask] = 0

    aniso_map, aniso_mean, aniso_var = st_anisotropy(filled_gray)
    stroke_p90 = stroke_width_proxy(edges)

    feats = [
        banding_score(filled_gray, mask),
        radial_spectrum_slope(filled_gray),
        stroke_p90,
        text_area_ratio(filled_gray, mask),
        fractal_dim_edge_512(edges),
        patchwise_edge_density(edges, mask),
        aniso_mean,
        aniso_var,
        st_aniso_spatial_gradient(aniso_map),
        flat_boundary_peri_area(flat_mask),
        stroke_p90,  # stroke_p90
        flat_hole_ratio(flat_mask),
        highfreq_spatial_autocorr(filled_gray),
        patch_vs_global_rank_entropy_gap(filled_gray, flat_mask, mask),
        float(flat_mask.sum() / (mask.sum() + 1e-6)),
        flat_ratio_variance_across_tiles(flat_mask, mask),
        patch_vs_global_st_aniso_gap(aniso_map, mask),
        patch_vs_global_spectrum_slope_gap(filled_gray, mask),
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
    mask = np.zeros((target_size, target_size), dtype=bool)
    mask[y0:y0 + nh, x0:x0 + nw] = True
    return np.array(canvas), mask


def extract_category(name, img_dir, limit=0, workers=1):
    img_dir = Path(img_dir)
    if not img_dir.exists():
        print(f"[SKIP] {name}: missing dir {img_dir}")
        return

    out_stats = EMBEDDINGS_DIR / f"{name}_cpu_stats_v2.npy"
    out_files = EMBEDDINGS_DIR / f"{name}_cpu_stats_v2_files.txt"
    out_meta = EMBEDDINGS_DIR / f"{name}_cpu_stats_v2_meta.json"
    checkpoint_file = EMBEDDINGS_DIR / f"{name}_cpu_stats_v2_ckpt.npz"

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
            else:
                print(f"[WARN] {name}: checkpoint files do not match current list, ignoring checkpoint")
    zero_feats = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    error_count = 0
    def process_one(p):
        try:
            img, mask = load_image(p)
            feats = extract_features(img, mask)
            return feats, True
        except Exception:
            return zero_feats, False

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for p, (feats, ok) in tqdm(
                zip(paths[start_index:], ex.map(process_one, paths[start_index:])),
                total=len(paths) - start_index,
                desc=name,
            ):
                stats_list.append(feats[None, :])
                files_list.append(str(p))
                if not ok:
                    error_count += 1
                if len(files_list) % CHECKPOINT_INTERVAL == 0:
                    _save_checkpoint(checkpoint_file, stats_list, files_list)
    else:
        for p in tqdm(paths[start_index:], desc=name):
            feats, ok = process_one(p)
            stats_list.append(feats[None, :])
            files_list.append(str(p))
            if not ok:
                error_count += 1
            if len(files_list) % CHECKPOINT_INTERVAL == 0:
                _save_checkpoint(checkpoint_file, stats_list, files_list)

    if not stats_list:
        print(f"[WARN] {name}: no samples")
        return

    stats_arr = np.concatenate(stats_list, axis=0).astype(np.float16)
    np.save(out_stats, stats_arr)
    out_files.write_text("\n".join(files_list) + "\n")
    out_meta.write_text(json.dumps({
        "features": FEATURE_NAMES,
        "samples": len(stats_arr),
        "dtype": "float16",
    }, indent=2))

    if error_count:
        print(f"[WARN] {name}: {error_count} failures (zero-filled to keep alignment)")
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
