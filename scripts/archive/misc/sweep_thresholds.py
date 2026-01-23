#!/usr/bin/env python3
"""
Sweep flat/canny/connectivity/DCT thresholds and report effect sizes.

Example:
  python scripts/sweep_thresholds.py \
    --ai-dir data/novelai \
    --real-dir data/animedl2m_dataset_release/real_images/images \
    --samples-per-class 500 \
    --flat-percentiles 15,20,25 \
    --canny "30:100,50:150,80:200" \
    --connectivity 4,8 \
    --dct-low 2,3 \
    --out logs/threshold_sweep.csv
"""

import argparse
import csv
import random
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.fftpack import dct


def list_images(root: Path):
    exts = (".jpg", ".jpeg", ".png", ".webp")
    return [p for p in root.rglob("*") if p.suffix.lower() in exts]


def rgb_to_ycbcr(img_rgb):
    img = img_rgb.astype(np.float32)
    y = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    cb = 128 - 0.168736 * img[:, :, 0] - 0.331264 * img[:, :, 1] + 0.5 * img[:, :, 2]
    cr = 128 + 0.5 * img[:, :, 0] - 0.418688 * img[:, :, 1] - 0.081312 * img[:, :, 2]
    return y, cb, cr


def cohen_d(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if len(a) < 2 or len(b) < 2:
        return 0.0
    mean_a = a.mean()
    mean_b = b.mean()
    var_a = a.var(ddof=1)
    var_b = b.var(ddof=1)
    pooled = np.sqrt(((len(a) - 1) * var_a + (len(b) - 1) * var_b) / (len(a) + len(b) - 2))
    if pooled == 0:
        return 0.0
    return (mean_a - mean_b) / pooled


def compute_flat_mask(grad_mag, percentile):
    threshold = np.percentile(grad_mag, percentile)
    return grad_mag <= threshold


def compute_flat_cluster_stats(flat_mask, connectivity, min_frac):
    flat_uint8 = flat_mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(flat_uint8, connectivity=connectivity)
    if num_labels <= 1:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    img_size = flat_mask.shape[0] * flat_mask.shape[1]
    cluster_sizes = stats[1:, cv2.CC_STAT_AREA].astype(np.float32) / img_size
    cluster_sizes = cluster_sizes[cluster_sizes >= min_frac]
    if cluster_sizes.size == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    count = float(cluster_sizes.size)
    count_per_mp = count / (img_size / 1_000_000.0)
    return (
        float(cluster_sizes.max()),
        float(cluster_sizes.mean()),
        float(np.median(cluster_sizes)),
        count,
        float(count_per_mp),
    )


def compute_flat_ratio_variance(grad_mag, percentile, patch_size):
    h, w = grad_mag.shape
    threshold = np.percentile(grad_mag, percentile)
    patch_flat_ratios = []
    for i in range(0, h - patch_size, patch_size):
        for j in range(0, w - patch_size, patch_size):
            patch_grad = grad_mag[i : i + patch_size, j : j + patch_size]
            flat_ratio = (patch_grad <= threshold).sum() / (patch_size * patch_size)
            patch_flat_ratios.append(flat_ratio)
    if len(patch_flat_ratios) > 1:
        return float(np.var(patch_flat_ratios))
    return 0.0


def compute_cbcr_lowfreq(cb, cr, low_size):
    if SKIP_DCT:
        return 0.0
    h, w = cb.shape
    block_size = 8
    low_energy = 0.0
    total_energy = 0.0
    for i in range(0, h - block_size, block_size):
        for j in range(0, w - block_size, block_size):
            block_cb = cb[i : i + block_size, j : j + block_size]
            block_cr = cr[i : i + block_size, j : j + block_size]
            dct_cb = dct(dct(block_cb.T, norm="ortho").T, norm="ortho")
            dct_cr = dct(dct(block_cr.T, norm="ortho").T, norm="ortho")
            low_cb = np.sum(dct_cb[:low_size, :low_size] ** 2)
            low_cr = np.sum(dct_cr[:low_size, :low_size] ** 2)
            total_cb = np.sum(dct_cb**2)
            total_cr = np.sum(dct_cr**2)
            low_energy += low_cb + low_cr
            total_energy += total_cb + total_cr
    return float(low_energy / (total_energy + 1e-8))


def compute_edge_lengths(edges):
    num_labels, labels = cv2.connectedComponents(edges.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return 0.0, 0.0, 0.0
    lengths = []
    for label_id in range(1, num_labels):
        lengths.append((labels == label_id).sum())
    lengths = np.asarray(lengths, dtype=np.float32)
    mean_val = float(lengths.mean())
    var_val = float(lengths.var()) if len(lengths) > 1 else 0.0
    if lengths.size == 0 or mean_val == 0.0:
        gini = 0.0
    else:
        diff_sum = np.abs(lengths[:, None] - lengths[None, :]).mean()
        gini = float(diff_sum / (2.0 * mean_val))
    return mean_val, var_val, gini


def compute_patch_edge_density_stats(edges, patch_size):
    h, w = edges.shape
    patch_densities = []
    for i in range(0, h - patch_size, patch_size):
        for j in range(0, w - patch_size, patch_size):
            patch = edges[i : i + patch_size, j : j + patch_size]
            density = patch.sum() / (patch_size * patch_size * 255)
            patch_densities.append(density)
    if len(patch_densities) > 0:
        mean_val = float(np.mean(patch_densities))
        var_val = float(np.var(patch_densities)) if len(patch_densities) > 1 else 0.0
        return mean_val, var_val
    return 0.0, 0.0


def compute_edge_crossing_rate(img_gray, edges):
    if edges.sum() < 100:
        return 0.0
    grad_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    angles = np.arctan2(grad_y, grad_x)
    edge_mask = edges > 0
    # Quantize angle to 8 bins to make comparison robust.
    bins = ((angles + np.pi) / (2 * np.pi) * 8).astype(np.int32) % 8
    h, w = edges.shape
    crossing = 0
    edge_count = int(edge_mask.sum())
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if not edge_mask[i, j]:
                continue
            local_bins = bins[i - 1 : i + 2, j - 1 : j + 2][edge_mask[i - 1 : i + 2, j - 1 : j + 2]]
            if local_bins.size < 2:
                continue
            if local_bins.max() - local_bins.min() >= 3:
                crossing += 1
    if edge_count == 0:
        return 0.0
    return float(crossing / edge_count)


def _resize_if_needed(img: Image.Image, max_side: int) -> Image.Image:
    if max_side <= 0:
        return img
    w, h = img.size
    longest = max(w, h)
    if longest <= max_side:
        return img
    scale = max_side / float(longest)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return img.resize((new_w, new_h), Image.BICUBIC)


def extract_metrics(
    img_path,
    flat_percentile,
    canny_low,
    canny_high,
    connectivity,
    dct_low,
    patch_size,
    max_side,
    cluster_min_frac,
):
    img = Image.open(img_path).convert("RGB")
    img = _resize_if_needed(img, max_side)
    img_rgb = np.array(img)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    grad_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    flat_mask = compute_flat_mask(grad_mag, flat_percentile)
    flat_ratio = float(flat_mask.mean())
    (
        flat_cluster_max,
        flat_cluster_mean,
        flat_cluster_median,
        flat_cluster_count,
        flat_cluster_count_per_mp,
    ) = compute_flat_cluster_stats(
        flat_mask, connectivity, cluster_min_frac
    )
    flat_ratio_var = compute_flat_ratio_variance(grad_mag, flat_percentile, patch_size)

    _, cb, cr = rgb_to_ycbcr(img_rgb)
    cbcr_lowfreq = compute_cbcr_lowfreq(cb, cr, dct_low)

    edges = cv2.Canny(img_gray, canny_low, canny_high)
    edge_len_mean, edge_len_var, edge_len_gini = compute_edge_lengths(edges)
    img_size = img_gray.shape[0] * img_gray.shape[1]
    edge_len_mean = edge_len_mean / np.sqrt(img_size) * 100
    edge_len_var = edge_len_var / img_size * 10000
    edge_crossing_rate = compute_edge_crossing_rate(img_gray, edges)
    patch_edge_density_mean, patch_edge_density_var = compute_patch_edge_density_stats(edges, patch_size)

    return {
        "flat_ratio": flat_ratio,
        "flat_cluster_max_size": flat_cluster_max,
        "flat_cluster_mean_size": flat_cluster_mean,
        "flat_cluster_median_size": flat_cluster_median,
        "flat_cluster_count": flat_cluster_count,
        "flat_cluster_count_per_mp": flat_cluster_count_per_mp,
        "flat_ratio_variance": flat_ratio_var,
        "cbcr_lowfreq_energy": cbcr_lowfreq,
        "edge_len_mean": edge_len_mean,
        "edge_len_var": edge_len_var,
        "edge_len_gini": edge_len_gini,
        "edge_crossing_rate": edge_crossing_rate,
        "patch_edge_density_mean": patch_edge_density_mean,
        "patch_edge_density_var": patch_edge_density_var,
    }


AI_SAMPLES = []
REAL_SAMPLES = []
PATCH_SIZE = 32
SKIP_DCT = False


def _init_pool(ai_samples, real_samples, patch_size, skip_dct):
    global AI_SAMPLES, REAL_SAMPLES, PATCH_SIZE, SKIP_DCT
    AI_SAMPLES = ai_samples
    REAL_SAMPLES = real_samples
    PATCH_SIZE = patch_size
    SKIP_DCT = skip_dct


def evaluate_combo(combo):
    flat_p, canny_low, canny_high, conn, dct_low, max_side, cluster_min_frac = combo
    ai_metrics = []
    real_metrics = []
    for p in AI_SAMPLES:
        ai_metrics.append(
            extract_metrics(
                p,
                flat_p,
                canny_low,
                canny_high,
                conn,
                dct_low,
                PATCH_SIZE,
                max_side,
                cluster_min_frac,
            )
        )
    for p in REAL_SAMPLES:
        real_metrics.append(
            extract_metrics(
                p,
                flat_p,
                canny_low,
                canny_high,
                conn,
                dct_low,
                PATCH_SIZE,
                max_side,
                cluster_min_frac,
            )
        )
    rows = []
    metrics = ai_metrics[0].keys()
    for name in metrics:
        ai_vals = [m[name] for m in ai_metrics]
        real_vals = [m[name] for m in real_metrics]
        rows.append(
            [
                flat_p,
                canny_low,
                canny_high,
                conn,
                dct_low,
                name,
                np.mean(ai_vals),
                np.mean(real_vals),
                cohen_d(ai_vals, real_vals),
            ]
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Sweep thresholds for extra stats.")
    parser.add_argument("--ai-dir", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--samples-per-class", type=int, default=500)
    parser.add_argument("--flat-percentiles", type=str, default="15,20,25")
    parser.add_argument("--canny", type=str, default="30:100,50:150,80:200")
    parser.add_argument("--connectivity", type=str, default="4,8")
    parser.add_argument("--dct-low", type=str, default="2,3")
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-side", type=int, default=0, help="Resize long edge to this size for speed (0=off)")
    parser.add_argument("--skip-dct", action="store_true", help="Skip DCT lowfreq calculation for speed")
    parser.add_argument("--resume", action="store_true", help="Skip combos already in output CSV")
    parser.add_argument("--cluster-min-frac", type=float, default=0.005, help="Min flat cluster area fraction to keep")
    args = parser.parse_args()

    ai_images = list_images(args.ai_dir)
    real_images = list_images(args.real_dir)
    if not ai_images or not real_images:
        raise SystemExit("AI or real image list is empty.")

    random.seed(args.seed)
    ai_samples = random.sample(ai_images, min(args.samples_per_class, len(ai_images)))
    real_samples = random.sample(real_images, min(args.samples_per_class, len(real_images)))

    flat_percentiles = [int(x) for x in args.flat_percentiles.split(",") if x.strip()]
    canny_pairs = []
    for part in args.canny.split(","):
        low, high = part.split(":")
        canny_pairs.append((int(low), int(high)))
    connectivities = [int(x) for x in args.connectivity.split(",") if x.strip()]
    dct_lows = [int(x) for x in args.dct_low.split(",") if x.strip()]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    combos = []
    for flat_p in flat_percentiles:
        for canny_low, canny_high in canny_pairs:
            for conn in connectivities:
                for dct_low in dct_lows:
                    combos.append(
                        (
                            flat_p,
                            canny_low,
                            canny_high,
                            conn,
                            dct_low,
                            args.max_side,
                            args.cluster_min_frac,
                        )
                    )

    completed = set()
    if args.resume and args.out.exists():
        with args.out.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    completed.add(
                        (
                            int(row["flat_percentile"]),
                            int(row["canny_low"]),
                            int(row["canny_high"]),
                            int(row["connectivity"]),
                            int(row["dct_low"]),
                            int(args.max_side),
                            float(args.cluster_min_frac),
                        )
                    )
                except Exception:
                    continue
        if completed:
            combos = [c for c in combos if c not in completed]

    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "flat_percentile",
                "canny_low",
                "canny_high",
                "connectivity",
                "dct_low",
                "metric",
                "ai_mean",
                "real_mean",
                "cohen_d",
            ]
        )

        if args.workers > 1:
            with Pool(
                processes=args.workers,
                initializer=_init_pool,
                initargs=(ai_samples, real_samples, args.patch_size, args.skip_dct),
            ) as pool:
                done = 0
                total = len(combos)
                for rows in pool.imap_unordered(evaluate_combo, combos):
                    writer.writerows(rows)
                    done += 1
                    if done % 1 == 0:
                        f.flush()
                        print(f"[PROGRESS] combos {done}/{total}")
        else:
            _init_pool(ai_samples, real_samples, args.patch_size, args.skip_dct)
            total = len(combos)
            for idx, combo in enumerate(combos, start=1):
                rows = evaluate_combo(combo)
                writer.writerows(rows)
                f.flush()
                print(f"[PROGRESS] combos {idx}/{total}")

    print(f"Saved sweep results to {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())
