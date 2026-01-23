#!/usr/bin/env python3
"""
Merge batch v3 stats (13d) with existing extra/boundary stats to create 15d bundle:
  - v3 13d (from *_cpu_stats_v3.npy)
  - extra edge_length_mean (extra_stats[9])
  - boundary rank_entropy (boundary_stats[3])

Output:
  embeddings/{category}_cpu_stats_v3_batch15.npy
  embeddings/{category}_cpu_stats_v3_batch15_files.txt
  embeddings/{category}_cpu_stats_v3_batch15_meta.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

EMBEDDINGS_DIR = Path("/home/techne/aicheckers/embeddings")

EXTRA_EDGE_LEN_MEAN_IDX = 9
BOUNDARY_RANK_ENTROPY_IDX = 3

FEATURE_NAMES = [
    # v3 13d
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
    # extra/boundary
    "edge_length_mean",
    "rank_entropy",
]


def merge_category(name: str):
    v3_path = EMBEDDINGS_DIR / f"{name}_cpu_stats_v3.npy"
    v3_files = EMBEDDINGS_DIR / f"{name}_cpu_stats_v3_files.txt"
    extra_path = EMBEDDINGS_DIR / f"{name}_extra_stats.npy"
    boundary_path = EMBEDDINGS_DIR / f"{name}_boundary_stats.npy"

    if not v3_path.exists():
        raise SystemExit(f"{name}: missing {v3_path}")
    if not extra_path.exists():
        raise SystemExit(f"{name}: missing {extra_path}")
    if not boundary_path.exists():
        raise SystemExit(f"{name}: missing {boundary_path}")

    v3 = np.load(v3_path)
    extra = np.load(extra_path)
    boundary = np.load(boundary_path)

    n_min = min(v3.shape[0], extra.shape[0], boundary.shape[0])
    if n_min == 0:
        raise SystemExit(f"{name}: empty arrays")

    v3 = v3[:n_min]
    extra_edge = extra[:n_min, EXTRA_EDGE_LEN_MEAN_IDX:EXTRA_EDGE_LEN_MEAN_IDX + 1]
    boundary_rank = boundary[:n_min, BOUNDARY_RANK_ENTROPY_IDX:BOUNDARY_RANK_ENTROPY_IDX + 1]

    stats = np.concatenate([v3, extra_edge, boundary_rank], axis=1).astype(np.float32)

    out_stats = EMBEDDINGS_DIR / f"{name}_cpu_stats_v3_batch15.npy"
    out_files = EMBEDDINGS_DIR / f"{name}_cpu_stats_v3_batch15_files.txt"
    out_meta = EMBEDDINGS_DIR / f"{name}_cpu_stats_v3_batch15_meta.json"

    np.save(out_stats, stats)
    if v3_files.exists():
        out_files.write_text(v3_files.read_text())
    else:
        out_files.write_text("\n".join([str(i) for i in range(n_min)]) + "\n")
    out_meta.write_text(json.dumps({
        "features": FEATURE_NAMES,
        "samples": int(n_min),
        "dtype": "float32",
        "source": {
            "v3": v3_path.name,
            "extra": extra_path.name,
            "boundary": boundary_path.name,
        },
        "extra_indices": {"edge_length_mean": EXTRA_EDGE_LEN_MEAN_IDX},
        "boundary_indices": {"rank_entropy": BOUNDARY_RANK_ENTROPY_IDX},
    }, indent=2))

    print(f"[DONE] {name}: {stats.shape}")


def main():
    parser = argparse.ArgumentParser(description="Merge v3 13d with extra/boundary to 15d bundle.")
    parser.add_argument("--category", type=str)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.category:
        merge_category(args.category)
        return

    if args.all:
        names = sorted({p.name.replace("_cpu_stats_v3.npy", "") for p in EMBEDDINGS_DIR.glob("*_cpu_stats_v3.npy")})
        for name in names:
            merge_category(name)
        return

    print("Use --category or --all")


if __name__ == "__main__":
    main()
