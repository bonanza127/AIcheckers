#!/usr/bin/env python3
"""
CPU Statistics Extraction for 29d/30d model inference.

Optimized version: loads image only once for both v2 and v3 extraction.

Uses the exact same implementations as training:
- cpu_stats_v2 (18d): from scripts/extract_cpu_stats_v2.py
- cpu_stats_v3_20d (20d): from scripts/extract_cpu_stats_v3_all.py

Usage:
    from lib.cpu_stats import compute_cpu_stats
    cpu_v2, cpu_v3_20d = compute_cpu_stats(pil_image_or_path)

    # Or for pre-loaded images (faster for batch processing):
    from lib.cpu_stats import compute_cpu_stats_from_array
    cpu_v2, cpu_v3_20d = compute_cpu_stats_from_array(img_rgb_array)
"""
import sys
from pathlib import Path

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
from PIL import Image

# Import the original extraction functions
from scripts.extract_cpu_stats_v2 import extract_features as _extract_v2_features
from scripts.extract_cpu_stats_v3_all import extract_unified as _extract_v3_unified

# ============================================================
# cpu_stats_v3 unified (27d) → 20d mapping
# ============================================================
UNIFIED_TO_20D_IDX = [1, 2, 3, 6, 8, 9, 10, 11, 12, 13, 14, 16, 18, 19, 20, 22, 23, 24, 25, 26]

# Target size for resizing
TARGET_SIZE = 512


def _load_and_resize(path_or_image):
    """
    Load and resize image to TARGET_SIZE x TARGET_SIZE with padding.
    Returns (img_rgb, mask) where mask indicates valid pixels.
    """
    # Handle different input types
    if isinstance(path_or_image, (str, Path)):
        img = Image.open(path_or_image).convert("RGB")
    elif isinstance(path_or_image, np.ndarray):
        # Assume RGB numpy array
        img = Image.fromarray(path_or_image)
    elif hasattr(path_or_image, 'convert'):  # PIL Image
        img = path_or_image.convert("RGB")
    else:
        raise ValueError(f"Unsupported input type: {type(path_or_image)}")

    # Resize maintaining aspect ratio
    w, h = img.size
    scale = TARGET_SIZE / max(h, w)
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))

    if (nw, nh) != img.size:
        img = img.resize((nw, nh), Image.LANCZOS)

    # Create padded canvas
    canvas = Image.new("RGB", (TARGET_SIZE, TARGET_SIZE), (128, 128, 128))
    x0 = (TARGET_SIZE - nw) // 2
    y0 = (TARGET_SIZE - nh) // 2
    canvas.paste(img, (x0, y0))

    # Create mask for valid pixels
    mask = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=bool)
    mask[y0:y0 + nh, x0:x0 + nw] = True

    return np.array(canvas), mask


def compute_cpu_stats_from_array(img_rgb, mask=None):
    """
    Compute CPU statistics from a pre-loaded RGB array.

    This is the fastest method for batch processing where you've already
    loaded the image.

    Args:
        img_rgb: numpy array (H, W, 3) in RGB format, should be 512x512
        mask: optional boolean mask for valid pixels (512x512)

    Returns:
        tuple: (cpu_v2_18d, cpu_v3_20d) as numpy arrays
    """
    if mask is None:
        mask = np.ones((img_rgb.shape[0], img_rgb.shape[1]), dtype=bool)

    # Extract CPU v2 (18d)
    cpu_v2 = _extract_v2_features(img_rgb, mask)

    # Extract CPU v3 unified (27d) and convert to 20d
    cpu_v3_unified = _extract_v3_unified(img_rgb)
    cpu_v3_20d = cpu_v3_unified[UNIFIED_TO_20D_IDX]

    return cpu_v2, cpu_v3_20d


def compute_cpu_stats(img_or_path):
    """
    Compute CPU statistics for 29d/30d model inference.

    Optimized to load the image only once for both v2 and v3 extraction.

    Args:
        img_or_path: PIL Image, numpy array (RGB), or path to image file

    Returns:
        tuple: (cpu_v2_18d, cpu_v3_20d) as numpy arrays
    """
    # Load and resize once
    img_rgb, mask = _load_and_resize(img_or_path)

    # Extract both from the same loaded image
    return compute_cpu_stats_from_array(img_rgb, mask)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        v2, v3_20d = compute_cpu_stats(img_path)
        print(f"cpu_v2 (18d): {v2.shape}")
        print(f"cpu_v3_20d (20d): {v3_20d.shape}")
        print(f"\nv2 values: {v2}")
        print(f"\nv3_20d values: {v3_20d}")
    else:
        print("Usage: python cpu_stats.py <image_path>")
