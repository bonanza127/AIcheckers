#!/usr/bin/env python3
"""
Boundary Stats計算モジュール（777d構成用）
フラクタル次元 + ランクエントロピー + 曲率分散 = 5次元

使用インデックス（777d構成）: [0, 3]
  - [0] fractal_dim_1x
  - [3] rank_entropy
"""
import numpy as np
import cv2

# Box-counting用サイズ
BOX_SIZES_1X = [2, 4, 8, 16, 32]
BOX_SIZES_05X = [2, 4, 8, 16]

# Flat領域判定の相対閾値
FLAT_PERCENTILE = 15


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
    """Box-counting法でフラクタル次元を計算"""
    counts = []
    h, w = binary_map.shape

    for box_size in box_sizes:
        if box_size > min(h, w):
            continue

        h_trim = (h // box_size) * box_size
        w_trim = (w // box_size) * box_size
        trimmed = binary_map[:h_trim, :w_trim]

        n_rows = h_trim // box_size
        n_cols = w_trim // box_size
        reshaped = trimmed.reshape(n_rows, box_size, n_cols, box_size)
        box_has_edge = reshaped.any(axis=(1, 3))
        count = box_has_edge.sum()
        counts.append((box_size, count))

    if len(counts) < 2:
        return 0.0

    log_sizes = np.log([1.0 / s for s, _ in counts])
    log_counts = np.log([max(c, 1) for _, c in counts])

    try:
        slope, _ = np.polyfit(log_sizes, log_counts, 1)
        return slope
    except:
        return 0.0


def compute_fractal_dim_multiscale(edge_map):
    """2スケールでフラクタル次元を計算"""
    fd_1x = box_counting_dimension(edge_map, BOX_SIZES_1X)

    edge_half = cv2.resize(edge_map, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    edge_half = (edge_half > 127).astype(np.uint8) * 255
    fd_05x = box_counting_dimension(edge_half, BOX_SIZES_05X)

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


def compute_boundary_stats(img_rgb, target_size=512):
    """
    RGB画像からboundary_statsを計算（5次元）

    Args:
        img_rgb: numpy array (H, W, 3) RGB画像
        target_size: リサイズ先のサイズ（デフォルト512）

    Returns:
        numpy array (5,) [fractal_dim_1x, fractal_dim_05x, fractal_dim_diff, rank_entropy, curvature_var]
    """
    # リサイズ
    h, w = img_rgb.shape[:2]
    if h != target_size or w != target_size:
        img_rgb = cv2.resize(img_rgb, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)

    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # Edge map
    edge_map = get_edge_map(img_gray)

    # Flat mask
    flat_mask = get_flat_mask(img_gray)

    # Fractal dim (2スケール + 差分)
    fd_1x, fd_05x, fd_diff = compute_fractal_dim_multiscale(edge_map)

    # Rank entropy
    rank_entropy = compute_rank_entropy(img_gray, flat_mask)

    # Curvature var
    curvature_var = compute_boundary_features(flat_mask)

    return np.array([fd_1x, fd_05x, fd_diff, rank_entropy, curvature_var], dtype=np.float32)
