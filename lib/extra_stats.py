#!/usr/bin/env python3
"""
Extra Stats計算モジュール（777d構成用）
CbCr統計 + Flat統計 + 線統計 = 15次元

使用インデックス（777d構成）: [4, 6, 9, 11, 14]
  - [4] flat_cluster_count
  - [6] flat_ratio
  - [9] edge_length_mean
  - [11] edge_length_gini
  - [14] flat_ratio_variance
"""
import numpy as np
import cv2
from scipy import ndimage

# 設定
FLAT_PERCENTILE = 15
CLUSTER_MIN_FRAC = 0.005


def rgb_to_ycbcr(img_rgb):
    """RGB -> YCbCr変換"""
    img = img_rgb.astype(np.float32)
    y = 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]
    cb = 128 - 0.168736 * img[:,:,0] - 0.331264 * img[:,:,1] + 0.5 * img[:,:,2]
    cr = 128 + 0.5 * img[:,:,0] - 0.418688 * img[:,:,1] + 0.081312 * img[:,:,2]
    return y, cb, cr


def get_flat_mask(img_gray, percentile=FLAT_PERCENTILE):
    """勾配が小さい領域（Flat領域）のマスクを取得"""
    grad_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    threshold = np.percentile(grad_mag, percentile)
    flat_mask = grad_mag <= threshold
    return flat_mask, grad_mag


def compute_flat_cluster_stats(flat_mask):
    """Flat領域のクラスタ統計を計算"""
    flat_uint8 = flat_mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(flat_uint8, connectivity=4)
    img_size = flat_mask.shape[0] * flat_mask.shape[1]

    if num_labels <= 1:
        return 0.0, 0.0, 0.0, 0.0

    cluster_sizes = stats[1:, cv2.CC_STAT_AREA].astype(np.float32) / img_size
    cluster_sizes = cluster_sizes[cluster_sizes >= CLUSTER_MIN_FRAC]
    if cluster_sizes.size == 0:
        return 0.0, 0.0, 0.0, 0.0

    return (
        float(cluster_sizes.max()),
        float(np.median(cluster_sizes)),
        float(np.mean(cluster_sizes)),
        float(cluster_sizes.size) / np.sqrt(img_size) * 100
    )


def compute_flat_ratio_variance(img_gray, grad_mag=None, patch_size=32):
    """パッチごとのflat_ratioの分散を計算"""
    h, w = img_gray.shape

    if grad_mag is None:
        grad_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    threshold = np.percentile(grad_mag, FLAT_PERCENTILE)

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
    cbcr = np.stack([cb, cr], axis=-1)

    cbcr_var_global = cbcr.var()

    if flat_mask.sum() > 100:
        cbcr_flat = cbcr[flat_mask]
        cbcr_var_flat = cbcr_flat.var()
    else:
        cbcr_var_flat = 0.0

    flat_cluster_size, flat_cluster_median, flat_cluster_mean, flat_cluster_count = compute_flat_cluster_stats(flat_mask)

    cb_shifted = np.roll(cb, 1, axis=1)
    cr_shifted = np.roll(cr, 1, axis=1)
    cb_corr = np.corrcoef(cb.flatten(), cb_shifted.flatten())[0, 1]
    cr_corr = np.corrcoef(cr.flatten(), cr_shifted.flatten())[0, 1]
    cbcr_autocorr = (cb_corr + cr_corr) / 2
    if np.isnan(cbcr_autocorr):
        cbcr_autocorr = 0.0

    flat_ratio = flat_mask.sum() / flat_mask.size

    return np.array([
        np.log1p(cbcr_var_global),
        np.log1p(cbcr_var_flat),
        flat_cluster_median,
        flat_cluster_mean,
        flat_cluster_count,
        cbcr_autocorr,
        flat_ratio,
        flat_cluster_size
    ], dtype=np.float32)


def compute_gini(values):
    """ジニ係数を計算"""
    if len(values) == 0:
        return 0.0
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    cumsum = np.cumsum(sorted_vals)
    return (2 * np.sum((np.arange(1, n+1) * sorted_vals)) / (n * cumsum[-1] + 1e-8)) - (n + 1) / n


def compute_line_stats(img_gray):
    """線統計を計算（6次元）"""
    edges = cv2.Canny(img_gray, 50, 150)

    if edges.sum() < 100:
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    grad_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)

    edge_mask = edges > 0
    angles = np.arctan2(grad_y[edge_mask], grad_x[edge_mask])
    hist, _ = np.histogram(angles, bins=8, range=(-np.pi, np.pi))
    hist = hist / (hist.sum() + 1e-8)
    edge_direction_entropy = -np.sum(hist * np.log(hist + 1e-8)) / np.log(8)

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

    img_size = img_gray.shape[0] * img_gray.shape[1]
    edge_length_mean = edge_length_mean / np.sqrt(img_size) * 100
    edge_length_var = edge_length_var / img_size * 10000

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    micro_loop_count = 0
    total_contours = len(contours)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter > 0 and area > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if area < 100 and circularity > 0.3:
                micro_loop_count += 1

    micro_loop_ratio = micro_loop_count / (total_contours + 1)

    h, w = img_gray.shape
    patch_size = 32
    patch_densities = []

    for i in range(0, h - patch_size, patch_size):
        for j in range(0, w - patch_size, patch_size):
            patch = edges[i:i+patch_size, j:j+patch_size]
            density = patch.sum() / (patch_size * patch_size * 255)
            patch_densities.append(density)

    if len(patch_densities) > 1:
        patch_edge_density_var = np.var(patch_densities)
    else:
        patch_edge_density_var = 0.0

    return np.array([
        edge_direction_entropy,
        edge_length_mean,
        edge_length_var,
        edge_length_gini,
        micro_loop_ratio,
        patch_edge_density_var
    ], dtype=np.float32)


def compute_extra_stats(img_rgb):
    """
    RGB画像からextra_statsを計算（15次元）

    Args:
        img_rgb: numpy array (H, W, 3) RGB画像

    Returns:
        numpy array (15,) extra stats
    """
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    flat_mask, grad_mag = get_flat_mask(img_gray)

    cbcr_stats = compute_cbcr_stats(img_rgb, flat_mask)
    line_stats = compute_line_stats(img_gray)
    flat_ratio_var = compute_flat_ratio_variance(img_gray, grad_mag)

    return np.concatenate([cbcr_stats, line_stats, [flat_ratio_var]])
