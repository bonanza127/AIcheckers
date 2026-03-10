"""
候補Bモデル用の拡張特徴量計算モジュール

リアルタイム推論時に使用する特徴量:
- multi_layer_pstats_136 (136d): DINOv3 block 3,6,9,11 からパッチ統計量
- patch_dist_256 (256d): block 6 のパッチ mean/std から上位256次元選択
- hog_27 (27d): HOGベース特徴量
- dct_65 (65d): DCTベース特徴量
- lbp_59 (59d): LBPヒストグラム (nri_uniform)

合計: 136 + 256 + 27 + 65 + 59 = 543d (cpu24は既存なので別途)
"""
import logging
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import scipy.fft as sfft
from scipy.stats import kurtosis as scipy_kurtosis
from skimage.feature import local_binary_pattern

from lib.patch_stats import compute_patch_stats_v3

logger = logging.getLogger(__name__)

# Constants
IMG_SIZE = 224
HOG_BINS = 9
HOG_CELL_SIZE = 8
NUM_DCT_FREQS = 32
EXTRACT_BLOCKS = [3, 6, 9, 11]  # multi_layer_pstats用
MID_BLOCK = 6  # patch_dist用
LBP_P = 8
LBP_R = 1
LBP_UNIFORM_BINS = 59  # method='nri_uniform'

# Top256 indices path
TOP256_INDICES_PATH = Path(__file__).parent.parent / "embeddings" / "patch_dist_top256_indices.npy"
_top256_indices: Optional[np.ndarray] = None


def get_top256_indices() -> np.ndarray:
    """top256インデックスをロード（遅延初期化）"""
    global _top256_indices
    if _top256_indices is None:
        if not TOP256_INDICES_PATH.exists():
            raise FileNotFoundError(f"top256 indices not found: {TOP256_INDICES_PATH}")
        _top256_indices = np.load(TOP256_INDICES_PATH)
        logger.info("Loaded top256 indices: %s", _top256_indices.shape)
    return _top256_indices


def _sanitize_array(arr: np.ndarray) -> np.ndarray:
    """NaN/Infをサニタイズ（replace NaN with 0, keep large values for log1p later）"""
    arr = np.nan_to_num(arr, nan=0.0, posinf=1e10, neginf=-1e10)
    return arr.astype(np.float32)


# ============================================================================
# CPU特徴量
# ============================================================================

def _zigzag_indices(n: int) -> np.ndarray:
    """nxn行列のジグザグスキャン順序を返す"""
    indices = []
    for s in range(2 * n - 1):
        if s % 2 == 0:
            for i in range(min(s, n - 1), max(-1, s - n), -1):
                j = s - i
                indices.append(i * n + j)
        else:
            for i in range(max(0, s - n + 1), min(s + 1, n)):
                j = s - i
                indices.append(i * n + j)
    return np.array(indices)


def _dct2(block: np.ndarray) -> np.ndarray:
    """2D DCT (type-II) via scipy"""
    return sfft.dctn(block, type=2, norm='ortho')


def compute_hog_features(gray: np.ndarray) -> np.ndarray:
    """
    HOG特徴量 (27d)

    8x8 cells, 9 orientation bins on 224x224 -> 28x28 cells
    Output: HOG global(9d) + cell var(9d) + cell std(9d) = 27d
    """
    gy = np.gradient(gray, axis=0)
    gx = np.gradient(gray, axis=1)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    angle = np.arctan2(gy, gx)  # [-pi, pi]
    angle = angle % np.pi  # [0, pi]

    h, w = gray.shape
    cell_size = HOG_CELL_SIZE
    num_cells_y = h // cell_size
    num_cells_x = w // cell_size
    bin_edges = np.linspace(0, np.pi, HOG_BINS + 1)

    cell_hists = np.zeros((num_cells_y, num_cells_x, HOG_BINS), dtype=np.float64)

    for cy in range(num_cells_y):
        for cx in range(num_cells_x):
            y0, y1 = cy * cell_size, (cy + 1) * cell_size
            x0, x1 = cx * cell_size, (cx + 1) * cell_size
            cell_mag = magnitude[y0:y1, x0:x1].ravel()
            cell_ang = angle[y0:y1, x0:x1].ravel()
            hist, _ = np.histogram(cell_ang, bins=bin_edges, weights=cell_mag)
            total = hist.sum() + 1e-10
            cell_hists[cy, cx] = hist / total

    # global: mean of all cell histograms (9d)
    hog_global = cell_hists.mean(axis=(0, 1))

    # cell variance per bin (9d)
    hog_cell_var = cell_hists.var(axis=(0, 1))

    # cell std per bin (9d)
    hog_cell_std = cell_hists.std(axis=(0, 1))

    return np.concatenate([hog_global, hog_cell_var, hog_cell_std]).astype(np.float32)


def compute_dct_features(gray: np.ndarray) -> np.ndarray:
    """
    DCT特徴量 (65d)

    8x8 DCT meanabs(32d) + kurtosis(32d) + blocking score(1d) = 65d
    """
    h, w = gray.shape
    bh, bw = h // 8, w // 8

    zigzag_order = _zigzag_indices(8)
    freq_indices = zigzag_order[1:NUM_DCT_FREQS + 1]

    coeff_matrix = np.zeros((bh * bw, NUM_DCT_FREQS), dtype=np.float64)
    block_idx = 0
    for i in range(bh):
        for j in range(bw):
            block = gray[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8]
            dct_block = _dct2(block)
            flat = dct_block.ravel()
            coeff_matrix[block_idx] = flat[freq_indices]
            block_idx += 1

    meanabs = np.mean(np.abs(coeff_matrix), axis=0).astype(np.float32)

    kurt = np.zeros(NUM_DCT_FREQS, dtype=np.float32)
    for fi in range(NUM_DCT_FREQS):
        col = coeff_matrix[:, fi]
        if col.std() > 1e-10:
            kurt[fi] = scipy_kurtosis(col, fisher=True)

    # Blocking score
    boundary_diff = 0.0
    non_boundary_diff = 0.0
    b_count = 0
    nb_count = 0

    for row in range(1, h):
        diff = np.mean(np.abs(gray[row].astype(np.float64) - gray[row - 1].astype(np.float64)))
        if row % 8 == 0:
            boundary_diff += diff
            b_count += 1
        else:
            non_boundary_diff += diff
            nb_count += 1

    for col in range(1, w):
        diff = np.mean(np.abs(gray[:, col].astype(np.float64) - gray[:, col - 1].astype(np.float64)))
        if col % 8 == 0:
            boundary_diff += diff
            b_count += 1
        else:
            non_boundary_diff += diff
            nb_count += 1

    blocking_score = np.float32(0.0)
    if nb_count > 0 and b_count > 0:
        avg_boundary = boundary_diff / b_count
        avg_non_boundary = non_boundary_diff / nb_count
        blocking_score = np.float32(avg_boundary / (avg_non_boundary + 1e-10))

    return np.concatenate([meanabs, kurt, [blocking_score]])


def compute_lbp_features(gray: np.ndarray) -> np.ndarray:
    """
    LBP特徴量 (59d)

    Local Binary Pattern (P=8, R=1, method='nri_uniform') histogram.
    """
    gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)
    lbp = local_binary_pattern(gray_u8, LBP_P, LBP_R, method="nri_uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=LBP_UNIFORM_BINS, range=(0, LBP_UNIFORM_BINS), density=True)
    return hist.astype(np.float32)


def compute_extended_cpu_features(image: Image.Image) -> np.ndarray:
    """
    CPU特徴量を計算（hog_27 + dct_65 + lbp_59 = 151d）

    Args:
        image: RGB PIL Image (任意サイズ)

    Returns:
        np.ndarray: shape (151,) float32
    """
    # 224x224グレースケールに変換
    gray = np.array(
        image.convert("L").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR),
        dtype=np.float64
    )

    hog = compute_hog_features(gray)
    dct = compute_dct_features(gray)
    lbp = compute_lbp_features(gray)

    result = np.concatenate([hog, dct, lbp])
    return _sanitize_array(result)


# ============================================================================
# GPU特徴量
# ============================================================================

def compute_extended_gpu_features(
    hidden_states: tuple,
    reg_offset: int = 5,
    num_patches: int = 196,
    top256_indices: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    GPU特徴量を計算（multi_layer_pstats_136 + patch_dist_256）

    Args:
        hidden_states: DINOv3のoutput.hidden_states
        reg_offset: register token offset (通常5: CLS + 4 registers)
        num_patches: パッチ数 (通常196 = 14x14)
        top256_indices: patch_dist_1536から選択するインデックス (256,)
                        Noneの場合はget_top256_indices()で自動ロード

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - multi_layer_pstats_136: shape (136,) float32
            - patch_dist_256: shape (256,) float32
    """
    if top256_indices is None:
        top256_indices = get_top256_indices()

    device = hidden_states[0].device

    # --- multi_layer_pstats_136: 4 blocks x 34d = 136d ---
    pstats_list = []
    for block_idx in EXTRACT_BLOCKS:
        hs = hidden_states[block_idx + 1]  # +1 for embedding layer offset
        block_patches = hs[:, reg_offset:reg_offset + num_patches, :]  # (B, 196, 768)
        block_cls = hs[:, 0, :]  # (B, 768)
        pstats = compute_patch_stats_v3(block_patches, block_cls)  # (B, 34)
        pstats_list.append(pstats.cpu().numpy())

    multi_pstats = np.concatenate(pstats_list, axis=-1)  # (B, 136)

    # --- patch_dist_256: block 6 patch mean/std -> top256 ---
    mid_hidden = hidden_states[MID_BLOCK + 1]
    mid_patches = mid_hidden[:, reg_offset:reg_offset + num_patches, :]  # (B, 196, 768)

    patch_mean = mid_patches.mean(dim=1)  # (B, 768)
    patch_std = mid_patches.std(dim=1)    # (B, 768)
    patch_dist_1536 = torch.cat([patch_mean, patch_std], dim=-1)  # (B, 1536)

    # Select top256 indices
    patch_dist_256 = patch_dist_1536[:, top256_indices].cpu().numpy()  # (B, 256)

    # Squeeze batch dimension if single sample
    if multi_pstats.shape[0] == 1:
        multi_pstats = multi_pstats.squeeze(0)
        patch_dist_256 = patch_dist_256.squeeze(0)

    return _sanitize_array(multi_pstats), _sanitize_array(patch_dist_256)


def compute_all_extended_features(
    image: Image.Image,
    hidden_states: tuple,
    reg_offset: int = 5,
    num_patches: int = 196,
) -> dict:
    """
    全ての拡張特徴量を一括計算（便利関数）

    Args:
        image: RGB PIL Image
        hidden_states: DINOv3のoutput.hidden_states
        reg_offset: register token offset
        num_patches: パッチ数

    Returns:
        dict: {
            'cpu_151': (151,) hog_27 + dct_65 + lbp_59,
            'multi_layer_pstats_136': (136,),
            'patch_dist_256': (256,),
        }
    """
    cpu_features = compute_extended_cpu_features(image)
    multi_pstats, patch_dist = compute_extended_gpu_features(
        hidden_states, reg_offset, num_patches
    )

    return {
        'cpu_151': cpu_features,
        'multi_layer_pstats_136': multi_pstats,
        'patch_dist_256': patch_dist,
    }
