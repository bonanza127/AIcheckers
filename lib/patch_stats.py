"""
パッチ統計量計算の共通モジュール

backend/main.py と scripts/extract_embeddings_v2.py の両方で使用
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 閾値定数
HIGH_SCORE_THRESHOLD = 0.8
HIGH_SIM_THRESHOLD = 0.85
GRID_SIZE = 14  # 14x14 = 196 patches


def compute_patch_stats_inference(
    patch_embeddings: torch.Tensor,
    classifier: nn.Linear,
    return_scores: bool = False
):
    """
    パッチ埋め込みから統計量を計算（推論時用、単一サンプル）

    Args:
        patch_embeddings: (1, 196, 768) パッチ埋め込み
        classifier: 768→2 または 775→2 の分類器
        return_scores: Trueの場合、パッチスコア配列も返す

    Returns:
        return_scores=False: (1, 7) パッチ統計量
        return_scores=True: ((1, 7), (196,)) パッチ統計量とパッチAIスコア
    """
    with torch.no_grad():
        # パッチごとのAIスコアを計算（768次元入力）
        # 775次元分類器の場合は先頭768次元のみ使用
        weight = classifier.weight[:, :768]  # (2, 768)
        bias = classifier.bias
        flat_patches = patch_embeddings.reshape(-1, 768)  # (196, 768)
        logits = torch.mm(flat_patches, weight.t()) + bias  # (196, 2)
        probs = torch.softmax(logits, dim=1)
        ai_scores = probs[:, 1]  # (196,)

        # 統計量計算
        patch_mean = ai_scores.mean()
        patch_max = ai_scores.max()
        patch_var = ai_scores.var()
        max_minus_mean = patch_max - patch_mean
        embed_var_mean = patch_embeddings[0].var(dim=0).mean()
        count_high_score = (ai_scores >= HIGH_SCORE_THRESHOLD).float().mean()

        # v_high_sim_85: 垂直方向の高類似度パッチ比率
        patch_emb = patch_embeddings[0]  # (196, 768)
        patches_grid = patch_emb.reshape(GRID_SIZE, GRID_SIZE, -1)
        v_sims = []
        for row in range(GRID_SIZE - 1):
            for col in range(GRID_SIZE):
                current = patches_grid[row, col]
                down = patches_grid[row + 1, col]
                sim = F.cosine_similarity(current.unsqueeze(0), down.unsqueeze(0)).item()
                v_sims.append(sim)
        v_high_sim_85 = torch.tensor(
            sum(1 for s in v_sims if s > HIGH_SIM_THRESHOLD) / len(v_sims),
            device=patch_embeddings.device
        )

        stats = torch.stack([
            patch_mean, patch_max, patch_var, max_minus_mean,
            embed_var_mean, count_high_score, v_high_sim_85
        ])

        if return_scores:
            return stats.unsqueeze(0), ai_scores
        return stats.unsqueeze(0)


def compute_patch_stats_batch(
    patch_embeddings: torch.Tensor,
    classifier: nn.Linear = None
) -> np.ndarray:
    """
    パッチ統計量を計算（バッチ抽出用）

    Args:
        patch_embeddings: (batch, 196, 768) パッチのembedding
        classifier: 分類器（Noneの場合はembedding-based統計のみ）

    Returns:
        stats: (batch, 7) パッチ統計量
    """
    batch_size = patch_embeddings.shape[0]
    num_patches = patch_embeddings.shape[1]
    stats = np.zeros((batch_size, 7), dtype=np.float32)

    with torch.no_grad():
        if classifier is not None:
            # 分類器を通してパッチごとのAIスコアを計算
            # 775次元分類器の場合は先頭768次元のみ使用
            flat_patches = patch_embeddings.reshape(-1, 768)
            weight = classifier.weight[:, :768]
            bias = classifier.bias
            logits = torch.mm(flat_patches, weight.t()) + bias
            probs = F.softmax(logits, dim=1)
            ai_scores = probs[:, 1].reshape(batch_size, -1)

            for i in range(batch_size):
                scores = ai_scores[i].cpu().numpy()
                stats[i, 0] = np.mean(scores)
                stats[i, 1] = np.max(scores)
                stats[i, 2] = np.var(scores)
                stats[i, 3] = stats[i, 1] - stats[i, 0]
                stats[i, 5] = np.sum(scores >= HIGH_SCORE_THRESHOLD) / num_patches
        else:
            # 分類器がない場合はembedding-based統計
            for i in range(batch_size):
                patch_emb = patch_embeddings[i].cpu().numpy()
                norms = np.linalg.norm(patch_emb, axis=1, keepdims=True)
                normalized = patch_emb / (norms + 1e-8)
                mean_patch = normalized.mean(axis=0)
                cos_sims = normalized @ mean_patch
                stats[i, 0] = np.mean(cos_sims)
                stats[i, 1] = np.max(cos_sims)
                stats[i, 2] = np.var(cos_sims)
                stats[i, 3] = stats[i, 1] - stats[i, 0]
                stats[i, 5] = np.sum(cos_sims >= HIGH_SCORE_THRESHOLD) / num_patches

        # embedding空間での多様性 + 垂直方向の高類似度（共通）
        for i in range(batch_size):
            patch_emb = patch_embeddings[i]
            stats[i, 4] = patch_emb.cpu().numpy().var(axis=0).mean()

            # v_high_sim_85
            patches_grid = patch_emb.reshape(GRID_SIZE, GRID_SIZE, -1)
            v_sims = []
            for row in range(GRID_SIZE - 1):
                for col in range(GRID_SIZE):
                    current = patches_grid[row, col]
                    down = patches_grid[row + 1, col]
                    sim = F.cosine_similarity(current.unsqueeze(0), down.unsqueeze(0)).item()
                    v_sims.append(sim)
            stats[i, 6] = np.sum(np.array(v_sims) > HIGH_SIM_THRESHOLD) / len(v_sims)

    return stats


# ============================================================================
# v2: 教師なし中間層パッチ統計量（2026-01 改訂）
# ============================================================================

HIGH_SIM_THRESHOLD_V2 = 0.9  # v2用の高類似度閾値


def compute_patch_stats_v2(
    patch_embeddings: torch.Tensor,
    return_heatmap: bool = False
) -> torch.Tensor:
    """
    中間層パッチから教師なし統計量を計算（7次元）

    分類器を使わず、パッチ埋め込みの構造的特徴のみから統計量を抽出。
    未知のAIモデルに対する汎化性能を高めることを目的とする。

    Args:
        patch_embeddings: (batch, 196, 768) または (196, 768) パッチ埋め込み
        return_heatmap: Trueの場合、可視化用のヒートマップも返す

    Returns:
        stats: (batch, 7) または (7,) パッチ統計量
        heatmap (optional): (batch, 196) または (196,) 可視化用スコア

    統計量の内訳 (7次元):
        0: adj_sim_mean   - 隣接パッチ平均コサイン類似度
        1: adj_sim_var    - 隣接パッチ類似度分散
        2: high_sim_ratio - 高類似度率（>0.9）
        3: patch_var      - パッチ埋め込み分散
        4: anisotropy     - 縦横類似度差
        5: norm_var       - ノルム分散
        6: norm_range     - ノルムレンジ（max - min）
    """
    # fp16入力をfp32に変換
    patch_embeddings = patch_embeddings.float()

    # 入力の次元を正規化
    if patch_embeddings.dim() == 2:
        # (196, 768) -> (1, 196, 768)
        patch_embeddings = patch_embeddings.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    batch_size = patch_embeddings.shape[0]
    device = patch_embeddings.device

    stats_list = []
    heatmap_list = []

    for b in range(batch_size):
        patches = patch_embeddings[b]  # (196, 768)
        grid = patches.reshape(GRID_SIZE, GRID_SIZE, -1)  # (14, 14, 768)
        
        # ① 隣接コサイン類似度（水平・垂直）
        # 水平方向: (14, 13) ペア
        sim_h = F.cosine_similarity(
            grid[:, :-1].reshape(-1, 768),
            grid[:, 1:].reshape(-1, 768),
            dim=-1
        )
        # 垂直方向: (13, 14) ペア
        sim_v = F.cosine_similarity(
            grid[:-1, :].reshape(-1, 768),
            grid[1:, :].reshape(-1, 768),
            dim=-1
        )
        
        adj_sim_mean = (sim_h.mean() + sim_v.mean()) / 2
        adj_sim_var = (sim_h.var() + sim_v.var()) / 2
        high_sim_ratio = (
            (sim_h > HIGH_SIM_THRESHOLD_V2).float().mean() +
            (sim_v > HIGH_SIM_THRESHOLD_V2).float().mean()
        ) / 2
        
        # ② パッチ埋め込み分散（次元ごと分散の平均）
        patch_var = patches.var(dim=0).mean()
        
        # ③ 方向性（縦 vs 横の差）
        anisotropy = (sim_v.mean() - sim_h.mean()).abs()

        # ④ ノルム分布
        norms = torch.norm(patches, dim=1)  # (196,)
        norm_var = norms.var()
        norm_range = norms.max() - norms.min()

        # 統計量をスタック
        stats = torch.stack([
            adj_sim_mean,    # 0
            adj_sim_var,     # 1
            high_sim_ratio,  # 2
            patch_var,       # 3
            anisotropy,      # 4
            norm_var,        # 5
            norm_range,      # 6
        ])
        stats_list.append(stats)
        
        # ヒートマップ用: 各パッチの「異常度」を計算
        if return_heatmap:
            # 隣接パッチとの類似度が低いほど異常（人間的）
            # → 1 - 平均類似度 = 異常度
            heatmap = torch.zeros(196, device=device)
            for i in range(GRID_SIZE):
                for j in range(GRID_SIZE):
                    idx = i * GRID_SIZE + j
                    neighbors = []
                    if i > 0:
                        neighbors.append(grid[i-1, j])
                    if i < GRID_SIZE - 1:
                        neighbors.append(grid[i+1, j])
                    if j > 0:
                        neighbors.append(grid[i, j-1])
                    if j < GRID_SIZE - 1:
                        neighbors.append(grid[i, j+1])
                    if neighbors:
                        neighbor_stack = torch.stack(neighbors)
                        sims = F.cosine_similarity(
                            grid[i, j].unsqueeze(0).expand(len(neighbors), -1),
                            neighbor_stack,
                            dim=-1
                        )
                        # 高類似度 = AI的（スコア高） → そのまま
                        heatmap[idx] = sims.mean()
            heatmap_list.append(heatmap)
    
    # バッチ結果をスタック
    result_stats = torch.stack(stats_list)  # (batch, 7)

    if squeeze_output:
        result_stats = result_stats.squeeze(0)  # (7,)
    
    if return_heatmap:
        result_heatmap = torch.stack(heatmap_list)  # (batch, 196)
        if squeeze_output:
            result_heatmap = result_heatmap.squeeze(0)
        return result_stats, result_heatmap
    
    return result_stats


def compute_patch_stats_v2_batch(
    patch_embeddings: torch.Tensor
) -> np.ndarray:
    """
    バッチ抽出用のv2パッチ統計量（NumPy出力）

    Args:
        patch_embeddings: (batch, 196, 768) パッチ埋め込み

    Returns:
        stats: (batch, 7) パッチ統計量（NumPy配列）
    """
    with torch.no_grad():
        stats_tensor = compute_patch_stats_v2(patch_embeddings, return_heatmap=False)
        return stats_tensor.cpu().numpy()


# ============================================================================
# v3: 拡張パッチ統計量（2026-01-11 新規）- 34次元
# ============================================================================

LOW_SIM_THRESHOLD_V3 = 0.7  # 低類似度の閾値
KNN_K = 8  # k近傍のk
PCA_RANK = 10  # pca_lowrankのランク

V3_STAT_NAMES = [
    # 既存v2 (7d)
    "adj_sim_mean", "adj_sim_var", "high_sim_ratio", "patch_var",
    "anisotropy", "norm_var", "norm_range",
    # 隣接低類似度 (2d)
    "low_sim_ratio_v", "low_sim_ratio_h",
    # CLS乖離系 (4d)
    "cls_sim_mean", "cls_sim_iqr", "cls_sim_high_ratio", "cls_angle_dispersion",
    # KNN系 (2d)
    "knn_sim_mean", "knn_sim_var",
    # スペクトル系 (3d) - pca_lowrank(k=10)で高速化
    "eigenvalue_ratio", "spectral_gap", "top3_ratio",
    # グラフ系 (2d)
    "degree_centrality", "clustering_coef",
    # 空間周波数系 (2d)
    "low_freq_energy_2x2", "high_freq_energy_2x2",
    # 分布系 (2d)
    "norm_skewness", "norm_kurtosis",
    # NEW: Spectral Context (2d)
    "attn_spectral_ratio", "attn_spectral_gap",
    # NEW: Band Energy Spatial Var (3d)
    "band_energy_spatial_var_low", "band_energy_spatial_var_mid", "band_energy_spatial_var_high",
    # NEW: Band Adj Sim Var (2d) - full scaleはadj_sim_varと重複するため除外
    "band_adj_sim_var_2x2", "band_adj_sim_var_4x4",
    # NEW: Band Entropy (1d)
    "band_entropy",
    # NEW: Center vs Edge Band Ratio (2d)
    "center_edge_low_ratio", "center_edge_high_ratio",
]


def compute_patch_stats_v3(
    patch_embeddings: torch.Tensor,
    cls_token: torch.Tensor = None,
) -> torch.Tensor:
    """
    拡張パッチ統計量v3（34次元）- GPU only

    Args:
        patch_embeddings: (batch, 196, 768) または (196, 768) パッチ埋め込み
        cls_token: (batch, 768) または (768,) CLSトークン（オプション）

    Returns:
        stats: (batch, 34) または (34,) パッチ統計量
    """
    # fp16入力をfp32に変換（quantile等がfp16非対応のため）
    patch_embeddings = patch_embeddings.float()
    if cls_token is not None:
        cls_token = cls_token.float()

    # 入力の次元を正規化
    if patch_embeddings.dim() == 2:
        patch_embeddings = patch_embeddings.unsqueeze(0)
        if cls_token is not None and cls_token.dim() == 1:
            cls_token = cls_token.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    batch_size = patch_embeddings.shape[0]
    device = patch_embeddings.device
    embed_dim = patch_embeddings.shape[-1]

    stats_list = []

    for b in range(batch_size):
        patches = patch_embeddings[b]  # (196, 768)
        grid = patches.reshape(GRID_SIZE, GRID_SIZE, -1)  # (14, 14, 768)

        # ============ 既存v2統計量 (7d) ============
        # 水平方向類似度
        sim_h = F.cosine_similarity(
            grid[:, :-1].reshape(-1, embed_dim),
            grid[:, 1:].reshape(-1, embed_dim),
            dim=-1
        )
        # 垂直方向類似度
        sim_v = F.cosine_similarity(
            grid[:-1, :].reshape(-1, embed_dim),
            grid[1:, :].reshape(-1, embed_dim),
            dim=-1
        )

        adj_sim_mean = (sim_h.mean() + sim_v.mean()) / 2
        adj_sim_var = (sim_h.var() + sim_v.var()) / 2
        high_sim_ratio = (
            (sim_h > HIGH_SIM_THRESHOLD_V2).float().mean() +
            (sim_v > HIGH_SIM_THRESHOLD_V2).float().mean()
        ) / 2
        patch_var = patches.var(dim=0).mean()
        anisotropy = (sim_v.mean() - sim_h.mean()).abs()
        norms = torch.norm(patches, dim=1)
        norm_var = norms.var()
        norm_range = norms.max() - norms.min()

        # ============ 隣接低類似度 (2d) ============
        low_sim_ratio_v = (sim_v < LOW_SIM_THRESHOLD_V3).float().mean()
        low_sim_ratio_h = (sim_h < LOW_SIM_THRESHOLD_V3).float().mean()

        # ============ CLS乖離系 (4d) ============
        if cls_token is not None:
            cls = cls_token[b]  # (768,)
            cls_sims = F.cosine_similarity(patches, cls.unsqueeze(0).expand(196, -1), dim=-1)
            cls_sim_mean = cls_sims.mean()
            q75 = torch.quantile(cls_sims, 0.75)
            q25 = torch.quantile(cls_sims, 0.25)
            cls_sim_iqr = q75 - q25
            cls_sim_high_ratio = (cls_sims > HIGH_SIM_THRESHOLD_V2).float().mean()
            # 角度分散: arccos(similarity)の分散
            cls_angles = torch.acos(cls_sims.clamp(-1, 1))
            cls_angle_dispersion = cls_angles.var()
        else:
            cls_sim_mean = torch.tensor(0.0, device=device)
            cls_sim_iqr = torch.tensor(0.0, device=device)
            cls_sim_high_ratio = torch.tensor(0.0, device=device)
            cls_angle_dispersion = torch.tensor(0.0, device=device)

        # ============ KNN系 (2d) ============
        patches_norm = F.normalize(patches, dim=-1)
        sim_matrix = torch.mm(patches_norm, patches_norm.t())  # (196, 196)
        sim_matrix.fill_diagonal_(-1)
        topk_sims, _ = sim_matrix.topk(KNN_K, dim=-1)  # (196, k)
        knn_sim_mean = topk_sims.mean()
        knn_sim_var = topk_sims.var()

        # ============ スペクトル系 (3d) - pca_lowrank高速化 ============
        patches_centered = patches - patches.mean(dim=0, keepdim=True)
        try:
            U, S, V = torch.pca_lowrank(patches_centered, q=PCA_RANK)
            eigvals = (S ** 2) / (196 - 1)
            eigvals = eigvals.clamp(min=1e-10)
            total_var = eigvals.sum()
            eigenvalue_ratio = eigvals[0] / total_var
            spectral_gap = (eigvals[0] - eigvals[1]) / eigvals[0]
            top3_ratio = eigvals[:3].sum() / total_var
        except:
            eigenvalue_ratio = torch.tensor(0.0, device=device)
            spectral_gap = torch.tensor(0.0, device=device)
            top3_ratio = torch.tensor(0.0, device=device)

        # ============ グラフ系 (2d) ============
        adj_matrix = (sim_matrix > HIGH_SIM_THRESHOLD_V2).float()
        degree_centrality = adj_matrix.sum(dim=-1).mean() / 195

        adj_sq = torch.mm(adj_matrix, adj_matrix)
        triangles = (adj_sq * adj_matrix).sum() / 6
        possible_triangles = adj_matrix.sum() * (adj_matrix.sum() - 1) / 2
        clustering_coef = triangles / (possible_triangles + 1e-10)

        # ============ 空間周波数系 (2d) ============
        grid_2x2 = grid.reshape(2, 7, 2, 7, embed_dim).mean(dim=(1, 3))  # (2, 2, 768)
        low_freq_energy = grid_2x2.var()

        diff_h = (grid[:, :-1] - grid[:, 1:]).pow(2).mean()
        diff_v = (grid[:-1, :] - grid[1:, :]).pow(2).mean()
        high_freq_energy = (diff_h + diff_v) / 2

        # ============ 分布系 (2d) ============
        norm_mean = norms.mean()
        norm_std = norms.std() + 1e-10
        norm_centered = (norms - norm_mean) / norm_std
        norm_skewness = (norm_centered ** 3).mean()
        norm_kurtosis = (norm_centered ** 4).mean() - 3

        # ============ NEW: Spectral Context (2d) ============
        # アテンション行列のスペクトル分析（pca_lowrank）
        try:
            U_attn, S_attn, V_attn = torch.pca_lowrank(patches_norm, q=PCA_RANK)
            pc_var = S_attn ** 2
            attn_spectral_ratio = pc_var[0] / pc_var.sum()
            attn_spectral_gap = (pc_var[0] - pc_var[1]) / pc_var[0]
        except:
            attn_spectral_ratio = torch.tensor(0.0, device=device)
            attn_spectral_gap = torch.tensor(0.0, device=device)

        # ============ NEW: Band Energy Spatial Var (3d) ============
        grid_t = grid.permute(2, 0, 1).unsqueeze(0)  # (1, 768, 14, 14)

        # Low: 7x7ブロック平均
        p_low = F.avg_pool2d(grid_t, 7, stride=7).squeeze()  # (768, 2, 2)
        low_energy_per_loc = p_low.pow(2).sum(dim=0)  # (2, 2)
        band_energy_spatial_var_low = low_energy_per_loc.var()

        # Mid: 2x2スケール差分
        p_2x2 = F.avg_pool2d(grid_t, 2, stride=2).squeeze()  # (768, 7, 7)
        p_2x2_up = F.interpolate(p_2x2.unsqueeze(0), size=(14, 14), mode='nearest').squeeze()
        mid_freq = grid_t.squeeze() - p_2x2_up  # (768, 14, 14)
        mid_energy_per_loc = mid_freq.pow(2).sum(dim=0)  # (14, 14)
        band_energy_spatial_var_mid = mid_energy_per_loc.var()

        # High: 隣接差分
        diff_h_sq = (grid[:, :-1] - grid[:, 1:]).pow(2).sum(dim=-1)  # (14, 13)
        diff_v_sq = (grid[:-1, :] - grid[1:, :]).pow(2).sum(dim=-1)  # (13, 14)
        band_energy_spatial_var_high = (diff_h_sq.var() + diff_v_sq.var()) / 2

        # ============ NEW: Band Adj Sim Var (2d) ============
        # 2x2スケール
        p_2x2_t = p_2x2.permute(1, 2, 0)  # (7, 7, 768)
        sim_h_2x2 = F.cosine_similarity(
            p_2x2_t[:, :-1].reshape(-1, embed_dim),
            p_2x2_t[:, 1:].reshape(-1, embed_dim),
            dim=-1
        )
        sim_v_2x2 = F.cosine_similarity(
            p_2x2_t[:-1, :].reshape(-1, embed_dim),
            p_2x2_t[1:, :].reshape(-1, embed_dim),
            dim=-1
        )
        band_adj_sim_var_2x2 = (sim_h_2x2.var() + sim_v_2x2.var()) / 2

        # 4x4スケール
        p_padded = F.pad(grid_t, (1, 1, 1, 1), mode='replicate')
        p_4x4 = F.avg_pool2d(p_padded, 4, stride=4).squeeze().permute(1, 2, 0)  # (4, 4, 768)
        sim_h_4x4 = F.cosine_similarity(
            p_4x4[:, :-1].reshape(-1, embed_dim),
            p_4x4[:, 1:].reshape(-1, embed_dim),
            dim=-1
        )
        sim_v_4x4 = F.cosine_similarity(
            p_4x4[:-1, :].reshape(-1, embed_dim),
            p_4x4[1:, :].reshape(-1, embed_dim),
            dim=-1
        )
        band_adj_sim_var_4x4 = (sim_h_4x4.var() + sim_v_4x4.var()) / 2

        # ============ NEW: Band Entropy (1d) ============
        p_7x7 = F.avg_pool2d(grid_t, 7).squeeze()  # (768, 2, 2)
        energy_low = p_7x7.pow(2).mean()
        energy_mid = (p_2x2.pow(2).mean() - energy_low).abs()
        energy_high = (grid.pow(2).mean() - p_2x2.pow(2).mean()).abs()
        energies = torch.stack([energy_low, energy_mid, energy_high]).clamp(min=1e-10)
        probs_band = energies / energies.sum()
        band_entropy = -(probs_band * torch.log(probs_band + 1e-10)).sum()

        # ============ NEW: Center vs Edge Band Ratio (2d) ============
        center = grid[5:9, 5:9]  # (4, 4, 768)
        edge = torch.cat([
            grid[0:2, :].reshape(-1, embed_dim),
            grid[-2:, :].reshape(-1, embed_dim),
            grid[2:-2, 0:2].reshape(-1, embed_dim),
            grid[2:-2, -2:].reshape(-1, embed_dim),
        ], dim=0)  # (~80, 768)

        center_low = center.mean(dim=(0, 1)).pow(2).sum()
        center_high = center.var(dim=(0, 1)).sum()
        edge_low = edge.mean(dim=0).pow(2).sum()
        edge_high = edge.var(dim=0).sum()
        center_edge_low_ratio = center_low / (edge_low + 1e-10)
        center_edge_high_ratio = center_high / (edge_high + 1e-10)

        # 統計量をスタック (34d)
        stats = torch.stack([
            adj_sim_mean, adj_sim_var, high_sim_ratio, patch_var,
            anisotropy, norm_var, norm_range,
            low_sim_ratio_v, low_sim_ratio_h,
            cls_sim_mean, cls_sim_iqr, cls_sim_high_ratio, cls_angle_dispersion,
            knn_sim_mean, knn_sim_var,
            eigenvalue_ratio, spectral_gap, top3_ratio,
            degree_centrality, clustering_coef,
            low_freq_energy, high_freq_energy,
            norm_skewness, norm_kurtosis,
            attn_spectral_ratio, attn_spectral_gap,
            band_energy_spatial_var_low, band_energy_spatial_var_mid, band_energy_spatial_var_high,
            band_adj_sim_var_2x2, band_adj_sim_var_4x4,
            band_entropy,
            center_edge_low_ratio, center_edge_high_ratio,
        ])
        stats_list.append(stats)

    result = torch.stack(stats_list)
    if squeeze_output:
        result = result.squeeze(0)
    return result


def compute_patch_stats_v3_batch(
    patch_embeddings: torch.Tensor,
    cls_token: torch.Tensor = None,
) -> np.ndarray:
    """
    バッチ抽出用のv3パッチ統計量（NumPy出力）

    Args:
        patch_embeddings: (batch, 196, 768) パッチ埋め込み
        cls_token: (batch, 768) CLSトークン（オプション）

    Returns:
        stats: (batch, 34) パッチ統計量（NumPy配列）
    """
    with torch.no_grad():
        stats_tensor = compute_patch_stats_v3(patch_embeddings, cls_token)
        return stats_tensor.cpu().numpy()

