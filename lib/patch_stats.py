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
    
    統計量の内訳:
        0: adj_sim_mean   - 隣接パッチ平均コサイン類似度
        1: adj_sim_var    - 隣接パッチ類似度分散
        2: high_sim_ratio - 高類似度率（>0.9）
        3: patch_var      - パッチ埋め込み分散
        4: anisotropy     - 縦横類似度差
        5: norm_var       - ノルム分散
        6: norm_range     - ノルムレンジ（max - min）
    """
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

