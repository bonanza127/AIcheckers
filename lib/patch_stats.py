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
