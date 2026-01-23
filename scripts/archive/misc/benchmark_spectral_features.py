#!/usr/bin/env python3
"""
スペクトル/アテンション系特徴量の速度ベンチマーク
"""
import torch
import torch.nn.functional as F
import time
import numpy as np

# 設定
GRID_SIZE = 14
NUM_PATCHES = 196
EMBED_DIM = 768
BATCH_SIZE = 8
NUM_RUNS = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ダミーデータ
patches = torch.randn(BATCH_SIZE, NUM_PATCHES, EMBED_DIM, device=device)
cls_token = torch.randn(BATCH_SIZE, EMBED_DIM, device=device)


def benchmark(func, name, runs=NUM_RUNS):
    """関数のベンチマーク"""
    # ウォームアップ
    for _ in range(5):
        func()
    torch.cuda.synchronize()

    times = []
    for _ in range(runs):
        start = time.perf_counter()
        func()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    mean_ms = np.mean(times) * 1000
    std_ms = np.std(times) * 1000
    print(f"{name:40s}: {mean_ms:7.3f} ± {std_ms:.3f} ms")
    return mean_ms


# ============================================================================
# 現行スペクトル系（eigvalsh）
# ============================================================================
def current_spectral():
    results = []
    for b in range(BATCH_SIZE):
        p = patches[b]
        p_centered = p - p.mean(dim=0, keepdim=True)
        cov = torch.mm(p_centered.t(), p_centered) / (NUM_PATCHES - 1)
        eigenvalues = torch.linalg.eigvalsh(cov)
        eigenvalues = eigenvalues.flip(0).clamp(min=1e-10)
        total_var = eigenvalues.sum()
        ratio = eigenvalues[0] / total_var
        gap = (eigenvalues[0] - eigenvalues[1]) / eigenvalues[0]
        results.append((ratio, gap))
    return results


# ============================================================================
# pca_lowrank版（高速化）
# ============================================================================
def pca_lowrank_spectral():
    results = []
    for b in range(BATCH_SIZE):
        p = patches[b]
        p_centered = p - p.mean(dim=0, keepdim=True)
        U, S, V = torch.pca_lowrank(p_centered, q=10)
        eigvals = (S ** 2) / (NUM_PATCHES - 1)
        total_var = eigvals.sum()
        ratio = eigvals[0] / total_var
        gap = (eigvals[0] - eigvals[1]) / eigvals[0]
        top3 = eigvals[:3].sum() / total_var
        results.append((ratio, gap, top3))
    return results


# ============================================================================
# 新規候補: Spectral Context Features
# ============================================================================

def spectral_context_attention():
    """
    スペクトルコンテキストアテンション風特徴量
    - パッチ間のアテンションスコアをスペクトル的に分析
    """
    results = []
    for b in range(BATCH_SIZE):
        p = patches[b]  # (196, 768)
        p_norm = F.normalize(p, dim=-1)

        # Self-attention的な類似度行列
        attn = torch.mm(p_norm, p_norm.t())  # (196, 196)

        # アテンション行列の固有値（スペクトル分析）
        attn_eigvals = torch.linalg.eigvalsh(attn)
        attn_eigvals = attn_eigvals.flip(0).clamp(min=1e-10)

        # アテンションスペクトル特徴
        attn_spectral_ratio = attn_eigvals[0] / attn_eigvals.sum()
        attn_spectral_gap = (attn_eigvals[0] - attn_eigvals[1]) / attn_eigvals[0]

        results.append((attn_spectral_ratio, attn_spectral_gap))
    return results


def spectral_context_attention_fast():
    """
    高速版: pca_lowrankでアテンション行列のスペクトル
    """
    results = []
    for b in range(BATCH_SIZE):
        p = patches[b]
        p_norm = F.normalize(p, dim=-1)

        # アテンション行列をそのままpca_lowrankには適用できないので
        # パッチを主成分空間に射影してから類似度を計算
        U, S, V = torch.pca_lowrank(p_norm, q=10)
        # 主成分空間での分散
        pc_var = S ** 2
        attn_spectral_ratio = pc_var[0] / pc_var.sum()
        attn_spectral_gap = (pc_var[0] - pc_var[1]) / pc_var[0]

        results.append((attn_spectral_ratio, attn_spectral_gap))
    return results


def cross_patch_spectral_correlation():
    """
    パッチ間スペクトル相関
    - 異なる空間位置のパッチのスペクトル特性の相関
    """
    results = []
    for b in range(BATCH_SIZE):
        p = patches[b].reshape(GRID_SIZE, GRID_SIZE, EMBED_DIM)

        # 4象限に分割
        q1 = p[:7, :7].reshape(-1, EMBED_DIM)   # 左上
        q2 = p[:7, 7:].reshape(-1, EMBED_DIM)   # 右上
        q3 = p[7:, :7].reshape(-1, EMBED_DIM)   # 左下
        q4 = p[7:, 7:].reshape(-1, EMBED_DIM)   # 右下

        # 各象限の主成分
        def get_pc1(x):
            x_c = x - x.mean(dim=0, keepdim=True)
            U, S, V = torch.pca_lowrank(x_c, q=3)
            return V[:, 0]  # 第1主成分方向

        pc1_q1 = get_pc1(q1)
        pc1_q2 = get_pc1(q2)
        pc1_q3 = get_pc1(q3)
        pc1_q4 = get_pc1(q4)

        # 象限間の主成分相関
        cross_corr_diag = F.cosine_similarity(pc1_q1.unsqueeze(0), pc1_q4.unsqueeze(0))
        cross_corr_anti = F.cosine_similarity(pc1_q2.unsqueeze(0), pc1_q3.unsqueeze(0))

        results.append((cross_corr_diag.item(), cross_corr_anti.item()))
    return results


def frequency_band_energy():
    """
    周波数帯域エネルギー（DCT風）
    - 2x2, 4x4, 7x7 のマルチスケール
    """
    results = []
    for b in range(BATCH_SIZE):
        p = patches[b].reshape(GRID_SIZE, GRID_SIZE, EMBED_DIM)

        # 各スケールでの平均プーリング
        # 2x2 (7x7グリッド)
        p_2x2 = F.avg_pool2d(p.permute(2, 0, 1).unsqueeze(0), 2).squeeze()
        # 4x4 (3x3グリッド、パディング必要)
        p_padded = F.pad(p.permute(2, 0, 1).unsqueeze(0), (1, 1, 1, 1), mode='replicate')
        p_4x4 = F.avg_pool2d(p_padded, 4).squeeze()
        # 7x7 (2x2グリッド)
        p_7x7 = F.avg_pool2d(p.permute(2, 0, 1).unsqueeze(0), 7).squeeze()

        # 各スケールでの分散（エネルギー）
        energy_fine = p.var()
        energy_2x2 = p_2x2.var()
        energy_4x4 = p_4x4.var()
        energy_7x7 = p_7x7.var()

        # 比率として正規化
        total_e = energy_fine + 1e-10
        ratio_2x2 = energy_2x2 / total_e
        ratio_4x4 = energy_4x4 / total_e
        ratio_7x7 = energy_7x7 / total_e

        results.append((ratio_2x2.item(), ratio_4x4.item(), ratio_7x7.item()))
    return results


def masked_response_simulation():
    """
    マスク応答シミュレーション
    - ランダムマスクした場合の特徴変化を測定
    """
    results = []
    for b in range(BATCH_SIZE):
        p = patches[b]  # (196, 768)

        # オリジナルの平均
        orig_mean = p.mean(dim=0)

        # 25%マスク（中央領域）
        mask = torch.ones(GRID_SIZE, GRID_SIZE, device=device)
        mask[5:9, 5:9] = 0
        mask = mask.reshape(-1, 1)

        masked_p = p * mask
        masked_mean = masked_p.sum(dim=0) / mask.sum()

        # マスク前後の変化
        mean_shift = F.cosine_similarity(orig_mean.unsqueeze(0), masked_mean.unsqueeze(0))

        # 周辺領域の分散変化
        edge_indices = torch.cat([
            torch.arange(14),  # 上辺
            torch.arange(14) + 14*13,  # 下辺
            torch.arange(1, 13) * 14,  # 左辺
            torch.arange(1, 13) * 14 + 13,  # 右辺
        ]).to(device)
        edge_var = p[edge_indices].var()
        center_indices = []
        for i in range(5, 9):
            for j in range(5, 9):
                center_indices.append(i * 14 + j)
        center_var = p[torch.tensor(center_indices, device=device)].var()

        var_ratio = center_var / (edge_var + 1e-10)

        results.append((mean_shift.item(), var_ratio.item()))
    return results


# ============================================================================
# ベンチマーク実行
# ============================================================================
print("\n" + "=" * 60)
print("スペクトル/アテンション系特徴量ベンチマーク")
print("=" * 60)
print(f"バッチサイズ: {BATCH_SIZE}, 実行回数: {NUM_RUNS}\n")

results = {}

print("--- 基本スペクトル ---")
results["current_eigvalsh"] = benchmark(current_spectral, "現行 (eigvalsh)")
results["pca_lowrank"] = benchmark(pca_lowrank_spectral, "pca_lowrank (q=10)")

print("\n--- スペクトルコンテキスト ---")
results["spectral_context_full"] = benchmark(spectral_context_attention, "Spectral Context (full eigvalsh)")
results["spectral_context_fast"] = benchmark(spectral_context_attention_fast, "Spectral Context (pca_lowrank)")

print("\n--- クロスパッチ ---")
results["cross_patch"] = benchmark(cross_patch_spectral_correlation, "Cross-patch Spectral Correlation")

print("\n--- 周波数帯域 ---")
results["freq_band"] = benchmark(frequency_band_energy, "Frequency Band Energy (multi-scale)")

print("\n--- マスク応答 ---")
results["masked_response"] = benchmark(masked_response_simulation, "Masked Response Simulation")

# サマリー
print("\n" + "=" * 60)
print("追加コスト分析（pca_lowrank基準）")
print("=" * 60)
baseline = results["pca_lowrank"]
for name, time_ms in results.items():
    if name != "pca_lowrank":
        overhead = time_ms - baseline
        print(f"{name:40s}: +{overhead:6.2f} ms ({overhead/baseline*100:5.1f}%)")

# 推奨構成
print("\n" + "=" * 60)
print("推奨: 追加する特徴量")
print("=" * 60)

fast_threshold = baseline * 0.5  # 基準の50%以下なら追加推奨
print(f"閾値: +{fast_threshold:.2f} ms 以下")
print()

recommended = []
for name, time_ms in results.items():
    overhead = time_ms - baseline
    if name != "pca_lowrank" and name != "current_eigvalsh" and overhead < fast_threshold:
        print(f"  ✓ {name} (+{overhead:.2f} ms)")
        recommended.append(name)

if not recommended:
    print("  （閾値内の特徴量なし。最も軽いものを検討）")
    sorted_results = sorted([(k, v) for k, v in results.items() if k not in ["pca_lowrank", "current_eigvalsh"]], key=lambda x: x[1])
    print(f"  → 最軽量: {sorted_results[0][0]} (+{sorted_results[0][1] - baseline:.2f} ms)")
