#!/usr/bin/env python3
"""
周波数帯域系特徴量の追加ベンチマーク
"""
import torch
import torch.nn.functional as F
import time
import numpy as np

GRID_SIZE = 14
NUM_PATCHES = 196
EMBED_DIM = 768
BATCH_SIZE = 8
NUM_RUNS = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

patches = torch.randn(BATCH_SIZE, NUM_PATCHES, EMBED_DIM, device=device)


def benchmark(func, name, runs=NUM_RUNS):
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
    print(f"{name:45s}: {mean_ms:7.3f} ± {std_ms:.3f} ms")
    return mean_ms


# ============================================================================
# 周波数帯域系特徴量
# ============================================================================

def band_energy_spatial_var():
    """
    各周波数帯域のエネルギーの空間的分散
    - 低/中/高周波それぞれで、空間位置ごとのエネルギー分散
    """
    results = []
    for b in range(BATCH_SIZE):
        p = patches[b].reshape(GRID_SIZE, GRID_SIZE, EMBED_DIM)

        # 低周波: 7x7ブロック平均
        p_low = F.avg_pool2d(p.permute(2,0,1).unsqueeze(0), 7, stride=7).squeeze()  # (768, 2, 2)
        low_energy_per_loc = p_low.pow(2).sum(dim=0)  # (2, 2)
        low_spatial_var = low_energy_per_loc.var()

        # 中周波: 差分（隣接 - 2x2平均）
        p_2x2 = F.avg_pool2d(p.permute(2,0,1).unsqueeze(0), 2, stride=2).squeeze()  # (768, 7, 7)
        p_2x2_up = F.interpolate(p_2x2.unsqueeze(0), size=(14, 14), mode='nearest').squeeze()
        mid_freq = p.permute(2,0,1) - p_2x2_up  # (768, 14, 14)
        mid_energy_per_loc = mid_freq.pow(2).sum(dim=0)  # (14, 14)
        mid_spatial_var = mid_energy_per_loc.var()

        # 高周波: 隣接差分
        diff_h = (p[:, :-1] - p[:, 1:]).pow(2).sum(dim=-1)  # (14, 13)
        diff_v = (p[:-1, :] - p[1:, :]).pow(2).sum(dim=-1)  # (13, 14)
        high_spatial_var = (diff_h.var() + diff_v.var()) / 2

        results.append((low_spatial_var.item(), mid_spatial_var.item(), high_spatial_var.item()))
    return results


def band_adj_sim_var():
    """
    各周波数帯域での隣接類似度の分散
    - 元の解像度、2x2縮約、4x4縮約での隣接類似度分散
    """
    results = []
    for b in range(BATCH_SIZE):
        p = patches[b].reshape(GRID_SIZE, GRID_SIZE, EMBED_DIM)

        # フルスケール (14x14)
        sim_h = F.cosine_similarity(
            p[:, :-1].reshape(-1, EMBED_DIM),
            p[:, 1:].reshape(-1, EMBED_DIM),
            dim=-1
        )
        sim_v = F.cosine_similarity(
            p[:-1, :].reshape(-1, EMBED_DIM),
            p[1:, :].reshape(-1, EMBED_DIM),
            dim=-1
        )
        full_adj_var = (sim_h.var() + sim_v.var()) / 2

        # 2x2縮約 (7x7)
        p_2x2 = F.avg_pool2d(p.permute(2,0,1).unsqueeze(0), 2).squeeze().permute(1,2,0)  # (7, 7, 768)
        sim_h_2x2 = F.cosine_similarity(
            p_2x2[:, :-1].reshape(-1, EMBED_DIM),
            p_2x2[:, 1:].reshape(-1, EMBED_DIM),
            dim=-1
        )
        sim_v_2x2 = F.cosine_similarity(
            p_2x2[:-1, :].reshape(-1, EMBED_DIM),
            p_2x2[1:, :].reshape(-1, EMBED_DIM),
            dim=-1
        )
        adj_var_2x2 = (sim_h_2x2.var() + sim_v_2x2.var()) / 2

        # 4x4縮約 (3x3、パディング)
        p_padded = F.pad(p.permute(2,0,1).unsqueeze(0), (1,1,1,1), mode='replicate')
        p_4x4 = F.avg_pool2d(p_padded, 4, stride=4).squeeze().permute(1,2,0)  # (4, 4, 768)
        sim_h_4x4 = F.cosine_similarity(
            p_4x4[:, :-1].reshape(-1, EMBED_DIM),
            p_4x4[:, 1:].reshape(-1, EMBED_DIM),
            dim=-1
        )
        sim_v_4x4 = F.cosine_similarity(
            p_4x4[:-1, :].reshape(-1, EMBED_DIM),
            p_4x4[1:, :].reshape(-1, EMBED_DIM),
            dim=-1
        )
        adj_var_4x4 = (sim_h_4x4.var() + sim_v_4x4.var()) / 2

        results.append((full_adj_var.item(), adj_var_2x2.item(), adj_var_4x4.item()))
    return results


def band_entropy():
    """
    周波数帯域エントロピー
    - 各帯域のエネルギー分布のエントロピー
    """
    results = []
    for b in range(BATCH_SIZE):
        p = patches[b].reshape(GRID_SIZE, GRID_SIZE, EMBED_DIM)

        # 各スケールでのエネルギー
        # Low: 7x7ブロック
        p_7x7 = F.avg_pool2d(p.permute(2,0,1).unsqueeze(0), 7).squeeze()  # (768, 2, 2)
        energy_low = p_7x7.pow(2).mean()

        # Mid: 2x2 - 7x7
        p_2x2 = F.avg_pool2d(p.permute(2,0,1).unsqueeze(0), 2).squeeze()  # (768, 7, 7)
        energy_mid = p_2x2.pow(2).mean() - energy_low

        # High: original - 2x2
        energy_high = p.pow(2).mean() - p_2x2.pow(2).mean()

        # 正規化してエントロピー計算
        energies = torch.stack([energy_low, energy_mid.abs(), energy_high.abs()])
        energies = energies.clamp(min=1e-10)
        probs = energies / energies.sum()
        entropy = -(probs * torch.log(probs + 1e-10)).sum()

        results.append(entropy.item())
    return results


def center_vs_edge_band_ratio():
    """
    中央 vs エッジの周波数帯域比
    - 中央領域とエッジ領域での低/高周波エネルギー比
    """
    results = []
    for b in range(BATCH_SIZE):
        p = patches[b].reshape(GRID_SIZE, GRID_SIZE, EMBED_DIM)

        # 中央 (4x4 center)
        center = p[5:9, 5:9]  # (4, 4, 768)
        # エッジ (外周)
        edge_top = p[0:2, :]
        edge_bottom = p[-2:, :]
        edge_left = p[2:-2, 0:2]
        edge_right = p[2:-2, -2:]
        edge = torch.cat([
            edge_top.reshape(-1, EMBED_DIM),
            edge_bottom.reshape(-1, EMBED_DIM),
            edge_left.reshape(-1, EMBED_DIM),
            edge_right.reshape(-1, EMBED_DIM),
        ], dim=0)  # (~80, 768)

        # 中央の低周波（平均）と高周波（分散）
        center_low = center.mean(dim=(0,1)).pow(2).sum()
        center_high = center.var(dim=(0,1)).sum()

        # エッジの低周波と高周波
        edge_low = edge.mean(dim=0).pow(2).sum()
        edge_high = edge.var(dim=0).sum()

        # 比率
        low_ratio = center_low / (edge_low + 1e-10)
        high_ratio = center_high / (edge_high + 1e-10)

        results.append((low_ratio.item(), high_ratio.item()))
    return results


def all_band_features_combined():
    """全band特徴量を一括計算（実際の実装に近い形）"""
    results = []
    for b in range(BATCH_SIZE):
        p = patches[b].reshape(GRID_SIZE, GRID_SIZE, EMBED_DIM)

        # === Band Energy Spatial Var (3d) ===
        p_low = F.avg_pool2d(p.permute(2,0,1).unsqueeze(0), 7, stride=7).squeeze()
        low_energy_per_loc = p_low.pow(2).sum(dim=0)
        low_spatial_var = low_energy_per_loc.var()

        p_2x2 = F.avg_pool2d(p.permute(2,0,1).unsqueeze(0), 2, stride=2).squeeze()
        p_2x2_up = F.interpolate(p_2x2.unsqueeze(0), size=(14, 14), mode='nearest').squeeze()
        mid_freq = p.permute(2,0,1) - p_2x2_up
        mid_energy_per_loc = mid_freq.pow(2).sum(dim=0)
        mid_spatial_var = mid_energy_per_loc.var()

        diff_h = (p[:, :-1] - p[:, 1:]).pow(2).sum(dim=-1)
        diff_v = (p[:-1, :] - p[1:, :]).pow(2).sum(dim=-1)
        high_spatial_var = (diff_h.var() + diff_v.var()) / 2

        # === Band Adj Sim Var (3d) ===
        sim_h = F.cosine_similarity(p[:, :-1].reshape(-1, EMBED_DIM), p[:, 1:].reshape(-1, EMBED_DIM), dim=-1)
        sim_v = F.cosine_similarity(p[:-1, :].reshape(-1, EMBED_DIM), p[1:, :].reshape(-1, EMBED_DIM), dim=-1)
        full_adj_var = (sim_h.var() + sim_v.var()) / 2

        p_2x2_t = p_2x2.permute(1,2,0)
        sim_h_2x2 = F.cosine_similarity(p_2x2_t[:, :-1].reshape(-1, EMBED_DIM), p_2x2_t[:, 1:].reshape(-1, EMBED_DIM), dim=-1)
        sim_v_2x2 = F.cosine_similarity(p_2x2_t[:-1, :].reshape(-1, EMBED_DIM), p_2x2_t[1:, :].reshape(-1, EMBED_DIM), dim=-1)
        adj_var_2x2 = (sim_h_2x2.var() + sim_v_2x2.var()) / 2

        p_padded = F.pad(p.permute(2,0,1).unsqueeze(0), (1,1,1,1), mode='replicate')
        p_4x4 = F.avg_pool2d(p_padded, 4, stride=4).squeeze().permute(1,2,0)
        sim_h_4x4 = F.cosine_similarity(p_4x4[:, :-1].reshape(-1, EMBED_DIM), p_4x4[:, 1:].reshape(-1, EMBED_DIM), dim=-1)
        sim_v_4x4 = F.cosine_similarity(p_4x4[:-1, :].reshape(-1, EMBED_DIM), p_4x4[1:, :].reshape(-1, EMBED_DIM), dim=-1)
        adj_var_4x4 = (sim_h_4x4.var() + sim_v_4x4.var()) / 2

        # === Band Entropy (1d) ===
        p_7x7 = F.avg_pool2d(p.permute(2,0,1).unsqueeze(0), 7).squeeze()
        energy_low = p_7x7.pow(2).mean()
        energy_mid = p_2x2.pow(2).mean() - energy_low
        energy_high = p.pow(2).mean() - p_2x2.pow(2).mean()
        energies = torch.stack([energy_low, energy_mid.abs(), energy_high.abs()]).clamp(min=1e-10)
        probs = energies / energies.sum()
        band_entropy = -(probs * torch.log(probs + 1e-10)).sum()

        # === Center vs Edge Band Ratio (2d) ===
        center = p[5:9, 5:9]
        edge = torch.cat([
            p[0:2, :].reshape(-1, EMBED_DIM),
            p[-2:, :].reshape(-1, EMBED_DIM),
            p[2:-2, 0:2].reshape(-1, EMBED_DIM),
            p[2:-2, -2:].reshape(-1, EMBED_DIM),
        ], dim=0)
        center_low = center.mean(dim=(0,1)).pow(2).sum()
        center_high = center.var(dim=(0,1)).sum()
        edge_low = edge.mean(dim=0).pow(2).sum()
        edge_high = edge.var(dim=0).sum()
        low_ratio = center_low / (edge_low + 1e-10)
        high_ratio = center_high / (edge_high + 1e-10)

        results.append((
            low_spatial_var, mid_spatial_var, high_spatial_var,
            full_adj_var, adj_var_2x2, adj_var_4x4,
            band_entropy,
            low_ratio, high_ratio
        ))
    return results


# ============================================================================
# ベンチマーク実行
# ============================================================================
print("\n" + "=" * 60)
print("周波数帯域系特徴量ベンチマーク")
print("=" * 60)
print(f"バッチサイズ: {BATCH_SIZE}, 実行回数: {NUM_RUNS}\n")

results = {}

print("--- 個別計測 ---")
results["band_energy_spatial_var"] = benchmark(band_energy_spatial_var, "Band Energy Spatial Var (3d)")
results["band_adj_sim_var"] = benchmark(band_adj_sim_var, "Band Adj Sim Var (3d)")
results["band_entropy"] = benchmark(band_entropy, "Band Entropy (1d)")
results["center_edge_ratio"] = benchmark(center_vs_edge_band_ratio, "Center vs Edge Band Ratio (2d)")

print("\n--- 一括計測 ---")
results["all_combined"] = benchmark(all_band_features_combined, "All Band Features Combined (9d)")

print("\n" + "=" * 60)
print("サマリー")
print("=" * 60)
total_individual = sum([results[k] for k in ["band_energy_spatial_var", "band_adj_sim_var", "band_entropy", "center_edge_ratio"]])
print(f"個別合計:     {total_individual:.2f} ms")
print(f"一括計算:     {results['all_combined']:.2f} ms")
print(f"削減率:       {(1 - results['all_combined']/total_individual)*100:.1f}%")
print(f"\n1枚あたり:    {results['all_combined']/BATCH_SIZE:.3f} ms")
print(f"追加次元:     9d")
