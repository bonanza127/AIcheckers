#!/usr/bin/env python3
"""
新特徴量の評価スクリプト（高速版）

- バッチ処理（GPU、OOM防止のため適度なサイズ）
- tqdm進捗表示
- サンプル数指定可能

Usage:
    python3 scripts/analyze_new_features.py --samples 10000
    python3 scripts/analyze_new_features.py --samples 5000 --batch-size 64
"""
import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from scipy import stats as scipy_stats

EMBEDDINGS_DIR = Path("/home/techne/aicheckers/embeddings")

AI_CATS = [
    "illustrious_ai", "pony_ai", "sdxl10_ai", "sd15_ai", "other_ai",
    "flux1d_ai", "novelai_ai", "pixai_ai", "novelai_combined_ai", "novelai_artist_tagged_ai"
]
REAL_CATS = ["danbooru_real"]

ELITE_EXISTING = ['cls_sim_mean', 'knn_sim_var', 'high_freq_energy_2x2', 'top3_ratio']


def compute_batch_stats(patches_batch: torch.Tensor) -> dict:
    """
    バッチ処理で統計量を計算

    Args:
        patches_batch: (B, 196, 768) tensor on GPU

    Returns:
        dict of lists, each list has B elements
    """
    B = patches_batch.shape[0]
    device = patches_batch.device

    # Normalize
    patches_norm = patches_batch / (patches_batch.norm(dim=-1, keepdim=True) + 1e-8)

    # CLS approximation (mean of patches)
    cls_approx = patches_batch.mean(dim=1, keepdim=True)  # (B, 1, 768)
    cls_norm = cls_approx / (cls_approx.norm(dim=-1, keepdim=True) + 1e-8)

    # cls_sim: (B, 196)
    cls_sim = torch.bmm(patches_norm, cls_norm.transpose(1, 2)).squeeze(-1)

    results = {k: [] for k in [
        'cls_sim_mean', 'knn_sim_var', 'high_freq_energy_2x2', 'top3_ratio',
        'cls_sim_min', 'cls_sim_max', 'cls_outlier_ratio', 'cls_sim_skewness',
        'cls_sim_kurtosis', 'cls_sim_bottom10_mean', 'local_uniformity_gap',
        'spectral_entropy', 'effective_rank', 'top10_ratio', 'spatial_gradient_var'
    ]}

    # Process each sample (some ops don't vectorize well)
    for i in range(B):
        p = patches_batch[i]  # (196, 768)
        p_norm = patches_norm[i]  # (196, 768)
        cs = cls_sim[i]  # (196,)

        # === Existing elite ===
        results['cls_sim_mean'].append(cs.mean().item())

        # knn_sim_var
        sim_matrix = p_norm @ p_norm.T
        sim_matrix.fill_diagonal_(0)
        knn_sim, _ = torch.topk(sim_matrix, k=5, dim=1)
        knn_mean_per_patch = knn_sim.mean(dim=1)
        results['knn_sim_var'].append(knn_mean_per_patch.var().item())

        # high_freq_energy_2x2
        p_2d = p.reshape(14, 14, 768)
        low_freq = p_2d.unfold(0, 2, 2).unfold(1, 2, 2).mean(dim=(-2, -1))
        low_up = low_freq.repeat_interleave(2, dim=0).repeat_interleave(2, dim=1)
        high_freq = p_2d - low_up
        results['high_freq_energy_2x2'].append((high_freq ** 2).sum().item())

        # top3_ratio, spectral
        p_centered = p - p.mean(dim=0, keepdim=True)
        try:
            _, s, _ = torch.linalg.svd(p_centered, full_matrices=False)
            total_var = (s ** 2).sum()
            results['top3_ratio'].append(((s[:3] ** 2).sum() / (total_var + 1e-8)).item())
            results['top10_ratio'].append(((s[:10] ** 2).sum() / (total_var + 1e-8)).item())
            s_norm = (s ** 2) / (total_var + 1e-8)
            entropy = -(s_norm * torch.log(s_norm + 1e-8)).sum().item()
            results['spectral_entropy'].append(entropy)
            results['effective_rank'].append(np.exp(entropy))
        except:
            results['top3_ratio'].append(0.0)
            results['top10_ratio'].append(0.0)
            results['spectral_entropy'].append(0.0)
            results['effective_rank'].append(0.0)

        # === NEW: cls_sim派生 ===
        results['cls_sim_min'].append(cs.min().item())
        results['cls_sim_max'].append(cs.max().item())
        results['cls_outlier_ratio'].append((cs < 0.3).float().mean().item())

        cs_np = cs.cpu().numpy()
        results['cls_sim_skewness'].append(float(scipy_stats.skew(cs_np)))
        results['cls_sim_kurtosis'].append(float(scipy_stats.kurtosis(cs_np)))

        bottom10 = torch.topk(cs, k=10, largest=False).values
        results['cls_sim_bottom10_mean'].append(bottom10.mean().item())

        # local_uniformity_gap
        results['local_uniformity_gap'].append(
            (knn_mean_per_patch.max() - knn_mean_per_patch.min()).item()
        )

        # spatial_gradient_var
        p_2d_flat = p.reshape(14, 14, 768)
        h_diff = (p_2d_flat[:, 1:, :] - p_2d_flat[:, :-1, :]).pow(2).sum(dim=-1)
        v_diff = (p_2d_flat[1:, :, :] - p_2d_flat[:-1, :, :]).pow(2).sum(dim=-1)
        results['spatial_gradient_var'].append(
            torch.cat([h_diff.flatten(), v_diff.flatten()]).var().item()
        )

    return results


def load_and_sample(cats: list, max_total: int, desc: str) -> torch.Tensor:
    """カテゴリからサンプリングしてロード"""
    all_patches = []

    for cat in cats:
        path = EMBEDDINGS_DIR / f"{cat}_mid_patches.npy"
        if not path.exists():
            continue
        patches = np.load(path, mmap_mode='r')
        all_patches.append((cat, patches, len(patches)))

    # Calculate samples per category
    total_available = sum(n for _, _, n in all_patches)
    samples_per_cat = []
    for cat, patches, n in all_patches:
        ratio = n / total_available
        samples_per_cat.append((cat, patches, min(n, int(max_total * ratio) + 1)))

    # Load and concatenate
    result = []
    for cat, patches, n_sample in tqdm(samples_per_cat, desc=f"Loading {desc}"):
        idx = np.random.choice(len(patches), min(n_sample, len(patches)), replace=False)
        result.append(patches[idx].astype(np.float32))

    return np.concatenate(result, axis=0)


def cohens_d(a, b):
    a, b = np.array(a), np.array(b)
    na, nb = len(a), len(b)
    var_a, var_b = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled_std = np.sqrt(((na-1)*var_a + (nb-1)*var_b) / (na+nb-2))
    return (np.mean(a) - np.mean(b)) / pooled_std if pooled_std > 0 else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=10000, help="Total samples per class")
    parser.add_argument("--batch-size", type=int, default=32, help="GPU batch size")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Samples per class: {args.samples}")
    print(f"Batch size: {args.batch_size}")

    # Load data
    ai_patches = load_and_sample(AI_CATS, args.samples, "AI")
    real_patches = load_and_sample(REAL_CATS, args.samples, "Real")

    # Balance
    n_min = min(len(ai_patches), len(real_patches))
    ai_patches = ai_patches[np.random.choice(len(ai_patches), n_min, replace=False)]
    real_patches = real_patches[np.random.choice(len(real_patches), n_min, replace=False)]

    print(f"\nBalanced: {n_min} samples each")

    # Compute stats
    def process_all(patches_np, desc):
        all_stats = {k: [] for k in [
            'cls_sim_mean', 'knn_sim_var', 'high_freq_energy_2x2', 'top3_ratio',
            'cls_sim_min', 'cls_sim_max', 'cls_outlier_ratio', 'cls_sim_skewness',
            'cls_sim_kurtosis', 'cls_sim_bottom10_mean', 'local_uniformity_gap',
            'spectral_entropy', 'effective_rank', 'top10_ratio', 'spatial_gradient_var'
        ]}

        n_batches = (len(patches_np) + args.batch_size - 1) // args.batch_size

        for i in tqdm(range(n_batches), desc=desc):
            start = i * args.batch_size
            end = min(start + args.batch_size, len(patches_np))
            batch = torch.tensor(patches_np[start:end], device=device)

            batch_stats = compute_batch_stats(batch)
            for k, v in batch_stats.items():
                all_stats[k].extend(v)

            # Free GPU memory
            del batch
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        return all_stats

    ai_stats = process_all(ai_patches, "Processing AI")
    real_stats = process_all(real_patches, "Processing Real")

    # Cohen's d analysis
    print(f"\n{'='*60}")
    print(f"{'特徴量':<25} {'Cohen d':>10} {'|d|':>8} {'判定':<12}")
    print("-" * 60)

    results = []
    for k in ai_stats.keys():
        d = cohens_d(ai_stats[k], real_stats[k])
        results.append((k, d))

    results.sort(key=lambda x: -abs(x[1]))

    for k, d in results:
        if abs(d) >= 0.8:
            verdict = "★Large"
        elif abs(d) >= 0.5:
            verdict = "◆Medium"
        elif abs(d) >= 0.2:
            verdict = "○Small"
        else:
            verdict = "×Negligible"

        marker = " [既存]" if k in ELITE_EXISTING else " [NEW]"
        print(f"{k:<25} {d:>10.4f} {abs(d):>8.4f} {verdict:<12}{marker}")

    # Correlation analysis
    print(f"\n{'='*60}")
    print("既存エリートとの相関（新候補のみ、|d|≥0.2）")
    print("-" * 80)

    new_candidates = [k for k, d in results if k not in ELITE_EXISTING and abs(d) >= 0.2]

    if new_candidates:
        print(f"{'新候補':<25} {'cls_sim_mean':>14} {'knn_sim_var':>12} {'high_freq':>10} {'top3_ratio':>11}")
        print("-" * 80)

        for nc in new_candidates:
            combined_nc = ai_stats[nc] + real_stats[nc]
            corrs = []
            for elite in ELITE_EXISTING:
                combined_elite = ai_stats[elite] + real_stats[elite]
                corr = np.corrcoef(combined_nc, combined_elite)[0, 1]
                corrs.append(corr)
            print(f"{nc:<25} {corrs[0]:>14.3f} {corrs[1]:>12.3f} {corrs[2]:>10.3f} {corrs[3]:>11.3f}")

    print(f"\n完了!")


if __name__ == "__main__":
    main()
