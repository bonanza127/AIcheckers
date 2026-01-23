#!/usr/bin/env python3
"""
Test 15d 2-head classifier on hard negatives
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# DINOv3 ローカルロード
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from lib.patch_stats import compute_patch_stats_v3


MODEL_DIR = Path("/home/techne/aicheckers/models/two_head_15d")
HARDNEG_CSV = Path("/home/techne/aicheckers/logs/civitai_hard_negatives.csv")

# 既存11d: v3統計量からのインデックス
KEEP_IDX_11D = [1, 2, 3, 5, 9, 12, 14, 17, 18, 19, 32]

# 新規4d特徴量名
NEW_FEAT_NAMES = [
    'local_efficiency', 'corner_coherence',
    'edge_interior_gap', 'cls_sim_center_bias'
]


def load_dinov3():
    """DINOv3をローカルからロード（transformers版）"""
    from transformers import AutoImageProcessor, AutoModel

    # 学習時と同じモデルを使用
    model_path = Path("/home/techne/aicheckers/models/dinov3-vitb16")

    print(f"Loading DINOv3 from: {model_path}")
    processor = AutoImageProcessor.from_pretrained(str(model_path))
    model = AutoModel.from_pretrained(str(model_path))
    model.eval()

    return model, processor


def get_dino_features(model, processor, img_path: Path, device, mid_layer: int = 6):
    """画像からDINOv3特徴量を抽出（transformers版）"""
    try:
        img = Image.open(img_path).convert('RGB')
        inputs = processor(images=img, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
    except Exception as e:
        return None, None, None

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        # DINOv3 (vitb16): [CLS(0), REG1-4(1-4), PATCH1-196(5-200)] = 201 tokens

        # 最終層からCLSトークン
        final_hidden = outputs.last_hidden_state  # (1, 201, 768)
        cls_final = final_hidden[:, 0, :]  # (1, 768)

        # 中間層からパッチトークンとCLS
        mid_hidden = outputs.hidden_states[mid_layer + 1]  # +1: index 0 is initial embedding
        mid_patches = mid_hidden[:, 5:5+196, :]  # (1, 196, 768) - skip CLS and REG tokens
        mid_cls = mid_hidden[:, 0, :]  # (1, 768)

    return cls_final, mid_patches, mid_cls


def compute_new_4d_single(patches: torch.Tensor, mid_cls: torch.Tensor) -> np.ndarray:
    """単一サンプル用の新規4d特徴量を計算

    Args:
        patches: (1, 196, 768) or (196, 768)
        mid_cls: (1, 768) or (768,)

    Returns:
        (4,) 新規4d特徴量
    """
    device = patches.device

    if patches.dim() == 2:
        patches = patches.unsqueeze(0)
    if mid_cls.dim() == 1:
        mid_cls = mid_cls.unsqueeze(0)

    B, N, D = patches.shape

    pn = F.normalize(patches, dim=-1)
    cn = F.normalize(mid_cls, dim=-1)
    sim = torch.bmm(pn, pn.transpose(1, 2))

    # グラフ構築（閾値0.7）
    adj = ((sim > 0.7).float() * (1 - torch.eye(N, device=device)))
    degree = adj.sum(dim=-1)

    results = []

    # 1. local_efficiency
    adj_sq = torch.bmm(adj, adj)
    triangles = (adj_sq * adj).sum(dim=(1, 2)) / 6
    possible = (degree * (degree - 1) / 2).sum(dim=-1)
    local_eff = triangles / (possible + 1e-8)
    results.append(local_eff.item())

    # 2. corner_coherence
    corners = [0, 13, 182, 195]
    corner_sims = []
    for c1 in range(len(corners)):
        for c2 in range(c1 + 1, len(corners)):
            corner_sims.append(sim[0, corners[c1], corners[c2]].item())
    results.append(np.mean(corner_sims))

    # 3. edge_interior_gap
    edge_idx = list(range(14)) + list(range(14, 182, 14)) + \
               list(range(27, 196, 14)) + list(range(182, 196))
    edge_idx = list(set(edge_idx))
    interior_idx = [i for i in range(196) if i not in edge_idx]

    edge_mean = sim[0, edge_idx, :][:, edge_idx].mean().item()
    interior_mean = sim[0, interior_idx, :][:, interior_idx].mean().item()
    results.append(interior_mean - edge_mean)

    # 4. cls_sim_center_bias
    cls_sims = torch.bmm(pn, cn.unsqueeze(-1)).squeeze(-1)
    cls_grid = cls_sims.view(B, 14, 14)

    coords = torch.stack(torch.meshgrid(
        torch.arange(14, device=device, dtype=torch.float32),
        torch.arange(14, device=device, dtype=torch.float32),
        indexing='ij'
    ), dim=-1)
    center = torch.tensor([6.5, 6.5], device=device)
    dist_from_center = ((coords - center) ** 2).sum(dim=-1).sqrt()
    dist_flat = dist_from_center.flatten()
    cls_flat = cls_grid[0].flatten()

    r = torch.corrcoef(torch.stack([dist_flat, cls_flat]))[0, 1]
    results.append(r.item() if not torch.isnan(r) else 0)

    return np.array(results, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=500, help='テストする最大サンプル数')
    parser.add_argument('--only-false-neg', action='store_true', help='False negative (ai_prob<0.5) のみ')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # モデルロード
    print("Loading 15d 2-head model...")
    head_a = nn.Linear(768, 1)
    head_a.load_state_dict(torch.load(MODEL_DIR / "head_a.pt", map_location='cpu', weights_only=False))
    head_a.to(device).eval()

    head_b = nn.Linear(15, 1)
    head_b.load_state_dict(torch.load(MODEL_DIR / "head_b.pt", map_location='cpu', weights_only=False))
    head_b.to(device).eval()

    with open(MODEL_DIR / "best_alpha.json") as f:
        best_alpha = json.load(f)['alpha']
    print(f"Best α: {best_alpha}")

    norm_stats = torch.load(MODEL_DIR / "norm_stats.pt", map_location='cpu', weights_only=False)
    cls_mean = torch.tensor(norm_stats['cls_mean'], device=device)
    cls_std = torch.tensor(norm_stats['cls_std'], device=device)
    stats_mean = torch.tensor(norm_stats['stats_mean'], device=device)
    stats_std = torch.tensor(norm_stats['stats_std'], device=device)
    print(f"Norm stats: CLS={cls_mean.shape[0]}d, Stats={stats_mean.shape[0]}d")

    # DINOv3ロード
    print("Loading DINOv3...")
    dino, processor = load_dinov3()
    dino = dino.to(device)

    # ハードネガCSV読み込み
    print(f"Loading hard negatives from {HARDNEG_CSV}...")
    df = pd.read_csv(HARDNEG_CSV)

    if args.only_false_neg:
        df = df[df['ai_prob'] < 0.5]
        print(f"Filtered to false negatives: {len(df)} samples")

    if args.limit and len(df) > args.limit:
        df = df.head(args.limit)

    print(f"Testing {len(df)} samples...")

    results = []
    old_correct = 0
    new_correct = 0

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        img_path = Path(row['path'])
        old_prob = row['ai_prob']

        if not img_path.exists():
            continue

        # DINOv3特徴抽出
        cls_final, mid_patches, mid_cls = get_dino_features(dino, processor, img_path, device)
        if cls_final is None:
            continue

        # v3統計量（torch tensor のまま渡す！）
        stats_v3 = compute_patch_stats_v3(mid_patches, mid_cls)  # (1, 34)

        # 11d抽出
        stats_11d = stats_v3[0, KEEP_IDX_11D]  # (11,)

        # 新規5d計算
        new_4d = compute_new_4d_single(mid_patches, mid_cls)
        new_4d = torch.tensor(new_4d, device=device)

        # 15d結合
        stats_15d = torch.cat([stats_11d, new_4d])  # (15,)

        # 正規化
        cls_norm = (cls_final - cls_mean) / (cls_std + 1e-8)
        stats_15d_norm = (stats_15d - stats_mean) / (stats_std + 1e-8)

        # 推論
        with torch.no_grad():
            logit_a = head_a(cls_norm)  # (1, 1)
            logit_b = head_b(stats_15d_norm.unsqueeze(0))  # (1, 1)
            logit_combined = (1 - best_alpha) * logit_a + best_alpha * logit_b
            prob = torch.sigmoid(logit_combined).item()

        results.append({
            'path': str(img_path),
            'old_prob': old_prob,
            'new_prob': prob,
            'improved': prob > old_prob,
        })

        # 正解率（AI画像なので prob > 0.5 が正解）
        if old_prob > 0.5:
            old_correct += 1
        if prob > 0.5:
            new_correct += 1

    # 結果サマリ
    n = len(results)
    print(f"\n=== Results ({n} samples) ===")
    print(f"Old model accuracy: {old_correct}/{n} ({100*old_correct/n:.1f}%)")
    print(f"New 15d accuracy:   {new_correct}/{n} ({100*new_correct/n:.1f}%)")

    improved_count = sum(1 for r in results if r['improved'])
    print(f"Improved samples: {improved_count}/{n} ({100*improved_count/n:.1f}%)")

    # スコア変化統計
    old_probs = [r['old_prob'] for r in results]
    new_probs = [r['new_prob'] for r in results]
    print(f"\nOld prob: mean={np.mean(old_probs):.3f}, std={np.std(old_probs):.3f}")
    print(f"New prob: mean={np.mean(new_probs):.3f}, std={np.std(new_probs):.3f}")
    print(f"Avg improvement: {np.mean(new_probs) - np.mean(old_probs):.3f}")

    # 結果CSV保存
    out_path = Path("/home/techne/aicheckers/logs/hardneg_15d_test.csv")
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
