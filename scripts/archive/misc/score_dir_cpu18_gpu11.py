#!/usr/bin/env python3
"""
Score a directory with CPU16 + GPU8 (CLS + stats) in realtime.

Uses:
  - CPU v2 stats (computed on-the-fly, 512 letterbox)
  - extra_stats (on original resolution)
  - boundary_stats (512 resized inside)
  - DINOv3 CLS + patch_stats_v3 + new4d
"""
import argparse
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from lib.patch_stats import compute_patch_stats_v3
from lib.extra_stats import compute_extra_stats
from lib.boundary_stats import compute_boundary_stats
from scripts.extract_cpu_stats_v2 import load_image as load_cpu_img
from scripts.extract_cpu_stats_v2 import extract_features as extract_cpu_v2

PROJECT_ROOT = Path("/home/techne/aicheckers")
MODEL_DIR = PROJECT_ROOT / "models" / "two_head_24d"
DINO_DIR = PROJECT_ROOT / "models" / "dinov3-vitb16"

# 元29d: [0, 1, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16]
# 削除: 11 (patch_vs_global_rank_entropy_gap), 13 (flat_ratio_variance_across_tiles)
CPU_V2_SELECT_IDX = [0, 1, 3, 4, 5, 6, 7, 8, 9, 12, 14, 15, 16]
EXTRA_SELECT_IDX = [5, 9]
BOUNDARY_SELECT_IDX = [3]
GPU_V3_IDX = [1, 3, 5, 17, 32]


def compute_new3_torch(mid_patches: torch.Tensor, mid_cls: torch.Tensor) -> torch.Tensor:
    """Compute new3d on device, returns (B, 3)."""
    pn = F.normalize(mid_patches, dim=-1)
    cn = F.normalize(mid_cls, dim=-1)
    sim = torch.bmm(pn, pn.transpose(1, 2))
    adj = ((sim > 0.7).float() * (1 - torch.eye(sim.shape[1], device=sim.device)))
    degree = adj.sum(dim=-1)

    # local_efficiency
    adj_sq = torch.bmm(adj, adj)
    triangles = (adj_sq * adj).sum(dim=(1, 2)) / 6
    possible = (degree * (degree - 1) / 2).sum(dim=-1)
    local_eff = triangles / (possible + 1e-8)

    # edge_interior_gap
    edge_idx = list(range(14)) + list(range(14, 182, 14)) + \
               list(range(27, 196, 14)) + list(range(182, 196))
    edge_idx = list(set(edge_idx))
    interior_idx = [i for i in range(196) if i not in edge_idx]
    edge_mean = sim[:, edge_idx, :][:, :, edge_idx].mean(dim=(1, 2))
    interior_mean = sim[:, interior_idx, :][:, :, interior_idx].mean(dim=(1, 2))
    edge_gap = interior_mean - edge_mean

    # cls_sim_center_bias
    cls_sims = torch.bmm(pn, cn.unsqueeze(-1)).squeeze(-1)
    cls_grid = cls_sims.view(sim.shape[0], 14, 14)
    coords = torch.stack(torch.meshgrid(
        torch.arange(14, device=sim.device, dtype=torch.float32),
        torch.arange(14, device=sim.device, dtype=torch.float32),
        indexing="ij"
    ), dim=-1)
    center = torch.tensor([6.5, 6.5], device=sim.device)
    dist_from_center = ((coords - center) ** 2).sum(dim=-1).sqrt().flatten()
    center_corr = []
    for b in range(cls_grid.shape[0]):
        cls_flat = cls_grid[b].flatten()
        r = torch.corrcoef(torch.stack([dist_from_center, cls_flat]))[0, 1]
        center_corr.append(0.0 if torch.isnan(r) else r)
    center_corr = torch.stack(center_corr)

    return torch.stack([local_eff, edge_gap, center_corr], dim=1)


def list_images(root: Path):
    exts = (".jpg", ".jpeg", ".png", ".webp")
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in exts])


def main():
    parser = argparse.ArgumentParser(description="Score directory with CPU18 + GPU11 model (realtime).")
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-images", type=int, default=0)
    args = parser.parse_args()

    if not args.dir.exists():
        raise SystemExit(f"Not found: {args.dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    processor = AutoImageProcessor.from_pretrained(str(DINO_DIR))
    model = AutoModel.from_pretrained(str(DINO_DIR)).to(device).eval()

    head_a = torch.nn.Linear(768, 1)
    head_b = torch.nn.Linear(24, 1)
    head_a.load_state_dict(torch.load(MODEL_DIR / "head_a.pt", map_location="cpu", weights_only=False))
    head_b.load_state_dict(torch.load(MODEL_DIR / "head_b.pt", map_location="cpu", weights_only=False))
    head_a.to(device).eval()
    head_b.to(device).eval()

    best_alpha = json.loads((MODEL_DIR / "best_alpha.json").read_text())["alpha"]
    ns = torch.load(MODEL_DIR / "norm_stats.pt", map_location="cpu", weights_only=False)
    cls_mean = torch.tensor(ns["cls_mean"], dtype=torch.float32, device=device)
    cls_std = torch.tensor(ns["cls_std"], dtype=torch.float32, device=device)
    stats_mean = torch.tensor(ns["stats_mean"], dtype=torch.float32, device=device)
    stats_std = torch.tensor(ns["stats_std"], dtype=torch.float32, device=device)

    paths = list_images(args.dir)
    if args.max_images:
        paths = paths[: args.max_images]
    print(f"Found {len(paths)} images")

    scores = []
    failed = 0

    def process_single_image(p):
        """単一画像のCPU特徴量を計算（並列化用）"""
        try:
            img = Image.open(p).convert("RGB")
            img_array = np.array(img)
            cpu_img, cpu_mask = load_cpu_img(p)
            cpu_feats = extract_cpu_v2(cpu_img, cpu_mask)[CPU_V2_SELECT_IDX]
            extra = compute_extra_stats(img_array)[EXTRA_SELECT_IDX]
            boundary = compute_boundary_stats(img_array)[BOUNDARY_SELECT_IDX]
            return (img, img_array, cpu_feats, extra, boundary, None)
        except Exception as e:
            return (None, None, None, None, None, e)

    total_batches = (len(paths) + args.batch_size - 1) // args.batch_size
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for i in tqdm(range(0, len(paths), args.batch_size), total=total_batches, desc="Processing"):
            batch_paths = paths[i:i + args.batch_size]
            imgs = []
            originals = []
            cpu_v2_list = []
            extra_list = []
            boundary_list = []

            # 並列でCPU特徴量計算
            futures = [executor.submit(process_single_image, p) for p in batch_paths]
            for future in futures:
                img, img_array, cpu_feats, extra, boundary, err = future.result()
                if err is not None:
                    failed += 1
                    continue
                imgs.append(img)
                originals.append(img_array)
                cpu_v2_list.append(cpu_feats)
                extra_list.append(extra)
                boundary_list.append(boundary)

            if not imgs:
                continue

            inputs = processor(images=imgs, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                final_hidden = outputs.last_hidden_state
                cls_final = final_hidden[:, 0, :]
                mid_hidden = outputs.hidden_states[7]
                mid_patches = mid_hidden[:, 5:5 + 196, :]
                mid_cls = mid_hidden[:, 0, :]

                stats_v3 = compute_patch_stats_v3(mid_patches, mid_cls)
                gpu_11d = stats_v3[:, GPU_V3_IDX]
                new3 = compute_new3_torch(mid_patches, mid_cls)
                gpu = torch.cat([gpu_11d, new3], dim=1)

                cpu = torch.tensor(
                    np.concatenate([
                        np.stack(cpu_v2_list, axis=0),
                        np.stack(extra_list, axis=0),
                        np.stack(boundary_list, axis=0),
                    ], axis=1),
                    dtype=torch.float32,
                    device=device,
                )

                stats = torch.cat([cpu, gpu], dim=1)

                cls_n = (cls_final - cls_mean) / (cls_std + 1e-8)
                stats_n = (stats - stats_mean) / (stats_std + 1e-8)
                logit = head_a(cls_n).squeeze(1) + best_alpha * head_b(stats_n).squeeze(1)
                probs = torch.sigmoid(logit).detach().cpu().numpy()

            for p, s in zip(batch_paths, probs):
                scores.append((str(p), float(s)))

    vals = np.array([s for _, s in scores], dtype=np.float32)
    print(f"Processed: {len(vals)}, Failed: {failed}")
    for lo, hi in [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]:
        if hi < 1.0:
            count = ((vals >= lo) & (vals < hi)).sum()
        else:
            count = ((vals >= lo) & (vals <= hi)).sum()
        pct = count / len(vals) * 100 if len(vals) else 0.0
        print(f"{int(lo*100):>2d}-{int(hi*100):>2d}% : {count:5d} ({pct:5.1f}%)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        f.write("path,ai_prob\n")
        for p, s in scores:
            f.write(f'"{p}",{s:.6f}\n')
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
