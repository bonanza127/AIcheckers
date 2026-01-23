#!/usr/bin/env python3
"""
Scan a directory of images and score AI probability using local DINOv3 + classifier.
Outputs a CSV sorted by score ascending (hard negatives first).
"""
import argparse
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

# CPU並列処理の上限
CPU_WORKERS = 6

PROJECT_ROOT = Path("/home/techne/aicheckers")
MODEL_DIR = PROJECT_ROOT / "models" / "dinov3-vitb16"
CKPT_PATH = PROJECT_ROOT / "models" / "dinov3_classifier.pt"
MID_LAYER_INDEX = 6

sys.path.insert(0, str(PROJECT_ROOT))
from lib.patch_stats import compute_patch_stats_v2
from lib.extra_stats import compute_extra_stats
from lib.boundary_stats import compute_boundary_stats


def list_images(root: Path):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.webp")
    images = []
    for ext in exts:
        images.extend(root.rglob(ext))
    return sorted(images)


def main():
    parser = argparse.ArgumentParser(description="Scan hard negatives directory")
    parser.add_argument("--dir", type=Path, required=True, help="Input image directory")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size (GPU-safe)")
    parser.add_argument("--out", type=Path, required=True, help="Output CSV path")
    parser.add_argument("--max-images", type=int, default=0, help="Limit images (0 = all)")
    parser.add_argument("--clear-every", type=int, default=200, help="CUDA cache clear interval")
    args = parser.parse_args()

    if not args.dir.exists():
        print(f"Folder not found: {args.dir}")
        return 1

    if not torch.cuda.is_available():
        print("CUDA not available; aborting (GPU required).")
        return 2

    device = torch.device("cuda")
    print(f"Device: {device}")

    images = list_images(args.dir)
    if args.max_images and args.max_images > 0:
        images = images[: args.max_images]
    print(f"Found {len(images)} images in {args.dir}")
    if not images:
        return 0

    processor = AutoImageProcessor.from_pretrained(str(MODEL_DIR))
    model = AutoModel.from_pretrained(str(MODEL_DIR)).to(device).eval()

    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    input_dim = ckpt.get("input_dim", 768)
    classifier = torch.nn.Linear(input_dim, 2).to(device)
    classifier.load_state_dict(ckpt["classifier"])
    classifier.eval()

    feature_mean = ckpt.get("feature_mean", None)
    feature_std = ckpt.get("feature_std", None)
    if feature_mean is not None and feature_std is not None:
        feat_mean = torch.as_tensor(feature_mean, device=device)
        feat_std = torch.as_tensor(feature_std, device=device)
    else:
        feat_mean = feat_std = None

    # 777d構成用インデックス
    patch_indices = ckpt.get("patch_indices", None)
    extra_indices = ckpt.get("extra_indices", None)
    boundary_indices = ckpt.get("boundary_indices", None)
    use_777d = patch_indices is not None and extra_indices is not None and boundary_indices is not None
    print(f"Model: {input_dim}d, 777d mode: {use_777d}")

    scores = []
    failed = 0
    total_batches = (len(images) + args.batch_size - 1) // args.batch_size

    for batch_idx, start in enumerate(range(0, len(images), args.batch_size)):
        batch_paths = images[start : start + args.batch_size]
        batch_imgs = []
        keep_paths = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                batch_imgs.append(img)
                keep_paths.append(p)
            except Exception:
                failed += 1
        if not batch_imgs:
            continue

        inputs = processor(images=batch_imgs, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = model(**inputs, output_hidden_states=True)
            hidden = outputs.last_hidden_state
            cls = hidden[:, 0, :]
            if input_dim > 768:
                mid_hidden = outputs.hidden_states[MID_LAYER_INDEX + 1]
                patch_embeddings = mid_hidden[:, 5:5 + 196, :]
                patch_stats = compute_patch_stats_v2(patch_embeddings)

                if use_777d:
                    # 777d: CPU並列でextra/boundary statsを計算
                    batch_rgb = [np.array(img) for img in batch_imgs]
                    with ThreadPoolExecutor(max_workers=CPU_WORKERS) as executor:
                        extra_results = list(executor.map(compute_extra_stats, batch_rgb))
                        boundary_results = list(executor.map(compute_boundary_stats, batch_rgb))

                    batch_features = []
                    for i in range(len(batch_imgs)):
                        patch_stats_np = patch_stats[i].cpu().numpy()
                        patch_sel = patch_stats_np[patch_indices]
                        extra_sel = extra_results[i][extra_indices]
                        boundary_sel = boundary_results[i][boundary_indices]
                        additional = np.concatenate([patch_sel, extra_sel, boundary_sel])
                        additional_tensor = torch.tensor(additional, dtype=torch.float32, device=device)
                        feat = torch.cat([cls[i], additional_tensor], dim=0)
                        batch_features.append(feat)
                    features = torch.stack(batch_features, dim=0)
                else:
                    features = torch.cat([cls, patch_stats], dim=1)
            else:
                features = cls
            if feat_mean is not None:
                features = (features - feat_mean) / feat_std
            logits = classifier(features)
            probs = F.softmax(logits, dim=1)
            ai_scores = (probs[:, 1] * 100).detach().cpu().numpy()

        for p, s in zip(keep_paths, ai_scores):
            scores.append((str(p), float(s)))

        # 進捗表示（10バッチごと）
        if (batch_idx + 1) % 10 == 0 or batch_idx == total_batches - 1:
            print(f"  Progress: {batch_idx + 1}/{total_batches} batches ({len(scores)} images)")

        if args.clear_every > 0 and batch_idx % args.clear_every == 0:
            torch.cuda.empty_cache()

    if not scores:
        print("No scores computed.")
        return 0

    scores_np = np.array([s for _, s in scores], dtype=np.float32)
    print(f"Processed: {len(scores)} images, Failed: {failed}")

    # 範囲別カウント
    print(f"\n{'範囲':>12} | {'枚数':>8} | {'割合':>8}")
    print("-" * 35)
    ranges = [(80, 100), (60, 80), (40, 60), (20, 40), (0, 20)]
    for low, high in ranges:
        count = ((scores_np >= low) & (scores_np < high)).sum()
        if high == 100:
            count = (scores_np >= low).sum()  # 100%も含める
        pct = count / len(scores_np) * 100
        print(f"{low:>5}-{high:>3}% | {count:>7}枚 | {pct:>6.1f}%")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        f.write("path,ai_score\n")
        for p, s in sorted(scores, key=lambda x: x[1]):
            f.write(f"\"{p}\",{s:.4f}\n")
    print(f"Saved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
