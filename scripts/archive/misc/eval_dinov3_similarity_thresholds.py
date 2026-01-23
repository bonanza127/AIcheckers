#!/usr/bin/env python3
"""
Evaluate DINOv3 cosine similarity thresholds using a dataset of AI images.

We compute:
- Positive pairs: original vs lightly augmented (resize + JPEG) version.
  This approximates "same image with minor perturbations".
- Negative pairs: random different-image pairs from the same dataset.

Thresholds are evaluated on TPR (positives) and FPR (negatives).
"""
import argparse
import io
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def list_images(root: Path) -> list[Path]:
    paths = [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    return sorted(paths)


def make_augmentation(image: Image.Image) -> Image.Image:
    width, height = image.size
    scale = 0.85
    resized = image.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.LANCZOS)
    restored = resized.resize((width, height), Image.LANCZOS)

    # Mild JPEG compression to simulate lossy re-encoding.
    buffer = io.BytesIO()
    restored.save(buffer, format="JPEG", quality=85)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def compute_embeddings(
    processor,
    model,
    paths: list[Path],
    device: str,
    batch_size: int,
) -> np.ndarray:
    embeddings = []
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i:i + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = processor(images=images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = model(**inputs)
            cls = outputs.last_hidden_state[:, 0, :]
            cls = torch.nn.functional.normalize(cls, dim=-1)
        embeddings.append(cls.cpu().numpy())
    return np.vstack(embeddings)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(a * b, axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/home/techne/aicheckers/data/novelai_combined"),
    )
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--neg-pairs", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/home/techne/aicheckers/models/dinov3-vitb16"),
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.85, 0.9, 0.95],
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if not args.data_dir.exists():
        raise SystemExit(f"Data dir not found: {args.data_dir}")
    if not args.model_path.exists():
        raise SystemExit(f"Model path not found: {args.model_path}")

    all_paths = list_images(args.data_dir)
    if not all_paths:
        raise SystemExit(f"No images found in {args.data_dir}")

    sample_count = min(args.num_samples, len(all_paths))
    sample_paths = random.sample(all_paths, sample_count)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Images: {sample_count} (from {len(all_paths)} total)")
    print(f"Model: {args.model_path}")

    processor = AutoImageProcessor.from_pretrained(str(args.model_path))
    model = AutoModel.from_pretrained(str(args.model_path), attn_implementation="eager")
    model.to(device)
    model.eval()

    # Original embeddings.
    emb_orig = compute_embeddings(processor, model, sample_paths, device, args.batch_size)

    # Augmented embeddings for positives (batched to keep memory stable).
    pos_sims = []
    for i in range(0, sample_count, args.batch_size):
        batch_paths = sample_paths[i:i + args.batch_size]
        aug_images = [make_augmentation(Image.open(p).convert("RGB")) for p in batch_paths]
        inputs = processor(images=aug_images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = model(**inputs)
            emb_aug = outputs.last_hidden_state[:, 0, :]
            emb_aug = torch.nn.functional.normalize(emb_aug, dim=-1).cpu().numpy()
        emb_slice = emb_orig[i:i + len(batch_paths)]
        pos_sims.extend(cosine_similarity(emb_slice, emb_aug).tolist())
    pos_sims = np.asarray(pos_sims)

    # Negative pairs: random different-image pairs.
    neg_sims = []
    for _ in range(args.neg_pairs):
        i, j = random.sample(range(sample_count), 2)
        neg_sims.append(float(np.dot(emb_orig[i], emb_orig[j])))
    neg_sims = np.asarray(neg_sims)

    print("\nPositive pairs (augmented) similarity:")
    print(f"  mean={pos_sims.mean():.4f}  p5={np.percentile(pos_sims, 5):.4f}  p1={np.percentile(pos_sims, 1):.4f}")
    print("Negative pairs (different) similarity:")
    print(f"  mean={neg_sims.mean():.4f}  p95={np.percentile(neg_sims, 95):.4f}  p99={np.percentile(neg_sims, 99):.4f}")

    print("\nThreshold evaluation:")
    best = None
    for t in sorted(args.thresholds):
        tpr = float((pos_sims >= t).mean())
        fpr = float((neg_sims >= t).mean())
        print(f"  thr={t:.2f}  TPR={tpr*100:5.2f}%  FPR={fpr*100:5.2f}%")
        score = tpr - (fpr * 3.0)
        if best is None or score > best[0]:
            best = (score, t, tpr, fpr)

    if best:
        _, t, tpr, fpr = best
        print(f"\nRecommendation: thr={t:.2f} (TPR {tpr*100:.2f}%, FPR {fpr*100:.2f}%)")


if __name__ == "__main__":
    main()
