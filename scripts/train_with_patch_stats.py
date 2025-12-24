#!/usr/bin/env python3
"""
CLS + パッチ統計量でLinear Probe分類器を学習

使い方:
  python scripts/train_with_patch_stats.py              # CLS + patch_stats (774次元)
  python scripts/train_with_patch_stats.py --cls-only   # CLSのみ (768次元、ベースライン)
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

EMBEDDINGS_DIR = Path("/home/techne/aicheckers/embeddings")
OUTPUT_PATH = Path("/home/techne/aicheckers/models/dinov3_classifier.pt")

# AIカテゴリ（1=AI）
AI_CATEGORIES = [
    "illustrious_ai",
    "pony_ai",
    "sdxl10_ai",
    "sd15_ai",
    "other_ai",
    "flux1d_ai",
    "novelai_ai",
    "pixai_ai",
    "novelai_aibooru_ai",  # aibooruのみ追加（品質良）
    # "novelai_pixiv_ai",   # 除外（人間の絵が多数混入）
    # "twitter_novelai_all_ai",  # 古い（重複多し）
    "novelai_combined_ai",  # dedup済みPixiv+Twitter (4,499枚)
]

# Realカテゴリ（0=Real）
REAL_CATEGORIES = [
    "danbooru_real",
]


def load_embeddings(use_patch_stats=True):
    """全embeddingsを読み込んで結合"""
    ai_embeddings = []
    real_embeddings = []

    print("Loading AI embeddings...")
    for cat in AI_CATEGORIES:
        cls_path = EMBEDDINGS_DIR / f"{cat}.npy"
        stats_path = EMBEDDINGS_DIR / f"{cat}_patch_stats.npy"

        if not cls_path.exists():
            print(f"  {cat}: CLS NOT FOUND, skipping")
            continue

        cls_emb = np.load(cls_path)

        if use_patch_stats:
            if not stats_path.exists():
                print(f"  {cat}: patch_stats NOT FOUND, skipping")
                continue
            stats = np.load(stats_path)
            # CLS + patch_stats を結合
            emb = np.concatenate([cls_emb, stats], axis=1)
            print(f"  {cat}: {emb.shape[0]} samples ({emb.shape[1]} dims)")
        else:
            emb = cls_emb
            print(f"  {cat}: {emb.shape[0]} samples")

        ai_embeddings.append(emb)

    print("\nLoading Real embeddings...")
    for cat in REAL_CATEGORIES:
        cls_path = EMBEDDINGS_DIR / f"{cat}.npy"
        stats_path = EMBEDDINGS_DIR / f"{cat}_patch_stats.npy"

        if not cls_path.exists():
            print(f"  {cat}: CLS NOT FOUND, skipping")
            continue

        cls_emb = np.load(cls_path)

        if use_patch_stats:
            if not stats_path.exists():
                print(f"  {cat}: patch_stats NOT FOUND, skipping")
                continue
            stats = np.load(stats_path)
            emb = np.concatenate([cls_emb, stats], axis=1)
            print(f"  {cat}: {emb.shape[0]} samples ({emb.shape[1]} dims)")
        else:
            emb = cls_emb
            print(f"  {cat}: {emb.shape[0]} samples")

        real_embeddings.append(emb)

    if not ai_embeddings or not real_embeddings:
        raise ValueError("No embeddings loaded!")

    # 結合
    ai_all = np.concatenate(ai_embeddings, axis=0)
    real_all = np.concatenate(real_embeddings, axis=0)

    print(f"\nTotal AI samples: {ai_all.shape[0]}")
    print(f"Total Real samples: {real_all.shape[0]}")
    print(f"Feature dimension: {ai_all.shape[1]}")

    # ラベル作成
    ai_labels = np.ones(ai_all.shape[0])
    real_labels = np.zeros(real_all.shape[0])

    X = np.concatenate([ai_all, real_all], axis=0)
    y = np.concatenate([ai_labels, real_labels], axis=0)

    return X, y


def train_classifier(X, y, epochs=30, lr=0.001):
    """Linear Probe分類器を学習"""
    from torch.utils.data import DataLoader, TensorDataset
    import torch.nn.functional as F

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTraining on {device}")

    # Shuffle
    perm = np.random.permutation(len(X))
    X, y = X[perm], y[perm]

    # Train/Val split (90/10)
    split_idx = int(len(X) * 0.9)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    input_dim = X_train.shape[1]
    print(f"Input dimension: {input_dim}")
    print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}")

    # Tensors
    X_train = torch.FloatTensor(X_train).to(device)
    y_train = torch.LongTensor(y_train.astype(int)).to(device)
    X_val = torch.FloatTensor(X_val).to(device)
    y_val = torch.LongTensor(y_val.astype(int)).to(device)

    # DataLoaders
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64)

    # Model
    model = nn.Linear(input_dim, 2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                logits = model(batch_x)
                preds = logits.argmax(dim=1)
                correct += (preds == batch_y).sum().item()
                total += len(batch_y)

        val_acc = correct / total if total > 0 else 0

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = model.state_dict().copy()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss/len(train_loader):.4f} - Val Acc: {val_acc*100:.2f}%")

    print(f"\nBest Validation Accuracy: {best_acc*100:.2f}%")

    model.load_state_dict(best_state)
    return model, best_acc, input_dim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cls-only", action="store_true", help="Use CLS only (no patch stats)")
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    use_patch_stats = not args.cls_only
    mode_str = "CLS + patch_stats (775 dims)" if use_patch_stats else "CLS only (768 dims)"

    print("=" * 50)
    print(f"Training: {mode_str}")
    print("=" * 50)

    # Load embeddings
    X, y = load_embeddings(use_patch_stats=use_patch_stats)

    # Train
    model, best_acc, input_dim = train_classifier(X, y, epochs=args.epochs)

    # Save
    output_path = OUTPUT_PATH if use_patch_stats else OUTPUT_PATH.with_name("dinov3_classifier_cls_only.pt")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "classifier": model.state_dict(),
        "val_acc": best_acc,
        "input_dim": input_dim,
        "use_patch_stats": use_patch_stats
    }, output_path)

    print(f"\nModel saved to {output_path}")
    print(f"Validation accuracy: {best_acc*100:.2f}%")


if __name__ == "__main__":
    main()
