#!/usr/bin/env python3
"""
保存済みembeddingsからLinear Probe分類器を学習
"""
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

EMBEDDINGS_DIR = Path("/home/techne/aicheckers/embeddings")
OUTPUT_PATH = Path("/home/techne/aicheckers/models/dinov3_classifier.pt")

# AIカテゴリ（1=AI）- 全データ
AI_CATEGORIES = [
    "illustrious_ai",
    "pony_ai",
    "sdxl10_ai",
    "sd15_ai",
    "other_ai",
    "flux1d_ai",
    "novelai_ai",
]

# Realカテゴリ（0=Real）
REAL_CATEGORIES = [
    "danbooru_real",
]


def load_embeddings():
    """全embeddingsを読み込んで結合"""
    ai_embeddings = []
    real_embeddings = []

    print("Loading AI embeddings...")
    for cat in AI_CATEGORIES:
        npy_path = EMBEDDINGS_DIR / f"{cat}.npy"
        if npy_path.exists():
            emb = np.load(npy_path)
            ai_embeddings.append(emb)
            print(f"  {cat}: {emb.shape[0]} samples")
        else:
            print(f"  {cat}: NOT FOUND")

    print("\nLoading Real embeddings...")
    for cat in REAL_CATEGORIES:
        npy_path = EMBEDDINGS_DIR / f"{cat}.npy"
        if npy_path.exists():
            emb = np.load(npy_path)
            real_embeddings.append(emb)
            print(f"  {cat}: {emb.shape[0]} samples")
        else:
            print(f"  {cat}: NOT FOUND")

    # 結合
    ai_all = np.concatenate(ai_embeddings, axis=0)
    real_all = np.concatenate(real_embeddings, axis=0)

    print(f"\nTotal AI samples: {ai_all.shape[0]}")
    print(f"Total Real samples: {real_all.shape[0]}")

    # ラベル作成
    ai_labels = np.ones(ai_all.shape[0])
    real_labels = np.zeros(real_all.shape[0])

    X = np.concatenate([ai_all, real_all], axis=0)
    y = np.concatenate([ai_labels, real_labels], axis=0)

    return X, y


def train_classifier(X, y, epochs=30, lr=0.001):
    """Linear Probe分類器を学習（元のアーキテクチャ）"""
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTraining on {device}")

    # Shuffle
    perm = np.random.permutation(len(X))
    X, y = X[perm], y[perm]

    # Train/Val split (90/10)
    split_idx = int(len(X) * 0.9)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}")
    print(f"Class balance - Real: {(y_train == 0).sum()}, AI: {(y_train == 1).sum()}")

    # Tensors (long for CrossEntropyLoss)
    X_train = torch.FloatTensor(X_train).to(device)
    y_train = torch.LongTensor(y_train.astype(int)).to(device)
    X_val = torch.FloatTensor(X_val).to(device)
    y_val = torch.LongTensor(y_val.astype(int)).to(device)

    # DataLoaders
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64)

    # Model - 元のアーキテクチャ
    model = nn.Linear(768, 2).to(device)
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

    # Save best model
    model.load_state_dict(best_state)
    return model, best_acc


def main():
    print("=" * 50)
    print("Linear Probe Training from Saved Embeddings")
    print("=" * 50)

    # Backup existing model first
    if OUTPUT_PATH.exists():
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = OUTPUT_PATH.with_name(f"dinov3_classifier_backup_{timestamp}.pt")
        import shutil
        shutil.copy(OUTPUT_PATH, backup_path)
        print(f"Existing model backed up to {backup_path}")

    # Load embeddings
    X, y = load_embeddings()

    # Train
    model, best_acc = train_classifier(X, y)  # epochs=30 (default)

    # Save (checkpoint形式: バックエンドが checkpoint["classifier"] を期待)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "classifier": model.state_dict(),
        "val_acc": best_acc
    }, OUTPUT_PATH)
    print(f"\nModel saved to {OUTPUT_PATH} (val_acc: {best_acc*100:.2f}%)")


if __name__ == "__main__":
    main()
