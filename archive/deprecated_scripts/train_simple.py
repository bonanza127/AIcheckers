#!/usr/bin/env python3
"""
オリジナルアルゴリズムでの学習（VAT/EM/Consistencyなし）
"""
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

EMBEDDINGS_DIR = Path("/home/techne/aicheckers/embeddings")

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


def train_classifier(X, y, epochs=20, lr=0.001):
    """オリジナルのシンプルなLinear Probe学習"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTraining on {device}")
    print(f"Epochs: {epochs}, LR: {lr}")

    # Shuffle
    perm = np.random.permutation(len(X))
    X, y = X[perm], y[perm]

    # Train/Val split (90/10)
    split_idx = int(len(X) * 0.9)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}")
    print(f"Class balance - Real: {(y_train == 0).sum()}, AI: {(y_train == 1).sum()}")

    # Tensors
    X_train = torch.FloatTensor(X_train).to(device)
    y_train = torch.LongTensor(y_train.astype(int)).to(device)
    X_val = torch.FloatTensor(X_val).to(device)
    y_val = torch.LongTensor(y_val.astype(int)).to(device)

    # DataLoaders
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64)

    # Model - オリジナルアーキテクチャ
    model = nn.Linear(768, 2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

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

    # Load best model
    model.load_state_dict(best_state)
    return model, best_acc


def test_category_accuracy(model, device):
    """各カテゴリごとのAI判定率を確認"""
    print("\n" + "="*50)
    print("Category-wise AI Detection Rate")
    print("="*50)

    model.eval()

    for cat in AI_CATEGORIES:
        npy_path = EMBEDDINGS_DIR / f"{cat}.npy"
        if not npy_path.exists():
            continue

        emb = np.load(npy_path)
        X = torch.FloatTensor(emb).to(device)

        with torch.no_grad():
            logits = model(X)
            preds = logits.argmax(dim=1)
            ai_rate = (preds == 1).float().mean().item() * 100

        status = "⚠️" if ai_rate < 90 else "✓"
        print(f"  {cat}: {ai_rate:.1f}% {status}")


def main():
    print("=" * 50)
    print("Simple Linear Probe Training (Original Algorithm)")
    print("=" * 50)

    # Load embeddings
    X, y = load_embeddings()

    # Train
    model, best_acc = train_classifier(X, y, epochs=20)

    # Test category accuracy
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_category_accuracy(model, device)

    # Save
    output_path = Path("/home/techne/aicheckers/models/dinov3_classifier_simple.pt")
    torch.save({
        "classifier": model.state_dict(),
        "val_acc": best_acc
    }, output_path)
    print(f"\nModel saved to {output_path}")


if __name__ == "__main__":
    main()
