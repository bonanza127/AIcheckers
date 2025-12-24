#!/usr/bin/env python3
"""
DINOv3 Linear Probe Classifier Training
embeddingsディレクトリの全.npyファイルを使って分類器を再学習
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
import argparse

EMBEDDINGS_DIR = Path("/home/techne/aicheckers/embeddings")
OUTPUT_PATH = Path("/home/techne/aicheckers/models/dinov3_classifier.pt")


def load_embeddings():
    """
    embeddingsディレクトリから全npyファイルを読み込み
    ファイル名規則: *_ai.npy → AI (label=1), *_real.npy → Real (label=0)
    """
    ai_embeddings = []
    real_embeddings = []

    for npy_file in EMBEDDINGS_DIR.glob("*.npy"):
        name = npy_file.stem.lower()
        emb = np.load(npy_file)

        if "_ai" in name:
            ai_embeddings.append(emb)
            print(f"  AI: {npy_file.name} ({len(emb)} samples)")
        elif "_real" in name:
            real_embeddings.append(emb)
            print(f"  Real: {npy_file.name} ({len(emb)} samples)")
        else:
            print(f"  Skip: {npy_file.name} (unknown category)")

    if not ai_embeddings:
        raise ValueError("No AI embeddings found (*_ai.npy)")
    if not real_embeddings:
        raise ValueError("No Real embeddings found (*_real.npy)")

    ai_all = np.concatenate(ai_embeddings, axis=0)
    real_all = np.concatenate(real_embeddings, axis=0)

    return ai_all, real_all


def train_classifier(ai_emb: np.ndarray, real_emb: np.ndarray,
                     epochs: int = 20, batch_size: int = 64, lr: float = 0.001,
                     balance: bool = True):
    """Linear Probe分類器を学習"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nTraining on {device}")

    # バランス調整（オプション）
    if balance:
        min_count = min(len(ai_emb), len(real_emb))
        print(f"Balancing: using {min_count} samples per class")

        # ランダムサンプリング
        np.random.seed(42)
        ai_idx = np.random.choice(len(ai_emb), min_count, replace=False)
        real_idx = np.random.choice(len(real_emb), min_count, replace=False)
        ai_emb = ai_emb[ai_idx]
        real_emb = real_emb[real_idx]

    # Tensor化
    X = torch.cat([
        torch.from_numpy(real_emb).float(),
        torch.from_numpy(ai_emb).float()
    ], dim=0)

    y = torch.cat([
        torch.zeros(len(real_emb), dtype=torch.long),  # Real = 0
        torch.ones(len(ai_emb), dtype=torch.long),     # AI = 1
    ])

    # シャッフル
    perm = torch.randperm(len(X))
    X, y = X[perm], y[perm]

    # Train/Val分割 (90/10)
    split_idx = int(len(X) * 0.9)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    print(f"Train: {len(X_train)}, Val: {len(X_val)}")
    print(f"Train class balance - Real: {(y_train == 0).sum().item()}, AI: {(y_train == 1).sum().item()}")

    # DataLoaders
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)

    # 分類器（768次元 → 2クラス）
    classifier = nn.Linear(768, 2).to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    best_state = None

    for epoch in range(epochs):
        # Train
        classifier.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = classifier(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validate
        classifier.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = classifier(batch_x)
                preds = logits.argmax(dim=1)
                correct += (preds == batch_y).sum().item()
                total += len(batch_y)

        val_acc = correct / total if total > 0 else 0
        avg_loss = train_loss / len(train_loader)

        # 進捗表示
        marker = " *" if val_acc > best_val_acc else ""
        print(f"Epoch {epoch+1:2d}/{epochs}: loss={avg_loss:.4f}, val_acc={val_acc:.4f}{marker}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = classifier.state_dict().copy()

    return best_state, best_val_acc


def main():
    parser = argparse.ArgumentParser(description="Train DINOv3 Linear Probe classifier")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--no-balance", action="store_true", help="Don't balance classes")
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH), help="Output path")
    args = parser.parse_args()

    print("=" * 50)
    print("DINOv3 Linear Probe Classifier Training")
    print("=" * 50)

    # Embedding読み込み
    print(f"\nLoading embeddings from {EMBEDDINGS_DIR}...")
    ai_emb, real_emb = load_embeddings()

    print(f"\nTotal AI samples: {len(ai_emb)}")
    print(f"Total Real samples: {len(real_emb)}")

    # 学習
    best_state, best_acc = train_classifier(
        ai_emb, real_emb,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        balance=not args.no_balance
    )

    # 保存
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "classifier": best_state,
        "val_acc": best_acc,
    }, output_path)

    print(f"\n" + "=" * 50)
    print(f"Best validation accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
    print(f"Saved to: {output_path}")
    print("=" * 50)

    print("\nTo apply changes:")
    print("  systemctl --user restart aicheckers-backend")


if __name__ == "__main__":
    main()
