"""
DINOv3 Linear Probe Training Script
ローカルで特徴量抽出 → Modal上でLinear Probe学習
"""
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import io

# Paths
ILLUSTRIOUS_DIR = Path("/home/techne/Downloads/animedl2m_dataset_release/civitai_subset/image/Illustrious")
REAL_IMAGES_DIR = Path("/home/techne/aicheckers/data/test_images/real")
DANBOORU_CACHE = Path("/home/techne/aicheckers/data/danbooru_real")

# HuggingFace token (for gated DINOv3)
HF_TOKEN = "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"


def download_real_images(num_images: int = 1000):
    """danbooru2024-sfwからreal imagesをダウンロード"""
    from huggingface_hub import hf_hub_download
    import tarfile

    DANBOORU_CACHE.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    existing = list(DANBOORU_CACHE.glob("*.jpg")) + list(DANBOORU_CACHE.glob("*.png"))
    if len(existing) >= num_images:
        print(f"Already have {len(existing)} real images")
        return list(DANBOORU_CACHE.iterdir())[:num_images]

    print(f"Downloading real images from danbooru2024-sfw...")
    source = "deepghs/danbooru2024-sfw"

    # Download tar files
    downloaded = 0
    tar_idx = 0
    while downloaded < num_images and tar_idx < 100:
        try:
            tar_path = hf_hub_download(
                repo_id=source,
                filename=f"images/{tar_idx:04d}.tar",
                repo_type="dataset",
            )

            # Extract images
            with tarfile.open(tar_path) as tar:
                for member in tar.getmembers():
                    if member.isfile() and member.name.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                        # Extract to cache dir
                        member.name = Path(member.name).name  # Remove path prefix
                        tar.extract(member, DANBOORU_CACHE)
                        downloaded += 1
                        if downloaded >= num_images:
                            break

            print(f"Extracted from tar {tar_idx}, total: {downloaded}")
            tar_idx += 1

        except Exception as e:
            print(f"Error downloading tar {tar_idx}: {e}")
            tar_idx += 1
            continue

    return list(DANBOORU_CACHE.iterdir())[:num_images]


def extract_embeddings_local(image_paths: list, device: str = "cuda"):
    """ローカルでDINOv3特徴量を抽出"""
    from transformers import AutoImageProcessor, AutoModel
    from huggingface_hub import login

    login(token=HF_TOKEN)

    model_name = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    print(f"Loading {model_name}...")

    processor = AutoImageProcessor.from_pretrained(model_name, token=HF_TOKEN)
    model = AutoModel.from_pretrained(model_name, token=HF_TOKEN)
    model.to(device)
    model.eval()

    features = []
    for p in tqdm(image_paths, desc="Extracting features"):
        try:
            img = Image.open(p).convert("RGB")
            inputs = processor(images=img, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                out = model(**inputs)
                feat = out.last_hidden_state[:, 0, :]  # CLS token
                features.append(feat.cpu())
        except Exception as e:
            print(f"Error {p}: {e}")

    return torch.cat(features, dim=0) if features else torch.empty(0, 768)


def train_locally(ai_emb: torch.Tensor, real_emb: torch.Tensor, epochs: int = 20):
    """ローカルでLinear Probe学習（GPUがあればローカルで完結）"""
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}")

    # Create dataset
    X = torch.cat([real_emb, ai_emb], dim=0)
    y = torch.cat([
        torch.zeros(len(real_emb), dtype=torch.long),
        torch.ones(len(ai_emb), dtype=torch.long),
    ])

    # Shuffle
    perm = torch.randperm(len(X))
    X, y = X[perm], y[perm]

    # Split
    split_idx = int(len(X) * 0.9)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    print(f"Train: {len(X_train)}, Val: {len(X_val)}")
    print(f"Class balance - Real: {(y_train == 0).sum().item()}, AI: {(y_train == 1).sum().item()}")

    # DataLoaders
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64)

    # Classifier
    classifier = nn.Linear(768, 2).to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    best_state = None

    for epoch in range(epochs):
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
        print(f"Epoch {epoch+1}/{epochs}: loss={train_loss/len(train_loader):.4f}, val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = classifier.state_dict()

    return best_state, best_val_acc


def upload_to_modal(classifier_state: dict, val_acc: float):
    """学習済み分類器をModal Volumeにアップロード"""
    import subprocess

    # Save locally
    local_path = Path("/tmp/linear_classifier.pt")
    torch.save({"classifier": classifier_state, "val_acc": val_acc}, local_path)
    print(f"Saved classifier to {local_path}")

    # Upload to Modal volume
    print("Uploading to Modal volume...")
    result = subprocess.run(
        ["modal", "volume", "put", "dinov3-embeddings", str(local_path), "linear_classifier.pt"],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        print("Upload successful!")
    else:
        print(f"Upload failed: {result.stderr}")
        print("You can manually upload with:")
        print(f"  modal volume put dinov3-embeddings {local_path} linear_classifier.pt")

    return local_path


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ai-dir", type=str, default=str(ILLUSTRIOUS_DIR))
    parser.add_argument("--real-dir", type=str, default=str(REAL_IMAGES_DIR))
    parser.add_argument("--max-ai", type=int, default=2000, help="Max AI images to use")
    parser.add_argument("--max-real", type=int, default=2000, help="Max real images to use")
    parser.add_argument("--download-real", action="store_true", help="Download real images from danbooru")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # Collect AI images
    ai_dir = Path(args.ai_dir)
    ai_images = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
        ai_images.extend(ai_dir.glob(ext))
    ai_images = ai_images[:args.max_ai]
    print(f"AI images: {len(ai_images)}")

    # Collect real images
    if args.download_real:
        real_images = download_real_images(args.max_real)
    else:
        real_dir = Path(args.real_dir)
        real_images = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
            real_images.extend(real_dir.glob(ext))
        real_images = real_images[:args.max_real]
    print(f"Real images: {len(real_images)}")

    if len(ai_images) == 0 or len(real_images) == 0:
        print("Error: Need both AI and real images")
        return

    # Balance dataset (use minimum of both)
    min_count = min(len(ai_images), len(real_images))
    print(f"Using {min_count} images per class for balanced training")
    ai_images = ai_images[:min_count]
    real_images = real_images[:min_count]

    # Extract embeddings
    print("\nExtracting AI image embeddings...")
    ai_emb = extract_embeddings_local(ai_images, args.device)

    print("\nExtracting real image embeddings...")
    real_emb = extract_embeddings_local(real_images, args.device)

    print(f"\nAI embeddings: {ai_emb.shape}")
    print(f"Real embeddings: {real_emb.shape}")

    # Train
    print("\nTraining Linear Probe...")
    best_state, best_acc = train_locally(ai_emb, real_emb, epochs=args.epochs)
    print(f"\nBest validation accuracy: {best_acc:.4f}")

    # Upload to Modal
    upload_to_modal(best_state, best_acc)

    print("\nDone! Deploy with: modal deploy app.py")


if __name__ == "__main__":
    main()
