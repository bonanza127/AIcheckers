"""
DINOv3 AI Image Detector - Modal App
Linear Probe方式: 凍結backbone + 学習済み分類器
"""
import modal

app = modal.App("dinov3-detector")

# Volume for trained classifier
volume = modal.Volume.from_name("dinov3-embeddings", create_if_missing=True)
CLASSIFIER_PATH = "/embeddings/linear_classifier.pt"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.40.0",
        "Pillow>=10.0.0",
        "numpy>=1.24.0",
        "huggingface_hub>=0.20.0",
    )
)


@app.cls(
    image=image,
    gpu="T4",
    volumes={"/embeddings": volume},
    scaledown_window=60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
class DINOv3Detector:
    @modal.enter()
    def load_model(self):
        import torch
        from transformers import AutoImageProcessor, AutoModel
        from pathlib import Path
        import os

        # HuggingFace login for gated model
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            from huggingface_hub import login
            login(token=hf_token)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading DINOv3 on {self.device}...")

        # Load DINOv3 (gated model)
        model_name = "facebook/dinov3-vitb16-pretrain-lvd1689m"
        self.processor = AutoImageProcessor.from_pretrained(model_name, token=hf_token)
        self.model = AutoModel.from_pretrained(model_name, token=hf_token)
        self.model.to(self.device)
        self.model.eval()
        print("DINOv3 loaded!")

        # Load linear classifier if available
        classifier_path = Path(CLASSIFIER_PATH)
        if classifier_path.exists():
            checkpoint = torch.load(classifier_path, map_location=self.device)
            self.classifier = torch.nn.Linear(768, 2).to(self.device)
            self.classifier.load_state_dict(checkpoint["classifier"])
            self.classifier.eval()
            print("Linear classifier loaded!")
        else:
            print("WARNING: No classifier found. Run train_linear_probe() first.")
            self.classifier = None

        print("DINOv3 ready!")

    def extract_features(self, image_bytes: bytes):
        """Extract CLS token features from image"""
        import torch
        from PIL import Image
        import io

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use CLS token (first token)
            features = outputs.last_hidden_state[:, 0, :]

        return features

    @modal.method()
    def detect(self, image_bytes: bytes) -> dict:
        """
        Detect if image is AI-generated using Linear Probe.

        Returns:
            dict with probability, is_ai, confidence
        """
        import torch

        if self.classifier is None:
            return {
                "error": "Linear classifier not loaded",
                "probability": 0.5,
                "is_ai": False,
                "confidence": 0.0,
            }

        # Extract features
        features = self.extract_features(image_bytes)

        # Classify
        with torch.no_grad():
            logits = self.classifier(features)
            probs = torch.softmax(logits, dim=1)[0]
            # Assuming class 0 = real, class 1 = AI
            ai_prob = probs[1].item()

        return {
            "probability": ai_prob,
            "is_ai": ai_prob > 0.5,
            "confidence": abs(ai_prob - 0.5) * 2,
        }


@app.function(
    image=image,
    gpu="T4",
    volumes={"/embeddings": volume},
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def train_linear_probe(
    ai_embeddings: bytes,
    real_embeddings: bytes,
    epochs: int = 10,
    lr: float = 0.001,
):
    """
    Train linear classifier on pre-extracted embeddings.

    Args:
        ai_embeddings: Serialized tensor of AI image embeddings
        real_embeddings: Serialized tensor of real image embeddings
        epochs: Number of training epochs
        lr: Learning rate
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    import io

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # Load embeddings
    ai_emb = torch.load(io.BytesIO(ai_embeddings))
    real_emb = torch.load(io.BytesIO(real_embeddings))

    print(f"AI embeddings: {ai_emb.shape}")
    print(f"Real embeddings: {real_emb.shape}")

    # Create dataset: label 0 = real, label 1 = AI
    X = torch.cat([real_emb, ai_emb], dim=0)
    y = torch.cat([
        torch.zeros(len(real_emb), dtype=torch.long),
        torch.ones(len(ai_emb), dtype=torch.long),
    ])

    # Shuffle
    perm = torch.randperm(len(X))
    X, y = X[perm], y[perm]

    # Split train/val (90/10)
    split_idx = int(len(X) * 0.9)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    print(f"Train: {len(X_train)}, Val: {len(X_val)}")

    # DataLoaders
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=64,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=64,
    )

    # Linear classifier
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

        val_acc = correct / total
        print(f"Epoch {epoch+1}/{epochs}: loss={train_loss/len(train_loader):.4f}, val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = classifier.state_dict()

    # Save best classifier
    torch.save({"classifier": best_state, "val_acc": best_val_acc}, CLASSIFIER_PATH)
    volume.commit()

    return {
        "best_val_acc": best_val_acc,
        "train_samples": len(X_train),
        "val_samples": len(X_val),
    }


@app.local_entrypoint()
def main(action: str = "test"):
    """
    Local entrypoint.

    Usage:
        modal run app.py --action test
    """
    if action == "test":
        print("Testing DINOv3 detector...")
        detector = DINOv3Detector()

        # Create dummy test image
        from PIL import Image
        import io
        img = Image.new("RGB", (224, 224), color="red")
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        result = detector.detect.remote(buf.getvalue())
        print(f"Result: {result}")
