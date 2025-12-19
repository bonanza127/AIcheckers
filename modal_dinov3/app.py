"""
DINOv3 AI Image Detector - Modal App
k-NN方式: 参照画像の特徴量との類似度で分類
"""
import modal

app = modal.App("dinov3-detector")

# Volume for reference embeddings
volume = modal.Volume.from_name("dinov3-embeddings", create_if_missing=True)
EMBEDDINGS_PATH = "/embeddings/reference_embeddings.pt"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.40.0",
        "Pillow>=10.0.0",
        "numpy>=1.24.0",
    )
)


@app.cls(
    image=image,
    gpu="T4",
    volumes={"/embeddings": volume},
    scaledown_window=60,
)
class DINOv3Detector:
    @modal.enter()
    def load_model(self):
        import torch
        from transformers import AutoImageProcessor, AutoModel
        from pathlib import Path

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading DINOv3 on {self.device}...")

        # Load DINOv3 or fallback to DINOv2
        model_name = "facebook/dinov3-vitb16-pretrain-lvd1689m"
        try:
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            print(f"Loaded DINOv3")
        except Exception as e:
            print(f"DINOv3 unavailable ({e}), using DINOv2")
            model_name = "facebook/dinov2-base"
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # Load reference embeddings if available
        embeddings_path = Path(EMBEDDINGS_PATH)
        if embeddings_path.exists():
            data = torch.load(embeddings_path, map_location=self.device)
            self.ai_embeddings = data["ai"].to(self.device)
            self.real_embeddings = data["real"].to(self.device)
            print(f"Loaded {len(self.ai_embeddings)} AI + {len(self.real_embeddings)} real reference embeddings")
        else:
            print("WARNING: No reference embeddings found. Run build_reference_embeddings() first.")
            self.ai_embeddings = None
            self.real_embeddings = None

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
            # Normalize for cosine similarity
            features = features / features.norm(dim=-1, keepdim=True)

        return features

    @modal.method()
    def detect(self, image_bytes: bytes) -> dict:
        """
        Detect if image is AI-generated using k-NN.

        Returns:
            dict with probability, is_ai, confidence
        """
        import torch

        if self.ai_embeddings is None or self.real_embeddings is None:
            return {
                "error": "Reference embeddings not loaded",
                "probability": 0.5,
                "is_ai": False,
                "confidence": 0.0,
            }

        # Extract query features
        query = self.extract_features(image_bytes)

        # Compute cosine similarities
        ai_sims = torch.mm(query, self.ai_embeddings.T)
        real_sims = torch.mm(query, self.real_embeddings.T)

        # k-NN: average top-k similarities
        k = min(10, len(self.ai_embeddings), len(self.real_embeddings))
        ai_score = ai_sims.topk(k).values.mean().item()
        real_score = real_sims.topk(k).values.mean().item()

        # Convert to probability
        # Higher similarity to AI refs = higher AI probability
        total = ai_score + real_score
        if total > 0:
            ai_prob = ai_score / total
        else:
            ai_prob = 0.5

        return {
            "probability": ai_prob,
            "is_ai": ai_prob > 0.5,
            "confidence": abs(ai_prob - 0.5) * 2,
            "ai_similarity": ai_score,
            "real_similarity": real_score,
        }

    @modal.method()
    def extract_and_save_embeddings(self, image_bytes_list: list, labels: list):
        """
        Extract embeddings from images and save to volume.

        Args:
            image_bytes_list: List of image bytes
            labels: List of labels (0=real, 1=ai)
        """
        import torch

        ai_features = []
        real_features = []

        for img_bytes, label in zip(image_bytes_list, labels):
            try:
                feat = self.extract_features(img_bytes)
                if label == 1:
                    ai_features.append(feat)
                else:
                    real_features.append(feat)
            except Exception as e:
                print(f"Error processing image: {e}")

        # Stack and save
        data = {
            "ai": torch.cat(ai_features, dim=0) if ai_features else torch.empty(0, 768),
            "real": torch.cat(real_features, dim=0) if real_features else torch.empty(0, 768),
        }

        torch.save(data, EMBEDDINGS_PATH)
        volume.commit()

        return {
            "ai_count": len(ai_features),
            "real_count": len(real_features),
            "saved_to": EMBEDDINGS_PATH,
        }


@app.function(image=image, gpu="T4", volumes={"/embeddings": volume}, timeout=1800)
def build_reference_embeddings(ai_image_paths: list, real_image_paths: list):
    """
    Build reference embeddings from local image files.

    Args:
        ai_image_paths: List of paths to AI-generated images
        real_image_paths: List of paths to real images
    """
    import torch
    from transformers import AutoImageProcessor, AutoModel
    from PIL import Image
    from pathlib import Path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model_name = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    def extract_batch(paths):
        features = []
        for p in paths:
            try:
                img = Image.open(p).convert("RGB")
                inputs = processor(images=img, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    out = model(**inputs)
                    feat = out.last_hidden_state[:, 0, :]
                    feat = feat / feat.norm(dim=-1, keepdim=True)
                    features.append(feat.cpu())
            except Exception as e:
                print(f"Error: {p}: {e}")
        return torch.cat(features, dim=0) if features else torch.empty(0, 768)

    print(f"Processing {len(ai_image_paths)} AI images...")
    ai_emb = extract_batch(ai_image_paths)

    print(f"Processing {len(real_image_paths)} real images...")
    real_emb = extract_batch(real_image_paths)

    # Save
    data = {"ai": ai_emb, "real": real_emb}
    torch.save(data, EMBEDDINGS_PATH)
    volume.commit()

    return {
        "ai_embeddings": len(ai_emb),
        "real_embeddings": len(real_emb),
    }


@app.local_entrypoint()
def main(action: str = "test"):
    """
    Local entrypoint.

    Usage:
        modal run app.py --action test
        modal run app.py --action build
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

    elif action == "build":
        print("Building reference embeddings...")
        # This would be called with actual image paths
        print("Please call build_reference_embeddings() with image paths")
