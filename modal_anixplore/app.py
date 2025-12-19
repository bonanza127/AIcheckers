"""Modal app for AniXplore inference"""
import modal

# Define the Modal image with required dependencies
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.0.0",
    "torchvision",
    "timm>=0.9.0",
    "Pillow",
).add_local_file(
    local_path="anixplore_model.py",
    remote_path="/root/anixplore_model.py",
)

app = modal.App("anixplore-detector")

# Create a volume for the checkpoint
volume = modal.Volume.from_name("anixplore-checkpoints", create_if_missing=True)
CHECKPOINT_PATH = "/checkpoints/checkpoint-29.pt"


@app.cls(
    image=image,
    gpu="T4",  # Start with T4, can upgrade to A10G if needed
    volumes={"/checkpoints": volume},
    scaledown_window=60,  # Keep warm for 60 seconds
)
class AniXploreDetector:
    @modal.enter()
    def load_model(self):
        import sys
        sys.path.insert(0, "/root")
        import torch
        from anixplore_model import AniXplore
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model on {self.device}...")
        
        # Initialize model
        self.model = AniXplore(seg_pretrain_path=None, conv_pretrain=False, image_size=512)
        
        # Load checkpoint
        from pathlib import Path
        checkpoint_path = Path(CHECKPOINT_PATH)
        if checkpoint_path.exists():
            print(f"Loading checkpoint from {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            # Handle different checkpoint formats
            if "model" in state_dict:
                state_dict = state_dict["model"]
            elif "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            self.model.load_state_dict(state_dict, strict=False)
            print("Checkpoint loaded successfully!")
        else:
            print(f"WARNING: Checkpoint not found at {checkpoint_path}")
            print("Please upload the checkpoint using: modal volume put anixplore-checkpoints /path/to/checkpoint-29 AniXplore/checkpoint-29")
        
        self.model.to(self.device)
        self.model.eval()
        print("Model ready!")

    @modal.method()
    def detect(self, image_bytes: bytes) -> dict:
        """Detect if an image is AI-generated.
        
        Args:
            image_bytes: Raw image bytes (PNG, JPG, etc.)
            
        Returns:
            dict with 'probability' (0-1, higher = more likely AI) and 'is_ai' (bool)
        """
        import torch
        from PIL import Image
        from torchvision import transforms
        import io
        
        # Load and preprocess image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Resize to 512x512 and normalize
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        # Run inference
        prob = self.model.predict(image_tensor)
        
        return {
            "probability": prob,
            "is_ai": prob > 0.5,
            "confidence": abs(prob - 0.5) * 2,  # 0-1 scale
        }


@app.function(image=image)
def test_model():
    """Test the model can be imported"""
    import sys
    sys.path.insert(0, "/root")
    from anixplore_model import AniXplore
    import torch
    
    model = AniXplore(seg_pretrain_path=None, conv_pretrain=False, image_size=512)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test with dummy input
    dummy = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        prob = model.predict(dummy)
    print(f"Test inference result: {prob}")
    return "OK"


@app.local_entrypoint()
def main():
    """Test the model locally"""
    print("Testing model import...")
    result = test_model.remote()
    print(f"Result: {result}")
