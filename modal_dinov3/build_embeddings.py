"""
DINOv3 参照画像の特徴量構築スクリプト
ローカルの画像からModal Volumeに特徴量を保存
"""
import modal
from pathlib import Path

# 参照画像のディレクトリ
AI_IMAGES_DIR = Path("/home/techne/aicheckers/data/test_images/ai_generated")
REAL_IMAGES_DIR = Path("/home/techne/aicheckers/data/test_images/real")

# AnimeDL-2Mからの追加（ダウンロード完了後）
ANIMEDL_AI_DIR = Path("/home/techne/aicheckers/data/animedl2m_full/fake_images")
ANIMEDL_REAL_DIR = Path("/home/techne/aicheckers/data/animedl2m_full/real_images")


def collect_images(directory: Path, max_images: int = 100) -> list:
    """ディレクトリから画像を収集"""
    if not directory.exists():
        print(f"Warning: {directory} does not exist")
        return []

    images = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
        images.extend(directory.glob(ext))
        images.extend(directory.glob(ext.upper()))

    images = images[:max_images]
    print(f"Found {len(images)} images in {directory}")
    return images


def build_local():
    """ローカルで特徴量を構築してModalにアップロード"""
    import torch
    from transformers import AutoImageProcessor, AutoModel
    from PIL import Image
    from tqdm import tqdm

    # 画像収集
    ai_images = collect_images(AI_IMAGES_DIR, max_images=100)
    real_images = collect_images(REAL_IMAGES_DIR, max_images=100)

    if not ai_images or not real_images:
        print("Error: Need both AI and real images")
        print(f"AI images: {len(ai_images)}, Real images: {len(real_images)}")
        return

    print(f"Total: {len(ai_images)} AI, {len(real_images)} real")

    # DINOv3ロード（CPUでも動作）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_name = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    print(f"Loading {model_name}...")
    try:
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device).eval()
    except Exception as e:
        print(f"Failed to load DINOv3: {e}")
        print("Trying DINOv2 as fallback...")
        model_name = "facebook/dinov2-base"
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device).eval()

    def extract_features(image_paths):
        features = []
        for p in tqdm(image_paths, desc="Extracting"):
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
                print(f"Error {p}: {e}")
        return torch.cat(features, dim=0) if features else torch.empty(0, 768)

    print("Extracting AI features...")
    ai_emb = extract_features(ai_images)

    print("Extracting real features...")
    real_emb = extract_features(real_images)

    # ローカル保存
    local_path = Path("/tmp/dinov3_reference_embeddings.pt")
    data = {"ai": ai_emb, "real": real_emb}
    torch.save(data, local_path)
    print(f"Saved to {local_path}")
    print(f"AI: {ai_emb.shape}, Real: {real_emb.shape}")

    # Modal Volumeにアップロード
    print("Uploading to Modal volume...")
    try:
        volume = modal.Volume.from_name("dinov3-embeddings", create_if_missing=True)
        volume.put_file(str(local_path), "reference_embeddings.pt")
        print("Upload complete!")
    except Exception as e:
        print(f"Modal upload failed: {e}")
        print(f"Local file saved at: {local_path}")

    return {"ai": len(ai_emb), "real": len(real_emb)}


def upload_existing(local_file: str):
    """既存の特徴量ファイルをModalにアップロード"""
    volume = modal.Volume.from_name("dinov3-embeddings", create_if_missing=True)
    volume.put_file(local_file, "reference_embeddings.pt")
    print(f"Uploaded {local_file} to Modal volume")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=["build", "upload"], default="build")
    parser.add_argument("--file", help="File to upload (for upload action)")
    args = parser.parse_args()

    if args.action == "build":
        build_local()
    elif args.action == "upload":
        if not args.file:
            print("Error: --file required for upload action")
        else:
            upload_existing(args.file)
