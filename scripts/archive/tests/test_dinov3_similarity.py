#!/usr/bin/env python3
"""
Test DINOv3 similarity between original and FastProtect-protected images.
This verifies if ViT-based hashing can track protected images.
"""
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import sys

# Config
DINOV3_MODEL = "facebook/dinov3-vitb16-pretrain-lvd1689m"
ORIGINAL_DIR = Path("target_images")
PROTECTED_DIR = Path("temp_protected_output")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    print(f"Device: {DEVICE}")
    print(f"Loading DINOv3 model: {DINOV3_MODEL}")
    
    processor = AutoImageProcessor.from_pretrained(DINOV3_MODEL, trust_remote_code=True)
    model = AutoModel.from_pretrained(DINOV3_MODEL, trust_remote_code=True).to(DEVICE)
    model.eval()
    
    print("\n" + "="*60)
    print("DINOv3 Similarity Test: Original vs FastProtect-Protected")
    print("="*60)
    
    results = []
    
    for orig_path in sorted(ORIGINAL_DIR.glob("*")):
        if orig_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
            continue
            
        protected_path = PROTECTED_DIR / orig_path.name
        if not protected_path.exists():
            print(f"[SKIP] {orig_path.name}: No protected version found")
            continue
        
        # Load images
        orig_img = Image.open(orig_path).convert("RGB")
        prot_img = Image.open(protected_path).convert("RGB")
        
        # Get embeddings
        with torch.no_grad():
            orig_inputs = processor(images=orig_img, return_tensors="pt").to(DEVICE)
            prot_inputs = processor(images=prot_img, return_tensors="pt").to(DEVICE)
            
            orig_emb = model(**orig_inputs).last_hidden_state[:, 0]  # CLS token
            prot_emb = model(**prot_inputs).last_hidden_state[:, 0]
            
            # Cosine similarity
            similarity = F.cosine_similarity(orig_emb, prot_emb).item()
        
        results.append((orig_path.name, similarity))
        
        status = "✓ MATCH" if similarity > 0.9 else "⚠ DRIFT" if similarity > 0.7 else "✗ FAIL"
        print(f"[{status}] {orig_path.name}: {similarity:.4f}")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if results:
        avg_sim = sum(s for _, s in results) / len(results)
        min_sim = min(s for _, s in results)
        max_sim = max(s for _, s in results)
        
        print(f"Average Similarity: {avg_sim:.4f}")
        print(f"Min Similarity:     {min_sim:.4f}")
        print(f"Max Similarity:     {max_sim:.4f}")
        
        if avg_sim > 0.95:
            print("\n✓ CONCLUSION: DINOv3 can reliably identify FastProtect images.")
            print("  → ViT-based hashing is VIABLE for tracking.")
        elif avg_sim > 0.85:
            print("\n⚠ CONCLUSION: DINOv3 shows some drift but may still work.")
            print("  → ViT-based hashing needs threshold tuning.")
        else:
            print("\n✗ CONCLUSION: DINOv3 CANNOT reliably identify FastProtect images.")
            print("  → ViT-based hashing is NOT viable. Consider pHash/ORB instead.")
    else:
        print("No images processed.")

if __name__ == "__main__":
    main()
