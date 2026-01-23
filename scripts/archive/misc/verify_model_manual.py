import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import sys
import glob
import random

# Add project root to path
sys.path.append(".")
from lib.patch_stats import compute_patch_stats_v2

# Config
DINO_PATH = "models/dinov3-vitb16"
CLASSIFIER_PATH = "models/dinov3_classifier.pt"
REAL_DIR = "data/animedl2m_dataset_release/real_images/images"
AI_DIR = "data/novelai"

def setup_model():
    print("Loading DINOv3...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(DINO_PATH)
    dino = AutoModel.from_pretrained(DINO_PATH).to(device)
    dino.eval()
    
    print(f"Loading Classifier from {CLASSIFIER_PATH}...")
    checkpoint = torch.load(CLASSIFIER_PATH, map_location=device)
    input_dim = checkpoint.get("input_dim", 775)
    classifier = nn.Linear(input_dim, 2).to(device)
    classifier.load_state_dict(checkpoint["classifier"])
    classifier.eval()
    
    return processor, dino, classifier, device

def process_image(image_path, processor, dino, classifier, device):
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error opening {image_path}: {e}")
        return None

    inputs = processor(images=img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = dino(**inputs, output_hidden_states=True)
        
        # CLS (Final)
        final_hidden = outputs.last_hidden_state
        cls_emb = final_hidden[:, 0, :] # (1, 768)
        
        # Patch Stats (Block 8)
        mid_hidden = outputs.hidden_states[8 + 1] # Index 9 = Block 8 output? 
        # Wait, Step 8 of previous session used MID_LAYER_INDEX=8.
        # hidden_states[0] is embeddings. [1] is layer 0/block 0?
        # Usually DINOv3 (ViT) has 12 blocks.
        # Check logic in `extract_embeddings_v2.py`.
        # It used `outputs.hidden_states[MID_LAYER_INDEX + 1]`.
        # So I stick to that.
        mid_patch_emb = mid_hidden[:, 5:5+196, :]
        
        # Compute Stats
        # compute_patch_stats_v2 expects (B, 196, 768)
        stats = compute_patch_stats_v2(mid_patch_emb, return_heatmap=False)
        # stats shape (7,) if batch=1 inside function?
        # lib/patch_stats.py: if input is 3D, returns 2D. 
        # My input is (1, 196, 768). Result will be (1, 7).
        
        # Concatenate
        features = torch.cat([cls_emb, stats], dim=1) # (1, 775)
        
        # Classify
        logits = classifier(features)
        probs = F.softmax(logits, dim=1)
        
        ai_prob = probs[0, 1].item()
        
    return ai_prob

def main():
    processor, dino, classifier, device = setup_model()
    
    # Get Lists
    real_files = glob.glob(f"{REAL_DIR}/*.jpg") + glob.glob(f"{REAL_DIR}/*.png")
    ai_files = glob.glob(f"{AI_DIR}/*.jpg") + glob.glob(f"{AI_DIR}/*.png")
    
    if not real_files or not ai_files:
        print("Could not find images.")
        return
        
    random.seed(42)
    test_real = random.sample(real_files, 10)
    test_ai = random.sample(ai_files, 10)
    
    print("\n" + "="*60)
    print("VERIFICATION TEST: 10 NovelAI vs 10 Real")
    print("="*60)
    print(f"{'Type':<10} | {'Filename':<30} | {'AI Probability':<15} | {'Judgement'}")
    print("-" * 75)
    
    results = []
    
    for f in test_ai:
        prob = process_image(f, processor, dino, classifier, device)
        judge = "✅ AI" if prob > 0.5 else "❌ Miss"
        print(f"{'NovelAI':<10} | {Path(f).name:<30} | {prob*100:.2f}%          | {judge}")
        results.append(('ai', prob))

    print("-" * 75)
    
    for f in test_real:
        prob = process_image(f, processor, dino, classifier, device)
        judge = "✅ Real" if prob < 0.5 else "❌ Miss"
        print(f"{'Real':<10} | {Path(f).name:<30} | {prob*100:.2f}%          | {judge}")
        results.append(('real', prob))
        
    print("="*60)
    
    ai_acc = sum(1 for t, p in results if t=='ai' and p > 0.5) / 10
    real_acc = sum(1 for t, p in results if t=='real' and p < 0.5) / 10
    print(f"AI Detection Rate: {ai_acc*100:.0f}%")
    print(f"Real Accuracy:     {real_acc*100:.0f}%")

if __name__ == "__main__":
    main()
