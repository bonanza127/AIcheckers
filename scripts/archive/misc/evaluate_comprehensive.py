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
import time

# Add project root
sys.path.append(".")
from lib.patch_stats import compute_patch_stats_v2

# Config
DINO_PATH = "models/dinov3-vitb16"
CLASSIFIER_PATH = "models/dinov3_classifier.pt"
BATCH_SIZE = 1 # Keep simle for now, speed is fine on GPU

CAT_PATHS = {
    "Real (Danbooru)": "data/animedl2m_dataset_release/real_images/images",
    "AI (NovelAI)": "data/novelai",
    "AI (SD 1.5)": "data/animedl2m_dataset_release/civitai_subset/image/SD 1.5",
    "AI (Pony)": "data/animedl2m_dataset_release/civitai_subset/image/Pony",
    "AI (SDXL)": "data/animedl2m_dataset_release/civitai_subset/image/SDXL 1.0"
}
SAMPLES_PER_CAT = 100

def setup_model():
    print("Loading Models...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(DINO_PATH)
    dino = AutoModel.from_pretrained(DINO_PATH).to(device)
    dino.eval()
    
    checkpoint = torch.load(CLASSIFIER_PATH, map_location=device)
    input_dim = checkpoint.get("input_dim", 775)
    classifier = nn.Linear(input_dim, 2).to(device)
    classifier.load_state_dict(checkpoint["classifier"])
    classifier.eval()
    
    return processor, dino, classifier, device

def predict(image_path, processor, dino, classifier, device):
    try:
        img = Image.open(image_path).convert("RGB")
    except:
        return None
        
    inputs = processor(images=img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = dino(**inputs, output_hidden_states=True)
        final_hidden = outputs.last_hidden_state
        cls_emb = final_hidden[:, 0, :]
        mid_hidden = outputs.hidden_states[8 + 1]
        mid_patch_emb = mid_hidden[:, 5:5+196, :]
        
        stats = compute_patch_stats_v2(mid_patch_emb, return_heatmap=False)
        features = torch.cat([cls_emb, stats], dim=1)
        
        logits = classifier(features)
        probs = F.softmax(logits, dim=1)
        ai_prob = probs[0, 1].item()
        
    return ai_prob

def main():
    processor, dino, classifier, device = setup_model()
    
    overall_stats = {}
    
    print(f"\nEvaluating {SAMPLES_PER_CAT} samples per category...")
    print(f"{'Category':<20} | {'Count':<5} | {'Acc':<8} | {'Avg Prob (AI)':<15} | {'Min':<6} | {'Max':<6}")
    print("-" * 80)
    
    total_ai_correct = 0
    total_ai_count = 0
    
    for cat_name, path_str in CAT_PATHS.items():
        base_path = Path(path_str)
        if not base_path.exists():
            print(f"{cat_name:<20} | NOT FOUND")
            continue
            
        all_files = list(base_path.glob("*.jpg")) + list(base_path.glob("*.jpeg")) + list(base_path.glob("*.png")) + list(base_path.glob("*.webp"))
        if not all_files:
            print(f"{cat_name:<20} | EMPTY")
            continue
            
        # Sample
        if len(all_files) > SAMPLES_PER_CAT:
            files = random.sample(all_files, SAMPLES_PER_CAT)
        else:
            files = all_files
            
        # Predict
        probs = []
        correct = 0
        is_ai_cat = "AI" in cat_name
        
        for f in files:
            p = predict(f, processor, dino, classifier, device)
            if p is not None:
                probs.append(p)
                if is_ai_cat:
                    if p > 0.5: correct += 1
                else:
                    if p < 0.5: correct += 1
        
        if not probs: continue
        
        acc = correct / len(probs) * 100
        avg_prob = sum(probs) / len(probs)
        min_prob = min(probs)
        max_prob = max(probs)
        
        print(f"{cat_name:<20} | {len(probs):<5} | {acc:.1f}%   | {avg_prob:.4f}          | {min_prob:.2f}  | {max_prob:.2f}")
        
        overall_stats[cat_name] = acc
        
        if is_ai_cat:
            total_ai_correct += correct
            total_ai_count += len(probs)
            
    print("-" * 80)
    # Summary
    if "Real (Danbooru)" in overall_stats:
        print(f"Real Accuracy: {overall_stats['Real (Danbooru)']:.2f}%")
        
    if total_ai_count > 0:
        print(f"AI Accuracy (Avg): {total_ai_correct/total_ai_count*100:.2f}%")

if __name__ == "__main__":
    main()
