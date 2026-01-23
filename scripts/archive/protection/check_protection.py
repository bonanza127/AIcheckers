#!/usr/bin/env python3
"""
Protection Effectiveness Checker
Uses WD14 Tagger (ConvNext V2) to analyze images and determine if protection is working.
Compares tags between Original and Protected images to quantify "Concept Erasure".
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from huggingface_hub import hf_hub_download
import onnxruntime as ort
from tqdm import tqdm

# Model Configuration
REPO_ID = "SmilingWolf/wd-v1-4-convnext-tagger-v2"
MODEL_FILENAME = "model.onnx"
TAGS_FILENAME = "selected_tags.csv"

def load_labels():
    print(f"Downloading tags from {REPO_ID}...")
    csv_path = hf_hub_download(repo_id=REPO_ID, filename=TAGS_FILENAME)
    df = pd.read_csv(csv_path)
    return df['name'].tolist()

def load_model():
    print(f"Downloading model from {REPO_ID}...")
    model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME)
    
    print("Loading ONNX model to GPU...")
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    try:
        session = ort.InferenceSession(model_path, providers=providers)
    except Exception as e:
        print(f"Warning: Failed to load CUDA provider, falling back to CPU. Error: {e}")
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        
    return session

def preprocess_image(image, size=448):
    # WD14 ConvNext expects 448x448
    # Keep aspect ratio, paste on white background (or just resize)
    # The standard implementation resizes with padding
    
    img = image.convert("RGB")
    w, h = img.size
    
    # Resize keeping aspect ratio
    scale = size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    img = img.resize((new_w, new_h), Image.BICUBIC)
    
    # Pad to square
    new_img = Image.new("RGB", (size, size), (255, 255, 255))
    new_img.paste(img, ((size - new_w) // 2, (size - new_h) // 2))
    
    # Convert to numpy, BGR, NCHW
    img_np = np.array(new_img, dtype=np.float32)
    img_np = img_np[:, :, ::-1] # RGB -> BGR
    img_np = np.expand_dims(img_np, 0) # BHWC
    
    return img_np

def get_tags(session, labels, image, threshold=0.35):
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    img_input = preprocess_image(image)
    
    probs = session.run([output_name], {input_name: img_input})[0][0]
    
    # Get general tags only (first 4 are meta tags usually, but let's filter by index if needed)
    # WD14 label structure: [general..., character..., copyright...]
    # For protection check, we care about ALL tags dropping confidence.
    
    found_tags = []
    for i, p in enumerate(probs):
        if p > threshold:
            found_tags.append((labels[i], float(p)))
            
    # Sort by confidence
    found_tags.sort(key=lambda x: x[1], reverse=True)
    return found_tags

def calculate_score(tags_original, tags_protected):
    """
    Calculate a simple protection score.
    Score = Average confidence drop of the top 10 original tags.
    """
    if not tags_original:
        return 0.0, []
        
    top_original = tags_original[:15] # Check top 15 tags
    orig_map = {t[0]: t[1] for t in top_original}
    
    prot_map = {t[0]: t[1] for t in tags_protected}
    
    total_drop = 0
    details = []
    
    for tag, orig_conf in top_original:
        prot_conf = prot_map.get(tag, 0.0) # If tag vanished, confidence is effectively 0 (or below threshold)
        drop = orig_conf - prot_conf
        total_drop += drop
        details.append(f"{tag}: {orig_conf:.2f} -> {prot_conf:.2f} (Drop: {drop:.2f})")
        
    avg_drop = total_drop / len(top_original)
    return avg_drop, details

def main():
    parser = argparse.ArgumentParser(description="Check protection effectiveness using WD14 Tagger")
    parser.add_argument("--protected", required=True, help="Directory containing protected images")
    parser.add_argument("--original", help="Directory containing original images (Optional). If not provided, assumes filenames match and are in the same folder or parent folder logic can be used?")
    # For now, let's just analyze the protected images and see what tags remain. 
    # But to measure DROP, we need the original.
    # Let's assume the user copies the PROTECTED files to a folder, and the ORIGINALS are available elsewhere.
    # OR, we can just run on the protected folder and print the confidence.
    # If confidence for '1girl' is 0.99, protection failed. If 0.1, it worked.
    
    parser.add_argument("--compare-with", help="Path to ORIGINAL images directory to compare against.")
    
    args = parser.parse_args()
    
    labels = load_labels()
    session = load_model()
    
    protected_path = Path(args.protected)
    images = list(protected_path.glob("*.jpg")) + list(protected_path.glob("*.png")) + list(protected_path.glob("*.webp"))
    images.sort()
    
    print(f"Analyzing {len(images)} images in {protected_path}...")
    
    # If comparison directory is provided
    orig_path = Path(args.compare_with) if args.compare_with else None
    
    for img_file in images:
        print(f"\n--- {img_file.name} ---")
        try:
            img = Image.open(img_file)
            tags_prot = get_tags(session, labels, img)
            
            # Print top 5 detected tags in PROTECTED image
            print("Detected Tags (Protected):")
            for t, c in tags_prot[:10]:
                print(f"  {t:<20} {c:.2f}")
                
            if orig_path:
                # Try to find matching original
                # Assuming same filename
                orig_file = orig_path / img_file.name
                if not orig_file.exists():
                    # Try typical variations if needed, or just skip
                    print(f"  (Original file {orig_file} not found, skipping comparison)")
                    continue
                    
                img_orig = Image.open(orig_file)
                tags_orig = get_tags(session, labels, img_orig)
                
                score, details = calculate_score(tags_orig, tags_prot)
                print(f"\nProtection Efficacy (Confidence Drop): {score*100:.1f}%")
                print("Top Tag Changes:")
                for d in details[:5]:
                    print(f"  {d}")
                    
        except Exception as e:
            print(f"Error analyzing {img_file}: {e}")

if __name__ == "__main__":
    main()
