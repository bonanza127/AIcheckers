
import sys
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from lib.patch_stats import compute_patch_stats_v2_batch

def main():
    print("Running quick extraction test...")
    DINOV3_MODEL_PATH = Path("/home/techne/aicheckers/models/dinov3-vitb16")
    IMAGE_DIR = Path("/home/techne/aicheckers/data/animedl2m_dataset_release/real_images/images")
    OUTPUT_PATH = Path("/home/techne/aicheckers/embeddings/danbooru_real_test_patch_stats.npy")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(str(DINOV3_MODEL_PATH))
    model = AutoModel.from_pretrained(str(DINOV3_MODEL_PATH))
    model.to(device)
    model.eval()
    
    files = list(IMAGE_DIR.glob("*.jpg"))[:100]
    if not files:
        print("No files found!")
        return

    print(f"Processing {len(files)} images...")
    
    batch_stats = []
    batch_size = 20
    
    for i in range(0, len(files), batch_size):
        batch_files = files[i:i+batch_size]
        images = []
        for f in batch_files:
            try:
                img = Image.open(f).convert("RGB")
                images.append(img)
            except: pass
            
        if not images: continue
        
        inputs = processor(images=images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # Mid layer 8 extraction
            mid_hidden = outputs.hidden_states[8 + 1]
            mid_patch_emb = mid_hidden[:, 5:5+196, :]
            stats = compute_patch_stats_v2_batch(mid_patch_emb)
            batch_stats.append(stats)
            
    all_stats = np.vstack(batch_stats)
    np.save(OUTPUT_PATH, all_stats)
    print(f"Saved stats to {OUTPUT_PATH}")
    print(f"Stats shape: {all_stats.shape}")

if __name__ == "__main__":
    main()
