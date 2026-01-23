import os
import sys
import io
import time
import tarfile
import numpy as np
import torch
from pathlib import Path
from PIL import Image
import modal
from modal.mount import Mount
from huggingface_hub import hf_hub_download

# Local imports (will be mounted)
from lib.patch_stats import compute_patch_stats_v2_batch

# Define Modal App
app = modal.App("dinov3-streaming-extractor")

# Volume for saving embeddings
vol = modal.Volume.from_name("aicheckers-embeddings", create_if_missing=True)

# Image definition
image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch", "torchvision", "transformers", "pillow", "numpy", 
        "huggingface_hub", "accelerate", "tqdm"
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})  # Fast download
    .add_local_dir("lib", remote_path="/root/lib")
    .add_local_dir("models", remote_path="/models")
)

# Constants
DINOV3_MODEL_NAME = "google/vit-base-patch16-224-in21k" # Fallback if local not used, but better to upload local model?
# Actually we usually map the local model path. 
# But for simplicity and speed, let's use the HuggingFace Hub version of DINOv3 if available?
# Techne's local model is at `models/dinov3-vitb16`. 
# Using the exact same weights is critical.
# I will mount the local model directory.

MODEL_DIR = "/models/dinov3-vitb16"
MID_LAYER_INDEX = 6
BATCH_SIZE = 64
HF_REPO = "deepghs/danbooru2024-sfw"
MID_PATCH_DTYPE = np.float16

@app.function(
    image=image,
    gpu="A10g",  # or T4, A100. A10G is good balance.
    volumes={"/vol/embeddings": vol},
    timeout=3600  # 1 hour per tar file is plenty
)
def extract_tar_shard(tar_index: int):
    import sys
    sys.path.insert(0, "/root") # for lib import
    
    # Check if already done
    cls_out = Path(f"/vol/embeddings/danbooru_real_{tar_index:04d}.npy")
    stats_out = Path(f"/vol/embeddings/danbooru_real_{tar_index:04d}_patch_stats.npy")
    mid_out = Path(f"/vol/embeddings/danbooru_real_{tar_index:04d}_mid_patches.npy")
    
    if cls_out.exists() and stats_out.exists() and mid_out.exists():
        print(f"Shard {tar_index} already exists. Skipping.")
        return
    
    print(f"=== Processing Shard {tar_index:04d} ===")
    device = torch.device("cuda")
    
    # Load Model (on every worker, but it's cached in image if properly done... 
    # here simple load is fine)
    from transformers import AutoImageProcessor, AutoModel
    
    # Load from mounted path
    try:
        processor = AutoImageProcessor.from_pretrained(MODEL_DIR)
        model = AutoModel.from_pretrained(MODEL_DIR)
    except Exception:
        # Fallback/Debug: If mount fails, maybe try generic? But we need specific DINOv3.
        # Assuming mount works.
        print("Error loading model from mount.")
        return

    model.to(device)
    model.eval()
    
    # Stream Download Tar
    filename = f"images/{tar_index:04d}.tar"
    print(f"Downloading {filename} from {HF_REPO}...")
    
    try:
        tar_path = hf_hub_download(
            repo_id=HF_REPO, 
            filename=filename, 
            repo_type="dataset"
        )
    except Exception as e:
        print(f"Failed to download {filename}: {e}")
        return

    # Pass 1: count valid images (for memmap sizing)
    valid_exts = (".jpg", ".jpeg", ".png", ".webp")
    total_images = 0
    with tarfile.open(tar_path, "r") as tar:
        for member in tar:
            if not member.isfile():
                continue
            if not member.name.lower().endswith(valid_exts):
                continue
            total_images += 1

    if total_images == 0:
        print(f"Shard {tar_index} is empty.")
        return

    cls_map = np.lib.format.open_memmap(
        cls_out, mode="w+", dtype=np.float32, shape=(total_images, 768)
    )
    stats_map = np.lib.format.open_memmap(
        stats_out, mode="w+", dtype=np.float32, shape=(total_images, 7)
    )
    mid_map = np.lib.format.open_memmap(
        mid_out, mode="w+", dtype=MID_PATCH_DTYPE, shape=(total_images, 196, 768)
    )
    write_idx = 0

    buffer_images = []
    
    # Pass 2: actual extraction
    with tarfile.open(tar_path, "r") as tar:
        for member in tar:
            if not member.isfile():
                continue
            if not member.name.lower().endswith(valid_exts):
                continue
                
            try:
                f = tar.extractfile(member)
                if f is None: continue
                img_bytes = f.read()
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                buffer_images.append(img)
            except Exception as e:
                print(f"Error reading image {member.name}: {e}")
                continue
            
            # Batch Processing
            if len(buffer_images) >= BATCH_SIZE:
                cls_emb, patch_stats, mid_patches = _process_batch(buffer_images, model, processor, device)
                batch_count = cls_emb.shape[0]
                cls_map[write_idx:write_idx + batch_count] = cls_emb
                stats_map[write_idx:write_idx + batch_count] = patch_stats
                mid_map[write_idx:write_idx + batch_count] = mid_patches
                write_idx += batch_count
                buffer_images = []
        
        # Process remaining
        if buffer_images:
            cls_emb, patch_stats, mid_patches = _process_batch(buffer_images, model, processor, device)
            batch_count = cls_emb.shape[0]
            cls_map[write_idx:write_idx + batch_count] = cls_emb
            stats_map[write_idx:write_idx + batch_count] = patch_stats
            mid_map[write_idx:write_idx + batch_count] = mid_patches
            write_idx += batch_count
            buffer_images = []
            
    if write_idx != total_images:
        print(f"Warning: processed {write_idx}/{total_images}, truncating outputs.")
        cls_map.flush()
        stats_map.flush()
        mid_map.flush()
        cls_map = np.lib.format.open_memmap(cls_out, mode="r", dtype=np.float32, shape=(total_images, 768))
        stats_map = np.lib.format.open_memmap(stats_out, mode="r", dtype=np.float32, shape=(total_images, 7))
        mid_map = np.lib.format.open_memmap(mid_out, mode="r", dtype=MID_PATCH_DTYPE, shape=(total_images, 196, 768))
        cls_tmp = cls_out.with_suffix(".tmp.npy")
        stats_tmp = stats_out.with_suffix(".tmp.npy")
        mid_tmp = mid_out.with_suffix(".tmp.npy")
        np.lib.format.open_memmap(cls_tmp, mode="w+", dtype=np.float32, shape=(write_idx, 768))[:] = cls_map[:write_idx]
        np.lib.format.open_memmap(stats_tmp, mode="w+", dtype=np.float32, shape=(write_idx, 7))[:] = stats_map[:write_idx]
        np.lib.format.open_memmap(mid_tmp, mode="w+", dtype=MID_PATCH_DTYPE, shape=(write_idx, 196, 768))[:] = mid_map[:write_idx]
        os.replace(cls_tmp, cls_out)
        os.replace(stats_tmp, stats_out)
        os.replace(mid_tmp, mid_out)

    vol.commit()
    print(f"Saved Shard {tar_index}: {write_idx} images")
        
    # Cleanup tar to save disk space on worker (ephemeral)
    try:
        os.remove(tar_path)
    except:
        pass


def _process_batch(images, model, processor, device):
    from lib.patch_stats import compute_patch_stats_v2_batch
    
    try:
        inputs = processor(images=images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.inference_mode():
            outputs = model(**inputs, output_hidden_states=True)
            
            # CLS (Final Layer)
            final_hidden = outputs.last_hidden_state
            cls_emb = final_hidden[:, 0, :].cpu().numpy()
            
            # Patch Stats (Mid Layer 8)
            mid_hidden = outputs.hidden_states[MID_LAYER_INDEX + 1]
            mid_patch_emb = mid_hidden[:, 5:5+196, :] # 196 patches
            
            patch_stats = compute_patch_stats_v2_batch(mid_patch_emb)
            mid_patches = mid_patch_emb.cpu().numpy().astype(MID_PATCH_DTYPE, copy=False)
        return cls_emb, patch_stats, mid_patches
    except Exception as e:
        print(f"Batch processing error: {e}")
        return np.empty((0, 768), dtype=np.float32), np.empty((0, 7), dtype=np.float32), np.empty((0, 196, 768), dtype=MID_PATCH_DTYPE)

@app.local_entrypoint()
def main(start_index: int = 0, num_shards: int = 200):
    # Process shards in parallel
    # range(0, 150) covers ~1M images (since 100 shards = 680k, 150 ~= 1M)
    # Be careful with concurrency limits on HF hub? usually fine.
    
    indices = range(start_index, start_index + num_shards)
    print(f"Starting extraction for shards: {list(indices)}")
    
    # Run in parallel (map)
    # Using 10 workers
    for _ in extract_tar_shard.map(indices):
        pass
    
    print("Done!")
