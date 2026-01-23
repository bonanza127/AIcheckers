import subprocess
import os
import re
from pathlib import Path

VOLUME_NAME = "aicheckers-embeddings"
REMOTE_DIR = "/vol/embeddings" # As defined in extract script
LOCAL_DIR = Path("embeddings/modal_shards")

def list_remote_files():
    print("Listing remote files...")
    # modal volume ls aicheckers-embeddings /vol/embeddings
    # Note: depends on where the script mounted the vol. 
    # In the extraction script, I mounted it at /vol/embeddings.
    # BUT `modal volume ls` takes the volume name and the PATH INSIDE THE VOLUME.
    # When volume is mounted at /vol/embeddings, writing to /vol/embeddings/foo.npy 
    # puts "foo.npy" at the ROOT of the volume? Or at "/vol/embeddings/"?
    # Usually: volumes={"/vol/embeddings": vol} means the volume root is mounted at /vol/embeddings.
    # So if I write to /vol/embeddings/foo.npy, I am writing to "foo.npy" at volume root.
    # So remote path for 'get' should be "/".
    
    import sys
    # Use python -m modal to ensure we use the installed package
    cmd = [sys.executable, "-m", "modal", "volume", "ls", VOLUME_NAME, "/"] 
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.splitlines()
    except subprocess.CalledProcessError as e:
        print(f"Error listing volume: {e.stderr}")
        return []

def download_file(remote_filename):
    local_path = LOCAL_DIR / remote_filename
    if local_path.exists():
        print(f"Skipping {remote_filename} (already exists)")
        return

    print(f"Downloading {remote_filename}...")
    import sys
    # modal volume get aicheckers-embeddings /remote_filename local_dir
    cmd = [sys.executable, "-m", "modal", "volume", "get", VOLUME_NAME, f"/{remote_filename}", str(LOCAL_DIR)]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to download {remote_filename}")

def main():
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    
    lines = list_remote_files()
    
    # Filter for npy files
    # Output of `ls` might accommodate permissions etc. "filename" usually at end.
    # Modal volume ls format:
    # ├── baz
    # ├── foo
    # └── bar
    # Or just names. 
    
    targets = []
    for line in lines:
        # Simple heuristic: look for .npy
        match = re.search(r'([\w\-\.]+\.npy)', line)
        if match:
            filename = match.group(1)
            if "danbooru_real" in filename:
                targets.append(filename)
    
    print(f"Found {len(targets)} target files.")
    
    for filename in sorted(targets):
        download_file(filename)
        
    print("\nDownload complete.")
    print(f"Files saved to {LOCAL_DIR}")
    
    # Verify count
    count = len(list(LOCAL_DIR.glob("*.npy")))
    print(f"Total local files: {count}")

if __name__ == "__main__":
    main()
