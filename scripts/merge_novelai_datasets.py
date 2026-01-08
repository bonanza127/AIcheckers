#!/usr/bin/env python3
"""
Merge images from pixiv_novelai and twitter_novelai into novelai_combined.
- Skips true duplicates (same content by hash)
- Renames files if same name but different content
"""

import os
import shutil
import hashlib
from pathlib import Path
from collections import defaultdict

# Directories
BASE_DIR = Path("/home/techne/aicheckers/data")
SOURCES = [
    BASE_DIR / "pixiv_novelai" / "images",
    BASE_DIR / "twitter_novelai" / "twitter",
]
DEST = BASE_DIR / "novelai_combined"

def get_file_hash(filepath: Path, chunk_size: int = 8192) -> str:
    """Calculate MD5 hash of a file."""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()

def get_unique_filename(dest_dir: Path, filename: str) -> Path:
    """Get a unique filename by appending _1, _2, etc if exists."""
    dest_path = dest_dir / filename
    if not dest_path.exists():
        return dest_path
    
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    counter = 1
    while True:
        new_name = f"{stem}_{counter}{suffix}"
        new_path = dest_dir / new_name
        if not new_path.exists():
            return new_path
        counter += 1

def main():
    print("=" * 60)
    print("NovelAI Dataset Merger")
    print("=" * 60)
    
    # Build hash index of existing files in destination
    print("\n[1/4] Indexing existing files in novelai_combined...")
    existing_hashes = {}  # hash -> filepath
    existing_count = 0
    
    for f in DEST.iterdir():
        if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp', '.gif']:
            file_hash = get_file_hash(f)
            existing_hashes[file_hash] = f
            existing_count += 1
    
    print(f"   Found {existing_count} existing files")
    
    # Process source directories
    stats = {
        'copied': 0,
        'skipped_duplicate': 0,
        'renamed': 0,
        'errors': 0,
    }
    
    for source_dir in SOURCES:
        print(f"\n[2/4] Processing: {source_dir.name}...")
        
        if not source_dir.exists():
            print(f"   WARNING: {source_dir} does not exist, skipping")
            continue
        
        # Find all image files (recursively for twitter which has subdirs)
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.gif', '*.JPG', '*.PNG']:
            image_files.extend(source_dir.rglob(ext))
        
        print(f"   Found {len(image_files)} images")
        
        for i, src_file in enumerate(image_files):
            if (i + 1) % 1000 == 0:
                print(f"   Progress: {i + 1}/{len(image_files)}")
            
            try:
                # Calculate hash
                file_hash = get_file_hash(src_file)
                
                # Check if duplicate content
                if file_hash in existing_hashes:
                    stats['skipped_duplicate'] += 1
                    continue
                
                # Get destination path
                dest_path = DEST / src_file.name
                
                # If name exists but different content, rename
                if dest_path.exists():
                    dest_path = get_unique_filename(DEST, src_file.name)
                    stats['renamed'] += 1
                
                # Copy file
                shutil.copy2(src_file, dest_path)
                existing_hashes[file_hash] = dest_path
                stats['copied'] += 1
                
            except Exception as e:
                print(f"   ERROR: {src_file}: {e}")
                stats['errors'] += 1
    
    # Final count
    print("\n[3/4] Counting final result...")
    final_count = sum(1 for f in DEST.iterdir() if f.is_file())
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"   Files copied:              {stats['copied']}")
    print(f"   Duplicates skipped:        {stats['skipped_duplicate']}")
    print(f"   Files renamed (same name): {stats['renamed']}")
    print(f"   Errors:                    {stats['errors']}")
    print(f"   ---")
    print(f"   Total in novelai_combined: {final_count}")
    print("=" * 60)

if __name__ == "__main__":
    main()
