#!/bin/bash
# 既存embeddings全てに対してパッチ統計量を追加抽出

set -e
cd /home/techne/aicheckers

BASE_DIR="data/animedl2m_dataset_release/civitai_subset/image"
REAL_DIR="data/animedl2m_dataset_release/real_images/images"

echo "=== illustrious_ai ==="
python3 scripts/extract_patch_stats_only.py \
    --name illustrious_ai \
    --image-dir "$BASE_DIR/Illustrious" \
    --batch-size 32

echo ""
echo "=== pony_ai ==="
python3 scripts/extract_patch_stats_only.py \
    --name pony_ai \
    --image-dir "$BASE_DIR/Pony" \
    --batch-size 32

echo ""
echo "=== sdxl10_ai ==="
# SDXLは複数ディレクトリにまたがる可能性があるので、親ディレクトリを指定
python3 scripts/extract_patch_stats_only.py \
    --name sdxl10_ai \
    --image-dir "$BASE_DIR" \
    --batch-size 32

echo ""
echo "=== sd15_ai ==="
python3 scripts/extract_patch_stats_only.py \
    --name sd15_ai \
    --image-dir "$BASE_DIR" \
    --batch-size 32

echo ""
echo "=== other_ai ==="
python3 scripts/extract_patch_stats_only.py \
    --name other_ai \
    --image-dir "$BASE_DIR/Other" \
    --batch-size 32

echo ""
echo "=== flux1d_ai ==="
python3 scripts/extract_patch_stats_only.py \
    --name flux1d_ai \
    --image-dir "$BASE_DIR" \
    --batch-size 32

echo ""
echo "=== novelai_ai ==="
python3 scripts/extract_patch_stats_only.py \
    --name novelai_ai \
    --image-dir "data/novelai" \
    --batch-size 32

echo ""
echo "=== pixai_ai ==="
python3 scripts/extract_patch_stats_only.py \
    --name pixai_ai \
    --image-dir "data/pixai" \
    --batch-size 32

echo ""
echo "=== danbooru_real ==="
python3 scripts/extract_patch_stats_only.py \
    --name danbooru_real \
    --image-dir "$REAL_DIR" \
    --batch-size 32

echo ""
echo "=== All Done! ==="
