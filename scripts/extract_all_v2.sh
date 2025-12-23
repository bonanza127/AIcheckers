#!/bin/bash
# 全画像からCLS + パッチ統計量を抽出

set -e
cd /home/techne/aicheckers

echo "=== AI画像 (Civitai) ==="
python3 scripts/extract_embeddings_v2.py \
    --dir data/animedl2m_dataset_release/civitai_subset/image \
    --name ai_civitai \
    --batch-size 32

echo ""
echo "=== Human画像 (Danbooru) ==="
python3 scripts/extract_embeddings_v2.py \
    --dir data/animedl2m_dataset_release/real_images/images \
    --name human_danbooru \
    --batch-size 32

echo ""
echo "=== 追加AI: NovelAI ==="
python3 scripts/extract_embeddings_v2.py \
    --dir data/novelai \
    --name novelai_v2 \
    --batch-size 32

echo ""
echo "=== 追加AI: PixAI ==="
python3 scripts/extract_embeddings_v2.py \
    --dir data/pixai \
    --name pixai_v2 \
    --batch-size 32

echo ""
echo "=== 追加AI: Pixiv NovelAI ==="
python3 scripts/extract_embeddings_v2.py \
    --dir data/pixiv_novelai \
    --name pixiv_novelai_v2 \
    --batch-size 32

echo ""
echo "=== Done! ==="
