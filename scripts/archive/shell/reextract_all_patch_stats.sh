#!/bin/bash
# 全カテゴリのpatch_statsを再抽出するスクリプト
# REGトークンバグ修正後の正しいパッチ位置で抽出

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ANIMEDL_BASE="/home/techne/aicheckers/data/animedl2m_dataset_release/civitai_subset/image"

echo "========================================"
echo "Patch Stats Re-extraction (REG bug fix)"
echo "========================================"
echo "Start time: $(date)"
echo ""

# AnimeDL-2M カテゴリ
declare -A ANIMEDL_CATEGORIES
ANIMEDL_CATEGORIES["illustrious_ai"]="Illustrious"
ANIMEDL_CATEGORIES["pony_ai"]="Pony"
ANIMEDL_CATEGORIES["sdxl10_ai"]="SDXL 1.0"
ANIMEDL_CATEGORIES["sd15_ai"]="SD 1.5"
ANIMEDL_CATEGORIES["other_ai"]="Other"
ANIMEDL_CATEGORIES["flux1d_ai"]="Flux.1 D"

# その他のカテゴリ
declare -A OTHER_CATEGORIES
OTHER_CATEGORIES["novelai_ai"]="/home/techne/aicheckers/data/novelai"
OTHER_CATEGORIES["pixai_ai"]="/home/techne/aicheckers/data/pixai"
OTHER_CATEGORIES["danbooru_real"]="/home/techne/aicheckers/data/animedl2m_dataset_release/real_images/images"
OTHER_CATEGORIES["novelai_aibooru_ai"]="/home/techne/aicheckers/data/novelai"
OTHER_CATEGORIES["twitter_novelai_all_ai"]="/home/techne/aicheckers/data/twitter_novelai"

# 進捗カウント
TOTAL=$((${#ANIMEDL_CATEGORIES[@]} + ${#OTHER_CATEGORIES[@]}))
CURRENT=0

# AnimeDL-2M カテゴリを処理
for name in "${!ANIMEDL_CATEGORIES[@]}"; do
    CURRENT=$((CURRENT + 1))
    dir="${ANIMEDL_BASE}/${ANIMEDL_CATEGORIES[$name]}"

    echo "[$CURRENT/$TOTAL] Processing $name..."

    if [ -d "$dir" ]; then
        python3 "$SCRIPT_DIR/extract_patch_stats_only.py" \
            --name "$name" \
            --image-dir "$dir" \
            --batch-size 32
        echo "  ✓ Done: $name"
    else
        echo "  ✗ Directory not found: $dir"
    fi
    echo ""
done

# その他のカテゴリを処理
for name in "${!OTHER_CATEGORIES[@]}"; do
    CURRENT=$((CURRENT + 1))
    dir="${OTHER_CATEGORIES[$name]}"

    echo "[$CURRENT/$TOTAL] Processing $name..."

    if [ -d "$dir" ]; then
        python3 "$SCRIPT_DIR/extract_patch_stats_only.py" \
            --name "$name" \
            --image-dir "$dir" \
            --batch-size 32
        echo "  ✓ Done: $name"
    else
        echo "  ✗ Directory not found: $dir"
    fi
    echo ""
done

echo "========================================"
echo "Re-extraction complete!"
echo "End time: $(date)"
echo "========================================"
