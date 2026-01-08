#!/usr/bin/env python3
"""
誤判定画像のメタデータ抽出
NovelAI画像のプロンプト情報を調査
"""
from PIL import Image
from pathlib import Path
import json
from collections import Counter

SAMPLE_DIR = Path("misclassified_analysis/score_range_samples/extreme_human_0.0-0.1")

def extract_metadata(image_path):
    """画像のメタデータを抽出"""
    try:
        img = Image.open(image_path)
        metadata = {}

        # PNG info (Convert bytes to string)
        if hasattr(img, 'info') and img.info:
            png_info = {}
            for key, value in img.info.items():
                if isinstance(value, bytes):
                    try:
                        png_info[str(key)] = value.decode('utf-8', errors='replace')
                    except:
                        png_info[str(key)] = str(value)
                else:
                    png_info[str(key)] = str(value)
            metadata['png_info'] = png_info

        return metadata
    except Exception as e:
        return {"error": str(e)}

def main():
    images = list(SAMPLE_DIR.glob("*.jpg")) + list(SAMPLE_DIR.glob("*.png"))

    print(f"Analyzing {len(images)} extreme Human-judged images...\n")

    results = []
    artist_tags = Counter()
    has_metadata_count = 0

    for img_path in sorted(images):
        score = float(img_path.name.split('_')[0])
        print(f"\n{'='*60}")
        print(f"File: {img_path.name}")
        print(f"Score: {score:.4f}")
        print('='*60)

        metadata = extract_metadata(img_path)

        # メタデータの有無をチェック
        prompt_text = None
        if metadata and not metadata.get("error"):
            png_info = metadata.get("png_info", {})
            if png_info:
                has_metadata_count += 1

                # プロンプト情報を探す
                for key, value in png_info.items():
                    key_lower = key.lower()
                    if any(k in key_lower for k in ["description", "comment", "prompt", "parameters"]):
                        print(f"\n{key}:")
                        print(value[:500] if len(value) > 500 else value)  # 最初の500文字
                        prompt_text = value

                        # アーティストタグを抽出
                        if isinstance(value, str):
                            import re
                            # {{{artist_name}}} パターンと (artist_name) パターン
                            artist_pattern = r'\{+([^}]+)\}+|\(([^)]+)\)'
                            matches = re.findall(artist_pattern, value)
                            for match in matches:
                                tag = match[0] if match[0] else match[1]
                                tag = tag.strip()
                                # アーティスト関連のキーワードを含むか、特定のパターンに一致
                                if tag and ('(' in value or len(tag.split()) <= 3):
                                    artist_tags[tag] += 1

        results.append({
            "filename": img_path.name,
            "score": score,
            "has_metadata": prompt_text is not None,
            "prompt": prompt_text
        })

        if not prompt_text:
            print("No metadata found")

    # サマリー
    print("\n\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total images analyzed: {len(images)}")
    print(f"Images with prompts: {has_metadata_count} ({has_metadata_count/len(images)*100:.1f}%)")

    if artist_tags:
        print(f"\nTop artist/style tags in extreme Human-judged images:")
        for tag, count in artist_tags.most_common(20):
            print(f"  {tag}: {count}")

    # 結果保存（テキスト形式）
    output_path = SAMPLE_DIR.parent / "metadata_analysis.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Metadata Analysis - Extreme Human Judged Images\n\n")
        for r in results:
            f.write(f"{'='*60}\n")
            f.write(f"File: {r['filename']}\n")
            f.write(f"Score: {r['score']:.4f}\n")
            f.write(f"{'='*60}\n")
            if r['prompt']:
                f.write(f"{r['prompt']}\n\n")
            else:
                f.write("No metadata\n\n")

    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main()
