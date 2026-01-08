#!/usr/bin/env python3
"""
極端なHuman判定画像のメタデータ抽出（フィルタリング付き）
明らかに無視すべき画像（モノクロ、ベクター等）を除外
"""
from PIL import Image
from pathlib import Path
import json
from collections import Counter

RESULTS_JSON = Path("misclassified_analysis/bulk_verification_results.json")
HUMAN_JUDGED_DIR = Path("misclassified_analysis/human_judged_images")
OUTPUT_DIR = Path("misclassified_analysis/critical_cases")

# 極端なHuman判定の閾値
EXTREME_THRESHOLD = 0.15

# 除外すべきキーワード（色彩・スタイル問題）
IGNORE_KEYWORDS = [
    "monochrome", "greyscale", "grayscale", "black and white",
    "{{{{{{no lineart", "vector art", "lineart only",
    "pixel art", "dot art", "mosaic",
    "sketch", "rough sketch", "line art only",
    "silhouette", "shadow only",
]

def extract_metadata(image_path):
    """画像のメタデータを抽出"""
    try:
        img = Image.open(image_path)
        metadata = {}

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

def should_ignore(prompt_text):
    """無視すべきプロンプトかどうか判定"""
    if not prompt_text:
        return False

    prompt_lower = prompt_text.lower()
    for keyword in IGNORE_KEYWORDS:
        if keyword.lower() in prompt_lower:
            return True
    return False

def extract_artist_tags(prompt_text):
    """アーティストタグを抽出"""
    if not prompt_text:
        return []

    artists = []

    # "artist xxx" パターン
    import re
    artist_pattern = r'artist\s+([a-zA-Z0-9_\-]+)'
    matches = re.findall(artist_pattern, prompt_text, re.IGNORECASE)
    artists.extend(matches)

    # プロンプトの最初の部分（通常アーティストタグが来る）
    first_tags = prompt_text.split(',')[:15]
    for tag in first_tags:
        tag = tag.strip()
        # アンダースコア含む、かつ短い（3単語以下）= アーティスト名の可能性
        if '_' in tag or '(' in tag:
            # {}を除去
            clean_tag = re.sub(r'[{}:]', '', tag).strip()
            if clean_tag and len(clean_tag.split()) <= 3:
                artists.append(clean_tag)

    return artists

def main():
    # 結果読み込み
    with open(RESULTS_JSON, "r") as f:
        results = json.load(f)

    # 極端なHuman判定のみ抽出
    extreme_cases = [r for r in results if r["human_judged"] and r["score"] < EXTREME_THRESHOLD]

    print(f"極端なHuman判定（スコア < {EXTREME_THRESHOLD}）: {len(extreme_cases)} 枚\n")

    # 出力ディレクトリ作成
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    critical_cases = []
    ignored_cases = []
    artist_counter = Counter()
    artist_combo_counter = Counter()

    for r in extreme_cases:
        img_path = HUMAN_JUDGED_DIR / f"{r['score']:.4f}_{r['filename']}"

        if not img_path.exists():
            continue

        metadata = extract_metadata(img_path)

        # プロンプト抽出
        prompt_text = None
        if metadata and not metadata.get("error"):
            png_info = metadata.get("png_info", {})
            for key, value in png_info.items():
                key_lower = key.lower()
                if any(k in key_lower for k in ["description", "comment", "prompt"]):
                    prompt_text = value
                    break

        # 無視すべきかチェック
        if should_ignore(prompt_text):
            ignored_cases.append({
                "filename": r["filename"],
                "score": r["score"],
                "reason": "Color/Style issue (monochrome, vector, etc.)",
                "prompt_snippet": prompt_text[:200] if prompt_text else None
            })
            continue

        # クリティカルケースとして記録
        artists = extract_artist_tags(prompt_text) if prompt_text else []

        # アーティスト数をカウント
        for artist in artists:
            artist_counter[artist] += 1

        # 複数アーティストの組み合わせ
        if len(artists) >= 3:
            combo = tuple(sorted(artists[:5]))  # 最初の5人
            artist_combo_counter[combo] += 1

        critical_cases.append({
            "filename": r["filename"],
            "score": r["score"],
            "source": r["source_dir"],
            "artist_count": len(artists),
            "artists": artists[:10],  # 最初の10人
            "prompt": prompt_text
        })

    # レポート生成
    print(f"{'='*60}")
    print(f"CRITICAL CASES ANALYSIS")
    print(f"{'='*60}")
    print(f"Total extreme cases: {len(extreme_cases)}")
    print(f"Ignored (color/style issues): {len(ignored_cases)}")
    print(f"Critical cases to investigate: {len(critical_cases)}")
    print()

    # アーティストタグ統計
    multi_artist_cases = [c for c in critical_cases if c["artist_count"] >= 5]
    print(f"Images with 5+ artist tags: {len(multi_artist_cases)} ({len(multi_artist_cases)/len(critical_cases)*100:.1f}%)")
    print()

    print(f"Top 20 most frequent artists in critical cases:")
    for artist, count in artist_counter.most_common(20):
        print(f"  {artist}: {count}")

    print()
    print(f"Top 10 artist combinations (3+ artists):")
    for combo, count in artist_combo_counter.most_common(10):
        print(f"  {count}x: {', '.join(combo[:3])}{'...' if len(combo) > 3 else ''}")

    # 詳細レポート保存
    report_path = OUTPUT_DIR / "critical_cases_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Critical Cases Analysis - Score < {EXTREME_THRESHOLD}\n\n")
        f.write(f"Total: {len(critical_cases)} cases\n")
        f.write(f"Ignored: {len(ignored_cases)} cases (color/style issues)\n\n")
        f.write("="*60 + "\n\n")

        # クリティカルケースを詳細に出力
        for case in sorted(critical_cases, key=lambda x: x["score"]):
            f.write(f"{'='*60}\n")
            f.write(f"File: {case['filename']}\n")
            f.write(f"Score: {case['score']:.4f}\n")
            f.write(f"Artist count: {case['artist_count']}\n")
            if case['artists']:
                f.write(f"Artists: {', '.join(case['artists'])}\n")
            f.write(f"{'='*60}\n")
            f.write(f"{case['prompt']}\n\n")

    # 無視したケースも記録
    ignored_path = OUTPUT_DIR / "ignored_cases.json"
    with open(ignored_path, "w", encoding="utf-8") as f:
        json.dump(ignored_cases, f, indent=2, ensure_ascii=False)

    # クリティカルケースをJSON保存
    critical_path = OUTPUT_DIR / "critical_cases.json"
    with open(critical_path, "w", encoding="utf-8") as f:
        json.dump(critical_cases, f, indent=2, ensure_ascii=False)

    print(f"\nReports saved to:")
    print(f"  - {report_path}")
    print(f"  - {critical_path}")
    print(f"  - {ignored_path}")

if __name__ == "__main__":
    main()
