#!/usr/bin/env python3
"""
誤判定画像のパターン分析
スコア範囲別にサンプリングして特徴を抽出
"""
import json
import random
import shutil
from pathlib import Path
from collections import defaultdict

RESULTS_JSON = Path("misclassified_analysis/bulk_verification_results.json")
SAMPLE_DIR = Path("misclassified_analysis/score_range_samples")

def main():
    # 結果読み込み
    with open(RESULTS_JSON, "r") as f:
        results = json.load(f)

    human_judged = [r for r in results if r["human_judged"]]

    # スコア範囲別に分類
    ranges = [
        (0.0, 0.1, "extreme_human_0.0-0.1"),
        (0.1, 0.2, "strong_human_0.1-0.2"),
        (0.2, 0.3, "moderate_human_0.2-0.3"),
        (0.3, 0.4, "weak_human_0.3-0.4"),
        (0.4, 0.5, "borderline_0.4-0.5"),
    ]

    by_range = defaultdict(list)
    for r in human_judged:
        score = r["score"]
        for low, high, label in ranges:
            if low <= score < high:
                by_range[label].append(r)
                break

    # サンプリング数（各範囲から最大20枚）
    SAMPLES_PER_RANGE = 20

    # サンプルディレクトリ作成
    SAMPLE_DIR.mkdir(exist_ok=True, parents=True)

    report = []
    report.append("# スコア範囲別サンプル分析\n\n")

    for low, high, label in ranges:
        images = by_range[label]
        sample_count = min(SAMPLES_PER_RANGE, len(images))

        # ランダムサンプリング
        sampled = random.sample(images, sample_count) if len(images) > sample_count else images

        # 範囲別ディレクトリ作成
        range_dir = SAMPLE_DIR / label
        range_dir.mkdir(exist_ok=True)

        report.append(f"## {label.replace('_', ' ').title()}\n\n")
        report.append(f"**総数**: {len(images)} 枚\n")
        report.append(f"**サンプル数**: {sample_count} 枚\n\n")

        # サンプル画像をコピー
        for r in sampled:
            src = Path(r["path"])
            if src.exists():
                dest = range_dir / f"{r['score']:.4f}_{src.name}"
                shutil.copy2(src, dest)

        # ソース別内訳
        source_counts = defaultdict(int)
        for r in images:
            source = Path(r["source_dir"]).name
            source_counts[source] += 1

        report.append("**ソース別内訳**:\n")
        for source, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True):
            pct = count / len(images) * 100
            report.append(f"- {source}: {count} 枚 ({pct:.1f}%)\n")

        report.append("\n---\n\n")

    # レポート保存
    report_path = SAMPLE_DIR / "ANALYSIS.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("".join(report))

    print(f"スコア範囲別サンプリング完了")
    print(f"サンプル保存先: {SAMPLE_DIR}")
    print(f"\n各範囲のサンプル数:")
    for low, high, label in ranges:
        count = len(by_range[label])
        sample_count = min(SAMPLES_PER_RANGE, count)
        print(f"  {label}: {sample_count}/{count} 枚")

if __name__ == "__main__":
    main()
