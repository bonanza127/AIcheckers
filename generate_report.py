#!/usr/bin/env python3
"""
大量検証の統計レポート生成
"""
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

# ファイルパス
RESULTS_JSON = Path("misclassified_analysis/bulk_verification_results.json")
REPORT_PATH = Path("misclassified_analysis/STATISTICAL_REPORT.md")

def main():
    # 結果読み込み
    with open(RESULTS_JSON, "r") as f:
        results = json.load(f)

    total = len(results)
    human_judged = [r for r in results if r["human_judged"]]
    ai_judged = [r for r in results if not r["human_judged"]]

    # ソース別統計
    by_source = defaultdict(lambda: {"total": 0, "human": 0, "ai": 0})
    for r in results:
        source = Path(r["source_dir"]).name
        by_source[source]["total"] += 1
        if r["human_judged"]:
            by_source[source]["human"] += 1
        else:
            by_source[source]["ai"] += 1

    # スコア統計
    human_scores = [r["score"] for r in human_judged]
    ai_scores = [r["score"] for r in ai_judged]

    # スコア範囲別
    ranges = [
        (0.0, 0.1, "Extreme Human (0.0-0.1)"),
        (0.1, 0.2, "Strong Human (0.1-0.2)"),
        (0.2, 0.3, "Moderate Human (0.2-0.3)"),
        (0.3, 0.4, "Weak Human (0.3-0.4)"),
        (0.4, 0.5, "Borderline (0.4-0.5)"),
    ]

    # 最低スコアTop20
    sorted_human = sorted(human_judged, key=lambda x: x["score"])[:20]

    # レポート生成
    report = []
    report.append("# NovelAI大量検証 統計レポート\n")
    report.append(f"**検証日時**: 2024-12-28\n")
    report.append(f"**総画像数**: {total:,} 枚\n\n")

    report.append("---\n\n")
    report.append("## 総合結果\n\n")
    report.append("| 判定 | 枚数 | 割合 |\n")
    report.append("|------|------|------|\n")
    report.append(f"| **AI判定（正解）** | {len(ai_judged):,} | {len(ai_judged)/total*100:.1f}% |\n")
    report.append(f"| **Human判定（誤判定）** | {len(human_judged):,} | {len(human_judged)/total*100:.1f}% |\n\n")

    report.append("---\n\n")
    report.append("## ソース別精度\n\n")
    report.append("| ソース | 総数 | AI判定 | Human判定 | 精度 |\n")
    report.append("|--------|------|--------|-----------|------|\n")
    for source, stats in sorted(by_source.items(), key=lambda x: x[1]["human"]/x[1]["total"], reverse=True):
        accuracy = stats["ai"] / stats["total"] * 100
        report.append(f"| {source} | {stats['total']:,} | {stats['ai']:,} | {stats['human']:,} | {accuracy:.1f}% |\n")

    report.append("\n---\n\n")
    report.append("## スコア分布（Human判定のみ）\n\n")
    report.append(f"**平均スコア**: {np.mean(human_scores):.4f}\n")
    report.append(f"**中央値**: {np.median(human_scores):.4f}\n")
    report.append(f"**最小値**: {min(human_scores):.4f}\n")
    report.append(f"**最大値**: {max(human_scores):.4f}\n\n")

    report.append("### スコア範囲別内訳\n\n")
    report.append("| 範囲 | 枚数 | 割合 |\n")
    report.append("|------|------|------|\n")
    for low, high, label in ranges:
        count = sum(1 for s in human_scores if low <= s < high)
        pct = count / len(human_scores) * 100
        report.append(f"| {label} | {count:,} | {pct:.1f}% |\n")

    report.append("\n---\n\n")
    report.append("## 最も確信度の高いHuman判定 Top20\n\n")
    report.append("| スコア | ファイル名 | ソース |\n")
    report.append("|--------|-----------|--------|\n")
    for r in sorted_human:
        source = Path(r["source_dir"]).name
        report.append(f"| {r['score']:.4f} | {r['filename']} | {source} |\n")

    report.append("\n---\n\n")
    report.append("## AI判定のスコア分布（参考）\n\n")
    report.append(f"**平均スコア**: {np.mean(ai_scores):.4f}\n")
    report.append(f"**中央値**: {np.median(ai_scores):.4f}\n")
    report.append(f"**最小値**: {min(ai_scores):.4f}\n")
    report.append(f"**最大値**: {max(ai_scores):.4f}\n\n")

    # 低信頼度AI判定（0.5-0.6）
    low_confidence_ai = [r for r in ai_judged if 0.5 <= r["score"] < 0.6]
    if low_confidence_ai:
        report.append(f"**低信頼度AI判定（0.5-0.6）**: {len(low_confidence_ai):,} 枚 ({len(low_confidence_ai)/len(ai_judged)*100:.1f}%)\n\n")

    report.append("---\n\n")
    report.append("## 結論\n\n")
    report.append(f"1. **全体精度**: {len(ai_judged)/total*100:.1f}%（AI判定率）\n")
    report.append(f"2. **誤判定率**: {len(human_judged)/total*100:.1f}%\n")
    report.append(f"3. **極端なHuman判定**: {sum(1 for s in human_scores if s < 0.1):,} 枚（スコア < 0.1）\n")
    report.append(f"4. **誤判定画像保存先**: `misclassified_analysis/human_judged_images/`\n\n")
    report.append("### 次のステップ\n\n")
    report.append("- 極端なHuman判定画像（スコア < 0.1）の目視確認\n")
    report.append("- 色空間バリエーション（白黒、ドット調など）の分析\n")
    report.append("- 色空間正規化の実装検討\n")

    # レポート保存
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("".join(report))

    print(f"統計レポート生成完了: {REPORT_PATH}")
    print(f"\n要約:")
    print(f"  総画像数: {total:,} 枚")
    print(f"  AI判定: {len(ai_judged):,} 枚 ({len(ai_judged)/total*100:.1f}%)")
    print(f"  Human判定: {len(human_judged):,} 枚 ({len(human_judged)/total*100:.1f}%)")
    print(f"  極端なHuman判定（< 0.1）: {sum(1 for s in human_scores if s < 0.1):,} 枚")

if __name__ == "__main__":
    main()
