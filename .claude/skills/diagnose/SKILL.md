---
name: diagnose
description: AIcheckersのモデル精度診断・システムヘルスチェック。カテゴリ別AI検出率、改善推奨、バックエンド状態を一覧表示。モデルの問題調査、精度確認、デプロイ前チェックに使用。
allowed-tools: Read, Glob, Grep, Bash
user-invocable: true
---

# /diagnose - AIcheckers精度診断スキル

モデルの精度とシステム状態を一括診断するスキル。

## 使用方法

```bash
/diagnose              # フル診断
/diagnose --quick      # 簡易チェック（バックエンド状態のみ）
/diagnose --verbose    # 詳細統計付き
```

## 診断内容

### 1. カテゴリ別AI検出率
各embeddingカテゴリに対するモデルの検出率を計測し、問題のあるカテゴリを特定。

### 2. 改善推奨
検出率が90%未満のカテゴリについて、サンプル数と改善提案を表示。

### 3. バックエンド状態
サービス稼働状態とAPI応答速度を確認。

## 実行スクリプト

診断は以下のスクリプトで実行：
```bash
python3 /home/techne/aicheckers/.claude/skills/diagnose/scripts/diagnose.py
```

## 出力例

```
🔍 Moonlight V1.3 診断レポート
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 カテゴリ別AI検出率
  pony_ai       ████████████████████ 99.3%  ✓ (19,857件)
  sd15_ai       ████████████████████ 98.9%  ✓ (9,985件)
  novelai_ai    ██████████████░░░░░░ 72.6%  ⚠️ (1,045件)

📈 全体精度: 98.12%

⚠️ 改善推奨:
  • novelai_ai: 1,045件 → 3,000件以上推奨

🏥 バックエンド: ✓ 稼働中 (応答 45ms)
```

## トラブルシューティング

### 特定カテゴリの精度が低い場合
1. サンプル数を確認（3,000件以上推奨）
2. embeddingの品質確認（NaN値チェック）
3. 追加データ収集を検討

### バックエンドが応答しない場合
```bash
systemctl --user restart aicheckers-backend
journalctl --user -u aicheckers-backend -f
```
