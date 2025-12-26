---
name: train-aicheckers
description: AIcheckersモデルの学習ワークフロー。ユーザーが「学習」「トレーニング」「train」「モデルを更新」と言ったとき、または新しいデータを追加して再学習する際に使用。データ準備→Embedding抽出→学習→テスト→デプロイの全工程をガイド。
---

# AIcheckers Training Workflow

## 絶対にやってはいけないこと

- **curlでAPIをテストしない** - レート制限でエラーになる
- **bashでJSONパースしない** - 必ずPythonを使う
- **Validation Accuracy（96%+）を最終精度と誤解しない** - 実世界の検出率（88-90%）とは別物
- **新しいテストスクリプトを作らない** - 必ず`diagnose`スキルを使う
- **パッチ統計計算を独自実装しない** - 既存のcompute_patch_stats関数を使う

## 重要: パッチ統計計算の一貫性（2024-12-25 修正済み）

パッチ統計（patch_stats）の計算は**必ず同じ分類器**を使う必要がある。

**修正済み:**
- `extract_embeddings_v2.py`: 775d分類器の先頭768dを使用（`classifier.weight[:, :768]`）
- `backend/main.py`: 同上
- 両者が同じ計算方法を使うため、学習と推論の一貫性が確保された

**重要な注意:**
- 分類器を再学習したら、**全embeddingを再抽出する必要がある**
- 抽出には `models/dinov3_classifier.pt`（775d）を使用する

---

## ワークフロー

### 0. 前提確認

```bash
# 現在のモデル状態を確認
python3 .claude/skills/diagnose/scripts/diagnose.py -v
```

### 1. データ準備

```bash
# 重複削除（新規データ追加前に必須）
# 閾値: Pixiv=9, Twitter=11, Danbooru=8
python3 scripts/dedup_images.py --dir /path/to/images --threshold 9
```

### 2. Embedding抽出

`extract_embeddings_v2.py`は775d分類器の先頭768dを使用してパッチ統計を計算する。
`backend/main.py`と同じ計算方法なので、学習と推論の一貫性が確保されている。

```bash
python3 scripts/extract_embeddings_v2.py --dir /path/to/images --name category_name
```

出力:
- `embeddings/{category_name}.npy` - CLSトークン (N, 768)
- `embeddings/{category_name}_patch_stats.npy` - パッチ統計 (N, 7)
- `embeddings/{category_name}_files.txt` - ファイル名リスト

### 3. 学習スクリプト更新

`scripts/train_from_embeddings.py`の`AI_CATEGORIES`に新カテゴリを追加：

```python
AI_CATEGORIES = [
    "illustrious_ai",
    "pony_ai",
    # ... 既存カテゴリ
    "new_category_ai",  # 追加
]
```

### 4. 学習実行

```bash
python3 scripts/train_from_embeddings.py
```

確認事項:
- Validation Accuracy: 96%+ であること
- ログに異常なNaNやエラーがないこと

### 5. テスト（必須）- diagnoseスキルを使用

**絶対に新しいテストスクリプトを作らない。既存のdiagnoseを使う。**

```bash
# Embeddingベースの検出率確認
python3 .claude/skills/diagnose/scripts/diagnose.py -v

# 実画像テスト（推奨）
python3 .claude/skills/diagnose/scripts/diagnose.py -t
```

目標値:
| 指標 | 目標 |
|------|------|
| AI検出率 (Embedding) | 90%+ |
| AI検出率 (実画像) | 85%+ |
| Human正解率 | 95%+ |

### 6. デプロイ

```bash
systemctl --user restart aicheckers-backend
journalctl --user -u aicheckers-backend -f  # ログ確認
```

---

## ファイル構成

| パス | 説明 |
|------|------|
| `scripts/train_from_embeddings.py` | 学習スクリプト |
| `scripts/extract_embeddings_v2.py` | Embedding抽出 |
| `scripts/dedup_images.py` | pHash重複削除 |
| `.claude/skills/diagnose/scripts/diagnose.py` | テスト・診断（これを使う）|
| `models/dinov3_classifier.pt` | 本番モデル (775次元) |
| `models/dinov3_classifier_cls_only.pt` | CLS-only分類器 (768次元) |
| `embeddings/*.npy` | 保存済みEmbedding |

---

## 精度の読み方

- **Validation Accuracy (96%+)**: 学習時のホールドアウト精度。高くて当然
- **AI検出率 (Embedding)**: 保存済みembeddingでの検出率。分類器更新で変動
- **AI検出率 (実画像)**: 実際の画像を推論した検出率。これが本当の精度
- **Human正解率**: 誤検知率の逆

---

## トラブルシューティング

### テスト結果が悪い場合

1. **embeddingの再抽出を検討**
   - 分類器を更新した後は、全embeddingを再抽出する必要がある
   - `python3 scripts/extract_embeddings_v2.py` を使用

2. **データ品質を確認**
   - 新規データにラベルミスがないか
   - 重複削除が正しく行われたか

3. **データバランスを確認**
   - 特定カテゴリが多すぎないか
   - AI/Realの比率が極端でないか

### バックエンドが起動しない場合

```bash
journalctl --user -u aicheckers-backend -f
```

---

## pHash重複削除 推奨閾値

| 媒体 | 推奨閾値 |
|------|----------|
| Pixiv | 9 |
| Twitter/X | 11 |
| Danbooru系 | 8 |
| AI生成サイト | 10〜11 |
