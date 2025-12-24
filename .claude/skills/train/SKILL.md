---
name: train
description: AIcheckersモデルの学習ワークフロー。ユーザーが「学習」「トレーニング」「train」「モデルを更新」と言ったとき、または新しいデータを追加して再学習する際に使用。データ準備→Embedding抽出→学習→テスト→デプロイの全工程をガイド。
---

# AIcheckers Training Workflow

## 絶対にやってはいけないこと

- **curlでAPIをテストしない** - レート制限でエラーになる
- **bashでJSONパースしない** - 必ずPythonを使う
- **Validation Accuracy（96%+）を最終精度と誤解しない** - 実世界の検出率（88-90%）とは別物

## ワークフロー

### 1. データ準備

```bash
# 重複削除（新規データ追加前に必須）
python3 scripts/dedup_images.py --dir /path/to/images --threshold 16

# Embedding抽出
python3 scripts/extract_embeddings_v2.py --dir /path/to/images --name category_name
```

出力: `embeddings/{category_name}.npy`, `embeddings/{category_name}_patch_stats.npy`

### 2. 学習スクリプト更新

`scripts/train_from_embeddings.py`の`AI_CATEGORIES`に新カテゴリを追加：

```python
AI_CATEGORIES = [
    "illustrious_ai",
    "pony_ai",
    # ... 既存カテゴリ
    "new_category_ai",  # 追加
]
```

### 3. 学習実行

```bash
python3 scripts/train_from_embeddings.py
```

確認事項:
- Validation Accuracy: 96%+ であること
- ログに異常なNaNやエラーがないこと

### 4. テスト（必須）

```bash
python3 scripts/test_model.py --model models/dinov3_classifier.pt
```

目標値:
| 指標 | 目標 |
|------|------|
| AI検出率 | 88%+ |
| Human正解率 | 98%+ |
| 平均AIスコア(AI画像) | 80%+ |
| 平均AIスコア(Human画像) | 20%以下 |

### 5. デプロイ

```bash
systemctl --user restart aicheckers-backend
journalctl --user -u aicheckers-backend -f  # ログ確認
```

## ファイル構成

| パス | 説明 |
|------|------|
| `scripts/train_from_embeddings.py` | 学習スクリプト |
| `scripts/test_model.py` | テストスクリプト |
| `scripts/extract_embeddings_v2.py` | Embedding抽出 |
| `scripts/dedup_images.py` | pHash重複削除 |
| `models/dinov3_classifier.pt` | 本番モデル (775次元) |
| `embeddings/*.npy` | 保存済みEmbedding |

## 精度の読み方

- **Validation Accuracy (96%+)**: 学習時のホールドアウト精度。高くて当然
- **AI検出率 (88%+)**: テスト画像でAI≥50%の割合。これが実世界の性能
- **Human正解率 (98%+)**: テスト画像でAI<50%の割合。誤検知率

## トラブルシューティング

### テスト結果が悪い場合
1. `test_model.py`が正しく動作しているか確認
2. モデルファイルのパスが正しいか確認
3. テスト用画像フォルダにデータがあるか確認

### バックエンドが起動しない場合
```bash
journalctl --user -u aicheckers-backend -f
```
でエラーログを確認
