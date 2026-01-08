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

## 学習後の必須作業

> ⚠️ **学習が完了したら、必ずこのスキルファイルの「データソースディレクトリ」テーブルを更新すること。**
> - 各カテゴリの枚数を最新の値に更新
> - 新しいデータソースを追加した場合はテーブルに追記
> - 日付を更新（例: 2026-01時点 → 2026-02時点）

## 重要: パッチ統計量v2アーキテクチャ（2026-01 改訂）

パッチ統計量v2は**教師なし（unsupervised）**で、中間層（Block 8）から抽出される。

**設計原則:**
- 中間層から「分類器を通さない」統計量を抽出
- 7次元: adj_sim_mean, adj_sim_var, high_sim_ratio, patch_var, anisotropy, norm_var, norm_range
- 未知のAIモデルに対する汎化性能を重視

**v1からの変更点:**
- `extract_embeddings_v2.py`: 分類器不要、中間層から直接統計量を計算
- `backend/main.py`: 同上、`output_hidden_states=True`で中間層取得
- **分類器再学習後もembedding再抽出が必要**（統計量の意味が変わったため）

> ⚠️ **v1からv2への移行時は、全embeddingの再抽出が必須です。**

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

`extract_embeddings_v2.py`は中間層（Block 8）から教師なしパッチ統計量を計算する。
分類器は不要。

```bash
python3 scripts/extract_embeddings_v2.py --dir /path/to/images --name category_name
```

オプション:
- `--mid-layer N`: 中間層インデックス (0-11, デフォルト: 8)
- `--degradation-prob P`: 劣化Augmentation確率 (0.0-1.0)

出力:
- `embeddings/{category_name}.npy` - CLSトークン (N, 768) - 最終層
- `embeddings/{category_name}_patch_stats.npy` - パッチ統計v2 (N, 7) - 中間層
- `embeddings/{category_name}_files.txt` - ファイル名リスト

#### データソースディレクトリ（2026-01時点）

| カテゴリ | ソースディレクトリ | 枚数 |
|----------|-------------------|------|
| **AI (civitai_subset)** | | |
| pony_ai | `data/animedl2m_dataset_release/civitai_subset/image/Pony` | ~19,857 |
| illustrious_ai | `data/animedl2m_dataset_release/civitai_subset/image/Illustrious` | ~4,824 |
| sdxl10_ai | `data/animedl2m_dataset_release/civitai_subset/image/SDXL 1.0` | ~8,916 |
| sd15_ai | `data/animedl2m_dataset_release/civitai_subset/image/SD 1.5` | ~9,985 |
| flux1d_ai | `data/animedl2m_dataset_release/civitai_subset/image/Flux.1 D` | ~1,843 |
| other_ai | `data/animedl2m_dataset_release/civitai_subset/image/Other` | ~4,555 |
| **AI (NovelAI系)** | | |
| novelai_ai | `data/novelai` | ~1,283 |
| novelai_combined_ai | `data/novelai_combined` | ~21,878 |
| novelai_artist_tagged_ai | `data/novelai_artist_tagged` | ~846 |
| pixai_ai | `data/pixai` | ~1,018 |
| **Real (人間の絵)** | | |
| danbooru_real | `data/animedl2m_dataset_release/real_images/images` | ~49,998 |

> 注: パスは `/home/techne/aicheckers/` からの相対パス

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
| `models/dinov3-vitb16/` | **DINOv3ベースモデル（ローカル、327MB）** |
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
