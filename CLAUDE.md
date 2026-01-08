# AIcheckers - AI Anime Image Detector

アニメ絵特化のAI生成画像判別ツール。日本市場向け。

---

## 📚 ドキュメント索引

| ドキュメント | 内容 |
|------------|------|
| **[docs/api.md](docs/api.md)** | Enterprise API、キー発行、開発者アカウント |
| **[docs/guard.md](docs/guard.md)** | Guard機能（SAP v3, FastProtect, Modal実験） |
| **[docs/patrol.md](docs/patrol.md)** | Patrol機能（TrustMark, ViTハッシュ, DMCA）|
| **[docs/training.md](docs/training.md)** | データセット、学習ワークフロー、ベースライン精度 |
| **[docs/environment.md](docs/environment.md)** | 環境情報、技術スタック、劣化Augmentation |

---

## ⚡ クイックリファレンス

### よく使うコマンド

```bash
# 診断（テスト）- 必ずこれを使う
python3 .claude/skills/diagnose/scripts/diagnose.py -v    # Embeddingベース
python3 .claude/skills/diagnose/scripts/diagnose.py -t    # 実画像テスト

# バックエンド再起動
systemctl --user restart aicheckers-backend

# 重複削除
python3 scripts/dedup_images.py --dir /path/to/images --threshold 9

# Embedding抽出（劣化Augmentation推奨）
python3 scripts/extract_embeddings_v2.py --dir /path/to/images --name category_name --degradation-prob 0.5

# 学習（オプション例）
python3 scripts/train_from_embeddings.py                    # 通常
python3 scripts/train_from_embeddings.py --no-em            # EM無効（推奨）
python3 scripts/train_from_embeddings.py --label-smoothing 0.1  # Label Smoothing
```

---

## 🚫 絶対にやってはいけないこと

1. **curlでAPIをテストしない** - レート制限でエラーになる
2. **新しいテストスクリプトを作らない** - `diagnose`スキルを使う
3. **パッチ統計計算を独自実装しない** - `lib/patch_stats.py`を使う
4. **バグの原因を外部のせいにしない** - コードに問題がある前提で調査
5. **Validation Accuracy（96%+）を最終精度と誤解しない**
6. **Modal CLIを直接実行しない** - タイムアウトループの原因になる（下記参照）

### Modal連携 - 非同期実行パターン

**問題**: `modal run`の同期実行は長時間ジョブでタイムアウトループに陥る

**解決策**: `spawn()` + ステータスファイルによる非同期実行

```bash
# ジョブ投入（即座に戻る）
modal run scripts/modal_kohya_lora.py --submit "train_sap_v3_variants:lora_sap_v3"

# ステータス確認
modal run scripts/modal_kohya_lora.py --status

# Modal Dashboardでも確認可能
# https://modal.com/apps
```

**フォーマット**: `--submit "訓練フォルダ名:出力名"`

| 例 | 説明 |
|-----|------|
| `train_normal:lora_normal` | 通常画像でLoRA学習 |
| `train_sap_v3:lora_sap_v3` | SAP v3攻撃画像でLoRA学習 |
| `train_sap_v3_variants:lora_sap_v3_perlin` | Perlin版で学習 |

**ステータスの見方**:
- `running` - 実行中
- `completed` - 完了（result に結果）
- `failed` - 失敗（result にエラー）

---

## 📂 ファイル構成インデックス

### scripts/ - 使用中のスクリプト
| ファイル | 用途 | 使用頻度 |
|----------|------|----------|
| `extract_embeddings_v2.py` | CLS + パッチ統計量抽出 | 高 |
| `train_from_embeddings.py` | 分類器学習 | 高 |
| `dedup_images.py` | pHash重複削除 | 中 |
| `extract_patch_stats_only.py` | パッチ統計のみ追加抽出 | 低 |
| `patch_analysis.py` | パッチ分析ツール | 低 |
| `train_with_patch_stats.py` | パッチ統計付き学習（実験用） | 低 |
| `fastprotect_train.py` | FastProtect摂動学習 | 中 |
| `fastprotect_inference.py` | FastProtect画像保護 | 中 |

### scripts/ - スクレイパー類
| ファイル | 用途 |
|----------|------|
| `smart_scraper.py` | 汎用スクレイパー |
| `aibooru_scraper.py` | AIBooru用 |
| `pixiv_scraper.py` | Pixiv用 |
| `pixai_scraper.py` | PixAI用 |
| `civitai_scraper.py` | CivitAI用 |
| `twitter_bot.py` | Twitter用 |

### archive/deprecated_scripts/ - 非推奨（使うな）
| ファイル | 非推奨理由 |
|----------|-----------|
| `test_model.py` | diagnoseスキルと重複、不整合の原因 |
| `extract_embeddings.py` | v2に置き換え済み |
| `train_simple.py` | train_from_embeddingsに統合 |
| `train_classifier.py` | 古い学習スクリプト |
| `batch_extract.py` | 使用されていない |
| `extract_real.py` | v2に統合済み |

### .claude/skills/ - スキル
| スキル | 用途 |
|--------|------|
| `train` | 学習ワークフロー全体 |
| `diagnose` | モデル診断・テスト |

### lib/ - 共通モジュール
| ファイル | 用途 |
|----------|------|
| `patch_stats.py` | パッチ統計量計算（backend/main.py, extract_embeddings_v2.pyで使用） |
| `vae_hooks.py` | VAE中間層フック（FastProtect用） |
| `mpl_loss.py` | Multi-Layer Protection Loss（FastProtect用） |
| `trustmark_helper.py` | TrustMark透かし埋め込み・抽出 |

### models/
| ファイル | 説明 |
|----------|------|
| `dinov3_classifier.pt` | **本番モデル** (775次元) |
| `dinov3_classifier_cls_only.pt` | CLS-only分類器 (768次元) |
| `baseline_before_gate/` | ベースライン保存 |

### embeddings/
```
{category}.npy              # CLSトークン (N, 768)
{category}_patch_stats.npy  # パッチ統計量 (N, 7)
{category}_files.txt        # ファイル名リスト
```

---

## 🔧 既知の問題と注意点

### パッチ統計計算の一貫性（2024-12-25 修正済み）

**修正内容:**
- `extract_embeddings_v2.py`を修正し、775d分類器の先頭768dを使用するようにした
- `backend/main.py`と同じ計算方法になり、学習と推論の一貫性が確保された

**重要:**
- モデルを再学習したら、**全embeddingを再抽出**する必要がある
- 抽出コマンド: `python3 scripts/extract_embeddings_v2.py --dir /path --name name`
- 抽出時は775d分類器（`models/dinov3_classifier.pt`）を使用する
