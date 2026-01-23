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
# Modalスクリプトはarchiveに移動済み
# 必要な場合は scripts/archive/modal/ から復元して使用

# ジョブ投入（即座に戻る）
modal run scripts/archive/modal/modal_kohya_lora.py --submit "train_sap_v3_variants:lora_sap_v3"

# Modal Dashboardで確認
# https://modal.com/apps
```

> **Note**: Modalスクリプトは実験的用途のためアーカイブ済み。通常の学習は`train_from_embeddings.py`を使用。

---

## 📂 ファイル構成インデックス

### scripts/ - 使用中のスクリプト（22ファイル）

#### 抽出・学習
| ファイル | 用途 | 使用頻度 |
|----------|------|----------|
| `extract_embeddings_v2.py` | CLS + パッチ統計量抽出 | 高 |
| `extract_cpu_stats_v2.py` | CPU統計量v2抽出（lib/cpu_stats.py依存） | 高 |
| `extract_cpu_stats_v3.py` | CPU統計量v3抽出 | 高 |
| `extract_cpu_stats_v3_all.py` | CPU統計量v3 unified抽出（lib/cpu_stats.py依存） | 高 |
| `extract_patch_stats_only.py` | パッチ統計のみ追加抽出 | 低 |
| `train_from_embeddings.py` | 汎用分類器学習 | 高 |
| `train_28d_plus_60.py` | **現行本番モデル学習** | 高 |
| `train_with_patch_stats.py` | パッチ統計付き学習（実験用） | 低 |

#### Guard（画像保護）
| ファイル | 用途 |
|----------|------|
| `moonknight_v3.py` | **本番推論エンジン**（バックエンドが使用） |
| `fastprotect_train.py` | FastProtect摂動学習 |
| `fastprotect_inference.py` | FastProtect画像保護（Modal用） |

#### スクレイパー
| ファイル | 用途 |
|----------|------|
| `smart_scraper.py` | 汎用スクレイパー |
| `aibooru_scraper.py` | AIBooru用 |
| `aibooru_artist_tagged_scraper.py` | AIBooru（アーティストタグ付き） |
| `pixiv_scraper.py` | Pixiv用 |
| `pixai_scraper.py` | PixAI用 |
| `civitai_scraper.py` | CivitAI用 |
| `twitter_bot.py` | Twitter用 |

#### ユーティリティ
| ファイル | 用途 |
|----------|------|
| `dedup_images.py` | pHash重複削除 |
| `patch_analysis.py` | パッチ分析ツール |
| `generate_magic_link.py` | VIP/開発者マジックリンク生成 |

#### シェルスクリプト
| ファイル | 用途 |
|----------|------|
| `backup_classifier.sh` | モデルバックアップ |
| `restore_classifier.sh` | モデル復元 |
| `backup_data.sh` | データバックアップ |
| `send_failure_email.sh` | 障害通知メール |
| `bot_control.sh` | Twitterボット制御 |
| `extract_all_v2.sh` | バッチ抽出 |

### scripts/archive/ - アーカイブ済み（151ファイル）

実験・旧バージョンのスクリプト。必要に応じて参照可能。

| ディレクトリ | 内容 | ファイル数 |
|-------------|------|-----------|
| `tests/` | テストスクリプト（→ diagnoseスキルに統合） | 33 |
| `training/` | 旧学習スクリプト（train_two_head_*, train_28d_* 等） | 21 |
| `extraction/` | 旧抽出スクリプト | 19 |
| `analysis/` | AB実験・分析・デバッグスクリプト | 13 |
| `modal/` | Modal実験スクリプト | 10 |
| `rendering/` | プロモ画像生成スクリプト | 12 |
| `protection/` | SAP/Ironclad等の保護実験 | 13 |
| `misc/` | その他一回限りスクリプト | 25 |
| `shell/` | 旧シェルスクリプト | 4 |

### .claude/skills/ - スキル
| スキル | 用途 |
|--------|------|
| `train` | 学習ワークフロー全体 |
| `diagnose` | モデル診断・テスト |

### lib/ - 共通モジュール
| ファイル | 用途 |
|----------|------|
| `patch_stats.py` | GPU パッチ統計量計算（v2, v3） |
| `cpu_stats.py` | CPU 統計量計算（Two-Head用） |
| `boundary_stats.py` | 境界統計量 |
| `extra_stats.py` | 追加統計量 |
| `trustmark_helper.py` | TrustMark透かし埋め込み・抽出 |
| `fastprotect_inference.py` | FastProtect推論 |
| `vae_hooks.py` | VAE中間層フック（FastProtect用） |
| `mpl_loss.py` | Multi-Layer Protection Loss |
| `signature.py` | 署名関連 |
| `cloudflare_bypass.py` | Cloudflare回避（スクレイピング用） |

### models/
| ファイル | 説明 |
|----------|------|
| `two_head_28d_plus_60/` | **本番モデル** (796次元: CLS 768d + GPU 4d + CPU 24d) |
| `dinov3-vitb16/` | DINOv3 ViT-B/16 ベースモデル |
| `dinov3_classifier_cls_only.pt` | CLS-only分類器 (768次元) - フォールバック用 |
| `baseline_before_gate/` | ベースライン保存 |

> **Note**: 旧775dモデル（`dinov3_classifier.pt`）は削除済み。Two-Head 796dが本番。

### embeddings/
```
{category}.npy              # CLSトークン (N, 768)
{category}_patch_stats.npy  # パッチ統計量 (N, 7)
{category}_cpu_stats_v2.npy # CPU統計量 v2 (N, 16)
{category}_cpu_stats_v3.npy # CPU統計量 v3 (N, 20)
{category}_files.txt        # ファイル名リスト
```

---

## 🔧 既知の問題と注意点

### Two-Head モデルへの移行（2026-01 完了）

旧775dモデルから Two-Head 796dモデルへ移行済み。

**重要:**
- 新規学習時は `train_28d_plus_60.py` または `train_from_embeddings.py` を使用
- Embeddingの再抽出には **CPU統計量も必要**
  ```bash
  # GPU統計量
  python3 scripts/extract_embeddings_v2.py --dir /path --name name
  # CPU統計量
  python3 scripts/extract_cpu_stats_v3.py --dir /path --name name
  ```

---

## 🧠 パッチ統計アーキテクチャ（参考情報）

> **Note**: 現在の本番モデルは **Two-Head 28d (796次元)** です。
> 以下は設計原則と統計量の参考情報。

### 設計原則

1. **中間層から「分類器を通さない」** - 必須条件
2. **統計量は必ず教師なし（unsupervised）** - cosine, variance, norm等
3. **過学習防止** - 次元数は必要最小限に

### GPU統計量（patch_stats_v3から4次元を選択）

本番で使用する4次元（`GPU_4D_IDX = [1, 3, 5, 6]`）:

| 統計量 | 説明 |
|--------|------|
| `adj_sim_var` | 隣接パッチ類似度分散 |
| `patch_var` | パッチ埋め込み分散 |
| `norm_var` | ノルム分散 |
| `norm_range` | ノルムレンジ |

### やってはいけないこと

- ❌ 中間層に線形分類器を新設（過学習の温床）
- ❌ 中間層CLSを使う（DINOは中間CLSを最適化していない）
- ❌ patchごとの「AI確率」を計算（定義不能）

### 実装ファイル

- `lib/patch_stats.py` - GPU統計量計算（v2, v3）
- `lib/cpu_stats.py` - CPU統計量計算
- `backend/main.py` - 推論（Two-Head対応）

---

## 🔢 Two-Head モデル特徴量インデックス（2026-01 更新）

### 現在の本番モデル: `two_head_28d_plus_60`

**アーキテクチャ**: CLS (768d) + GPU (4d) + CPU (24d) = **796次元**

### 特徴量インデックス定義

```python
# GPU特徴量: patch_stats_v3から選択（4次元）
GPU_4D_IDX = [1, 3, 5, 6]

# CPU特徴量 v2: cpu_stats_v2から選択（13次元）
CPU16_13D_IDX = [0, 1, 2, 4, 5, 7, 8, 9, 11, 12, 13, 14, 15]

# CPU特徴量 v3: cpu_stats_v3_20dから選択（11次元）
CPU20_11D_IDX = [0, 1, 2, 3, 4, 5, 8, 10, 15, 16, 17]

# CPU合計: 13 + 11 = 24次元
```

### GPU特徴量詳細 (4d) - `patch_stats_v3[GPU_4D_IDX]`

| 位置 | 元idx | 特徴量名 | 説明 |
|------|-------|----------|------|
| 0 | 1 | adj_sim_var | 隣接パッチ類似度分散 |
| 1 | 3 | patch_var | パッチ埋め込み分散 |
| 2 | 5 | norm_var | ノルム分散 |
| 3 | 6 | norm_range | ノルムレンジ |

### CPU特徴量詳細 Part1 (13d) - `cpu_stats_v2[CPU16_13D_IDX]`

| 位置 | 元idx | 特徴量名 |
|------|-------|----------|
| 0 | 0 | banding_score |
| 1 | 1 | radial_spectrum_slope |
| 2 | 2 | stroke_width_proxy |
| 3 | 4 | fractal_dim_edge_512 |
| 4 | 5 | patchwise_edge_density |
| 5 | 7 | st_aniso_var |
| 6 | 8 | st_aniso_spatial_gradient |
| 7 | 9 | flat_boundary_peri_area |
| 8 | 11 | flat_hole_ratio |
| 9 | 12 | highfreq_spatial_autocorr |
| 10 | 13 | patch_vs_global_rank_entropy_gap |
| 11 | 14 | flat_ratio |
| 12 | 15 | flat_ratio_variance_across_tiles |

### CPU特徴量詳細 Part2 (11d) - `cpu_stats_v3_20d[CPU20_11D_IDX]`

| 位置 | 20d idx | unified idx | 特徴量名 |
|------|---------|-------------|----------|
| 0 | 0 | 1 | histogram_modality |
| 1 | 1 | 2 | color_palette_entropy |
| 2 | 2 | 3 | luminance_layer_count |
| 3 | 3 | 6 | luminance_skewness |
| 4 | 4 | 8 | value_bimodality |
| 5 | 5 | 9 | multiscale_variance_ratio |
| 6 | 8 | 12 | luminance_mean |
| 7 | 10 | 14 | saturation_mean |
| 8 | 15 | 22 | radial_spectrum_slope_patch_gap |
| 9 | 16 | 23 | color_banding_score |
| 10 | 17 | 24 | compression_artifact_pattern |

### 変換チェーン

```
cpu_stats_v3_unified (27d)
    ↓ UNIFIED_TO_20D_IDX
cpu_stats_v3_20d (20d)
    ↓ CPU20_11D_IDX
最終CPU Part2 (11d)
```

```python
# lib/cpu_stats.py
UNIFIED_TO_20D_IDX = [1, 2, 3, 6, 8, 9, 10, 11, 12, 13, 14, 16, 18, 19, 20, 22, 23, 24, 25, 26]
```

### モデル比較

| モデル | CLS | GPU | CPU | 合計 | 備考 |
|--------|-----|-----|-----|------|------|
| `two_head_28d_plus_60` | 768d | 4d | 24d | **796d** | **本番（推奨）** |
| `two_head_29d_ep30` | 768d | 5d | 24d | 797d | 29d = 28d + mid_adj_sim_var |

### 28d vs 29d の違い

- **28d**: `gpu_dim=4`、GPU特徴量は `GPU_4D_IDX` のみ
- **29d**: `gpu_dim=5`、28dに `mid_adj_sim_var` を追加

```python
# 29d専用: 中間層の隣接類似度分散を追加
mid_adj_var = compute_mid_adj_sim_var(patch_embeddings_mid)
gpu_5d = torch.cat([gpu_4d, mid_adj_var], dim=1)
```

### 正規化パラメータ

```python
STD_FLOOR = 1e-3  # 標準偏差の下限（ゼロ除算防止）
# 正規化: (x - mean) / clamp(std, min=STD_FLOOR)
```

### 関連ファイル

| ファイル | 役割 |
|----------|------|
| `models/two_head_28d_plus_60/model.pt` | 本番モデル |
| `scripts/train_28d_plus_60.py` | 学習スクリプト |
| `lib/cpu_stats.py` | CPU特徴量計算 |
| `lib/patch_stats.py` | GPU特徴量計算（v3） |
| `backend/main.py` | 推論（Two-Head対応）|
