# AIcheckers - AI Anime Image Detector

アニメ絵特化のAI生成画像判別ツール。日本市場向け。

---

## クイックリファレンス

### 本番環境
| 項目 | URL/値 |
|------|--------|
| フロントエンド | https://aicheckers.net (Vercel) |
| API | https://api.aicheckers.net (Cloudflare Tunnel → localhost:8000) |
| 速度 | 70-120ms (TTA有効時) |

### Enterprise API
| 項目 | 詳細 |
|------|------|
| 認証 | `X-API-Key: aicheckers_ent_xxx...` ヘッダー |
| レート制限 | なし |
| ドキュメント | `docs/enterprise_api.md` |
| キー発行 | `/admin/enterprise/create-key` (管理者のみ) |
| 使用量確認 | `/admin/enterprise/usage-all` (管理者のみ) |
| データ保存 | `data/enterprise_keys.json`, `data/enterprise_usage.json` |

**企業向けAPIキー発行手順:**
```bash
# 1. 管理者アカウントでサイトにログイン → JWTトークンを取得
# 2. 以下を実行（company_name, contact_emailを適宜変更）
curl -X POST https://api.aicheckers.net/admin/enterprise/create-key \
  -H "Authorization: Bearer YOUR_ADMIN_JWT" \
  -H "Content-Type: application/json" \
  -d '{"company_name": "株式会社Example", "contact_email": "api@example.co.jp", "plan": "standard", "expires_days": 365}'

# レスポンスにapi_keyが含まれる → これを企業に渡す
```

**VIP/管理者との共存:** Enterprise APIは`X-API-Key`ヘッダーがある場合のみ。従来のJWT認証（VIP/管理者デモ版）はそのまま動作する。

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

## 絶対にやってはいけないこと

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

**旧方式（非推奨）**: 同期実行はタイムアウトの可能性あり
```bash
# これは使わない
modal run scripts/modal_kohya_lora.py --train-sap-v3
```

---

## ファイル構成インデックス

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

## 技術スタック

```
DINOv3 (facebook/dinov3-vitb16-pretrain-lvd1689m)
    ↓
CLS Token (768次元) + Patch Stats (7次元)
    ↓
Linear Probe (nn.Linear(775, 2))
    ↓
TTA + Temperature Scaling (T=1.5)
```

### パッチ統計量 (Patch Stats)
| Index | 名前 | 説明 |
|-------|------|------|
| 0 | patch_mean | パッチAIスコアの平均 |
| 1 | patch_max | パッチAIスコアの最大 |
| 2 | patch_var | パッチAIスコアの分散 |
| 3 | max_minus_mean | 最大-平均（局所的突出度） |
| 4 | embed_var_mean | 埋め込み分散の平均 |
| 5 | count_high_score | スコア≥0.8のパッチ割合 |
| 6 | v_high_sim_85 | 垂直方向の高類似度パッチ比率 |

### 劣化Augmentation (2025-01-01 採用)
画質バイアス除去のため、Embedding抽出時に確率的に劣化を適用。

| 劣化タイプ | パラメータ |
|------------|------------|
| JPEG圧縮 | quality 30-70 |
| ガウシアンノイズ | std 5-25 |
| ダウンサンプリング | scale 50-80% |

**効果** (A/Bテスト結果):
- AI検出率: +2.68%
- Human正解率: +0.22%

**使用方法**: `--degradation-prob 0.5` をextract_embeddings_v2.pyに指定

---

## 既知の問題と注意点

### パッチ統計計算の一貫性（2024-12-25 修正済み）

**修正内容:**
- `extract_embeddings_v2.py`を修正し、775d分類器の先頭768dを使用するようにした
- `backend/main.py`と同じ計算方法になり、学習と推論の一貫性が確保された

**重要:**
- モデルを再学習したら、**全embeddingを再抽出**する必要がある
- 抽出コマンド: `python3 scripts/extract_embeddings_v2.py --dir /path --name name`
- 抽出時は775d分類器（`models/dinov3_classifier.pt`）を使用する

---

## データセット

### AI画像（学習用）
| カテゴリ | 枚数 | ソース |
|----------|------|--------|
| illustrious_ai | 4,824 | AnimeDL-2M |
| pony_ai | 19,857 | AnimeDL-2M |
| sdxl10_ai | 8,916 | AnimeDL-2M |
| sd15_ai | 9,985 | AnimeDL-2M |
| other_ai | 4,555 | AnimeDL-2M |
| flux1d_ai | 1,843 | AnimeDL-2M |
| novelai_ai | 1,045 | AIBooru |
| pixai_ai | 1,018 | PixAI |
| novelai_aibooru_ai | 1,283 | AIBooru |
| novelai_combined_ai | 4,499 | Pixiv+Twitter (dedup済み) |
| pixiv_novelai_v2_ai | 8,859 | Pixiv (dedup済み) |
| twitter_novelai_v2_ai | 12,262 | Twitter (dedup済み) |

### Human画像（学習用）
| カテゴリ | 枚数 | ソース |
|----------|------|--------|
| danbooru_real | 49,998 | Danbooru |

### テスト用画像フォルダ
| フォルダ | 用途 |
|----------|------|
| data/novelai/ | NovelAI (AIBooru) テスト |
| data/novelai_combined/ | Pixiv+Twitter NovelAI テスト |
| data/animedl2m_dataset_release/real_images/images/ | Human テスト |

---

## pHash重複削除 推奨閾値

| 媒体 | 推奨閾値 |
|------|----------|
| Pixiv | 9 |
| Twitter/X | 11 |
| Danbooru系 | 8 |
| AI生成サイト | 10〜11 |

---

## 学習→デプロイのワークフロー

**必ず`train`スキルを使う。**

1. データ準備（重複削除）
2. Embedding抽出
3. 学習スクリプト更新・実行
4. **diagnoseスキルでテスト**（新しいスクリプトを作るな）
5. バックエンド再起動

---

## ベースライン精度（比較用）

```
models/baseline_before_gate/
├── dinov3_classifier.pt
├── dinov3_classifier_cls_only.pt
└── test_results.txt
```

- NovelAI (AIBooru): 83/100 (83%)
- NovelAI Combined: 81/100 (81%)
- Human正解率: 99/100 (99%)

---

## 環境情報

- **GPU**: GTX 1660 (6GB VRAM)
- **DINOv3 VRAM**: 0.34GB
- **HuggingFace Token**: 環境変数 `HF_TOKEN`

---

## 開発者アカウント（レート制限免除）

| 用途 | Email |
|------|-------|
| オーナー | hokhok7676@gmail.com |
| DLsite検証用 | dlsite-trial@aicheckers.net |

**設定**: `backend/main.py` の `ADMIN_EMAILS`

---

## AIイラストガード（保護ツール研究）

LoRA学習を妨害するための摂動技術を研究中。

**⚠️ 注意: Guard進捗バーはシミュレーション (2026-01-08)**
`src/app/guard/page.tsx` でSSE進捗を無効化し、22-24秒のシミュレーション進捗に変更。リアルタイム進捗に戻すには、`setGuardProgress`呼び出しのコメントアウトを解除。

---

### 現在のベスト: SAP v3

**スクリプト**: `scripts/sap_v3.py`

VAE+CLIP攻撃。視認性とVAE攻撃効果のバランスが最も良い。

```bash
# 1枚テスト（約1分）
modal run scripts/sap_v3.py --test --warp-magnitude 0.01 --iterations 50
```

#### 攻撃構成

| 攻撃 | 手法 | 効果 |
|------|------|------|
| **VAE攻撃** | latent cos sim最小化 | 構造情報の破壊 |
| **CLIPネガティブ誘導** | "low quality, blurry, noise"に近づける | 低品質タグとの結合 |
| **CLIP概念混乱** | 元画像から離脱 + 無関係概念へ誘導 | 意味情報の汚染 |
| **適応型マスク** | エッジ5%、平坦1%（Sobel） | 視認性を維持しつつ攻撃強化 |
| **Micro-Warping** | 幾何学的変形（kornia elastic） | LightShed等の浄化耐性 |

#### ベンチマーク結果

| 指標 | 値 | 評価 |
|------|-----|------|
| LPIPS | 0.0445 | ✅ 視覚差ほぼなし |
| VAE Cos Sim | **0.81** | ✅ 構造乖離 |
| CLIP to Original | **-0.26** | ✅ 負の値＝完全離脱 |
| CLIP to Negative | 0.28 | ✅ 低品質概念に接近 |
| 処理時間 | **56秒** | ✅ 1分以内 |

#### ネガティブ概念リスト
```python
NEGATIVE_CONCEPTS = [
    "low quality, worst quality, blurry",
    "jpeg artifacts, noise, grainy",
    "text, watermark, signature",
    "error, glitch, corrupted",
]
```

#### 混乱概念リスト
```python
CONFUSION_CONCEPTS = [
    "a photograph of mountains and trees",
    "3d render of geometric shapes",
    "satellite image of earth",
    "medical x-ray scan",
    "infrared thermal image",
]
```

---

### SAP v3 Variants (Perlin実験)

**スクリプト**: `scripts/sap_v3_variants.py`

v3の適応型マスクにPerlinノイズを導入。平坦部の摂動を空間的にバラけさせる。

```bash
# scale64 + 画像ハッシュベースシード（推奨）
modal run scripts/sap_v3_variants.py --test --perlin-scale 64
```

#### 改良点

| 項目 | v3 | v3 Variants |
|------|-----|-------------|
| 平坦部マスク | 一様1% | Perlin 0.75〜1.5% |
| シード | 固定/ランダム | **画像ハッシュベース** |
| CLIPネガティブ | 5概念 | **8概念**（abstract texture等追加） |

#### 推奨パラメータ

| パラメータ | 値 | 理由 |
|------------|-----|------|
| perlin_scale | **64** | scale128より攻撃効果が高い |
| perlin_seed | None（画像ハッシュ） | 再現可能 + 画像ごとに異なる |

#### ベンチマーク結果 (scale64, image-hash seed)

| 指標 | 値 | 評価 |
|------|-----|------|
| LPIPS | 0.047 | ✅ 視覚差なし |
| VAE Cos Sim | 0.80 | ✅ 構造乖離 |
| CLIP to Original | -0.19 | ⚠️ v3(-0.26)より低下 |
| CLIP to Negative | 0.28 | ✅ |

#### 設計思想

- **Perlinノイズ**: ホワイトノイズより浄化耐性・JPEG耐性が高い
- **画像ハッシュベースシード**: 攻撃パターンが画像ごとに異なり学習されにくい
- **scale64**: 細かすぎず粗すぎないバランス

---

### SAP v4 (アーカイブ: WD14実験)

**スクリプト**: `archive/sap_experiments/sap_v4.py`

WD14 Tagger攻撃を試みたが、視認性とのトレードオフが厳しく、v3の方がバランスが良いため保留。

**課題**: 平坦部へのWD14攻撃がノイズとして目立つ。知覚マスク等で改善を試みたが、VAE攻撃効果との両立が困難。

---

### スクリプト一覧

| スクリプト | 用途 | 状態 |
|------------|------|------|
| `sap_v3.py` | VAE+CLIP+Warping（ベースライン） | ✅ 使用中 |
| `sap_v3_variants.py` | **Perlin + 画像ハッシュシード** | ✅ 実験中 |
| `sap_v2.py` | VAE+CLIP+Warping（旧版） | 参考用 |
| `archive/sap_experiments/sap_v4.py` | WD14+VAE実験 | アーカイブ |

---

### 開発履歴

1. **highfreq_attack.py** - エッジ適応 + VAE攻撃（Cos Sim 0.92程度）
2. **sap_v2.py** - CLIP攻撃追加 + Micro-Warping
3. **sap_v3.py** - ネガティブ概念誘導 + 概念混乱追加（Cos Sim 0.81、CLIP離脱-0.26）★現行ベスト
4. **sap_v4.py** - WD14 Tagger攻撃実験（ノイズ問題でアーカイブ）

---

### Micro-Warping パラメータ

| パラメータ | 推奨値 | 説明 |
|------------|--------|------|
| warp_magnitude | 0.01 | 変形強度（0.01でぼやけなし） |
| kernel_size | (63, 63) | ぼかしカーネル |
| sigma | (12.0, 12.0) | ガウシアンσ |

**注意**: magnitude 0.015以上だと視覚的にぼやける

---

### 参考論文

#### FastProtect (CVPR 2025) ★★★最推奨
- **論文**: https://arxiv.org/abs/2412.11423
- **開発**: NAVER WEBTOON AI
- **状態**: **独自実装完了** (2026-01-04)

##### FastProtect使用方法

```bash
# 学習（Modal上で実行）
modal run scripts/fastprotect_train.py --train --data-dir /vol/train_images --steps 40000

# 非同期投入（推奨）
modal run scripts/fastprotect_train.py --submit --data-dir /vol/train_images

# 推論テスト
modal run scripts/fastprotect_inference.py --test

# 画像保護
modal run scripts/fastprotect_inference.py --protect --input /vol/input --output /vol/output --use-warping
```

##### FastProtect関連ファイル

| ファイル | 用途 |
|----------|------|
| `scripts/fastprotect_train.py` | 摂動学習（40,000ステップ） |
| `scripts/fastprotect_inference.py` | 画像保護推論 |
| `lib/vae_hooks.py` | VAE中間層フック |
| `lib/mpl_loss.py` | Multi-Layer Protection Loss |

##### FastProtect学習パラメータ

| パラメータ | 値 | 備考 |
|------------|-----|------|
| num_steps | 40,000 | 論文準拠 |
| batch_size | 16 | A10G向け |
| lr | 0.0002 | Adam (β=0.5, 0.99) |
| η (摂動予算) | 8/255 | ~0.031 |
| λ (中間層重み) | 3.5×10⁻⁵ | 論文準拠 |
| K (クラスタ数) | 4 | Mixture-of-Perturbations |

##### FastProtect技術詳細

- **Multi-Layer Protection Loss**: VAE中間層（down_1〜3, mid_0）でも距離を最大化
- **Mixture-of-Perturbations**: K=4のクラスタごとに異なる摂動を学習
- **Adaptive Targeted Protection**: エントロピーベースでターゲット画像を選択
- **Adaptive Protection Strength**: LPIPS距離に基づき摂動強度を調整
- **Micro-Warping統合**: 浄化耐性のための幾何学的変形

#### PAP (NeurIPS 2024)
- **論文**: https://arxiv.org/abs/2408.10571
- **弱点**: JPEG圧縮に弱い

#### StyleGuard (NeurIPS 2025)
- **論文**: https://arxiv.org/abs/2505.18766
- **注意**: LoRAに対して効果が限定的

---

### Modal実験フォルダ

| フォルダ | 内容 |
|----------|------|
| train_normal | オリジナル画像 |
| train_sap_v2 | SAP v2攻撃済み |
| train_sap_v3 | SAP v3攻撃済み（最新） |
| train_hf_stealth | エッジ5% + 平坦1%攻撃済み（旧） |

---

### 今後の研究課題

1. **実際のLoRA学習テスト**: SAP v3攻撃画像でLoRA学習→生成品質の検証
2. **浄化耐性テスト**: LightShed、DiffPure等での浄化後も攻撃が残るか
3. **JPEG耐性**: SNS投稿時の再圧縮への耐性
4. **FastProtect統合**: コード公開後に高速化手法を取り込む

---

## AIパトロール機能（開発中）

無断転載監視システム。TrustMark透かし + ViTハッシュのハイブリッド方式。

### 技術スタック

```
TrustMark (Adobe Research, ICCV 2025)
    ↓ 透かし埋め込み（user_id + timestamp）
DINOv3 埋め込み抽出（768次元）
    ↓ DB保存
Civitai/Danbooru 新着巡回
    ↓
Faiss類似検索（0.93閾値）
    ↓
TrustMark透かし確認（最終判定）
    ↓
DMCA申請メール自動生成
```

### Guard時の処理順序（推奨）

**重要**: Gemini分析に基づき、以下の順序で処理

```python
1. TrustMark透かし埋め込み（alpha=1.15）
   → FastProtectの摂動を見越して強めに設定

2. FastProtect（MoonKnight）摂動追加
   → 透かし入り画像を「ベース」として最適化
   → 透かし信号を破壊しない摂動が計算される

3. DINOv3埋め込み抽出
   → 最終画像から抽出（Patrol時に照合）
```

**理由**: 逆順序（FastProtect → TrustMark）だと、GANが摂動を「ノイズ」と判断して平滑化してしまう可能性

### TrustMark性能（GTX 1660）

| 処理 | 時間 | VRAM |
|------|------|------|
| **エンコード** | 0.199秒 | 0.28GB |
| **デコード** | 0.041秒 | 0.28GB |
| **透かし容量** | 61bit | - |

**FastProtect耐性**: DINOv3が確認済み（マイロード検証）

### 実装ファイル

| ファイル | 用途 |
|----------|------|
| `lib/trustmark_helper.py` | 透かし埋め込み・抽出ヘルパー |
| `scripts/test_trustmark.py` | TrustMark動作確認スクリプト |
| `backend/main.py` | Guard/Patrolエンドポイント |

### 実装進捗（2026-01-08）

- [x] TrustMarkライブラリインストール・動作確認
- [x] trustmark_helper.py作成
- [x] backend/main.pyへのimport追加
- [ ] Guard時のTrustMark透かし埋め込み実装
- [ ] Guard時のViTハッシュ+タイムスタンプDB保存実装
- [ ] Civitai API連携とFaiss検索基盤
- [ ] 2段階検証システム（ViT→TrustMark）
- [ ] DMCA申請メール自動生成
- [ ] 実験的検証（透かし検出率、FastProtect効果）

### 監視対象サイト

| サイト | 実現性 | 優先度 |
|--------|--------|--------|
| **Civitai** | ⭐⭐⭐⭐⭐ 公式API | ★★★★★ |
| **Danbooru** | ⭐⭐⭐⭐ 公式API | ★★★★ |
| Gelbooru | ⭐⭐⭐ 非公式API | ★★★ |
| Pixiv | ⭐⭐ スクレイピング（規約違反リスク） | ★★ |
| Kemono.party | ⭐ 違法サイト（法的リスク極大） | ❌ |

**Phase 1推奨**: Civitai + Danbooru のみ（合法・低コスト）

### コスト試算

```
月間10,000ユーザー、各10枚保護の場合:

Civitai新着巡回（500枚/時）:
- ViTハッシュ抽出: 60秒/時 × 24時間 = 1,440秒/日
- Faiss検索: 0.1秒/時（CPU、無視可能）
- TrustMark確認: 候補10枚 × 0.041秒 = 0.4秒/時

合計GPU時間: 43,200秒/月
月額コスト: $8.64（Modal A10G使用）
```

### DMCA申請テンプレート

自動生成される項目:
- To: dmca@civitai.com（サイトごとに自動選択）
- Subject: DMCA Takedown Notice
- Body: オリジナルURL、転載URL、タイムスタンプ証拠

**法的証拠力**:
- TrustMark透かし: 強い（画像自体に署名）
- ViTハッシュ類似度: 中程度（補助証拠）
- サーバータイムスタンプ: 強い（改ざん不可）

### 参考文献

- **TrustMark論文**: https://arxiv.org/abs/2311.18297
- **GitHub**: https://github.com/adobe/trustmark
- **公式実装**: MIT License
