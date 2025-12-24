# AIcheckers - AI Anime Image Detector

アニメ絵特化のAI生成画像判別ツール。日本市場向け。

## 本番環境

| 項目 | URL/値 |
|------|--------|
| フロントエンド | https://aicheckers.net (Vercel) |
| API | https://api.aicheckers.net (Cloudflare Tunnel → localhost:8000) |
| 精度 | 98.10% |
| 速度 | 70-120ms (TTA有効時) |

---

## 技術スタック（現行）

```
DINOv3 (facebook/dinov3-vitb16-pretrain-lvd1689m)
    ↓
CLS Token (768次元)
    ↓
Linear Probe (nn.Linear(768, 2))
    ↓
TTA + Temperature Scaling
```

### パッチ統計量 (Patch Stats) - 現行775次元
CLSに加えて196パッチから7つの統計量を追加：
| Index | 名前 | AI-Real差分 | 有効性 |
|-------|------|-------------|--------|
| 0 | patch_mean | +0.23 | **有効** - パッチAIスコアの平均 |
| 1 | patch_max | +0.02 | 弱い - パッチAIスコアの最大 |
| 2 | patch_var | +0.02 | 弱い - パッチAIスコアの分散 |
| 3 | max_minus_mean | -0.22 | **有効** - 最大-平均（局所的突出度、Realが高い） |
| 4 | embed_var_mean | +0.003 | 無効 - 埋め込み分散の平均 |
| 5 | count_high_score | +0.22 | **有効** - スコア≥0.8のパッチ割合 |
| 6 | v_high_sim_85 | +0.02 | 弱い - 垂直方向の高類似度パッチ比率 |

### 将来の最適化プラン: 771次元化
有効な3特徴量のみ残す案：
- 残す: patch_mean, max_minus_mean, count_high_score
- 削除: patch_max, patch_var, embed_var_mean, v_high_sim_85
- メリット: 理論上のノイズ削減（ただし実用上はほぼ変わらない）
- 優先度: 低（害がないため現状維持で運用）

---

## ファイル構成

```
aicheckers/
├── src/app/                        # Next.js フロントエンド
├── backend/
│   └── main.py                     # FastAPI
├── scripts/
│   ├── extract_embeddings_v2.py    # CLS + パッチ統計量抽出
│   ├── extract_patch_stats_only.py # パッチ統計量のみ追加抽出
│   ├── train_from_embeddings.py    # 分類器学習（CLSのみ）
│   └── train_with_patch_stats.py   # 分類器学習（CLS+パッチ統計量）
├── embeddings/
│   ├── {category}.npy              # CLSトークン (N, 768)
│   ├── {category}_patch_stats.npy  # パッチ統計量 (N, 7)
│   └── {category}_files.txt        # ファイル名リスト
└── models/
    └── dinov3_classifier.pt        # 本番モデル
```

---

## データセット

### AI画像 (51,025枚)
| カテゴリ | 枚数 | ソース |
|----------|------|--------|
| illustrious_ai | 4,824 | AnimeDL-2M |
| pony_ai | 19,857 | AnimeDL-2M |
| sdxl10_ai | 8,916 | AnimeDL-2M |
| sd15_ai | 9,985 | AnimeDL-2M |
| other_ai | 4,555 | AnimeDL-2M |
| flux1d_ai | 1,843 | AnimeDL-2M |
| novelai_ai | 1,045 | AIBooru/Pixiv |
| pixai_ai | 1,018 | PixAI |

### Human画像 (49,998枚)
| カテゴリ | 枚数 | ソース |
|----------|------|--------|
| danbooru_real | 49,998 | Danbooru |

**元画像の場所**: `data/animedl2m_dataset_release/civitai_subset/image/`

---

## 分類器の保存形式（重要）

```python
# バックエンドが期待する形式
checkpoint = torch.load(path)
classifier.load_state_dict(checkpoint["classifier"])

# 正しい保存方法
torch.save({
    "classifier": model.state_dict(),
    "val_acc": best_acc
}, OUTPUT_PATH)
```

---

## よく使うコマンド

```bash
# バックエンド再起動
systemctl --user restart aicheckers-backend
journalctl --user -u aicheckers-backend -f

# Embedding抽出（CLS + パッチ統計量）
python3 scripts/extract_embeddings_v2.py --dir /path/to/images --name category_name

# 既存Embeddingにパッチ統計量を追加
python3 scripts/extract_patch_stats_only.py --name category_name --image-dir /path/to/images

# 分類器学習
python3 scripts/train_with_patch_stats.py           # CLS + パッチ統計量
python3 scripts/train_with_patch_stats.py --cls-only # CLSのみ（ベースライン）
```

---

## 精度向上テクニック

### 推論時
| 技術 | 効果 | 設定 |
|------|------|------|
| TTA | +0.5〜1% | `TTA_ENABLED` 環境変数 |
| Temperature Scaling | 過信防止 | `TEMPERATURE` 環境変数 (default: 1.5) |

### 学習時
| 技術 | タイミング | 効果 |
|------|------------|------|
| VAT | 全エポック | 決定境界を滑らかに |
| Entropy Minimization | epoch 15〜 | 曖昧な予測を減らす |
| Consistency Regularization | epoch 5〜 | embedding空間での一貫性 |

---

## 環境情報

- **GPU**: GTX 1660 (6GB VRAM)
- **DINOv3 VRAM**: 0.34GB
- **HuggingFace Token**: 環境変数 `HF_TOKEN`（gatedモデル）
- **レート制限**: `backend/main.py` の `RATE_LIMIT_ENABLED`

---

## 開発者アカウント（レート制限免除）

| 用途 | Email | Password |
|------|-------|----------|
| オーナー | hokhok7676@gmail.com | (Google OAuth) |
| DLsite検証用 | dlsite-trial@aicheckers.net | `*z!3kFAD3QXtXx` |

**レート制限免除設定**: `backend/main.py` の `ADMIN_EMAILS` に追加

---

## TODO: VIP機能

Stripe決済でVIP会員（レート制限解除）を実装予定。
- VIPモーダルUI: 完了
- Stripe Checkoutバックエンド: 完了
- Stripeダッシュボード設定: 未
- OAuth認証: 未

---

## 進行中タスク

### 775次元モデル運用中（2024/12/24）
**状況**: 本番稼働中、精度97.35%

v_high_sim_85を追加して7次元化完了。検証の結果、有効な特徴量は3つのみ（patch_mean, max_minus_mean, count_high_score）だが、無効な特徴量も害にならないため現状維持で運用。

---

### Twitter NovelAIデータ収集
- gallery-dlで`#NovelAI`タグの画像をダウンロード中
- 保存先: `data/twitter_novelai/`
- 現在: 約1200枚
- gallery-dl設定: `~/.config/gallery-dl/config.json`（Twitter認証済み）
- 完了後: `scripts/extract_embeddings_v2.py`でembedding抽出 → 学習データに追加

### スマートスクレイパー（未完成）
- `scripts/smart_scraper.py`: ダウンロード→即評価→フィルタリングの自動パイプライン
- gallery-dlで一時フォルダにDL → patch_stats抽出 → 低スコア画像を削除 → 合格のみ保存
- 課題: 775次元分類器ではパッチ評価不可（パッチは768次元）。768次元分類器を別途使う必要あり

---

## 注意事項

- 新規スクリプト作成前に、その出力を使う側のコードを必ず読め
- 問題発生時は手を止めて原因特定。焦って修正するな
- モデル更新後は必ず `systemctl --user restart aicheckers-backend`
- **バグの原因を外部（キャッシュ、ユーザー）のせいにするな。コードに問題がある前提で調査せよ**
