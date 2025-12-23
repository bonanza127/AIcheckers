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

### 実験中: パッチ統計量 (Patch Stats)
CLSに加えて196パッチから6つの統計量を追加し精度向上を検証中：
| Index | 名前 | 説明 |
|-------|------|------|
| 0 | patch_mean | パッチAIスコアの平均 |
| 1 | patch_max | パッチAIスコアの最大 |
| 2 | patch_var | パッチAIスコアの分散 |
| 3 | max_minus_mean | 最大 - 平均（局所的突出度） |
| 4 | embed_var_mean | 埋め込み分散の平均（パッチ多様性） |
| 5 | count_high_score | スコア≥0.8のパッチ割合 |

効果が確認されれば774次元（768+6）のモデルに移行予定。

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
│   ├── {category}_patch_stats.npy  # パッチ統計量 (N, 6)
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

## TODO: VIP機能

Stripe決済でVIP会員（レート制限解除）を実装予定。
- VIPモーダルUI: 完了
- Stripe Checkoutバックエンド: 完了
- Stripeダッシュボード設定: 未
- OAuth認証: 未

---

## 注意事項

- 新規スクリプト作成前に、その出力を使う側のコードを必ず読め
- 問題発生時は手を止めて原因特定。焦って修正するな
- モデル更新後は必ず `systemctl --user restart aicheckers-backend`
