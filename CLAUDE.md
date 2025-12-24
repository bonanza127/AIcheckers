# AIcheckers - AI Anime Image Detector

アニメ絵特化のAI生成画像判別ツール。日本市場向け。

## 本番環境

| 項目 | URL/値 |
|------|--------|
| フロントエンド | https://aicheckers.net (Vercel) |
| API | https://api.aicheckers.net (Cloudflare Tunnel → localhost:8000) |
| 速度 | 70-120ms (TTA有効時) |

---

## 技術スタック（現行）

```
DINOv3 (facebook/dinov3-vitb16-pretrain-lvd1689m)
    ↓
CLS Token (768次元) + Patch Stats (7次元)
    ↓
Linear Probe (nn.Linear(775, 2))
    ↓
TTA + Temperature Scaling
```

### パッチ統計量 (Patch Stats) - 現行775次元
| Index | 名前 | 説明 |
|-------|------|------|
| 0 | patch_mean | パッチAIスコアの平均 |
| 1 | patch_max | パッチAIスコアの最大 |
| 2 | patch_var | パッチAIスコアの分散 |
| 3 | max_minus_mean | 最大-平均（局所的突出度） |
| 4 | embed_var_mean | 埋め込み分散の平均 |
| 5 | count_high_score | スコア≥0.8のパッチ割合 |
| 6 | v_high_sim_85 | 垂直方向の高類似度パッチ比率 |

---

## モデル評価の正しい手順（重要）

### やってはいけないこと
- **curlでAPIを叩いてテストしない** - レート制限に引っかかり、エラーが返ってくる
- **bashでJSONパースしない** - 複雑なレスポンスはPythonで処理せよ
- Validation Accuracy（96%+）と実世界の検出率（88-90%）を混同しない

### 正しいテスト方法
学習後は以下のPythonスクリプトでテストせよ：

```python
# scripts/test_model.py として保存済み
python3 scripts/test_model.py --model models/dinov3_classifier.pt

# 出力例:
# NovelAI (AIBooru): 44/50 detected (84.5% avg)
# NovelAI Combined: 45/50 detected (83.4% avg)
# Human (Danbooru): 99/100 correct (10.2% avg)
```

### 精度の読み方
| 指標 | 意味 | 目標 |
|------|------|------|
| Validation Accuracy | 学習時のホールドアウト精度 | 96%+ |
| AI検出率 | テスト画像でAI≥50%の割合 | 88%+ |
| Human正解率 | テスト画像でAI<50%の割合 | 98%+ |
| 平均AIスコア(AI画像) | AI画像の平均スコア | 80%+ |
| 平均AIスコア(Human画像) | Human画像の平均スコア | 20%以下 |

### CLS-only vs CLS+Patch Stats
- 両者の差は誤差範囲内（0.1-0.2%程度）
- どちらを使っても実用上の差はない
- 現行は775次元（CLS+Patch Stats）を採用

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
│   ├── train_from_embeddings.py    # 分類器学習
│   ├── test_model.py               # モデルテスト（学習後に使用）
│   └── dedup_images.py             # pHashによる重複削除
├── embeddings/
│   ├── {category}.npy              # CLSトークン (N, 768)
│   ├── {category}_patch_stats.npy  # パッチ統計量 (N, 7)
│   └── {category}_files.txt        # ファイル名リスト
└── models/
    └── dinov3_classifier.pt        # 本番モデル (775次元)
```

---

## データセット

### AI画像 (57,825枚)
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

### Human画像 (49,998枚)
| カテゴリ | 枚数 | ソース |
|----------|------|--------|
| danbooru_real | 49,998 | Danbooru |

### テスト用画像フォルダ
| フォルダ | 用途 |
|----------|------|
| data/novelai/ | NovelAI画像（AIBooruソース）テスト用 |
| data/novelai_combined/ | Pixiv+Twitter NovelAI画像 |
| data/animedl2m_dataset_release/real_images/images/ | Human画像テスト用 |

---

## 分類器の保存形式

```python
# 保存
torch.save({
    "classifier": model.state_dict(),
    "val_acc": best_acc,
    "input_dim": input_dim,      # 775 or 768
    "use_patch_stats": True/False
}, OUTPUT_PATH)

# ロード（バックエンドが期待する形式）
checkpoint = torch.load(path)
input_dim = checkpoint.get("input_dim", 768)
classifier = nn.Linear(input_dim, 2)
classifier.load_state_dict(checkpoint["classifier"])
```

---

## よく使うコマンド

```bash
# バックエンド再起動（モデル更新後は必須）
systemctl --user restart aicheckers-backend
journalctl --user -u aicheckers-backend -f

# Embedding抽出（新規カテゴリ追加時）
python3 scripts/extract_embeddings_v2.py --dir /path/to/images --name category_name

# 分類器学習
python3 scripts/train_from_embeddings.py

# モデルテスト（学習後に必ず実行）
python3 scripts/test_model.py --model models/dinov3_classifier.pt

# 画像重複削除（新規データ追加前）
python3 scripts/dedup_images.py --dir /path/to/images --threshold 16
```

---

## 学習→デプロイのワークフロー

1. **データ準備**
   - 新規画像を収集
   - `dedup_images.py`で重複削除（threshold 16推奨）
   - `extract_embeddings_v2.py`でembedding抽出

2. **学習**
   - `train_from_embeddings.py`のAI_CATEGORIESに新カテゴリ追加
   - 学習実行：`python3 scripts/train_from_embeddings.py`
   - ログ確認：Validation Accuracy 96%+を確認

3. **テスト（必須）**
   - `python3 scripts/test_model.py --model models/dinov3_classifier.pt`
   - AI検出率88%+、Human正解率98%+を確認
   - **curlでテストしない**（レート制限あり）

4. **デプロイ**
   - `systemctl --user restart aicheckers-backend`
   - 本番サイトで動作確認

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
| Entropy Minimization | epoch 20〜 | 曖昧な予測を減らす |
| Consistency Regularization | epoch 5〜 | embedding空間での一貫性 |

---

## 環境情報

- **GPU**: GTX 1660 (6GB VRAM)
- **DINOv3 VRAM**: 0.34GB
- **HuggingFace Token**: 環境変数 `HF_TOKEN`
- **レート制限**: `backend/main.py` の `RATE_LIMIT_ENABLED`

---

## 開発者アカウント（レート制限免除）

| 用途 | Email |
|------|-------|
| オーナー | hokhok7676@gmail.com |
| DLsite検証用 | dlsite-trial@aicheckers.net |

**設定**: `backend/main.py` の `ADMIN_EMAILS`

---

## 注意事項

- 新規スクリプト作成前に、その出力を使う側のコードを必ず読め
- 問題発生時は手を止めて原因特定。焦って修正するな
- モデル更新後は必ず `systemctl --user restart aicheckers-backend`
- **バグの原因を外部（キャッシュ、ユーザー）のせいにするな。コードに問題がある前提で調査せよ**
- **APIをcurlでテストするな。Pythonでテストせよ**
