# AIcheckers - AI Anime Image Detector

## 概要
アニメ絵特化のAI生成画像判別ツール。日本市場向け。

## 本番URL
- **フロントエンド**: https://aicheckers.net (Vercel)
- **API**: https://api.aicheckers.net (Cloudflare Tunnel → localhost:8000)

## 現在のアーキテクチャ（2024-12-20）

```
aicheckers.net
    ↓
┌──────────────────────────────────────────────────────┐
│ DINOv3 (ローカル)  │ ← 98.35%精度、37-56ms、推奨     │
│ AniXplore (Modal)  │ ← F1: 0.9999、コールドスタート25秒│
│ legekka (ローカル) │ ← 94.68%、汎化性能高            │
└──────────────────────────────────────────────────────┘
```

**モデル切り替え**: UI上でクリックして手動切り替え可能

## モデル比較

| モデル | 方式 | 精度 | 速度 | 強み |
|--------|------|------|------|------|
| **DINOv3** | Linear Probe (ローカル) | 98.35% | **37-56ms** | 高速・高精度・拡張容易 |
| AniXplore | 周波数分析 (Modal) | F1: 0.9999 | 2秒+ | AnimeDL-2M学習データに最強 |
| legekka | ViT分類 (ローカル) | 94.68% | ~200ms | 汎化性能高、未知モデルに対応 |

## ファイル構成

```
aicheckers/
├── src/app/              # Next.js フロントエンド
├── backend/
│   └── main.py           # FastAPI（3モデル統合）
├── scripts/
│   └── extract_embeddings.py  # Embedding抽出スクリプト
├── embeddings/           # 保存済み特徴量 ★重要
│   ├── illustrious_ai.npy      # 4824枚分
│   ├── illustrious_ai_files.txt
│   └── (今後追加: pony_ai.npy, sdxl_ai.npy, danbooru_real.npy)
├── modal_anixplore/      # Modal用AniXplore
├── modal_dinov3/         # DINOv3関連
│   ├── app.py            # Modal版（バックアップ）
│   └── train_linear_probe.py  # 学習スクリプト
└── models/
    ├── AniXplore/        # チェックポイント（gitignore）
    └── dinov3_classifier.pt  # Linear Probe分類器
```

---

## Embedding保存戦略 ★重要

### 概念
DINOv3のLinear Probe方式では、特徴量（embedding）を一度抽出すれば保存・再利用可能。
新データ追加時は新規分のみ抽出し、既存embeddingと結合して分類器を再学習。

### 保存場所
```
embeddings/
├── illustrious_ai.npy     # (4824, 768) float32 ~15MB ✅保存済み
├── illustrious_ai_files.txt
├── pony_ai.npy            # 次に追加予定
├── sdxl_ai.npy            # 次に追加予定
├── danbooru_real.npy      # 必要に応じて追加
└── ...
```

### 新カテゴリ追加手順
```bash
# 1. 新カテゴリのembedding抽出（数分）
python scripts/extract_embeddings.py \
  --dir "/path/to/new_category" \
  --name "category_name"

# 2. 全embeddingを結合して分類器再学習
python scripts/train_classifier.py

# 3. 本番反映
cp models/dinov3_classifier.pt /home/techne/aicheckers/models/
systemctl --user restart aicheckers-backend
```

---

## コマンド

```bash
# バックエンド
systemctl --user restart aicheckers-backend
journalctl --user -u aicheckers-backend -f

# Embedding抽出
python scripts/extract_embeddings.py \
  --dir "/path/to/images" \
  --name "category_name" \
  --batch-size 32

# Modal (AniXplore)
cd ~/aicheckers/modal_anixplore
modal deploy app.py
```

---

## DINOv3 Linear Probe 詳細

### 概要
- **ベースモデル**: facebook/dinov3-vitb16-pretrain-lvd1689m (86M params)
- **方式**: 凍結backbone + 学習済み線形分類器
- **学習データ**: Illustrious 4844枚 + danbooru real 4844枚
- **Validation Accuracy**: 98.35%

### なぜDINOv3を選んだか
1. **学習コストが低い** - 特徴抽出は1回、分類器学習は数分
2. **スケーリングが容易** - 特徴量を蓄積して分類器を再学習するだけ
3. **ローカル推論が高速** - 37-56ms (Modal不要)

### スケーリング方法
```bash
# 既存の特徴量は保存済み、新規データのみ追加抽出
# 例: Ponyデータを追加
python train_linear_probe.py \
  --ai-dir /path/to/pony_images \
  --real-dir /path/to/more_real_images \
  --epochs 20

# 分類器を上書き
cp /tmp/linear_classifier.pt ~/aicheckers/models/dinov3_classifier.pt
systemctl --user restart aicheckers-backend
```

### パラメータ増加について
- **ViT-B (86M)**: 現在使用、十分な性能
- **ViT-L (300M)**: 特徴量再抽出が必要、精度+2%程度
- **ViT-G (1.1B)**: A100必須、コスト高

**結論**: データ増加で対応できる限りViT-Bを維持。データで限界が来たらViT-Lを検討。

---

## AnimeDL-2M データセット

### ダウンロード済み場所
```
/home/techne/aicheckers/data/animedl2m_dataset_release/
├── civitai_subset/image/    # AI生成画像（展開済み）
│   ├── Illustrious/         # 4844枚 ← embedding抽出済み
│   ├── Pony/                # 19882枚 ← 次の追加候補
│   ├── SDXL 1.0/            # 8924枚
│   ├── Other/               # 4555枚
│   └── ...
├── real_images/images/      # danbooru real
│   ├── 0000.tar ~ 0160.tar  # ★ダウンロード済み（17個、約11万枚）
│   └── (展開すると *.jpg, *.png)
└── fake_images/             # 未展開（23GB tar.gz、ユーザーがDL中）
```

### Real Images ダウンロード状況
- **完了**: 0000.tar ~ 0160.tar（17ファイル、step=10）
- **推定画像数**: 約11万枚
- **次回再開位置**: 0170.tar から

### ダウンロード再開方法
```bash
cd /home/techne/aicheckers/data/animedl2m_dataset_release/real_images
# download_more.py を編集して開始位置を 170 に変更
python3 download_more.py
```

### Tar展開方法
```bash
cd /home/techne/aicheckers/data/animedl2m_dataset_release/real_images/images
for tar in *.tar; do tar -xf "$tar"; done
```

### モデル別AI画像数
| モデル | 枚数 | Embedding |
|--------|------|-----------|
| **Illustrious** | 4,824 | ✅ 抽出済み |
| Pony | 19,882 | 次の追加候補 |
| SDXL 1.0 | 8,924 | |
| Other | 4,555 | |
| SD 1.5 | 63,583 | |

---

## 今後の方針

### 優先度高
1. **Pony/Other追加学習** - 特徴抽出して分類器再学習（コスト: 電気代のみ）
2. **精度モニタリング** - 新しいAIモデルが出たら評価

### 中長期
- **fake_images全量展開** - 全230万枚でLinear Probe
- **legekkaファインチューニング** - DINOv3と比較

### 見送り
- AniXploreのファインチューニング（複雑すぎる）
- ViT-L/G（データ増加で対応できる間は不要）

---

## 技術メモ

### ローカル環境
- **GPU**: GTX 1660 (6GB VRAM)
- **DINOv3 VRAM使用**: 0.34GB（余裕あり）
- **同時処理**: ~15-20リクエスト可能

### Modal
- $30/月無料枠（毎月リセット）
- T4 GPU: $0.59/時間
- DINOv3 Modal版はバックアップとして維持

### HuggingFace
- DINOv3はgatedモデル（Metaの承認必要）
- Token: 環境変数 `HF_TOKEN` で管理

## 参考リンク
- AnimeDL-2M: https://github.com/FlyTweety/AnimeDL2M
- DINOv3: https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m
- Modal: https://modal.com/

---

## ブランディング

- **モデル名**: Moonlight V1.3（DINOv3ベースのLinear Probe）
- **ロジック表記**: カスケード方式
- **キャッチコピー**: 「二次元に特化した日本のためのAIイラストチェッカー」

---

## 連絡先・メール

- **ドメインメール**: contact@aicheckers.net
- **設定**: Cloudflare Email Routing（受信専用、転送）
- **用途**: お問い合わせ、FANBOX連絡用

---

## 収益化方針

### 短期（3ヶ月後目安）
- **広告**: 忍者AdMax または Google AdSense
- AdSenseは審査厳しめ（コンテンツ不足で落ちやすい）

### 中期
- **pixivFANBOX**: クリエイター支援型
  - 支援者特典: スキャン無制限
  - 一般ユーザー: 1日10枚まで無料
- pixiv OAuth連携で支援者判定可能

### 検討中
- CAMPFIRE単発クラファン（開発資金・サーバー維持費）
- AI絵を忌避するクリエイター層からの支援が見込める

---

## 動的OGP

- **エンドポイント**: `/api/og?verdict=AI&score=98`
- **シェアページ**: `/share?verdict=AI&score=98`
- **用途**: X共有時に判定結果をバナー表示

---

## NovelAI対応

### 現状
- Moonlight V1.3はNovelAI画像に弱い（学習データに含まれていない）

### データ収集
- **ソース**: aibooru.online（NovelAIタグあり）
- **スクリプト**: `scripts/aibooru_scraper.py`
- **保存先**: `data/novelai/`

```bash
# NovelAI画像収集（Cloudflare bypass使用）
python scripts/aibooru_scraper.py --count 1000 --rating s --skip 10

# 完了後: embedding抽出
python scripts/extract_embeddings.py --dir data/novelai --name novelai_ai

# 分類器再学習
python modal_dinov3/train_linear_probe.py
```

### Cloudflare Bypass
- `curl_cffi`ライブラリでChrome偽装
- Skills: `~/.claude/skills/cloudflare-bypass-scraper/`

---

## セッション履歴 (2024-12-20)

### 実施内容
1. DINOv3について調査（legekkaとの比較）
2. Linear Probe方式を選択（k-NNより高精度）
3. AnimeDL-2Mからデータ準備
   - Illustrious: 4844枚（AI）
   - danbooru: 6340枚（Real）
4. ローカルGPU (GTX 1660) で学習
   - 特徴抽出: ~5分
   - 分類器学習: ~10秒
   - Validation Accuracy: **98.35%**
5. バックエンドに統合（ローカル推論）
6. 本番テスト完了

### 学んだこと
- DINOv3のLinear Probeは学習コストが非常に低い
- ローカル推論はModal (コールドスタート25秒) より圧倒的に速い
- 特徴量は蓄積可能、スケーリングが容易

---

## Civitai API

### 認証情報
- **API Key**: `.env` に保存済み (`CIVITAI_API_KEY`)
- **ドキュメント**: https://developer.civitai.com/

### 画像収集スクリプト
```bash
# 使い方
python scripts/civitai_scraper.py --model pony_v6 --count 1000
python scripts/civitai_scraper.py --version-id 290640 --model my_model --count 500

# 既知のモデルバージョンID
# - pony_v6: 290640
# - illustrious_xl: 1215460
```

### 注意事項
- レート制限: 2〜5秒のランダム待機を入れている
- NSFWフィルタ: デフォルトでSFWのみ
- 新モデル対応時はCivitaiでバージョンIDを調べる
