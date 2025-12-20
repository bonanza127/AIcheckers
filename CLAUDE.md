# AIcheckers - AI Anime Image Detector

## 概要
アニメ絵特化のAI生成画像判別ツール。日本市場向け。

## 本番URL
- **フロントエンド**: https://aicheckers.net (Vercel)
- **API**: https://api.aicheckers.net (Cloudflare Tunnel → localhost:8000)

## 現在のアーキテクチャ（2024-12-20更新）

```
aicheckers.net
    ↓
┌──────────────────────────────────────────────────────┐
│ Moonlight V1.3 (ローカル) │ ← 98.29%精度、37-56ms、推奨│
│ AniXplore (Modal)         │ ← F1: 0.9999、コールドスタート25秒│
│ legekka (ローカル)        │ ← 94.68%、汎化性能高      │
└──────────────────────────────────────────────────────┘
```

## モデル比較

| モデル | 方式 | 精度 | 速度 | 強み |
|--------|------|------|------|------|
| **Moonlight V1.3** | Linear Probe (ローカル) | 98.29% | **37-56ms** | 高速・高精度・NovelAI対応 |
| AniXplore | 周波数分析 (Modal) | F1: 0.9999 | 2秒+ | AnimeDL-2M学習データに最強 |
| legekka | ViT分類 (ローカル) | 94.68% | ~200ms | 汎化性能高、未知モデルに対応 |

---

## ブランディング

- **モデル名**: Moonlight V1.3（DINOv3ベースのLinear Probe）
- **ロジック表記**: カスケード方式
- **キャッチコピー**: 「二次元に特化した、日本のためのAIイラストチェッカー」

---

## ファイル構成

```
aicheckers/
├── src/app/                    # Next.js フロントエンド
│   ├── page.tsx                # メインページ
│   ├── how-it-works/page.tsx   # 仕組み説明ページ
│   └── disclaimer/page.tsx     # 免責事項ページ
├── backend/
│   └── main.py                 # FastAPI（3モデル統合）
├── scripts/
│   ├── extract_embeddings.py   # Embedding抽出
│   ├── train_from_embeddings.py # 分類器学習 ★メイン
│   └── aibooru_scraper.py      # NovelAI画像収集
├── embeddings/                 # 保存済み特徴量
└── models/
    └── dinov3_classifier.pt    # Linear Probe分類器
```

---

## Embeddings（全て抽出済み）

```
embeddings/
├── illustrious_ai.npy   # 4,824枚
├── pony_ai.npy          # 19,857枚
├── sdxl10_ai.npy        # 8,916枚
├── sd15_ai.npy          # 9,985枚
├── other_ai.npy         # 4,555枚
├── flux1d_ai.npy        # 1,843枚
├── novelai_ai.npy       # 1,045枚 ★追加済み
└── danbooru_real.npy    # 49,998枚
```

**合計**: AI 51,025枚 + Real 49,998枚 = 約10万枚

---

## 分類器学習

### アーキテクチャ（重要）
```python
# 正しいアーキテクチャ
model = nn.Linear(768, 2)
criterion = nn.CrossEntropyLoss()
```

### 学習コマンド
```bash
# 全embeddingsを結合して分類器を再学習
python scripts/train_from_embeddings.py

# バックエンド再起動
systemctl --user restart aicheckers-backend
```

### 新カテゴリ追加手順
```bash
# 1. 画像収集（例: aibooru）
python scripts/aibooru_scraper.py --count 1000 --rating s --skip 10

# 2. Embedding抽出
python scripts/extract_embeddings.py --dir data/novelai --name novelai_ai

# 3. 分類器再学習（自動でバックアップ作成）
python scripts/train_from_embeddings.py

# 4. バックエンド再起動
systemctl --user restart aicheckers-backend
```

---

## コマンド

```bash
# バックエンド
systemctl --user restart aicheckers-backend
journalctl --user -u aicheckers-backend -f

# Embedding抽出
python scripts/extract_embeddings.py --dir "/path/to/images" --name "category_name"

# 分類器学習
python scripts/train_from_embeddings.py
```

---

## 技術メモ

### ローカル環境
- **GPU**: GTX 1660 (6GB VRAM)
- **DINOv3 VRAM使用**: 0.34GB（余裕あり）

### モデルバックアップ
- 学習スクリプト実行時、自動でタイムスタンプ付きバックアップ作成
- 場所: `models/dinov3_classifier_backup_YYYYMMDD_HHMMSS.pt`

### HuggingFace
- DINOv3はgatedモデル（Metaの承認必要）
- Token: 環境変数 `HF_TOKEN` で管理

---

## 連絡先・メール

- **ドメインメール**: contact@aicheckers.net
- **設定**: Cloudflare Email Routing（受信専用、転送）

---

## 収益化方針

### 短期（3ヶ月後目安）
- **広告**: 忍者AdMax または Google AdSense

### 中期
- **pixivFANBOX**: クリエイター支援型
  - 支援者特典: スキャン無制限
  - 一般ユーザー: 1日10枚まで無料

---

## 動的OGP

- **エンドポイント**: `/api/og?verdict=AI&score=98`
- **シェアページ**: `/share?verdict=AI&score=98`

---

## Cloudflare Bypass

NovelAI画像収集など、Cloudflare保護サイトからのスクレイピング用
- `curl_cffi`ライブラリでChrome偽装
- Skills: `~/.claude/skills/cloudflare-bypass-scraper/`

---

## 参考リンク
- AnimeDL-2M: https://github.com/FlyTweety/AnimeDL2M
- DINOv3: https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m
