# AIcheckers - AI Anime Image Detector

## 次回TODO: VIP機能の方針決定

### 現状
- VIPモーダル実装済み（FANBOX連携 + 直接支援のUI）
- FANBOX連携: pixiv OAuthが非公開のため、毎回pixiv ID手動入力が必要（UX悪い）
- 直接支援: Stripe実装が必要だが、OAuth（Google/Twitter）でスムーズにログイン可能

### 方針候補
1. **直接支援のみ**: FANBOX連携を削除し、Stripe + OAuthで実装
2. **両方残す**: FANBOXは既存支援者向けとして残し、メインは直接支援

### 次のアクション
- 方針決定後、不要な方を削除
- 直接支援を実装する場合: Stripe決済 + Google/Twitter OAuth設定

---

## 概要
アニメ絵特化のAI生成画像判別ツール。日本市場向け。

## 本番URL
- **フロントエンド**: https://aicheckers.net (Vercel)
- **API**: https://api.aicheckers.net (Cloudflare Tunnel → localhost:8000)

## 現在のアーキテクチャ（2025-12-20更新）

```
aicheckers.net → Moonlight (DINOv3 Linear Probe) のみ
```

**精度**: 98.13% | **速度**: 37-56ms

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
# 現在のアーキテクチャ（変更する場合はバックアップを取ってから）
model = nn.Linear(768, 2)
criterion = nn.CrossEntropyLoss()
```

### 保存形式（絶対厳守）
```python
# バックエンド (backend/main.py:111-112) が期待する形式：
checkpoint = torch.load(path)
classifier.load_state_dict(checkpoint["classifier"])  # ← "classifier" キー必須

# 正しい保存方法：
torch.save({
    "classifier": model.state_dict(),
    "val_acc": best_acc
}, OUTPUT_PATH)

# ❌ 間違い（これをやるとバックエンドがロードできない）：
torch.save(model.state_dict(), OUTPUT_PATH)
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

### レート制限
- `backend/main.py` 47行目: `RATE_LIMIT_ENABLED = True/False`
- 現在: **無効**（開発中）
- 有効時: 上限24枚、1時間刻みで1枚回復
- 変更後: `systemctl --user restart aicheckers-backend`

### FANBOX VIP連携
- **フロー**: ユーザーがFANBOXで支援 → pixiv ID入力 → 同期確認 → VIP付与
- **VIPデータ**: `data/vip_users.json`
- **環境変数**: `FANBOXSESSID` にクリエイターのFANBOXセッションCookieを設定
- **取得方法**: ブラウザでFANBOXにログイン → DevTools → Application → Cookies → `FANBOXSESSID`
- **API**: `/verify-fanbox` (POST), `/check-vip/{pixiv_id}` (GET)

### 画像キャッシュ
- SHA256ハッシュでLRU 10,000件（同一画像のGPU処理スキップ）
- メモリ内のみ（再起動でクリア）

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

## 注意事項

- **新規スクリプト作成前に、その出力を使う側のコードを必ず読め**
- **問題発生時は手を止めて原因特定。焦って修正するな**

---

## 参考リンク
- AnimeDL-2M: https://github.com/FlyTweety/AnimeDL2M
- DINOv3: https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m
