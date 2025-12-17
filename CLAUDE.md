# AIcheckers - AI Anime Image Detector

## 概要
アニメ絵特化のAI生成画像判別ツール。日本市場向け。

## 本番URL
- **フロントエンド**: https://aicheckers.net (Vercel)
- **API**: https://api.aicheckers.net (Cloudflare Tunnel → localhost:8000)

## 構成

```
aicheckers/
├── src/app/           # Next.js フロントエンド
│   ├── page.tsx       # メインUI
│   ├── globals.css    # スタイル
│   ├── layout.tsx     # メタタグ/OGP
│   ├── sitemap.ts     # サイトマップ自動生成
│   └── robots.ts      # robots.txt自動生成
├── backend/           # FastAPI バックエンド
│   ├── main.py        # APIサーバー
│   ├── venv/          # Python仮想環境
│   └── requirements.txt
└── CLAUDE.md
```

## 起動方法

### バックエンド（systemdサービス / 自動起動）
```bash
# 状態確認
systemctl --user status aicheckers-backend

# 再起動
systemctl --user restart aicheckers-backend

# ログ確認
journalctl --user -u aicheckers-backend -f
```

### Cloudflare Tunnel（systemdサービス / 自動起動）
```bash
# 状態確認
systemctl --user status cloudflared

# 再起動
systemctl --user restart cloudflared

# ログ確認
journalctl --user -u cloudflared -f
```

### フロントエンド（ローカル開発）
```bash
cd ~/aicheckers
npm run dev
```

### 環境変数
`.env.local` でAPIエンドポイントを設定:
```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## 技術スタック
- **Frontend**: Next.js 16 + React + TypeScript + Tailwind CSS
- **Backend**: FastAPI (Python) + PyTorch
- **Model**: legekka/AI-Anime-Image-Detector-ViT
- **推論環境**: ローカル GTX 1660 (6GB VRAM)

## API

| Endpoint | Method | 説明 |
|----------|--------|------|
| `/health` | GET | ヘルスチェック |
| `/analyze` | POST | 画像判定（multipart/form-data） |

## ロードマップ

### Phase 1: MVP ✅
- [x] フロントエンドUI
- [x] legekkaモデルでバックエンドAPI
- [x] フロントエンド連携

### Phase 2: 精度向上
- [ ] AnimeDL-2Mでファインチューニング（クラウドGPU: $30-50）

### Phase 3: 本格運用
- [ ] AniXplore実装（学習済みモデル配布あり）
- [ ] スケール時VPS移行（RTX 4090: $200-400/月）

## モデル情報

| モデル | パラメータ | 精度 | VRAM |
|--------|-----------|------|------|
| legekka | 87.6M | 94.68% | ~2GB |
| AniXplore | - | F1 0.9999 | 8-16GB |

## 参考リンク
- legekka: https://huggingface.co/legekka/AI-Anime-Image-Detector-ViT
- AniXplore論文: https://arxiv.org/html/2504.11015
- AnimeDL-2M: https://github.com/FlyTweety/AnimeDL2M

## ブランチ管理
- **main**: 本番ブランチ。Vercelに自動デプロイ
- **backup/plan-c-pixel-art-sunset**: デザイン案C（ピクセルアート夕焼けテーマ）
