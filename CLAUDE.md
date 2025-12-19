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

---

## 今後の自分への申し送り（2024-12-17更新）

### 現在の状態
- **ヘッダー**: `AIチェッカー // AI-art-integrity V4.2`
- **モデル表示名**: `Mirror_of_Ra-Vit V1.1`（実体はlegekka）
- **判定ロジック**: 80%以上=AI、50-80%=UNKNOWN、50%未満=HUMAN

### 本日実装した機能
1. **Attention Map可視化**: ViTの注意マップをヒートマップとしてオーバーレイ表示
2. **バックエンド接続状態表示**: ONLINE/OFFLINE をヘッダーに表示
3. **X共有ボタン**: 最終判定ボックス右上、ホバーでぼんやり光る
4. **SEO改善**:
   - JSON-LD構造化データ（WebApplication）
   - 動的OGP画像生成（`opengraph-image.tsx`）
   - H1タグ最適化
5. **リアルな分析ログ**: スキャン中にステージ別ログを表示

### 既知の課題
- **legekkaモデル（2023年）は最新AIに弱い可能性あり**
  - SD系（Animagine, Waifu Diffusion）は99%以上で検出可能（テスト済み）
  - NovelAI V4、FLUX等の2024年モデルは未検証
- **AniXplore移行を検討中**
  - F1スコア0.9999の最新モデル
  - VRAM 8-16GB必要（現GTX 1660では厳しい）
  - Google Colabで動かす案あり

### SEO状況
- sitemap.xml、robots.txt設置済み
- Google Search Console未登録（要対応）
- 検索上位を狙うならコンテンツ（ブログ記事）追加が有効

### バックエンド注意点
- `attn_implementation="eager"` でモデルロードしないとAttention Map取れない
- matplotlib必須（ヒートマップ生成用）

### ファイル構成追加
```
src/app/
├── opengraph-image.tsx  # 動的OGP画像生成
```

### 今後やるかもしれないこと
- [ ] AniXploreへのモデル移行（Colab or クラウドGPU）
- [ ] ブログ/解説ページ追加（SEO用）
- [ ] 判定結果の画像付きシェア機能
- [ ] ユーザー報告機能（誤判定フィードバック収集）

---

## AniXplore + Modal 導入メモ（2024-12-19）

### 背景・調査結果

#### モデル比較
| モデル | 精度 | VRAM | 学習データ | 状態 |
|--------|------|------|-----------|------|
| legekka (現在) | 94.68% | ~2GB | 120万枚 (2023) | GTX 1660で稼働中 |
| AniXplore | F1: 0.9999 | 重い | AnimeDL-2M (2025) | Modal導入中 |
| saltacc/anime-ai-detect | 96% (自称) | ~2GB | 22,000枚のみ | 信頼性低い、放置 |

#### データセット比較
| データセット | 規模 | 内容 | 年代 |
|-------------|------|------|------|
| AnimeDL-2M | 200万枚+ | リアル/部分改変/完全AI生成 | 2025 |
| legekka学習データ | 120万枚 | リアル100万 + AI 21.7万 | 2023 |

**結論**: AniXploreが最新かつ最高精度。ただしVRAMが重いのでサーバーレスGPUで動かす。

### Modal導入

#### なぜModalか
- **$30/月の無料枠**（毎月リセット）
- サーバーレス = 使った分だけ課金、アイドル時は$0
- T4 GPUで約$0.0006/秒、1回の推論で$0.003程度
- 常時稼働VPS（$150-400/月）より大幅に安い

#### 2段構成アーキテクチャ
```
┌─────────────────────────────────────────────┐
│  aicheckers.net                             │
│  ┌─────────────────────────────────────┐    │
│  │ 1次判定: legekka (GTX 1660)        │    │
│  │ → 明確な判定（80%以上 or 50%未満）  │    │
│  └────────────┬────────────────────────┘    │
│               ↓ 微妙なケースのみ             │
│  ┌─────────────────────────────────────┐    │
│  │ 2次判定: AniXplore (Modal)         │    │
│  │ → サーバーレスGPUで呼び出し         │    │
│  └─────────────────────────────────────┘    │
└─────────────────────────────────────────────┘
```

### ファイル構成

```
aicheckers/
├── modal_anixplore/           # Modal用AniXploreデプロイ
│   ├── app.py                 # Modalアプリ定義
│   ├── anixplore_model.py     # AniXploreモデル（推論用簡略版）
│   └── test_image.py          # テストスクリプト
├── models/
│   └── AniXplore/
│       └── checkpoint-29/     # AniXploreチェックポイント (608MB)
```

### Modal操作コマンド

```bash
# Modalにログイン（初回のみ）
modal token new

# アプリをデプロイ
cd ~/aicheckers/modal_anixplore
modal deploy app.py

# テスト実行（ダミー入力）
modal run app.py

# チェックポイントをvolumeにアップロード
modal volume put anixplore-checkpoints ~/aicheckers/models/AniXplore/checkpoint-29 AniXplore/checkpoint-29

# 画像でテスト
python3 test_image.py /path/to/image.png

# デプロイ状況確認
# https://modal.com/apps/hokhok7676/main/deployed/anixplore-detector
```

### AniXploreモデル詳細

#### アーキテクチャ
- **ConvNeXt**: 高周波特徴抽出（DCT+DWT）
- **MixVisionTransformer (SegFormer系)**: 低周波特徴抽出
- **Feature Pyramid Network**: マルチスケール特徴統合
- **分類ヘッド**: AI生成確率を出力

#### 入力
- 512x512 RGB画像（AniXploreHRは1024x1024）
- 自動リサイズ対応

#### 出力
- `probability`: 0-1のAI生成確率
- `is_ai`: probability > 0.5
- `confidence`: 確信度

### チェックポイント

- **ダウンロード元**: https://drive.google.com/drive/folders/1HQWMh0SSOL1rWTNgbTdS8Jokm-Imq0CQ
- **AniXplore**: 608MB（512x512版）
- **AniXploreHR**: 608MB x 2（1024x1024版、より高精度だがVRAM重い）

**重要**: Google Driveからダウンロードすると `checkpoint-29` ディレクトリ形式になる。
PyTorchで読み込むには **zipに圧縮** する必要がある：
```bash
cd ~/aicheckers/models/AniXplore
zip -r checkpoint-29.pt checkpoint-29/
# これでtorch.load('checkpoint-29.pt')で読める
```

チェックポイント構造:
```python
{
    'model': {...},      # モデル重み
    'optimizer': {...},  # オプティマイザ状態
    'epoch': 29,         # エポック数
    'scaler': {...},     # AMP scaler
    'args': {...}        # 学習引数
}
```

### 参考リンク
- Modal: https://modal.com/
- Modal Pricing: https://modal.com/pricing
- AniXplore GitHub: https://github.com/FlyTweety/AnimeDL2M
- AnimeDL-2M論文: https://arxiv.org/abs/2504.11015

### 次のステップ
1. [ ] Modal上でのAniXplore動作確認（チェックポイントロード）
2. [ ] aicheckersバックエンドとModal統合
3. [ ] 2段構成の実装（legekka → AniXplore fallback）
4. [ ] コスト監視とアラート設定
