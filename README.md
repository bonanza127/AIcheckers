<div align="center">

# 🔍 AI Checkers

### 二次元に特化したAIイラスト検出エンジン

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Accuracy](https://img.shields.io/badge/Detection%20Accuracy-98.35%25-blue)]()
[![Status](https://img.shields.io/badge/Status-Active-brightgreen)]()
[![Users](https://img.shields.io/badge/Active%20Users-500%2B-orange)]()

**[🌐 Live Site](https://aicheckers.net/)** · **[📖 Docs](#documentation)** · **[💬 Contact](#contact)**

</div>

---

## 概要

AI Checkers は、AI生成されたアニメ・イラストを高精度で検出するオープンソースの判定エンジンです。二次元イラストに特化して独自にファインチューニングした Vision Transformer (ViT) モデルと手作り特徴量のハイブリッド手法により、AI生成画像と人間の手描きイラストを **98.35%** の精度で判別します。

日本のイラストコミュニティにおいて、AI生成作品の人間作品コンテストへの不正参加や出所偽装が深刻化する中、本プロジェクトはクリエイターの権利を技術的に保護するためのインフラとして開発・運用されています。

## 主な機能

- **高精度AI画像検出** — ViT + handcrafted features のハイブリッド判定
- **リアルタイム解析コンソール** — アップロードから判定まで数秒
- **バッチ処理対応** — 複数画像の一括スキャン
- **検出痕跡の可視化** — どの特徴量がAI生成の兆候を示したかを表示
- **REST API** — サードパーティサービスへの組み込み可能
- **フリーティア + VIP** — 1日24枚無料、VIPで無制限

## アーキテクチャ

```
┌─────────────────────────────────────────────────┐
│                  Frontend (Web)                  │
│         Upload UI · Console · History             │
└──────────────────────┬──────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────┐
│              API Gateway / Backend               │
│    Rate limiting · Auth · Queue management       │
└──────┬───────────────────────────┬──────────────┘
       │                           │
┌──────▼──────────┐    ┌───────────▼──────────────┐
│  ViT Inference   │    │  Feature Extractor       │
│  (Moonlight V1)  │    │  (Handcrafted features)  │
│  CLS token       │    │  Edge / noise / freq     │
└──────┬───────────┘    └───────────┬──────────────┘
       │                            │
┌──────▼────────────────────────────▼──────────────┐
│              Fusion / Decision Layer              │
│       Weighted ensemble → final verdict           │
└──────────────────────────────────────────────────┘
```

## モデル

### Moonlight V1.3.6

| 項目 | 仕様 |
|------|------|
| アーキテクチャ | Vision Transformer (ViT) |
| 学習データ | 10,000+ アニメイラスト (AI生成 + 手描き) |
| 判定方式 | CLS token + handcrafted features |
| 精度 | 98.35% (10,000枚検証) |
| 推論時間 | ~0.3s / image (GPU) |

### 検出対象の痕跡

- 周波数ドメインの不自然なパターン
- エッジの滑らかさの分布異常
- ノイズテクスチャの一貫性欠如
- ピクセルレベルの統計的バイアス
- 背景と主体の境界の不自然さ

## セットアップ

### 必要環境

- Python 3.10+
- CUDA 11.8+ (GPU推論の場合)
- 4GB+ VRAM (推奨)

### インストール

```bash
git clone https://github.com/bonanza127/aicheckers.git
cd aicheckers
pip install -r requirements.txt
```

### モデルのダウンロード

```bash
# モデルウェイトは リリースページ から取得
python scripts/download_model.py --version v1.3.6
```

### ローカル実行

```bash
# Web UI を起動
python app.py --host 0.0.0.0 --port 8080

# API のみ起動
python api.py --host 0.0.0.0 --port 5000
```

## API 使用例

```python
import requests

response = requests.post(
    "https://aicheckers.net/api/v1/detect",
    files={"image": open("illustration.png", "rb")}
)

result = response.json()
print(f"AI Probability: {result['ai_probability']}%")
print(f"Classification: {result['classification']}")
print(f"Model: {result['model']}")
print(f"Processing Time: {result['processing_time']}s")
```

## ロードマップ

- [x] ViT ベースラインモデル (v1.0)
- [x] Handcrafted features 統合 (v1.2)
- [x] Web コンソール リリース
- [x] REST API 公開
- [x] バッチ処理対応 (v1.3.6)
- [ ] Stable Diffusion XL / SD3 対応強化
- [ ] モバイルアプリ (iOS / Android)
- [ ] ブラウザ拡張機能
- [ ] 多言語対応 (英語 / 韓国語 / 中国語)

## コントリビュート

プルリクエストを歓迎します。特に以下の領域で支援を求めています：

- **モデル改善** — 新しいAI生成手法に対する検出精度向上
- **データセット拡充** — 多様なスタイルのアニメイラストサンプル
- **フロントエンド** — UI/UX の改善、アクセシビリティ対応
- **インフラ** — 推論パイプラインの最適化、レイテンシ削減

### 開発フロー

1. Issue を作成して変更内容を議論
2. フィーチャーブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add amazing feature'`)
4. PR を作成 — CI が自動的にモデル精度テストを実行
5. レビュー通過後にマージ

## ライセンス

MIT License — 詳細は [LICENSE](LICENSE) を参照

## 謝辞

- 日本のイラストレーター community の皆様からのフィードバックに感謝します
- モデル学習に使用したデータセットの作成者の方々に感謝します

## Contact

- **Website**: [aicheckers.net](https://aicheckers.net/)
- **Inquiries**: サイトのお問い合わせフォームから
- **Issues**: [GitHub Issues](https://github.com/bonanza127/aicheckers/issues)

---

<div align="center">

**AI生成技術の進化は止められない。だからこそ、クリエイターを守る技術も進化し続けなければならない。**

</div>