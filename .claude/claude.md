# AIcheckers

## 概要
アニメ絵特化のAI生成画像判別ツール。DINOv3ベースのLinear Probe分類器「Moonlight」を使用。

## 現在の作業
- [x] パッチ統計量（774次元）モデル導入完了
- [ ] VIP機能実装（Stripe決済、OAuth認証）

## 重要ファイル
- `backend/main.py` - FastAPI バックエンド（Moonlight推論、verdict判定）
- `scripts/train_with_patch_stats.py` - 774次元分類器学習
- `scripts/extract_embeddings_v2.py` - CLS + パッチ統計量抽出
- `models/dinov3_classifier.pt` - 本番モデル（774次元）

## 詳細ドキュメント
- [technical.md](technical.md) - 技術詳細
- [changes.md](changes.md) - 変更履歴
- [kento.md](kento.md) - 検討事項
