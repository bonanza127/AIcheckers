# 変更履歴

## 2024-12-23 - feat: CLS + パッチ統計量 (774次元) モデル導入 (6d2df00)
- 従来のCLSのみ(768次元)から、パッチ統計量6次元を追加
- 精度向上: 98.20% → 98.40% (+0.20%)
- バックエンド: 動的に768/774次元を判別してロード
- verdict判定: 5段階しきい値を`get_verdict()`関数に統一
- 新規スクリプト追加:
  - `scripts/train_with_patch_stats.py`
  - `scripts/extract_embeddings_v2.py`
  - `scripts/extract_patch_stats_only.py`

## 2024-12-23 - feat: 5段階判定システム導入 (e597a12)
- verdict を2段階から5段階に拡張
- フロントエンドに色分け対応

## 2024-12-23 - feat: VAT + Entropy Minimization (e683799)
- VAT: 全エポック適用
- Entropy Minimization: epoch 15から投入
- NaN検出時のスキップ機構追加
