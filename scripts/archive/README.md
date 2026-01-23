# scripts/archive/

実験・開発過程で作成されたスクリプトのアーカイブ。
2026-01-23 に整理。

## ディレクトリ構成

| ディレクトリ | 内容 |
|-------------|------|
| `tests/` | テストスクリプト。diagnoseスキルに機能統合済み |
| `training/` | 旧学習スクリプト（train_two_head_*, train_28d_* 等） |
| `extraction/` | 旧抽出スクリプト（extract_*の古いバージョン） |
| `analysis/` | AB実験・分析・デバッグ用スクリプト |
| `modal/` | Modal.com上での実験スクリプト |
| `rendering/` | プロモ画像・マーケティング素材生成 |
| `protection/` | SAP/Ironclad等の画像保護実験 |
| `misc/` | その他一回限りのユーティリティ |

## 注意

- これらのスクリプトは**非推奨**です
- テストには `diagnose` スキルを使用してください
- 学習には `train_from_embeddings.py` または `train_28d_plus_60.py` を使用してください
- 必要に応じて参照は可能ですが、そのまま実行すると動かない可能性があります
