# 学習・データセット

## データセット

### AI画像（学習用）
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
| pixiv_novelai_v2_ai | 8,859 | Pixiv (dedup済み) |
| twitter_novelai_v2_ai | 12,262 | Twitter (dedup済み) |

### Human画像（学習用）
| カテゴリ | 枚数 | ソース |
|----------|------|--------|
| danbooru_real | 49,998 | Danbooru |

### テスト用画像フォルダ
| フォルダ | 用途 |
|----------|------|
| data/novelai/ | NovelAI (AIBooru) テスト |
| data/novelai_combined/ | Pixiv+Twitter NovelAI テスト |
| data/animedl2m_dataset_release/real_images/images/ | Human テスト |

## pHash重複削除 推奨閾値

| 媒体 | 推奨閾値 |
|------|----------|
| Pixiv | 9 |
| Twitter/X | 11 |
| Danbooru系 | 8 |
| AI生成サイト | 10〜11 |

## 学習→デプロイのワークフロー

**必ず`train`スキルを使う。**

1. データ準備（重複削除）
2. Embedding抽出
3. 学習スクリプト更新・実行
4. **diagnoseスキルでテスト**（新しいスクリプトを作るな）
5. バックエンド再起動

## ベースライン精度（比較用）

```
models/baseline_before_gate/
├── dinov3_classifier.pt
├── dinov3_classifier_cls_only.pt
└── test_results.txt
```

- NovelAI (AIBooru): 83/100 (83%)
- NovelAI Combined: 81/100 (81%)
- Human正解率: 99/100 (99%)

## パッチ統計計算の一貫性（2024-12-25 修正済み）

**修正内容:**
- `extract_embeddings_v2.py`を修正し、775d分類器の先頭768dを使用するようにした
- `backend/main.py`と同じ計算方法になり、学習と推論の一貫性が確保された

**重要:**
- モデルを再学習したら、**全embeddingを再抽出**する必要がある
- 抽出コマンド: `python3 scripts/extract_embeddings_v2.py --dir /path --name name`
- 抽出時は775d分類器（`models/dinov3_classifier.pt`）を使用する
