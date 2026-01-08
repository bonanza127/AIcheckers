# AIパトロール機能

無断転載監視システム。TrustMark透かし + ViTハッシュのハイブリッド方式。

## 技術スタック

```
TrustMark (Adobe Research, ICCV 2025)
    ↓ 透かし埋め込み（user_id + timestamp）
DINOv3 埋め込み抽出（768次元）
    ↓ DB保存
Civitai/Danbooru 新着巡回
    ↓
Faiss類似検索（0.93閾値）
    ↓
TrustMark透かし確認（最終判定）
    ↓
DMCA申請メール自動生成
```

## Guard時の処理順序（推奨）

**重要**: Gemini分析に基づき、以下の順序で処理

```python
1. TrustMark透かし埋め込み（alpha=1.15）
   → FastProtectの摂動を見越して強めに設定

2. FastProtect（MoonKnight）摂動追加
   → 透かし入り画像を「ベース」として最適化
   → 透かし信号を破壊しない摂動が計算される

3. DINOv3埋め込み抽出
   → 最終画像から抽出（Patrol時に照合）
```

**理由**: 逆順序（FastProtect → TrustMark）だと、GANが摂動を「ノイズ」と判断して平滑化してしまう可能性

## TrustMark性能（GTX 1660）

| 処理 | 時間 | VRAM |
|------|------|------|
| **エンコード** | 0.199秒 | 0.28GB |
| **デコード** | 0.041秒 | 0.28GB |
| **透かし容量** | 61bit | - |

**FastProtect耐性**: DINOv3が確認済み（マイロード検証）

## 実装ファイル

| ファイル | 用途 |
|----------|------|
| `lib/trustmark_helper.py` | 透かし埋め込み・抽出ヘルパー |
| `scripts/test_trustmark.py` | TrustMark動作確認スクリプト |
| `backend/main.py` | Guard/Patrolエンドポイント |

## 実装進捗（2026-01-09）

**Phase 1: TrustMark + ViT Hash統合（完了）**
- [x] TrustMarkライブラリインストール・動作確認
- [x] trustmark_helper.py作成
- [x] backend/main.pyへのimport追加
- [x] Guard時のTrustMark透かし埋め込み実装（alpha=1.15）
- [x] Guard時のViTハッシュ+タイムスタンプDB保存実装
- [x] 処理順序実装（TrustMark → MoonKnight → DINOv3）
- [x] patrol_embeddings.json DB構造実装
- [x] 実験的検証（透かし検出率、FastProtect効果）

**Phase 2: Patrol機能（未着手）**
- [ ] Civitai API連携とFaiss検索基盤
- [ ] 2段階検証システム（ViT→TrustMark）
- [ ] DMCA申請メール自動生成
- [ ] Firecrawl MCP統合（スクレイピング支援）

## 実験結果（2026-01-09）

**Guard API統合テスト:**
- ✅ TrustMark透かし埋め込み成功（0.199秒/画像）
- ✅ MoonKnight V3保護適用成功
- ✅ DINOv3埋め込み抽出成功（768次元）
- ✅ patrol_embeddings.json保存成功（タイムスタンプ + 透かしハッシュ）
- ✅ 透かし検出率: 100%（MoonKnight適用後も維持）
- ✅ 処理時間: 約25秒/画像（TrustMark + MoonKnight + DINOv3）

**結論:**
TrustMark透かしはMoonKnight V3摂動の影響を受けず、完全に検出可能。ハイブリッド方式（透かし + ViTハッシュ）の技術的実現性を確認。

## 監視対象サイト

| サイト | 実現性 | 優先度 |
|--------|--------|--------|
| **Civitai** | ⭐⭐⭐⭐⭐ 公式API | ★★★★★ |
| **Danbooru** | ⭐⭐⭐⭐ 公式API | ★★★★ |
| Gelbooru | ⭐⭐⭐ 非公式API | ★★★ |
| Pixiv | ⭐⭐ スクレイピング（規約違反リスク） | ★★ |
| Kemono.party | ⭐ 違法サイト（法的リスク極大） | ❌ |

**Phase 1推奨**: Civitai + Danbooru のみ（合法・低コスト）

## コスト試算

```
月間10,000ユーザー、各10枚保護の場合:

Civitai新着巡回（500枚/時）:
- ViTハッシュ抽出: 60秒/時 × 24時間 = 1,440秒/日
- Faiss検索: 0.1秒/時（CPU、無視可能）
- TrustMark確認: 候補10枚 × 0.041秒 = 0.4秒/時

合計GPU時間: 43,200秒/月
月額コスト: $8.64（Modal A10G使用）
```

## DMCA申請テンプレート

自動生成される項目:
- To: dmca@civitai.com（サイトごとに自動選択）
- Subject: DMCA Takedown Notice
- Body: オリジナルURL、転載URL、タイムスタンプ証拠

**法的証拠力**:
- TrustMark透かし: 強い（画像自体に署名）
- ViTハッシュ類似度: 中程度（補助証拠）
- サーバータイムスタンプ: 強い（改ざん不可）

## 参考文献

- **TrustMark論文**: https://arxiv.org/abs/2311.18297
- **GitHub**: https://github.com/adobe/trustmark
- **公式実装**: MIT License
