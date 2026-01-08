# 環境情報・技術スタック

## ハードウェア環境

- **GPU**: GTX 1660 (6GB VRAM)
- **DINOv3 VRAM**: 0.34GB
- **HuggingFace Token**: 環境変数 `HF_TOKEN`

## 本番環境

| 項目 | URL/値 |
|------|--------|
| フロントエンド | https://aicheckers.net (Vercel) |
| API | https://api.aicheckers.net (Cloudflare Tunnel → localhost:8000) |
| 速度 | 70-120ms (TTA有効時) |

## 技術スタック

```
DINOv3 (facebook/dinov3-vitb16-pretrain-lvd1689m)
    ↓
CLS Token (768次元) + Patch Stats (7次元)
    ↓
Linear Probe (nn.Linear(775, 2))
    ↓
TTA + Temperature Scaling (T=1.5)
```

### パッチ統計量 (Patch Stats)
| Index | 名前 | 説明 |
|-------|------|------|
| 0 | patch_mean | パッチAIスコアの平均 |
| 1 | patch_max | パッチAIスコアの最大 |
| 2 | patch_var | パッチAIスコアの分散 |
| 3 | max_minus_mean | 最大-平均（局所的突出度） |
| 4 | embed_var_mean | 埋め込み分散の平均 |
| 5 | count_high_score | スコア≥0.8のパッチ割合 |
| 6 | v_high_sim_85 | 垂直方向の高類似度パッチ比率 |

### 劣化Augmentation (2025-01-01 採用)
画質バイアス除去のため、Embedding抽出時に確率的に劣化を適用。

| 劣化タイプ | パラメータ |
|------------|------------|
| JPEG圧縮 | quality 30-70 |
| ガウシアンノイズ | std 5-25 |
| ダウンサンプリング | scale 50-80% |

**効果** (A/Bテスト結果):
- AI検出率: +2.68%
- Human正解率: +0.22%

**使用方法**: `--degradation-prob 0.5` をextract_embeddings_v2.pyに指定
