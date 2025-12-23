# 技術詳細

## モデルアーキテクチャ

### Moonlight V1.3（現行）
- **ベース**: DINOv3 ViT-B/16
- **分類器**: Linear Probe (774 → 2)
- **精度**: 98.40%

### 特徴量構成（774次元）
| 次元 | 内容 |
|------|------|
| 0-767 | CLS token（画像全体の表現） |
| 768 | patch_mean（パッチ平均AIスコア） |
| 769 | patch_max（最大AIスコア） |
| 770 | patch_var（スコアばらつき） |
| 771 | max_minus_mean（局所異常度） |
| 772 | embed_var_mean（テクスチャ均一性） |
| 773 | count_high_score（高スコア領域割合） |

### DINOv3トークン構造
```
[CLS, REG1-4, PATCH1-196]
  0     1-4      5-200
```
- パッチ抽出: `hidden_states[:, 5:5+196, :]`

## 推論パラメータ

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| TTA_ENABLED | true | 水平反転で2回推論→平均 |
| TEMPERATURE | 1.5 | softmax平滑化 |
| HIGH_SCORE_THRESHOLD | 0.8 | パッチ統計計算用 |

## verdict判定しきい値

| AIスコア | Verdict | 色 |
|----------|---------|-----|
| 80%以上 | AI DETECTED | 赤 |
| 60-80% | HIGH ALERT | 黄 |
| 40-60% | UNKNOWN | グレー |
| 20-40% | MINOR CAUTION | 青 |
| 20%未満 | HUMAN CONFIRMED | 緑 |

## 学習テクニック
- VAT (Virtual Adversarial Training): 全エポック適用
- Entropy Minimization: epoch 15から投入
- AdamW + CosineAnnealing

## 環境
- GPU: GTX 1660 (6GB VRAM)
- DINOv3 VRAM: 0.34GB
- 推論時間: 70-120ms（TTA有効時）
