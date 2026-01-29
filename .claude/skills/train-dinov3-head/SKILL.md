---
name: train-dinov3-head
description: Use when training DINOv3 Two-Head classifier heads, evaluating model quality with hardneg datasets, or selecting best seed from multi-seed experiments. Also use after training to update this skill with latest architecture changes.
---

# DINOv3 Two-Head Classifier Training

## ⚠️ PR-AUCを信じすぎない

| 指標 | 信頼度 |
|------|--------|
| val_pr_auc (0.99+) | 低（過信禁止） |
| **hardneg det@0.5** | **高（最重要）** |
| **hardneg p10** | **高（ロバスト性）** |

### Multi-Seed実験結果（2026-01-29）

| Seed | val_pr_auc | det@0.5 | p10 |
|------|------------|---------|-----|
| 7 | 0.9994 | 89.21% | 0.418 |
| 17 | 0.9995 | 89.02% | 0.397 |
| 27 | 0.9994 | 87.93% | 0.309 |
| **37** | 0.9994 | **91.29%** | **0.695** |
| 47 | 0.9995 | 89.46% | 0.442 |

**val_pr_aucは差なし。det@0.5で3.36%差。必ず複数seedで比較。**

---

## Multi-Seed学習

```bash
for seed in 7 17 27 37 47; do
    python3 scripts/train_and_save_candidate_b.py \
        --seed $seed --epochs 100 --patience 15 \
        --output-dir models/candidate_b_seed${seed} \
        > logs/candidate_b_seed${seed}.log 2>&1 &
done; wait
```

**選定基準:** det@0.5 > p10 > val_pr_auc

---

## 現行本番モデル（2026-01-29）

**candidate_b_nolbp_seed37 (v6): 1280d**

| 特徴量 | 次元 |
|--------|------|
| CLS | 768d |
| GPU (patch_stats_v3[1,3,5,6]) | 4d |
| cpu24 (v2:13d + v3:11d) | 24d |
| multi_layer_pstats (block 3,6,9,11) | 136d |
| hog_27, dct_65, patch_dist_256 | 348d |

**サブモデル (v5): 796d** = CLS 768d + GPU 4d + cpu24 24d

---

## 特徴量インデックス

```python
GPU_4D_IDX = [1, 3, 5, 6]  # adj_sim_var, patch_var, norm_var, norm_range
CPU_V2_13_IDX = [0, 1, 2, 4, 5, 7, 8, 9, 11, 12, 13, 14, 15]
CPU_V3_11_IDX = [0, 1, 2, 3, 4, 5, 8, 10, 15, 16, 17]
```

---

## 前処理（学習/推論共通）

```python
def _prep_block(arr):
    arr = np.clip(np.nan_to_num(arr, 0), -1e4, 1e4)  # sanitize
    if np.max(np.abs(arr)) > 200:
        arr = np.sign(arr) * np.log1p(np.abs(arr))  # log1p
    return arr
```

---

## 学習パラメータ

| パラメータ | 値 |
|-----------|-----|
| lr | 1e-3 (AdamW) |
| weight_decay | 1e-5 |
| epochs/patience | 100/15 |
| scheduler | ReduceLROnPlateau(0.5, 5) |

---

## ファイル

| パス | 用途 |
|------|------|
| `scripts/train_and_save_candidate_b.py` | 本番学習 |
| `scripts/train_28d_plus_60.py` | サブモデル |
| `lib/extended_features.py` | 拡張特徴量 |

---

## 禁止事項

- ❌ PR-AUCだけで判断
- ❌ 単一seedで終了
- ❌ hardneg評価スキップ
- ❌ 学習/推論で異なる前処理
- ❌ **学習後にスキル更新せず放置**

---

## チェックリスト

- [ ] 複数seed（5+）で学習
- [ ] det@0.5で比較・選定
- [ ] config.json保存
- [ ] backend/main.py更新
- [ ] バックエンド再起動
- [ ] 本番動作確認
- [ ] **このスキルを更新**

---

## 🔄 学習後のスキル更新（必須）

**このスキルは常に最新を維持。**

| 変更 | 更新箇所 |
|------|----------|
| seed結果 | Multi-Seed実験結果テーブル |
| モデル構成 | 現行本番モデル |
| 特徴量 | 特徴量インデックス |
| パラメータ | 学習パラメータ |

**古いスキル → 次回学習で誤設定リスク。**
