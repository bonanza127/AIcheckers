---
name: train-dinov3-head
description: DINOv3 Two-Head分類器の学習スキル。Multi-Seed学習とhardneg評価を重視。PR-AUCを信じすぎない。
---

# DINOv3 Two-Head Classifier Training

## ⚠️ 重要警告: PR-AUCを信じすぎない

**Validation PR-AUC（0.99+）は汎化性能を示さない。**

| 指標 | 説明 | 信頼度 |
|------|------|--------|
| val_pr_auc | 学習時のホールドアウト精度 | 低（過信禁止） |
| **hardneg det@0.5** | 未知の難しいAI画像での検出率 | **高（最重要）** |
| **hardneg p10** | 下位10%の確信度 | **高（ロバスト性指標）** |

### Multi-Seed実験で判明した事実

同じハイパーパラメータでもseedによってhardneg性能が大きく変動する:

| Seed | val_pr_auc | hardneg det@0.5 | hardneg p10 |
|------|------------|-----------------|-------------|
| 7    | 0.9994     | 89.21%          | 0.418       |
| 17   | 0.9995     | 89.02%          | 0.397       |
| 27   | 0.9994     | 87.93%          | 0.309       |
| **37** | 0.9994   | **91.29%**      | **0.695**   |
| 47   | 0.9995     | 89.46%          | 0.442       |

- val_pr_aucは0.9994〜0.9995で**差がない**
- hardneg det@0.5は87.93%〜91.29%で**3.36%の差**
- hardneg p10は0.309〜0.695で**2倍以上の差**

**結論: 必ず複数seedで学習し、hardneg検出率で最良を選ぶ**

---

## Multi-Seed学習手順

### 1. 学習の実行

```bash
# 5つのseedで学習（並列推奨）
for seed in 7 17 27 37 47; do
    python3 scripts/train_and_save_candidate_b.py \
        --seed $seed \
        --epochs 100 \
        --patience 15 \
        --output-dir models/candidate_b_seed${seed} \
        > logs/candidate_b_seed${seed}.log 2>&1 &
done
wait
```

### 2. 結果の比較

各seedの結果を比較:

```bash
for seed in 7 17 27 37 47; do
    echo "=== Seed $seed ==="
    python3 -c "
import json
with open('models/candidate_b_seed${seed}/training_result.json') as f:
    d = json.load(f)
    print(f\"val_pr_auc: {d.get('val_pr_auc', 'N/A')}\")
    if 'hardneg_eval' in d:
        h = d['hardneg_eval']
        print(f\"det@0.5: {h.get('det_at_0.5', 'N/A')}\")
        print(f\"p10: {h.get('p10', 'N/A')}\")
"
done
```

### 3. 最良seedの選定

**選定基準（優先順位）:**
1. hardneg det@0.5 が最大
2. 同等なら hardneg p10 が最大
3. val_pr_auc は参考程度（差がつかない）

### 4. 本番採用

```bash
# 例: seed37が最良の場合
cp -r models/candidate_b_seed37 models/candidate_b_production
# backend/main.pyのTWO_HEAD_DIR_MAINを更新
systemctl --user restart aicheckers-backend
```

---

## アーキテクチャ

### 現行本番モデル: candidate_b_nolbp_seed37 (v6)

**入力次元: 1280d**

| 特徴量 | 次元数 | 説明 |
|--------|--------|------|
| CLS | 768d | DINOv3最終層のCLSトークン |
| GPU | 4d | patch_stats_v3から選択 |
| cpu24 | 24d | cpu_stats_v2(13d) + cpu_stats_v3(11d) |
| multi_layer_pstats | 136d | block 3,6,9,11のpatch_stats_v3 |
| hog_27 | 27d | HOG特徴量 |
| dct_65 | 65d | DCT特徴量 |
| patch_dist_256 | 256d | パッチ分布top256 |

**合計: 768 + 4 + 508 = 1280d**

### サブモデル: two_head_28d_plus_60 (v5)

**入力次元: 796d**
- CLS: 768d
- GPU: 4d
- CPU: 24d（cpu24のみ）

---

## 特徴量インデックス定義

```python
# GPU特徴量（patch_stats_v3から4次元選択）
GPU_4D_IDX = [1, 3, 5, 6]
# adj_sim_var, patch_var, norm_var, norm_range

# CPU特徴量 Part1（cpu_stats_v2から13次元選択）
CPU_V2_13_IDX = [0, 1, 2, 4, 5, 7, 8, 9, 11, 12, 13, 14, 15]

# CPU特徴量 Part2（cpu_stats_v3_20dから11次元選択）
CPU_V3_11_IDX = [0, 1, 2, 3, 4, 5, 8, 10, 15, 16, 17]
```

---

## 前処理関数

学習時と推論時で同じ前処理を適用する必要がある:

```python
def _sanitize_np(x: np.ndarray, clip: float = 1e4) -> np.ndarray:
    """NaN/Infを0に、値を±clipにクリップ"""
    x = np.asarray(x, dtype=np.float32)
    bad = ~np.isfinite(x)
    if bad.any():
        x = x.copy()
        x[bad] = 0.0
    return np.clip(x, -clip, clip)

def _maybe_log1p_np(x: np.ndarray, absmax_thresh: float = 200.0) -> np.ndarray:
    """absmax > threshなら符号付きlog1p変換"""
    absmax = float(np.max(np.abs(x))) if x.size else 0.0
    if absmax <= absmax_thresh:
        return x
    return np.sign(x) * np.log1p(np.abs(x))

def _prep_block(arr: np.ndarray) -> np.ndarray:
    """学習/推論共通の前処理"""
    return _maybe_log1p_np(_sanitize_np(arr))
```

---

## 学習パラメータ

| パラメータ | 値 | 備考 |
|-----------|-----|------|
| lr | 1e-3 | AdamW |
| weight_decay | 1e-5 | |
| epochs | 100 | max |
| patience | 15 | early stopping |
| batch_size | 256 | |
| val_split | 0.1 | |
| scheduler | ReduceLROnPlateau | factor=0.5, patience=5 |

---

## モデル保存形式

```
models/candidate_b_seed37/
├── model.pt          # state_dict
└── config.json       # ハイパーパラメータ
```

config.json例:
```json
{
  "cls_dim": 768,
  "gpu_dim": 4,
  "cpu_dim": 508,
  "hidden_dim": 256,
  "dropout": 0.3,
  "seed": 37
}
```

---

## ファイル構成

| パス | 説明 |
|------|------|
| `scripts/train_and_save_candidate_b.py` | 本番学習スクリプト |
| `scripts/train_28d_plus_60.py` | サブモデル学習スクリプト |
| `lib/extended_features.py` | 拡張特徴量計算（リアルタイム推論用） |
| `lib/patch_stats.py` | GPU統計量計算 |
| `lib/cpu_stats.py` | CPU統計量計算 |
| `embeddings/aibooru_hardneg*.npy` | hardneg評価データ |

---

## 比較表テンプレート

学習後、以下の形式で結果を記録:

| Seed | val_pr_auc | hardneg det@0.5 | hardneg p10 | 備考 |
|------|------------|-----------------|-------------|------|
| 7    |            |                 |             |      |
| 17   |            |                 |             |      |
| 27   |            |                 |             |      |
| 37   |            |                 |             |      |
| 47   |            |                 |             |      |

---

## 禁止事項

- ❌ PR-AUCだけで学習成功と判断する
- ❌ 単一seedで学習を終える
- ❌ hardneg評価をスキップする
- ❌ 学習時と推論時で異なる前処理を使う
- ❌ 新しいテストスクリプトを作る（既存を使う）

---

## チェックリスト

- [ ] 複数seed（最低5つ）で学習した
- [ ] hardneg det@0.5で比較した
- [ ] 最良seedを選定した
- [ ] config.jsonを保存した
- [ ] backend/main.pyのモデルパスを更新した
- [ ] バックエンドを再起動した
- [ ] 本番環境で動作確認した
