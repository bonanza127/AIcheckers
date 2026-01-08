# Ironclad Phase Scrambling 実装提案

**出典**: Gemini提案 (2026-01)
**ステータス**: 未実装

---

## 1. Phase Scrambling（位相攪乱）

DWTにおける「位相」= 各帯域内での**係数の配置パターン（構造情報）**

### 実装手法

#### A. 係数の符号反転
```python
# 秘密鍵ベースの符号制御
# 中周波（LH, HL）の各係数に対し、秘密鍵から生成したバイナリマスクで
# 一定割合の係数の正負（+/-）を反転

def apply_sign_flip(coeffs, key, flip_ratio=0.3):
    rng = np.random.default_rng(key)
    mask = rng.random(coeffs.shape) < flip_ratio
    coeffs[mask] *= -1
    return coeffs
```

#### B. 空間的シャッフル
```python
# 係数行列を2x2などの極小パッチ単位で秘密鍵ベースで入れ替え
# torch.roll や index_select を使用

def spatial_shuffle(coeffs, key, patch_size=2):
    # パッチ単位でシャッフル
    h, w = coeffs.shape
    patches = coeffs.reshape(h//patch_size, patch_size, w//patch_size, patch_size)
    # 秘密鍵でシャッフル順序を決定
    ...
```

### なぜ効くか

クリーナー（平滑化フィルタ）は「値の大きさ」を均そうとするが、
**「どこがプラスでどこがマイナスか」という配置の矛盾を直すアルゴリズムは一般的ではない**

---

## 2. 追加改善案

### 非対称な摂動制限（Constraint Clipping）

一律に `clamp(-0.06, 0.06)` ではなく、
**VAEが復元に失敗しやすい特定の周波数成分**に対してのみ強めに摂動を許容

```python
# 例: 中周波は強め、低周波は弱め
clamp_low = 0.02   # LL帯域
clamp_mid = 0.10   # LH, HL帯域
clamp_high = 0.06  # HH帯域
```

### 勾配の反転（Gradient Sign Inversion）

単純に類似度を下げるだけでなく、
**元の画像とは逆の性質を持つ特徴量**をわずかに混ぜ込む
→ 学習時に「逆方向の偽の特徴」を植え付ける

---

## 3. 不採用とした手法

| 手法 | 不採用理由 |
|------|-----------|
| garbage_latent | 破壊に特化しすぎ、画像が死ぬリスク |
| Rprop | 不安定、学習率スケジューラの方が高品質 |

**代替案**:
- Cosine Dissimilarity維持 + Lossの重みをエッジ部分に集中
- Adamで不安定なら学習率スケジューラを導入

---

## 4. 評価指標

LoRA学習後の検証で確認すべき3点:

1. **構造の崩れ**: 線が二重になる、手足が異常に増える
2. **テクスチャの砂嵐化**: 塗りの部分にノイズが浮き出る
3. **プロンプト無視**: 指定プロンプトが効かず、常に特定の「ゴミ」が出力される

---

## 5. 現在の検証結果 (2026-01-03)

| 指標 | Normal | Ironclad (DWTのみ) | 差 |
|------|--------|-------------------|-----|
| 学習Loss | 0.0805 | 0.111 | +38% |
| CLIPスコア | 0.2530 | 0.2550 | -0.79% |

**結論**: DWTのみでは学習を難しくするが、長時間学習で克服される
→ **Phase Scrambling追加が有効な可能性**
