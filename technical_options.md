# Technical Options - AI画像判別モデル向け精度向上テクニック集

このドキュメントは、DINOv3で実装・検証したテクニックを別モデルに移植する際のリファレンス。

---

## 現在実装済み（Production）

### 1. TTA (Test-Time Augmentation)

**フェーズ**: 推論時
**場所**: `backend/main.py` L533-547
**効果**: +0.5〜1%の精度向上

```python
# 実装
TTA_ENABLED = os.getenv("TTA_ENABLED", "true").lower() == "true"

# 推論時
ai_prob_original = model(image)
ai_prob_flipped = model(image.transpose(Image.FLIP_LEFT_RIGHT))
ai_prob = (ai_prob_original + ai_prob_flipped) / 2
```

**原理**:
- 水平反転画像で2回推論し、確率を平均化
- 反転に対して不変であるべき判定を安定化
- 速度影響: 約2倍（40-50ms増）

**適用条件**:
- ✅ 画像分類全般に有効
- ✅ バッチ推論でも適用可能
- ⚠️ 左右非対称が重要なタスクには不向き

---

### 2. Temperature Scaling

**フェーズ**: 推論時
**場所**: `backend/main.py` L58-59, L529
**効果**: 過信防止、50-60%付近の判定安定化

```python
# 実装
TEMPERATURE = float(os.getenv("TEMPERATURE", "1.5"))

# 推論時
probs = torch.softmax(logits / TEMPERATURE, dim=1)
```

**原理**:
- T > 1.0: 確率分布を平滑化（過信を抑制）
- T < 1.0: 確率分布を尖鋭化（自信を増幅）
- T = 1.0: 通常のsoftmax

**推奨値**:
| フェーズ | T値 | 用途 |
|---------|-----|------|
| 推論（通常） | 1.5 | 過信防止、キャリブレーション |
| 推論（攻め） | 1.0 | 高信頼度判定優先 |
| SS-VAT teacher | 0.02-0.04 | Sharpening（下記参照）|

**適用条件**:
- ✅ 確率出力のキャリブレーションに有効
- ✅ 検証データで最適値を調整推奨
- ⚠️ T < 1.0は学習済みモデルにのみ適用

---

### 3. VAT (Virtual Adversarial Training)

**フェーズ**: 学習時
**場所**: `scripts/train_from_embeddings.py`
**効果**: 決定境界の平滑化、汎化性能向上

```python
# 実装
vat_epsilon = 0.005  # 摂動半径
vat_alpha = 0.05 → 0.3  # 重み（線形ウォームアップ）

# Step 1: ランダム方向ノイズで勾配計算
d = torch.randn_like(x)
d = d / (d.norm(dim=1, keepdim=True) + 1e-8)
d.requires_grad = True

logits_perturbed = model(x + d * epsilon)
p_clean = F.softmax(model(x).detach(), dim=1)
p_perturbed = F.log_softmax(logits_perturbed, dim=1)
kl = F.kl_div(p_perturbed, p_clean, reduction='batchmean')
kl.backward()

# Step 2: 最悪方向への敵対的ノイズ
r_adv = d.grad / (d.grad.norm(dim=1, keepdim=True) + 1e-8) * epsilon

# Step 3: 敵対的摂動を加えて学習
loss = criterion(model(x + r_adv), y) + alpha * vat_loss
```

**原理**:
- 「モデルが最も迷う方向」に摂動を加えて学習
- 局所的なリプシッツ連続性を強制
- 決定境界付近の安定性向上

**パラメータ調整**:
| パラメータ | 推奨値 | 説明 |
|-----------|--------|------|
| ε (epsilon) | 0.005-0.01 | DINOv3など高品質embeddings向け |
| ε (epsilon) | 0.01-0.05 | 一般的なCNN特徴量向け |
| α (alpha) | 0.05→0.3 | 線形ウォームアップ推奨 |

**適用条件**:
- ✅ 教師あり/半教師あり学習どちらでも有効
- ✅ embeddings学習に特に効果的
- ⚠️ NaN発生時はεを下げる

---

### 4. Entropy Minimization

**フェーズ**: 学習時
**場所**: `scripts/train_from_embeddings.py`
**効果**: 曖昧な予測を減らし、自信を持った判定を促進

```python
# 実装
entropy_start_epoch = 15
entropy_alpha = 0.0 → 0.1  # 線形ウォームアップ
LOG_NUM_CLASSES = math.log(2)  # 正規化用

# エントロピー損失（正規化: 0.0〜1.0）
probs = F.softmax(logits, dim=1)
entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
entropy_loss = torch.mean(entropy) / LOG_NUM_CLASSES

loss = main_loss + alpha * entropy_loss
```

**原理**:
- 予測分布のエントロピーを最小化
- 「迷っている」予測を「確信」へ誘導
- log(C)で割って正規化（クラス数非依存）

**適用タイミング**:
- 中盤から投入（epoch 15〜）
- 最初から入れると早期収束のリスク

**適用条件**:
- ✅ 教師ありラベルが十分にある場合
- ✅ クラスバランスが均等な場合
- ⚠️ 不均衡データでは特定クラスへの偏りに注意

---

## 未実装（推奨・検討中）

### 5. Label Smoothing

**フェーズ**: 学習時
**状態**: 🟡 未実装（推奨）

```python
# 実装（簡単）
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

**原理**:
- ハードラベル [1, 0] → ソフトラベル [0.9, 0.1]
- モデルの「過信」を防止
- SS-VATへの「伸び代」を確保

**推奨値**: 0.1

**適用条件**:
- ✅ 教師あり学習全般に有効
- ✅ SS-VATの前段階として推奨
- ⚠️ 値を大きくしすぎると精度低下

---

### 6. Sharpening Temperature（SS-VAT用）

**フェーズ**: 学習時（teacher側のみ）
**状態**: 🔴 SS-VATで使用予定

```python
# teacher側（detached）でのみ使用
T_sharp = 0.02  # 極めて低い温度
p_teacher = F.softmax(teacher_logits / T_sharp, dim=1).detach()

# pseudo-label生成
pseudo_label = p_teacher.argmax(dim=1)
```

**原理**:
- T → 0 で argmax に近づく
- 決定境界を「カミソリの刃」のように鋭利化
- DINO系ではteacher側でsharpening、student側はマイルド

**⚠️ 絶対条件**:
| ❌ やってはいけない | ✅ 正しい使い方 |
|-------------------|----------------|
| student側の出力を直接sharpen | teacher/EMA/detached head側のみ |
| unlabeled全体に一律適用 | confidence filter併用 |
| VAT/consistencyより先に効かせる | EM/pseudo-label用 |

**適用条件**:
- ✅ 10万枚以上のfine-tuned済みモデル
- ✅ 表現空間が安定している場合
- ❌ SSL初期段階では狂気

---

### 7. Consistency Regularization（Strong/Weak Augmentation）

**フェーズ**: 学習時（unlabeledのみ）
**状態**: 🟡 SS-VATで最優先実装

```python
# Weak augmentation（軽い変換）
weak_aug = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
])

# Strong augmentation（強い変換）
strong_aug = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.1)),  # Cutout
    # JPEG compression（別途実装）
    # GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))
])

# Consistency Loss
x_weak = weak_aug(x)
x_strong = strong_aug(x)

with torch.no_grad():
    p_weak = F.softmax(model(x_weak) / T_sharp, dim=1)  # teacher側

p_strong = F.log_softmax(model(x_strong), dim=1)  # student側
L_cons = F.kl_div(p_strong, p_weak.detach(), reduction='batchmean')

loss = main_loss + lambda_cons * L_cons
```

**原理**:
- 同じ画像の weak/strong 変換で出力を一致させる
- AIイラスト判別の本質的特徴:
  - 微妙なテクスチャ統計
  - 塗りの一貫性
  - 局所構造の歪み
- strong augでこれらが部分的に壊れる → 本質的特徴だけが残る
- 表層ノイズへの依存を排除

**VAT との関係**:
| VAT | Consistency |
|-----|-------------|
| 微小・最悪方向の摂動 | 現実的な変形 |
| embedding空間で作用 | 画像空間で作用 |
| 両者は衝突しない、補完的 |

**推奨 Strong Augmentation（AIイラスト向け）**:
- `ColorJitter` - 色調変化
- `RandomErase / Cutout` - 局所情報欠落
- `JPEG compression` - 圧縮ノイズ
- `GaussianBlur (slight)` - 微細テクスチャ破壊

**パラメータ**:
| パラメータ | 推奨値 | 説明 |
|-----------|--------|------|
| λ_cons | 0.1〜0.3 | Consistency重み |
| 適用対象 | unlabeledのみ | labeledには不要 |

**実装コスト**: 低（augmentation + KL追加のみ）

---

### 8. パッチ別スコア分析（Patch-wise Score Analysis）

**フェーズ**: 推論時
**状態**: 🟡 未実装（推奨）

```python
# DINOv3は画像を197トークンに分割: [CLS] + 196パッチ (14×14)
# 現在は[CLS]のみ使用 → 全パッチを活用して局所的AI度を検出

# 推論時（オンライン計算、保存不要）
with torch.no_grad():
    # 全トークン出力を取得
    outputs = dino_model(image, return_dict=True)
    patch_features = outputs.last_hidden_state[:, 1:, :]  # [1, 196, 768]

    # パッチごとにAIスコア計算
    patch_logits = classifier(patch_features.squeeze(0))  # [196, 2]
    patch_scores = F.softmax(patch_logits, dim=1)[:, 1].numpy()  # [196]

    # 統計量計算
    mean_score = np.mean(patch_scores)
    max_score = np.max(patch_scores)
    variance = np.var(patch_scores)

    # 実務的指標
    max_minus_mean = max_score - mean_score
    practical_score = max_minus_mean + variance

# 出力例
{
    "ai_score": 0.85,           # 従来の[CLS]ベーススコア
    "patch_variance": 0.12,     # パッチ間ばらつき
    "max_minus_mean": 0.25,     # 局所的AI度の突出
    "lora_blend_flag": true     # 複数LoRA使用疑いフラグ
}
```

**原理**:
- 人間の絵: 全体的に均一なスコア分布（筆致が一貫）
- LoRA重ね合わせ: 顔は99%、背景は60%のようなムラが出る
- 「顔LoRA」「服LoRA」「背景LoRA」の境界でスコアが乖離

**統計量の選択**:
| 指標 | 用途 |
|------|------|
| Variance | 基本指標 |
| Max − Mean | 一部だけ異常にAIっぽい場合を検出 |
| Skewness（歪度） | LoRA顔が突出するケースに強い |
| Top-k mean − global mean | 上位10%パッチだけ異常か |

**実務的推奨**: `(max - mean) + variance` で十分効く

**顔検出は不要**:
- DINOパッチは空間構造を保持
- 高スコアパッチが自然に空間クラスタを形成
- 顔検出は可視化用に後から使えばいい

**ストレージ**:
- ❌ 全パッチembedding保存: 197 × 768 × 4bytes = 605KB/画像（現状の200倍）
- ✅ 推論時オンライン計算 → 最終スコア+フラグのみ保存

**適用条件**:
- ✅ 複数LoRA重ね合わせ検出に有効
- ✅ 追加ストレージ不要（オンライン計算）
- ⚠️ 推論時間が若干増加（パッチ分類 × 196）

---

### 9. バッチ内クラスバランス制約 H(Batch)

**フェーズ**: 学習時
**状態**: 🟡 クラス不均衡時に検討

```python
# バッチ全体の予測分布
batch_probs = F.softmax(logits, dim=1)
avg_probs = batch_probs.mean(dim=0)  # [C]

# バッチエントロピーを最大化（均等分布へ誘導）
batch_entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-8))
loss = main_loss - lambda_batch * batch_entropy  # マイナスで最大化
```

**原理**:
- 個別サンプルのEntropy Minimization（確信を持たせる）
- バッチ全体のEntropy Maximization（偏りを防ぐ）
- この2つを組み合わせてバランス維持

**適用条件**:
- ✅ クラス不均衡なデータセット
- ✅ EMを使う場合のセーフガード
- ⚠️ 現状（AI 51k vs Real 50k）ではほぼ不要

---

## パラメータ早見表

| テクニック | 適用フェーズ | 主要パラメータ | 推奨値 |
|-----------|-------------|---------------|--------|
| TTA | 推論 | - | 水平反転のみ |
| Temperature Scaling | 推論 | T | 1.5（通常）、0.02-0.04（SS-VAT teacher） |
| Patch-wise Analysis | 推論 | - | (max-mean)+variance ⭐LoRA対策 |
| VAT | 学習 | ε, α | ε=0.005-0.01, α=0.05→0.3 |
| Entropy Minimization | 学習 | α, start_epoch | α=0→0.1, start=15 |
| Label Smoothing | 学習 | smoothing | 0.1 |
| Consistency Reg. | 学習(unlabeled) | λ_cons | 0.1〜0.3 ⭐最優先 |
| H(Batch) | 学習 | λ | 0.1程度 |

---

## 新モデル導入時のチェックリスト

1. [ ] embedding抽出パイプライン構築
2. [ ] 分類ヘッド設計（Linear Probeで十分か？）
3. [ ] VAT適用（εの最適値を探索）
4. [ ] Temperature Scaling（検証データでTを調整）
5. [ ] TTA有効化（推論時）
6. [ ] Patch-wise Analysis（LoRA重ね合わせ対策）
7. [ ] Label Smoothing検討
8. [ ] クラスバランス確認（不均衡ならH(Batch)追加）
9. [ ] SS-VAT移行時: Consistency Regularization追加（最優先）

---

## 参考文献

- VAT: Miyato et al., "Virtual Adversarial Training" (2018)
- DINO: Caron et al., "Emerging Properties in Self-Supervised Vision Transformers" (2021)
- DINOv2/v3: Oquab et al., "DINOv2: Learning Robust Visual Features" (2023)
