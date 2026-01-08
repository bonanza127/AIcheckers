# AIイラストガード（保護ツール）

LoRA学習を妨害するための摂動技術を研究中。

**⚠️ 注意: Guard進捗バーはシミュレーション (2026-01-08)**
`src/app/guard/page.tsx` でSSE進捗を無効化し、22-24秒のシミュレーション進捗に変更。リアルタイム進捗に戻すには、`setGuardProgress`呼び出しのコメントアウトを解除。

---

## 現在のベスト: SAP v3

**スクリプト**: `scripts/sap_v3.py`

VAE+CLIP攻撃。視認性とVAE攻撃効果のバランスが最も良い。

```bash
# 1枚テスト（約1分）
modal run scripts/sap_v3.py --test --warp-magnitude 0.01 --iterations 50
```

### 攻撃構成

| 攻撃 | 手法 | 効果 |
|------|------|------|
| **VAE攻撃** | latent cos sim最小化 | 構造情報の破壊 |
| **CLIPネガティブ誘導** | "low quality, blurry, noise"に近づける | 低品質タグとの結合 |
| **CLIP概念混乱** | 元画像から離脱 + 無関係概念へ誘導 | 意味情報の汚染 |
| **適応型マスク** | エッジ5%、平坦1%（Sobel） | 視認性を維持しつつ攻撃強化 |
| **Micro-Warping** | 幾何学的変形（kornia elastic） | LightShed等の浄化耐性 |

### ベンチマーク結果

| 指標 | 値 | 評価 |
|------|-----|------|
| LPIPS | 0.0445 | ✅ 視覚差ほぼなし |
| VAE Cos Sim | **0.81** | ✅ 構造乖離 |
| CLIP to Original | **-0.26** | ✅ 負の値＝完全離脱 |
| CLIP to Negative | 0.28 | ✅ 低品質概念に接近 |
| 処理時間 | **56秒** | ✅ 1分以内 |

### ネガティブ概念リスト
```python
NEGATIVE_CONCEPTS = [
    "low quality, worst quality, blurry",
    "jpeg artifacts, noise, grainy",
    "text, watermark, signature",
    "error, glitch, corrupted",
]
```

### 混乱概念リスト
```python
CONFUSION_CONCEPTS = [
    "a photograph of mountains and trees",
    "3d render of geometric shapes",
    "satellite image of earth",
    "medical x-ray scan",
    "infrared thermal image",
]
```

---

## SAP v3 Variants (Perlin実験)

**スクリプト**: `scripts/sap_v3_variants.py`

v3の適応型マスクにPerlinノイズを導入。平坦部の摂動を空間的にバラけさせる。

```bash
# scale64 + 画像ハッシュベースシード（推奨）
modal run scripts/sap_v3_variants.py --test --perlin-scale 64
```

### 改良点

| 項目 | v3 | v3 Variants |
|------|-----|-------------|
| 平坦部マスク | 一様1% | Perlin 0.75〜1.5% |
| シード | 固定/ランダム | **画像ハッシュベース** |
| CLIPネガティブ | 5概念 | **8概念**（abstract texture等追加） |

### 推奨パラメータ

| パラメータ | 値 | 理由 |
|------------|-----|------|
| perlin_scale | **64** | scale128より攻撃効果が高い |
| perlin_seed | None（画像ハッシュ） | 再現可能 + 画像ごとに異なる |

### ベンチマーク結果 (scale64, image-hash seed)

| 指標 | 値 | 評価 |
|------|-----|------|
| LPIPS | 0.047 | ✅ 視覚差なし |
| VAE Cos Sim | 0.80 | ✅ 構造乖離 |
| CLIP to Original | -0.19 | ⚠️ v3(-0.26)より低下 |
| CLIP to Negative | 0.28 | ✅ |

### 設計思想

- **Perlinノイズ**: ホワイトノイズより浄化耐性・JPEG耐性が高い
- **画像ハッシュベースシード**: 攻撃パターンが画像ごとに異なり学習されにくい
- **scale64**: 細かすぎず粗すぎないバランス

---

## SAP v4 (アーカイブ: WD14実験)

**スクリプト**: `archive/sap_experiments/sap_v4.py`

WD14 Tagger攻撃を試みたが、視認性とのトレードオフが厳しく、v3の方がバランスが良いため保留。

**課題**: 平坦部へのWD14攻撃がノイズとして目立つ。知覚マスク等で改善を試みたが、VAE攻撃効果との両立が困難。

---

## スクリプト一覧

| スクリプト | 用途 | 状態 |
|------------|------|------|
| `sap_v3.py` | VAE+CLIP+Warping（ベースライン） | ✅ 使用中 |
| `sap_v3_variants.py` | **Perlin + 画像ハッシュシード** | ✅ 実験中 |
| `sap_v2.py` | VAE+CLIP+Warping（旧版） | 参考用 |
| `archive/sap_experiments/sap_v4.py` | WD14+VAE実験 | アーカイブ |

---

## 開発履歴

1. **highfreq_attack.py** - エッジ適応 + VAE攻撃（Cos Sim 0.92程度）
2. **sap_v2.py** - CLIP攻撃追加 + Micro-Warping
3. **sap_v3.py** - ネガティブ概念誘導 + 概念混乱追加（Cos Sim 0.81、CLIP離脱-0.26）★現行ベスト
4. **sap_v4.py** - WD14 Tagger攻撃実験（ノイズ問題でアーカイブ）

---

## Micro-Warping パラメータ

| パラメータ | 推奨値 | 説明 |
|------------|--------|------|
| warp_magnitude | 0.01 | 変形強度（0.01でぼやけなし） |
| kernel_size | (63, 63) | ぼかしカーネル |
| sigma | (12.0, 12.0) | ガウシアンσ |

**注意**: magnitude 0.015以上だと視覚的にぼやける

---

## 参考論文

### FastProtect (CVPR 2025) ★★★最推奨
- **論文**: https://arxiv.org/abs/2412.11423
- **開発**: NAVER WEBTOON AI
- **状態**: **独自実装完了** (2026-01-04)

#### 使用方法

```bash
# 学習（Modal上で実行）
modal run scripts/fastprotect_train.py --train --data-dir /vol/train_images --steps 40000

# 非同期投入（推奨）
modal run scripts/fastprotect_train.py --submit --data-dir /vol/train_images

# 推論テスト
modal run scripts/fastprotect_inference.py --test

# 画像保護
modal run scripts/fastprotect_inference.py --protect --input /vol/input --output /vol/output --use-warping
```

#### 関連ファイル

| ファイル | 用途 |
|----------|------|
| `scripts/fastprotect_train.py` | 摂動学習（40,000ステップ） |
| `scripts/fastprotect_inference.py` | 画像保護推論 |
| `lib/vae_hooks.py` | VAE中間層フック |
| `lib/mpl_loss.py` | Multi-Layer Protection Loss |

#### 学習パラメータ

| パラメータ | 値 | 備考 |
|------------|-----|------|
| num_steps | 40,000 | 論文準拠 |
| batch_size | 16 | A10G向け |
| lr | 0.0002 | Adam (β=0.5, 0.99) |
| η (摂動予算) | 8/255 | ~0.031 |
| λ (中間層重み) | 3.5×10⁻⁵ | 論文準拠 |
| K (クラスタ数) | 4 | Mixture-of-Perturbations |

#### 技術詳細

- **Multi-Layer Protection Loss**: VAE中間層（down_1〜3, mid_0）でも距離を最大化
- **Mixture-of-Perturbations**: K=4のクラスタごとに異なる摂動を学習
- **Adaptive Targeted Protection**: エントロピーベースでターゲット画像を選択
- **Adaptive Protection Strength**: LPIPS距離に基づき摂動強度を調整
- **Micro-Warping統合**: 浄化耐性のための幾何学的変形

### PAP (NeurIPS 2024)
- **論文**: https://arxiv.org/abs/2408.10571
- **弱点**: JPEG圧縮に弱い

### StyleGuard (NeurIPS 2025)
- **論文**: https://arxiv.org/abs/2505.18766
- **注意**: LoRAに対して効果が限定的

---

## Modal実験フォルダ

| フォルダ | 内容 |
|----------|------|
| train_normal | オリジナル画像 |
| train_sap_v2 | SAP v2攻撃済み |
| train_sap_v3 | SAP v3攻撃済み（最新） |
| train_hf_stealth | エッジ5% + 平坦1%攻撃済み（旧） |

---

## 今後の研究課題

1. **実際のLoRA学習テスト**: SAP v3攻撃画像でLoRA学習→生成品質の検証
2. **浄化耐性テスト**: LightShed、DiffPure等での浄化後も攻撃が残るか
3. **JPEG耐性**: SNS投稿時の再圧縮への耐性
4. **FastProtect統合**: コード公開後に高速化手法を取り込む
