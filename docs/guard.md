# AIイラストガード（画像保護ツール）

LoRA学習を妨害するための摂動技術。見た目はほぼ変わらないのに、AI学習に使うと品質が劣化する「毒」を画像に仕込む。

**最終更新**: 2026-01-23

---

## 防御スタック一覧

```
┌──────────────────────┬──────────────────────────────┬───────────────────────┐
│         手法         │          操作の本質          │     デフォルト値      │
├──────────────────────┼──────────────────────────────┼───────────────────────┤
│ FastProtect          │ ピクセル値にノイズを加算     │ strength=0.5          │
├──────────────────────┼──────────────────────────────┼───────────────────────┤
│ Micro-Warp           │ ピクセル位置を弾性変形       │ magnitude=0.004-0.007 │
├──────────────────────┼──────────────────────────────┼───────────────────────┤
│ Gamma Fluctuation    │ 明るさを局所的に変調         │ gamma_strength=0.03   │
├──────────────────────┼──────────────────────────────┼───────────────────────┤
│ Chromatic Aberration │ R/Bチャンネルを放射状にずらす│ magnitude=0.003       │
├──────────────────────┼──────────────────────────────┼───────────────────────┤
│ Hue Rotation         │ 色相を局所的に回転           │ max_degrees=2.0       │
└──────────────────────┴──────────────────────────────┴───────────────────────┘
```

---

## 仕組み（3行まとめ）

1. **VAE攻撃**: 画像をSDXLのVAEに通したとき、本来と違うlatentになるよう摂動を加える
2. **適応スケーリング**: 目立つ部分は弱く、目立たない部分は強く摂動をかける（LPIPS空間マップ）
3. **幾何学・色空間変形**: Micro-Warp、Chromatic Aberration、Hue Rotationで浄化ツール耐性を確保

---

## 本番システム構成

```
┌─────────────────────────────────────────────────────────────┐
│  Web UI (Guard ページ)                                       │
│  src/app/guard/page.tsx                                      │
└─────────────────────┬───────────────────────────────────────┘
                      │ POST /guard-stream
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  Backend API                                                 │
│  backend/main.py                                             │
│  └── MoonKnightV3 インスタンス                               │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  MoonKnight V3 推論エンジン                                  │
│  scripts/moonknight_v3.py                                    │
│                                                              │
│  必要モデル:                                                 │
│  └── models/fastprotect/                                     │
│      ├── checkpoint_step25000.pt  (学習済み摂動)             │
│      ├── kmeans_model.pkl         (クラスタリング)           │
│      └── target_entropies.json    (ターゲット選択用)         │
└─────────────────────────────────────────────────────────────┘
```

---

## MoonKnight V3 詳細仕様

### 処理フロー

```
入力画像 (任意サイズ)
    │
    ▼
512x512 にリサイズ
    │
    ▼
SDXL VAE で Encode → latent (64x64x4)
    │
    ▼
latent の Entropy (分散) を計算
    │
    ▼
Entropy に基づいてターゲット選択 (低/中/高)
    │
    ▼
K-means でクラスタ割り当て (K=4)
    │
    ▼
学習済み摂動を適用: delta_g(target) + Delta[target, k]
    │
    ▼
パッチLPIPS で「どこが目立つか」を空間マップ化
    │
    ▼
目立つ部分は弱く、目立たない部分は強くスケーリング
    │
    ▼
元解像度に Bicubic アップスケール
    │
    ▼
(Optional) Micro-Warping で微細変形
    │
    ▼
出力画像
```

### 主要パラメータ

| パラメータ | デフォルト | 説明 |
|------------|-----------|------|
| `strength` | 0.5 | 摂動強度（0.0〜1.0）。高いほど攻撃効果大だが視認性に影響 |
| `use_adaptive` | True | パッチLPIPSによる適応スケーリング |
| `use_warping` | True | Micro-Warping有効化 |
| `warp_magnitude` | None | 指定時は全画像で同じmagnitude。Noneならconfig依存 |

### Micro-Warping 多様化（2026-01-23）

画像ごとに3パターンからランダム選択（画像ハッシュベースで決定的）：

| パターン | kernel_size | sigma | magnitude | 特徴 |
|----------|-------------|-------|-----------|------|
| 細かい・弱い | 31 | 6.0 | 0.005 | 繊細な絵向け |
| 中程度 | 63 | 12.0 | 0.006 | 標準 |
| 粗い・強い | 95 | 18.0 | 0.007 | 浄化耐性重視 |

**選択ロジック**:
```python
img_bytes = image_tensor.numpy().tobytes()
seed = int(hashlib.sha256(img_bytes).hexdigest()[:8], 16)
config = warp_configs[seed % 3]
```

同じ画像なら常に同じconfigが選ばれる（再現可能）。

### Edge-Aware Warp（2026-01-23）

エッジ（輪郭線）付近のwarp強度を減衰させ、線画の崩れを防止。

| パラメータ | デフォルト | 説明 |
|------------|-----------|------|
| `use_edge_aware_warp` | True | Sobel法でエッジ検出し、warp減衰を適用 |
| `edge_avoid_strength` | 0.7 | エッジでの減衰率（0.0=減衰なし, 1.0=エッジ完全回避） |

**処理フロー**:
```python
Y = 0.299*R + 0.587*G + 0.114*B  # 輝度
edge_mag = kornia.filters.sobel(Y)
edge_mag = edge_mag / (edge_mag.max() + 1e-8)  # 正規化
attenuation = 1.0 - edge_avoid_strength * edge_mag
noise = noise * attenuation  # エッジ付近のwarpを抑制
```

### Chrominance-Only Warp（2026-01-23）

LAB色空間の色度チャンネル（a/b）のみに変形を適用し、輝度（L）を保持。

**根拠**: 人間の視覚は輝度に敏感だが色度には鈍感（Weber-Fechner則）。視認性への影響を最小化しつつ浄化耐性を維持。

| パラメータ | デフォルト | 説明 |
|------------|-----------|------|
| `chrominance_only_warp` | True | True時はLAB a/bのみにwarp適用 |

**処理フロー**:
```
RGB → LAB → Lを保存 → a,bにelastic transform → L + warped(a,b) → RGB
```

### CoupledTPS（2026-01-23）

Thin Plate Spline（薄板スプライン）による滑らかなグローバル変形。Elastic Transformがローカルな歪みを生成するのに対し、TPSは画像全体を滑らかに変形。

**特徴**:
- 少数の制御点（4×4=16点）で自然な変形
- 複数回（steps）適用することで多様性を確保
- デフォルトOFF（実験的機能）

| パラメータ | デフォルト | 説明 |
|------------|-----------|------|
| `use_coupled_tps` | False | TPSを適用するか |
| `tps_steps` | 2 | TPS適用回数（累積変形） |
| `tps_grid` | 4 | 制御点グリッドサイズ（4→16点） |
| `tps_magnitude` | 0.004 | 制御点の移動量（正規化座標） |
| `tps_margin` | 0.08 | 画像端からのマージン（端のアーティファクト防止） |

**処理フロー**:
```python
points_src = make_grid(4x4, margin=0.08)  # 正規化座標[0,1]
for _ in range(steps):
    offsets = randn() * magnitude
    points_dst = clamp(points_src + offsets, 0, 1)
    kernel, affine = get_tps_transform(points_dst, points_src)
    image = warp_image_tps(image, points_src, kernel, affine)
```

**Elastic Transform vs TPS**:
| 項目 | Elastic Transform | CoupledTPS |
|------|-------------------|------------|
| 変形タイプ | ローカル（ピクセル単位） | グローバル（制御点ベース） |
| 滑らかさ | ガウシアンフィルタ依存 | 数学的に滑らか |
| 計算量 | 軽い | やや重い |
| 用途 | 基本的な浄化耐性 | 高度な浄化ツール対策 |

### 低周波ガンマゆらぎ（2026-01-23）

輝度チャンネルに微細な明暗変化を追加。Denoise系ツールで消えにくい低周波成分。

| パラメータ | デフォルト | 説明 |
|------------|-----------|------|
| `use_gamma` | True | ガンマゆらぎを適用するか |
| `gamma_strength` | 0.03 | ガンマ変動幅（±3%） |

**内部パラメータ**:
- 低解像度ノイズ: 16×16 → Bicubicアップスケール
- LPIPS制御: 目立つ部分は80%減衰
- strength連動: protection strengthに応じて0.5〜1.5倍にスケール

**処理フロー**:
```
16x16ノイズ生成 → Bicubicアップスケール → gamma_map
RGB → 輝度Y計算 → Y' = Y^(1+gamma_map) → 輝度比でRGBスケール
```

### Chromatic Aberration（2026-01-23）

RGBチャンネルを放射状にシフト。カメラレンズの色収差を模倣。

| パラメータ | デフォルト | 説明 |
|------------|-----------|------|
| `use_chromatic_aberration` | True | 色収差を適用するか |
| `chromatic_magnitude` | 0.003 | シフト量（正規化座標） |

**処理フロー**:
```
中心からの距離r を計算
R: 外側方向へ r × magnitude シフト
G: 固定（基準）
B: 内側方向へ r × magnitude シフト
grid_sample で各チャンネルをリサンプル
```

### Hue Micro-Rotation（2026-01-23）

HSV色空間で色相を局所的に微回転。低周波ノイズマップで滑らかに変化。

| パラメータ | デフォルト | 説明 |
|------------|-----------|------|
| `use_hue_rotation` | True | 色相回転を適用するか |
| `hue_rotation_max_degrees` | 2.0 | 最大回転角度（±度） |

**処理フロー**:
```
RGB → HSV
8x8低周波ノイズ → Bicubicアップスケール → rotation_map
H' = (H + rotation_map × max_degrees/360) mod 1.0
HSV → RGB
※ 彩度 < 0.05 の部分は回転スキップ（グレー保護）
```

### Entropy計算

latentの分散で計算（学習時と統一）：
```python
z_flat = latent.view(B, -1)
entropy = z_flat.var(dim=1)
```

### MoP（Mixture of Perturbations）

- **K=4**: 画像をK-meansで4クラスタに分類
- **ターゲット3種**: 低/中/高エントロピー画像に向けて誘導
- **摂動構成**: `delta_g`（共通） + `Delta[k]`（クラスタ別） + `delta_t`（ターゲット別）
- 実装上は `delta_g` がターゲット別に保持されるため、`delta_g(target)` + `Delta[target,k]` の形

### バックエンド設定（backend/main.py）

```python
moonknight_engine = MoonKnightV3(
    model_dir="/home/techne/aicheckers/models/fastprotect",
    device="cuda",
    use_adaptive=True,
    use_warping=True,  # Micro-Warping有効
)

# 保護実行
protected_image = moonknight_engine.poison(image, strength=0.5)
```

---

## FastProtect 学習仕様

### 概要

摂動を「学習」しておき、推論時は適用するだけ。1枚あたり数秒で保護可能。

### 学習スクリプト

```bash
# Modal上で実行
modal run scripts/fastprotect_train.py --train --data-dir /vol/train_images --steps 40000
```

### 学習パラメータ

| パラメータ | 値 | 説明 |
|------------|-----|------|
| steps | 40,000 | 学習ステップ数 |
| batch_size | 16 | A10G向け |
| lr | 0.0002 | Adam (β=0.5, 0.99) |
| η | 8/255 (~0.031) | 摂動予算（L∞） |
| λ | 3.5×10⁻⁵ | Multi-Layer Loss重み |
| K | 4 | MoPクラスタ数 |

### Differentiable Augmentation（2026-01-23 強化）

学習中にランダム変換を適用し、圧縮耐性を向上：

| 変換 | 実装 | 説明 |
|------|------|------|
| **resize** | bilinear | 480-544px → 512px（微小リサイズ） |
| **jpeg** | **kornia RandomJPEG** | DCTベースの本物のJPEG圧縮シミュレーション（quality 60-90） |
| **crop** | 4-16px切り取り | 端のクロップ |

**JPEGシミュレーション改善**:
- 旧: ガウシアンブラーで近似
- 新: kornia `RandomJPEG`（DCTベースの微分可能JPEG）

### 出力ファイル

```
models/fastprotect/
├── checkpoint_step25000.pt   # 学習済み摂動
│   ├── delta_g              # ターゲット別摂動 (num_targets, 3, 512, 512)
│   ├── Delta                # クラスタ別摂動 (K, 3, 512, 512)
│   ├── K                    # クラスタ数
│   └── num_targets          # ターゲット数
├── kmeans_model.pkl          # K-meansモデル
└── target_entropies.json     # ターゲットエントロピー値
```

---

## 依存パッケージ

| パッケージ | 用途 | 必須 |
|-----------|------|------|
| torch | テンソル演算 | ✅ |
| diffusers | SDXL VAE | ✅ |
| lpips | 知覚距離計算 | ✅ |
| kornia | Micro-Warping, JPEG | ✅ |
| scikit-learn | K-means | ✅ |

---

## ベンチマーク目安

| 指標 | 目標値 | 説明 |
|------|--------|------|
| LPIPS | < 0.05 | 視覚差がほぼ分からない |
| VAE Cos Sim | < 0.85 | latent空間での乖離 |
| 処理時間 | < 10秒/枚 | 本番推論 |

---

## トラブルシューティング

### Micro-Warpingがスキップされる

```
[MoonKnight] Warning: kornia not found. Micro-warping skipped.
```

**解決**: `pip install kornia`

### MoP整合性エラー

```
ValueError: Checkpoint K (4) != KMeans n_clusters (3)
```

**原因**: 学習済みモデルとK-meansモデルの不整合

**解決**: 同じ学習セッションで生成されたファイルセットを使用

### 502エラー（バックエンド起動失敗）

**確認**: `journalctl --user -u aicheckers-backend -n 50`

**よくある原因**:
- kornia未インストール
- モデルファイル欠損
- GPU OOM

---

## 参考論文

| 論文 | 会議 | 概要 |
|------|------|------|
| **FastProtect** | CVPR 2025 | 本実装のベース。学習済み摂動による高速保護 |
| **CAT** | ICML 2025 | VAE攻撃を破る手法。要監視 |
| **GAP-Diff** | NDSS 2025 | JPEG耐性向上手法 |
| **DCT-Shield** | ICCV 2025 | 周波数ドメインでの保護 |

---

## 開発履歴

| 日付 | 内容 |
|------|------|
| 2026-01-04 | FastProtect独自実装完了 |
| 2026-01-08 | Guard UI進捗バーをシミュレーションに変更 |
| 2026-01-23 | MoonKnight V3: Entropy統一, MoP整合性チェック, パッチLPIPS, Micro-Warping多様化 |
| 2026-01-23 | FastProtect学習: kornia RandomJPEGによるJPEGシミュレーション強化 |
| 2026-01-23 | 低周波ガンマゆらぎ追加（denoise耐性向上） |
| 2026-01-23 | Edge-Aware Warp: Sobel法でエッジ付近のwarp減衰 |
| 2026-01-23 | Chrominance-Only Warp: LAB色空間でa/bのみ変形（視認性向上） |
| 2026-01-23 | CoupledTPS: 薄板スプラインによるグローバル変形（実験的、デフォルトOFF） |
| 2026-01-23 | Chromatic Aberration: RGBチャンネルの放射状シフト（幾何学的変形） |
| 2026-01-23 | Hue Micro-Rotation: 色相の局所回転（色空間変換） |

---

## 今後の課題

1. **CAT攻撃対策**: VAE-only保護の限界。CLIP/DINO併用検討
2. **JPEG耐性強化**: GAP-Diff統合
3. **実LoRA学習テスト**: 保護画像でLoRA学習→生成品質検証
4. **浄化耐性テスト**: LightShed, DiffPure等への耐性確認
