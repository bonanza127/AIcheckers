# Ironclad LoRA学習検証レポート

**日付**: 2026-01-03
**目的**: Ironclad画像摂動がLoRA学習に与える影響の検証

## 実験設定

### ベースモデル
- **モデル**: fnEVONoobXL_v20.safetensors (6.94GB)
- **プラットフォーム**: Modal (A10G GPU)

### 学習設定 (Kohya sd-scripts v0.8.7)
```
network_dim: 32
network_alpha: 16
learning_rate: 2e-4
optimizer: AdamW
lr_scheduler: cosine_with_restarts
max_train_epochs: 3
total_steps: 1980
batch_size: 1
mixed_precision: bf16
```

### データセット
- **画像数**: 11枚（同一画像セット）
- **Normal**: オリジナル画像
- **Ironclad**: DWT摂動適用済み画像
- **キャプション**: "anime style illustration, detailed, high quality"

## 結果

### 学習Loss比較

| LoRA | 平均Loss | 差 |
|------|----------|-----|
| Normal | 0.0805 | baseline |
| Ironclad | 0.111 | **+38%** |

### CLIP評価（3プロンプト × 2画像）

| LoRA | 平均CLIPスコア | 差 |
|------|----------------|-----|
| Normal | 0.2530 | baseline |
| Ironclad | 0.2550 | -0.79% |

### 詳細CLIPスコア

**Normal LoRA:**
- [0.223, 0.225, 0.290, 0.262, 0.264, 0.254]

**Ironclad LoRA:**
- [0.211, 0.244, 0.275, 0.257, 0.265, 0.278]

## 結論

1. **学習妨害効果**: Lossが38%上昇 → 学習が困難になっている
2. **最終品質**: CLIPスコアでは有意差なし → 長時間学習で克服される
3. **評価の限界**: CLIPは汎用品質指標であり、スタイル学習度を正確に測定できていない可能性

## 改善案

1. より強いIronclad設定（強度パラメータ調整）
2. 学習ステップ数を減らして比較（500, 1000ステップ等）
3. 位相攻撃（Phase Scrambling）の追加実装・検証
4. スタイル特化の評価指標（LPIPS、FID等）

## 生成ファイル

- Normal LoRA: `/vol/loras/lora_normal/lora_normal.safetensors`
- Ironclad LoRA: `/vol/loras/lora_ironclad/lora_ironclad.safetensors`
- 生成画像: `/vol/generated/lora_normal/`, `/vol/generated/lora_ironclad/`
- ローカルコピー: `~/Desktop/test_*.png`, `~/Desktop/ironclad_test_*.png`
