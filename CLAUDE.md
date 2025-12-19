# AIcheckers - AI Anime Image Detector

## 概要
アニメ絵特化のAI生成画像判別ツール。日本市場向け。

## 本番URL
- **フロントエンド**: https://aicheckers.net (Vercel)
- **API**: https://api.aicheckers.net (Cloudflare Tunnel → localhost:8000)

## 現在のアーキテクチャ（2024-12-19）

```
aicheckers.net
    ↓
┌─────────────────────────────────────┐
│ メイン: AniXplore (Modal T4)       │ ← F1: 0.9999、コールドスタート25秒、ウォーム2秒
│ フォールバック: legekka (ローカル)  │ ← 汎化性能高、Pony系に強い
└─────────────────────────────────────┘
```

**モデル切り替え**: UI上でクリックして手動切り替え可能

## モデル比較

| モデル | 精度 | 強み | 弱み |
|--------|------|------|------|
| AniXplore | F1: 0.9999 | AnimeDL-2M学習データに強い | Pony/FLUX等2024年後半モデルに弱い |
| legekka | 94.68% | 汎化性能高、未知モデルにも対応 | 全体的な精度はAniXploreに劣る |

**発見**: AniXploreは学習データ外に弱い。legekkaの方が未知のAIモデルに対応できる。

## ファイル構成

```
aicheckers/
├── src/app/              # Next.js フロントエンド
├── backend/
│   └── main.py           # FastAPI（AniXplore+legekka統合済み）
├── modal_anixplore/      # Modal用AniXploreデプロイ
│   ├── app.py
│   ├── anixplore_model.py
│   └── test_image.py
└── models/AniXplore/     # チェックポイント（gitignore）
```

## コマンド

```bash
# バックエンド
systemctl --user restart aicheckers-backend
journalctl --user -u aicheckers-backend -f

# Modal
cd ~/aicheckers/modal_anixplore
modal deploy app.py
python3 test_image.py /path/to/image.png
```

## 今後の方針

### 優先度高
1. **legekkaをAnimeDL-2Mでファインチューニング**
   - コスト: $5-15（Modal T4で10-20時間）
   - 効果: 確実に精度向上
   - legekkaの汎化性能 + AnimeDL-2Mの最新データ = 最強の可能性

### 検討中
- Ateeqq/ai-vs-human-image-detector追加（FLUX/SD3.5対応、Apache 2.0）
- 複数モデルのアンサンブル（多数決）

### 見送り
- AniXploreのファインチューニング（複雑すぎる）
- GradCAM（Attention Mapで十分）
- AIDE（商用禁止）

## 技術メモ

- Modal $30/月無料枠（毎月リセット）
- T4 GPU: $0.59/時間
- keep_warm=1で常時起動可（~$15-20/月消費）
- AniXploreチェックポイントはzip圧縮必須（torch.load用）

## 参考リンク
- AnimeDL-2M: https://github.com/FlyTweety/AnimeDL2M
- Ateeqq: https://huggingface.co/Ateeqq/ai-vs-human-image-detector
- Modal: https://modal.com/
