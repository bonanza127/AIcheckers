# AIcheckers Enterprise API

企業向けAPI仕様書。レート制限なし、使用量ベース課金に対応。

---

## 認証

すべてのリクエストに `X-API-Key` ヘッダーを付与してください。

```bash
curl -X POST https://api.aicheckers.net/analyze \
  -H "X-API-Key: aicheckers_ent_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" \
  -F "file=@image.png"
```

---

## エンドポイント

### 1. 画像解析 (ファイルアップロード)

**POST** `/analyze`

画像ファイルをアップロードしてAI生成判定を行います。

**リクエスト:**
```bash
curl -X POST https://api.aicheckers.net/analyze \
  -H "X-API-Key: YOUR_API_KEY" \
  -F "file=@image.png"
```

**レスポンス:**
```json
{
  "is_ai": true,
  "ai_score": 87.5,
  "human_score": 12.5,
  "confidence": 75.0,
  "verdict": "AI DETECTED",
  "processing_time": 0.095,
  "filename": "image.png",
  "model_used": "Moonlight",
  "attention_map": "base64_encoded_image...",
  "forensic_logs": [
    "TTA検証: 元画像 88.2% ↔ 反転画像 86.8% → 統合値 87.5%",
    "注目パターン: ヘッド多様性50%（6/12）、中央集中68%",
    "..."
  ],
  "detected_traces": "マルチヘッドの6個が単一の特徴量に収束。AI特有の画一的な演算パターンを検出"
}
```

---

### 2. 画像解析 (URL指定)

**POST** `/analyze-url`

画像URLを指定してAI生成判定を行います。

**リクエスト:**
```bash
curl -X POST https://api.aicheckers.net/analyze-url \
  -H "X-API-Key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/image.png"}'
```

**レスポンス:** `/analyze` と同じ形式（`source_url` が追加）

---

### 3. APIキー検証

**GET** `/enterprise/verify`

APIキーの有効性と今月の使用量を確認します。

**リクエスト:**
```bash
curl https://api.aicheckers.net/enterprise/verify \
  -H "X-API-Key: YOUR_API_KEY"
```

**レスポンス:**
```json
{
  "valid": true,
  "company_name": "Example Corp",
  "plan": "standard",
  "expires_at": "2026-01-01T00:00:00",
  "current_month_usage": 1234
}
```

---

## レスポンス詳細

### verdict (判定結果)

| verdict | ai_score範囲 | 説明 |
|---------|-------------|------|
| `AI DETECTED` | 80-100% | AI生成と判定 |
| `HIGH ALERT` | 60-80% | AI生成の可能性が高い |
| `MIDDLE CAUTION` | 40-60% | どちらとも言えない |
| `LOW SIMILARITY` | 20-40% | 人間作成の可能性が高い |
| `HUMAN CONFIRMED` | 0-20% | 人間作成と判定 |

### attention_map

Base64エンコードされたPNG画像。モデルが注目した領域をヒートマップで可視化。

### forensic_logs

AI判定の根拠となる技術的分析ログ（配列）。

### detected_traces

AI痕跡または人間らしさの検出サマリー（文字列）。

---

## エラーレスポンス

```json
{
  "detail": "エラーメッセージ"
}
```

| HTTPステータス | 説明 |
|--------------|------|
| 400 | 無効なリクエスト（画像形式エラー等） |
| 401 | 無効または期限切れのAPIキー |
| 500 | サーバー内部エラー |

---

## レート制限

Enterprise APIにはレート制限がありません。

ただし、過度なリクエストはサーバー負荷軽減のため制限される場合があります。
推奨: 並列リクエストは10件以下

---

## 使用量と課金

- 使用量は月次で集計されます
- 詳細な使用量レポートは管理者経由で提供されます

---

## サポート

技術的な問題やAPIキーに関するお問い合わせ:
- Email: support@aicheckers.net (準備中)

---

## 変更履歴

- 2026-01-02: Enterprise API v1.0 リリース
